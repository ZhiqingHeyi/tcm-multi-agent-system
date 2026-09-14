import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from ..config import settings

class LLMUnavailable(Exception):
    pass

def _endpoint() -> str:
    base = (settings.llm_base_url or "").rstrip("/")
    if not base:
        raise LLMUnavailable("未配置 OpenAI 兼容接口地址")
    if not settings.llm_api_key:
        raise LLMUnavailable("未配置 API Key")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"

def is_configured() -> bool:
    return bool(settings.llm_base_url and settings.llm_api_key and settings.llm_model_fast and settings.llm_model_pro)

def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}

def _model_for(role: str = "pro") -> str:
    return settings.llm_model_pro if role == "pro" else settings.llm_model_fast

async def chat(system_prompt: str, user_prompt: str, temperature: float = 0.4, max_tokens: int = 1200, role: str = "pro") -> str:
    payload = {
        "model": _model_for(role),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=15.0)) as client:
        response = await client.post(_endpoint(), headers=_headers(), json=payload)
        response.raise_for_status()
        data = response.json()
    choices = data.get("choices")
    if not choices:
        raise LLMUnavailable(f"网关返回异常响应（无 choices 字段）：{str(data)[:200]}")
    return (choices[0]["message"]["content"] or "").strip()

CONCISE_HINT = (
    "\n\n注意：上一次输出在传输中被截断，导致 JSON 不完整。请务必精简作答，"
    "直接输出完整可解析的 JSON，各字段控制在 80 字以内，不要输出多余解释。"
)


async def chat_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    role: str = "pro",
    max_tokens: int = 4096,
    attempts: int = 3,
) -> dict:
    """调用模型并解析 JSON，失败时最多重试三次（指数退避）。

    上游网关 vectide.cn 实测有两种偶发故障，且都不稳定：
    1. 连接层 ConnectError（表现为返回 0 字符 content）；
    2. 响应被中途切断，JSON 不完整（JSONDecodeError: Unterminated string）。
    二者都是瞬时故障，单次重试不足以救回。此前这些异常被静默吞掉并降级为规则引擎，
    导致六派辨证里混入"证据不足"的兜底结论却毫无提示——现在会打印日志、退避重试，
    仍失败才向上抛出，由调用方决定是否兜底。
    """
    prompt = user_prompt
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = await chat(system_prompt, prompt, temperature=temperature, max_tokens=max_tokens, role=role)
            return parse_json_block(raw)
        except (ValueError, KeyError, IndexError, httpx.HTTPError, LLMUnavailable) as exc:
            last_error = exc
            print(f"[llm] 第 {attempt}/{attempts} 次调用失败（{type(exc).__name__}: {str(exc)[:80]}），退避后重试", flush=True)
            if attempt < attempts:
                await asyncio.sleep(1.5 * attempt)
                prompt = user_prompt + CONCISE_HINT
    raise last_error if last_error else ValueError("模型未返回可解析的 JSON")

def parse_json_block(raw: str) -> dict:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)

async def chat_stream(system_prompt: str, user_prompt: str, temperature: float = 0.5, role: str = "fast") -> AsyncIterator[str]:
    payload = {
        "model": _model_for(role),
        "temperature": temperature,
        "stream": True,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        async with client.stream("POST", _endpoint(), headers=_headers(), json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    delta = json.loads(chunk)["choices"][0]["delta"].get("content", "")
                except (ValueError, KeyError, IndexError):
                    delta = ""
                if delta:
                    yield delta

async def test_connection(role: str = "pro") -> str:
    return await chat("你是一个连通性测试助手。", "只回复两个字：正常", temperature=0, max_tokens=10, role=role)
