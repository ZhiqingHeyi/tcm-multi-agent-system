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
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        response = await client.post(_endpoint(), headers=_headers(), json=payload)
        response.raise_for_status()
        data = response.json()
    return (data["choices"][0]["message"]["content"] or "").strip()

async def chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.3, role: str = "pro") -> dict:
    raw = await chat(system_prompt, user_prompt, temperature=temperature, role=role)
    return parse_json_block(raw)

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
