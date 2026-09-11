"""向量化层。

现实约束：当前接入的 OpenAI 兼容网关只开放了对话模型，没有 embedding 模型
（实测 /v1/models 仅 2 个 chat 模型，embedding 调用返回权限错误）。
因此这里采用「远程优先 + 本地兜底」的双通道设计：

- 配置了 embedding 模型（settings.embedding_model）时走远程 API；
- 否则使用本地哈希向量化（hashing vectorizer）。

本地向量的定位要说清楚：它**不是**语义嵌入，而是一个带次线性词频加权的
哈希词袋向量。它提供的是"字面近似"的召回通道。系统的精确召回主要交给
稀疏侧（bigram BM25 + GIN 索引）和后续的 LLM 重排，这是有意的架构分工：
稠密侧给语义泛化，稀疏侧给方名条文精确匹配，LLM 重排做最终裁决。
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re

import httpx

from ..config import settings

BATCH_SIZE = 32
MAX_RETRIES = 3
RETRY_BASE_DELAY = 0.8

ASCII_WORD_RE = re.compile(r"[a-z0-9]+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SKIP_CHARS = set(" \t\n\r，。、；：！？（）「」『』《》【】·—…-,.;:!?()[]{}\"'")


class EmbeddingUnavailable(Exception):
    pass


def is_remote_configured() -> bool:
    return bool(settings.embedding_model and (settings.embedding_api_key or settings.llm_api_key))


def _endpoint() -> str:
    base = (settings.embedding_base_url or settings.llm_base_url or "").rstrip("/")
    if not base:
        raise EmbeddingUnavailable("未配置 Embedding 接口地址")
    return base if base.endswith("/embeddings") else f"{base}/embeddings"


def _api_key() -> str:
    return settings.embedding_api_key or settings.llm_api_key


async def embed_remote(texts: list[str]) -> list[list[float]]:
    payload = {"model": settings.embedding_model, "input": texts}
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=8.0)) as client:
        response = await client.post(_endpoint(), headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    items = sorted(data["data"], key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


# --------------------------------------------------------------------------
# 本地哈希向量化
# --------------------------------------------------------------------------
def tokenize_features(text: str) -> list[str]:
    """中文取一元 + 二元词，英文数字取整词。比纯 bigram 多保留单字辨识度。"""
    tokens: list[str] = []
    cjk_run: list[str] = []

    def flush() -> None:
        if not cjk_run:
            return
        tokens.extend(cjk_run)
        tokens.extend(cjk_run[i] + cjk_run[i + 1] for i in range(len(cjk_run) - 1))
        cjk_run.clear()

    for char in text.lower():
        if CJK_RE.match(char):
            cjk_run.append(char)
        else:
            flush()
            if char not in SKIP_CHARS and not char.isspace():
                continue
    flush()
    tokens.extend(ASCII_WORD_RE.findall(text.lower()))
    return tokens


def local_hash_embedding(text: str, dim: int | None = None) -> list[float]:
    dim = dim or settings.embedding_dim
    weights = [0.0] * dim
    counts: dict[str, int] = {}
    for token in tokenize_features(text):
        counts[token] = counts.get(token, 0) + 1

    for token, tf in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        weights[index] += sign * (1.0 + math.log(tf))  # 次线性词频，抑制高频字主导

    norm = math.sqrt(sum(w * w for w in weights)) or 1.0
    return [w / norm for w in weights]


def _fit_dim(vector: list[float]) -> list[float]:
    dim = settings.embedding_dim
    if len(vector) == dim:
        trimmed = vector
    elif len(vector) > dim:
        trimmed = vector[:dim]
    else:
        trimmed = vector + [0.0] * (dim - len(vector))
    norm = math.sqrt(sum(w * w for w in trimmed)) or 1.0
    return [w / norm for w in trimmed]


# --------------------------------------------------------------------------
# 对外入口：分批 + 指数退避重试
# --------------------------------------------------------------------------
async def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if not is_remote_configured():
        return [local_hash_embedding(text) for text in texts]

    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vectors.extend(await _embed_batch_with_retry(batch))
    return vectors


async def _embed_batch_with_retry(batch: list[str]) -> list[list[float]]:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return [_fit_dim(vector) for vector in await embed_remote(batch)]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    # 远程不可用时降级为本地向量，保证入库不中断（可观测：日志会打印原因）
    print(f"[embeddings] 远程向量化失败，降级本地向量：{last_error}")
    return [local_hash_embedding(text) for text in batch]
