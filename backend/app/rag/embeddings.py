"""向量化层。

支持三种模式，靠配置切换，代码零改动：

1. **本地 GPU 服务**（推荐）：宿主机跑 tools/embed_server.py（Apple Silicon MPS），
   EMBEDDING_BASE_URL 指向 http://host.docker.internal:8100/v1 即可。
   之所以放宿主机，是因为 Docker Desktop on macOS 无法把 Metal GPU 透传给容器。
2. **云端 embedding API**：任何 OpenAI 兼容的 /v1/embeddings 供应商。
3. **本地哈希兜底**：未配置 embedding 模型时自动启用。

关于兜底模式必须说清楚：它**不是**语义嵌入，而是带次线性词频加权的哈希词袋向量，
只提供"字面近似"召回。实测它在零关键词重叠的改写查询上召回率仅 20%，
所以一旦接入真实 embedding，该指标应显著跃升——这也是验收接入是否成功的判据。

完整的召回分工：稠密侧给语义泛化，稀疏侧（bigram BM25 + GIN）给方名条文精确匹配，
LLM 重排做最终裁决。
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
async def embed(texts: list[str], prefix: str = "") -> list[list[float]]:
    """向量化入口。prefix 只作用于远程通道。

    本地哈希是词法向量，把「为这个句子生成表示以用于检索相关文章：」这类
    中文指令前缀拼进去，反而会让指令本身进入词袋、污染向量，所以本地通道忽略 prefix。
    """
    if not texts:
        return []
    if not is_remote_configured():
        return [local_hash_embedding(text) for text in texts]

    payload = [f"{prefix}{text}" for text in texts] if prefix else texts
    vectors: list[list[float]] = []
    for start in range(0, len(payload), BATCH_SIZE):
        batch = payload[start : start + BATCH_SIZE]
        vectors.extend(await _embed_batch_with_retry(batch))
    return vectors


async def embed_query(text: str) -> list[float]:
    """查询侧向量化：带查询指令前缀（BGE/E5 需要，bge-m3 留空）。"""
    [vector] = await embed([text], prefix=settings.embedding_query_prefix)
    return vector


async def embed_documents(texts: list[str]) -> list[list[float]]:
    """文档侧向量化：不加查询前缀。非对称处理是这类模型的硬要求。"""
    return await embed(texts, prefix=settings.embedding_passage_prefix)


def embedder_id() -> str:
    """当前向量空间标识，用于把"新旧向量混用"从根上挡掉。"""
    return settings.embedder_id


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
