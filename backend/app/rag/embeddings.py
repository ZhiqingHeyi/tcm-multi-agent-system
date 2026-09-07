import hashlib
import math

import httpx

from ..config import settings


class EmbeddingUnavailable(Exception):
    pass


def is_remote_configured() -> bool:
    return bool(settings.embedding_model and (settings.embedding_api_key or settings.llm_api_key))


def _endpoint() -> str:
    base = (settings.embedding_base_url or settings.llm_base_url or "").rstrip("/")
    if not base:
        raise EmbeddingUnavailable("未配置 Embedding 接口地址")
    if base.endswith("/embeddings"):
        return base
    return f"{base}/embeddings"


def _api_key() -> str:
    return settings.embedding_api_key or settings.llm_api_key


async def embed_remote(texts: list[str]) -> list[list[float]]:
    payload = {"model": settings.embedding_model, "input": texts}
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=8.0)) as client:
        response = await client.post(_endpoint(), headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
    items = sorted(data["data"], key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


def local_hash_embedding(text: str, dim: int | None = None) -> list[float]:
    dim = dim or settings.embedding_dim
    weights = [0.0] * dim
    tokens = _bigrams(text)
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weights[index] += sign
    norm = math.sqrt(sum(w * w for w in weights)) or 1.0
    return [w / norm for w in weights]


def _bigrams(text: str) -> list[str]:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


async def embed(texts: list[str]) -> list[list[float]]:
    if is_remote_configured():
        try:
            vectors = await embed_remote(texts)
            return [_fit_dim(v) for v in vectors]
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            pass
    return [local_hash_embedding(text) for text in texts]


def _fit_dim(vector: list[float]) -> list[float]:
    dim = settings.embedding_dim
    if len(vector) == dim:
        return vector
    if len(vector) > dim:
        trimmed = vector[:dim]
    else:
        trimmed = vector + [0.0] * (dim - len(vector))
    norm = math.sqrt(sum(w * w for w in trimmed)) or 1.0
    return [w / norm for w in trimmed]
