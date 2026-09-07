import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql_text

from ..config import settings
from ..db.session import SessionLocal
from .chunker import Chunk
from .embeddings import embed

_K1 = 1.5
_B = 0.75
_RRF_K = 60


@dataclass
class Retrieved:
    content: str
    school: str
    source: str
    section: str
    score: float
    method: str

    @property
    def key(self) -> str:
        return f"{self.source}::{self.section}::{self.content[:60]}"


async def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    vectors = await embed([f"{c.section}。{c.text}" for c in chunks])
    async with SessionLocal() as session:
        for chunk, vector in zip(chunks, vectors, strict=True):
            await session.execute(
                sql_text(
                    "INSERT INTO knowledge_chunks (school, source, section, content, embedding, token_key)"
                    " VALUES (:school, :source, :section, :content, CAST(:embedding AS vector), :token_key)"
                    " ON CONFLICT (token_key) DO UPDATE SET embedding = EXCLUDED.embedding"
                ),
                {
                    "school": chunk.school,
                    "source": chunk.source[:200],
                    "section": chunk.section[:300],
                    "content": chunk.text,
                    "embedding": json.dumps(vector),
                    "token_key": chunk.token_key[:640],
                },
            )
        await session.commit()
    return len(chunks)


async def vector_search(query: str, top_k: int, schools: list[str] | None = None) -> list[Retrieved]:
    [query_vector] = await embed([query])
    filter_sql = ""
    params: dict[str, Any] = {"query": json.dumps(query_vector), "limit": top_k * 4}
    if schools:
        filter_sql = "WHERE school = ANY(:schools)"
        params["schools"] = schools
    stmt = sql_text(
        "SELECT content, school, source, section, 1 - (embedding <=> CAST(:query AS vector)) AS similarity"
        f" FROM knowledge_chunks {filter_sql} ORDER BY embedding <=> CAST(:query AS vector) LIMIT :limit"
    )
    async with SessionLocal() as session:
        rows = (await session.execute(stmt, params)).mappings().all()
    return [
        Retrieved(
            content=row["content"],
            school=row["school"],
            source=row["source"],
            section=row["section"],
            score=float(row["similarity"]),
            method="vector",
        )
        for row in rows
    ]


def tokenize(text: str) -> list[str]:
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[i : i + 2] for i in range(len(cleaned) - 1)]


def bm25_scores(query: str, contents: list[str]) -> list[float]:
    query_tokens = set(tokenize(query))
    if not query_tokens or not contents:
        return [0.0] * len(contents)
    tokenized = [tokenize(content) for content in contents]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    avg_len = sum(len(tokens) for tokens in tokenized) / len(tokenized) or 1.0
    total = len(contents)
    scores: list[float] = []
    for tokens in tokenized:
        counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if not tf:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            score += idf * (tf * (_K1 + 1.0)) / (tf + _K1 * (1.0 - _B + _B * doc_len / avg_len))
        scores.append(score)
    return scores


async def keyword_search(query: str, top_k: int, schools: list[str] | None = None) -> list[Retrieved]:
    filter_sql = ""
    params: dict[str, Any] = {}
    if schools:
        filter_sql = "WHERE school = ANY(:schools)"
        params["schools"] = schools
    stmt = sql_text(f"SELECT content, school, source, section FROM knowledge_chunks {filter_sql} LIMIT 4000")
    async with SessionLocal() as session:
        rows = (await session.execute(stmt, params)).mappings().all()
    if not rows:
        return []
    scores = bm25_scores(query, [row["content"] for row in rows])
    scored = [
        Retrieved(row["content"], row["school"], row["source"], row["section"], score, "keyword")
        for row, score in zip(rows, scores, strict=True)
        if score > 0
    ]
    scored.sort(key=lambda item: -item.score)
    return scored[:top_k]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = _RRF_K) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, token in enumerate(ranked, start=1):
            fused[token] = fused.get(token, 0.0) + 1.0 / (k + position)
    return fused


async def hybrid_search(query: str, top_k: int | None = None, schools: list[str] | None = None) -> list[Retrieved]:
    top_k = top_k or settings.rag_top_k
    candidates = max(top_k * 3, 12)
    vector_results = await vector_search(query, candidates, schools)
    keyword_results = await keyword_search(query, candidates, schools)

    by_key: dict[str, Retrieved] = {}
    for item in vector_results + keyword_results:
        by_key.setdefault(item.key, item)
    vector_rank = [item.key for item in vector_results]
    keyword_rank = [item.key for item in keyword_results]
    fused = reciprocal_rank_fusion([vector_rank, keyword_rank])

    merged: list[Retrieved] = []
    for key, score in fused.items():
        base = by_key[key]
        in_vector = key in set(vector_rank)
        in_keyword = key in set(keyword_rank)
        method = "hybrid" if in_vector and in_keyword else base.method
        merged.append(Retrieved(base.content, base.school, base.source, base.section, score, method))
    merged.sort(key=lambda item: -item.score)
    return merged[:top_k]


async def count_chunks() -> int:
    async with SessionLocal() as session:
        row = (await session.execute(sql_text("SELECT COUNT(*) AS total FROM knowledge_chunks"))).mappings().one()
    return int(row["total"])
