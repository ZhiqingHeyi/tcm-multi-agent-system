"""知识库检索层：稠密 + 稀疏双通道召回，RRF 融合，LLM 重排。

工业级检索的三段式结构（这也是各家 RAG 系统的通用范式）：

    召回 Recall  →  融合 Fusion  →  精排 Rerank
    （要快、要多）  （消除量纲差异）  （要准，样本少可以用大模型）

- **召回**：两条互补通道并行。
  · 稠密通道用 pgvector HNSW 索引做余弦近邻，擅长"换了说法"的语义命中；
  · 稀疏通道用中文 bigram + PostgreSQL GIN 索引做候选生成，再在应用层
    精确计算 BM25 打分，擅长方名、药名、条文号的字面精确匹配。
- **融合**：RRF（Reciprocal Rank Fusion）。向量相似度（0~1）与 BM25 分数
  （无上界）量纲完全不同，直接加权需要调参且不稳定；RRF 只用**排名**，
  score = Σ 1/(k + rank)，天然免调参、对异常值鲁棒，是工业默认选择。
- **精排**：用 LLM 充当 cross-encoder，对候选逐个判相关性。
  大模型读 query 与 passage 的交互信息，精度显著高于双塔向量模型，
  代价是慢，所以只对 20 条以内的候选做，并做结果缓存。

另一个容易被忽略但极其重要的设计：**学派隔离检索**。温病派不该被火神派的
大剂量附子医案带偏，因此所有查询都支持按 school / role 元数据预过滤。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql_text

from ..config import settings
from ..db.session import SessionLocal
from .chunker import Chunk
from .embeddings import embed, embed_documents, embed_query

_K1 = 1.5
_B = 0.75
_RRF_K = 60
_RERANK_CACHE: dict[str, list[int]] = {}

# 共享基础层。内经、本草、脉经、局方这类典籍是六派共同的源头，
# 检索某一派时必须一并纳入，否则温病派问"营卫"就找不到素问。
COMMON_SCHOOL = "common"

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
NON_WORD_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


@dataclass
class Retrieved:
    content: str
    school: str
    source: str
    section: str
    score: float
    method: str
    title: str = ""
    role: str = "classic"

    @property
    def key(self) -> str:
        return f"{self.source}::{self.section}::{self.content[:60]}"

    def as_dict(self) -> dict[str, Any]:
        label = f"{self.title}·{self.section}" if self.title else self.section
        return {
            "content": self.content,
            "source": self.source,
            "section": self.section,
            "title": self.title,
            "role": self.role,
            "school": self.school,
            "score": round(self.score, 4),
            "method": self.method,
            "label": label,
        }


# --------------------------------------------------------------------------
# 中文 bigram 稀疏表示
# --------------------------------------------------------------------------
def to_bigrams(text: str, unique: bool = True) -> str:
    """中文取二元组、英文数字取整词。中文无空格，bigram 是最省事且有效的索引单位。"""
    tokens: list[str] = []
    for segment in NON_WORD_RE.sub(" ", text).split():
        if CJK_RE.search(segment):
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
        else:
            tokens.append(segment.lower())
    return " ".join(dict.fromkeys(tokens)) if unique else " ".join(tokens)


def tokenize(text: str) -> list[str]:
    return to_bigrams(text, unique=False).split()


# --------------------------------------------------------------------------
# 写入：幂等 upsert + 源级清理
# --------------------------------------------------------------------------
async def upsert_chunks(chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    vectors = await embed_documents([chunk.text for chunk in chunks])
    marker = settings.embedder_id[:80]
    rows = [
        {
            "school": chunk.school,
            "source": chunk.source[:200],
            "section": chunk.section[:300],
            "title": (chunk.title or chunk.source)[:200],
            "role": chunk.role[:20],
            "chunk_index": chunk.index,
            "content": chunk.text,
            "content_bigrams": to_bigrams(chunk.text)[:20000],
            "embedding": json.dumps(vector),
            "embedder_id": marker,
            "token_key": chunk.token_key[:640],
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    statement = sql_text(
        "INSERT INTO knowledge_chunks"
        " (school, source, section, title, role, chunk_index, content, content_bigrams, embedding, embedder_id, token_key)"
        " VALUES (:school, :source, :section, :title, :role, :chunk_index, :content,"
        " CAST(:content_bigrams AS text), CAST(:embedding AS vector), :embedder_id, :token_key)"
        " ON CONFLICT (token_key) DO UPDATE SET"
        "   embedding = EXCLUDED.embedding,"
        "   embedder_id = EXCLUDED.embedder_id,"
        "   content_bigrams = EXCLUDED.content_bigrams,"
        "   title = EXCLUDED.title,"
        "   role = EXCLUDED.role,"
        "   chunk_index = EXCLUDED.chunk_index"
    )
    async with SessionLocal() as session:
        for start in range(0, len(rows), 200):
            await session.execute(statement, rows[start : start + 200])
        await session.commit()
    return len(rows)


async def prune_source(source: str, keep_keys: list[str]) -> int:
    """删除该文档已被淘汰的旧块——文档改动后不做清理，向量库里会残留脏数据。"""
    async with SessionLocal() as session:
        result = await session.execute(
            sql_text(
                "DELETE FROM knowledge_chunks WHERE source = :source"
                " AND NOT (token_key = ANY(:keep))"
            ),
            {"source": source[:200], "keep": keep_keys},
        )
        await session.commit()
        return result.rowcount or 0


async def sweep_orphan_sources(active_sources: list[str]) -> int:
    """清理已被整篇删除或更名的文档残留块。

    文档级别的 prune_source 只能清「仍在处理中的文档」的旧块；
    如果整篇文档被删了（比如合订本拆书后旧的 bcl_readings.md 消失），
    它的所有块都会成为库里的"孤儿块"，持续污染检索，必须全局清理。
    """
    if not active_sources:
        return 0
    async with SessionLocal() as session:
        result = await session.execute(
            sql_text("DELETE FROM knowledge_chunks WHERE NOT (source = ANY(:sources))"),
            {"sources": [s[:200] for s in active_sources]},
        )
        await session.commit()
        return result.rowcount or 0


async def source_stats() -> list[dict[str, Any]]:
    async with SessionLocal() as session:
        rows = (
            await session.execute(
                sql_text(
                    "SELECT school, source, COUNT(*) AS chunks, MAX(title) AS title, MAX(role) AS role"
                    " FROM knowledge_chunks GROUP BY school, source ORDER BY school, source"
                )
            )
        ).mappings().all()
    return [dict(row) for row in rows]


async def count_chunks() -> int:
    async with SessionLocal() as session:
        row = (await session.execute(sql_text("SELECT COUNT(*) AS total FROM knowledge_chunks"))).mappings().one()
    return int(row["total"])


# --------------------------------------------------------------------------
# 召回通道
# --------------------------------------------------------------------------
def _filter_sql(
    schools: list[str] | None,
    roles: list[str] | None,
    alias: str = "",
    with_common: bool = True,
) -> tuple[str, dict[str, Any]]:
    prefix = f"{alias}." if alias else ""
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if schools:
        scope = list(schools)
        if with_common and COMMON_SCHOOL not in scope:
            scope.append(COMMON_SCHOOL)
        clauses.append(f"{prefix}school = ANY(:schools)")
        params["schools"] = scope
    if roles:
        clauses.append(f"{prefix}role = ANY(:roles)")
        params["roles"] = roles
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _embedder_clause(where: str) -> str:
    """只召回与当前向量化模型同源的块。

    换 embedding 模型等于换向量空间：旧向量与新查询向量算余弦是纯噪声，
    而且不报错，只会表现为"召回莫名变差"。用 embedder_id 硬隔离，
    宁可暂时查不到，也不返回错答案。稀疏通道（BM25）与模型无关，不需要过滤。
    """
    return where + (" AND " if where else " WHERE ") + "embedder_id = :embedder"


async def vector_search(
    query: str,
    top_k: int,
    schools: list[str] | None = None,
    roles: list[str] | None = None,
    with_common: bool = True,
) -> list[Retrieved]:
    query_vector = await embed_query(query)
    where, params = _filter_sql(schools, roles, with_common=with_common)
    where = _embedder_clause(where)
    params.update({"query": json.dumps(query_vector), "limit": top_k, "embedder": settings.embedder_id[:80]})
    statement = sql_text(
        "SELECT content, school, source, section, title, role,"
        " 1 - (embedding <=> CAST(:query AS vector)) AS similarity"
        f" FROM knowledge_chunks{where}"
        " ORDER BY embedding <=> CAST(:query AS vector) LIMIT :limit"
    )
    async with SessionLocal() as session:
        rows = (await session.execute(statement, params)).mappings().all()
    return [
        Retrieved(
            content=row["content"], school=row["school"], source=row["source"],
            section=row["section"], title=row["title"], role=row["role"],
            score=float(row["similarity"]), method="vector",
        )
        for row in rows
    ]


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


async def keyword_search(
    query: str,
    top_k: int,
    schools: list[str] | None = None,
    roles: list[str] | None = None,
    candidate_limit: int = 800,
    with_common: bool = True,
) -> list[Retrieved]:
    """稀疏通道：GIN 索引下推候选生成 → 应用层精确 BM25 打分。

    这是本项目相对"Python 全表扫 BM25"的关键升级：10 万块规模下，
    全表扫描 + 逐条分词会把接口拖到秒级；索引召回把候选压到数百条，
    精确算分只在这几百条上做，兼顾速度与精度。
    """
    tsquery = " | ".join(to_bigrams(query).split()[:80])
    if not tsquery:
        return []
    where, params = _filter_sql(schools, roles, with_common=with_common)
    extra = " AND " if where else " WHERE "
    params.update({"tsq": tsquery, "limit": candidate_limit})
    statement = sql_text(
        "SELECT content, school, source, section, title, role"
        f" FROM knowledge_chunks{where}{extra}to_tsvector('simple', content_bigrams) @@ to_tsquery('simple', :tsq)"
        " LIMIT :limit"
    )
    async with SessionLocal() as session:
        rows = (await session.execute(statement, params)).mappings().all()
    if not rows:
        return []
    scores = bm25_scores(query, [row["content"] for row in rows])
    scored = [
        Retrieved(
            content=row["content"], school=row["school"], source=row["source"],
            section=row["section"], title=row["title"], role=row["role"],
            score=score, method="keyword",
        )
        for row, score in zip(rows, scores, strict=True)
        if score > 0
    ]
    scored.sort(key=lambda item: -item.score)
    return scored[:top_k]


# --------------------------------------------------------------------------
# 融合
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = _RRF_K, weights: list[float] | None = None
) -> dict[str, float]:
    """加权 RRF。weights 为各通道可信度系数，默认等权（保持向后兼容）。

    等权 RRF 的隐含假设是"两条通道一样准"。评测数据否定了这个假设：
    稀疏通道单独召回率明显低于稠密通道，等权会让它的噪声排名挤掉稠密的正确结果。
    """
    weights = weights or [1.0] * len(ranked_lists)
    fused: dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for position, key in enumerate(ranked, start=1):
            fused[key] = fused.get(key, 0.0) + weight / (k + position)
    return fused


async def random_chunks(
    limit: int,
    schools: list[str] | None = None,
    roles: list[str] | None = None,
    with_common: bool = True,
) -> list[Retrieved]:
    """在检索允许的范围内均匀随机抽样。

    评测用的随机基线必须**从全量语料采样**，而不是从检索结果的候选池里打乱。
    候选池本身就经过相关性筛选，打乱后仍会命中，会得出"随机基线 62.5%"这种
    荒谬结论，让可失败性检验彻底失效。
    """
    where, params = _filter_sql(schools, roles, with_common=with_common)
    params["limit"] = limit
    statement = sql_text(
        "SELECT content, school, source, section, title, role"
        f" FROM knowledge_chunks{where} ORDER BY random() LIMIT :limit"
    )
    async with SessionLocal() as session:
        rows = (await session.execute(statement, params)).mappings().all()
    return [
        Retrieved(row["content"], row["school"], row["source"], row["section"], 0.0, "random", row["title"], row["role"])
        for row in rows
    ]


async def _fused_search(
    query: str, limit: int, schools: list[str] | None, roles: list[str] | None, with_common: bool
) -> list[Retrieved]:
    """单层内的双通道召回 + 加权 RRF 融合。"""
    candidates = max(limit * 4, 16)
    vector_results = await vector_search(query, candidates, schools, roles, with_common=with_common)
    keyword_results = await keyword_search(query, candidates, schools, roles, with_common=with_common)

    by_key: dict[str, Retrieved] = {}
    for item in vector_results + keyword_results:
        by_key.setdefault(item.key, item)
    fused = reciprocal_rank_fusion(
        [[item.key for item in vector_results], [item.key for item in keyword_results]],
        weights=[1.0, settings.rag_sparse_weight],
    )
    vector_keys = {item.key for item in vector_results}
    keyword_keys = {item.key for item in keyword_results}

    merged: list[Retrieved] = []
    for key, score in fused.items():
        base = by_key[key]
        if key in vector_keys and key in keyword_keys:
            method = "hybrid"
        elif key in keyword_keys:
            method = "keyword"
        else:
            method = "vector"
        merged.append(
            Retrieved(base.content, base.school, base.source, base.section, score, method, base.title, base.role)
        )
    merged.sort(key=lambda item: -item.score)
    return merged[:limit]


# --------------------------------------------------------------------------
# LLM 精排（cross-encoder 效应）
# --------------------------------------------------------------------------
async def _rerank_all(query: str, items: list[Retrieved]) -> list[Retrieved]:
    """对候选全量做 LLM 精排并返回完整重排序列（不截断）。

    分层配额必须在精排之后施加：若先按扁平排名截断再配额，等于让"共享层挤占"
    这个错误排名先决定了候选池，配额就救不回来了。
    """
    if len(items) <= 1 or not settings.rag_rerank:
        return items
    from ..agents import llm as llm_module

    if not llm_module.is_configured():
        return items

    cache_key = hashlib.blake2b(
        (query + "|" + "|".join(item.key for item in items)).encode("utf-8"), digest_size=16
    ).hexdigest()
    if cache_key in _RERANK_CACHE:
        order = _RERANK_CACHE[cache_key]
    else:
        catalogue = "\n\n".join(
            f"[{index}] {item.title or item.source}·{item.section}\n{item.content[:300]}"
            for index, item in enumerate(items)
        )
        prompt = (
            f"用户问题：{query}\n\n候选文献片段：\n{catalogue}\n\n"
            "请按与问题的相关性从高到低排序，只输出 JSON："
            '{"ordered":[最相关的片段编号, ...]}'
        )
        try:
            data = await llm_module.chat_json(
                "你是中医文献检索的相关性评审员。只依据片段是否能为该问题提供直接依据来排序，"
                "不为文风或长度加分。",
                prompt,
                temperature=0,
                role="fast",
                max_tokens=300,
            )
            raw = data.get("ordered") or []
            order = [int(value) for value in raw if isinstance(value, (int, float, str)) and str(value).isdigit()]
            order = [index for index in order if 0 <= index < len(items)]
        except Exception:  # noqa: BLE001 - 精排失败不能影响主链路
            return items
        _RERANK_CACHE[cache_key] = order
        if len(_RERANK_CACHE) > 512:
            _RERANK_CACHE.pop(next(iter(_RERANK_CACHE)))

    ranked = [items[index] for index in order]
    ranked.extend(item for index, item in enumerate(items) if index not in set(order))
    return ranked


async def llm_rerank(query: str, items: list[Retrieved], top_k: int) -> list[Retrieved]:
    if len(items) <= top_k:
        return items[:top_k]
    return (await _rerank_all(query, items))[:top_k]


async def hybrid_search(
    query: str,
    top_k: int | None = None,
    schools: list[str] | None = None,
    roles: list[str] | None = None,
    rerank: bool = True,
    with_common: bool = True,
    tiered: bool | None = None,
) -> list[Retrieved]:
    """融合检索。tiered=True 启用分层配额，避免共享基础层挤占本派席位。

    扁平并集过滤（tiered=False）的缺陷：common 层占语料约三分之一，而《景岳全书》
    《医贯》这类泛中医典籍与本派典籍复用同一套术语（实测"补中益气+黄芪"在景岳全书
    共现 25 块，仅 6 块出自脾胃论的《脾胃论》），Top-K 席位会被泛典占满，
    本派权威典籍反而落榜，且精排样本也被污染。

    分层配额把「本派」与「共享层」分开检索、各自融合排序，精排后再按席位合并：
    本派占 top_k - quota 席，共享层保底 quota 席。
    """
    top_k = top_k or settings.rag_top_k
    tiered = settings.rag_tiered if tiered is None else tiered
    quota = max(min(settings.rag_common_quota, top_k - 1), 0)
    layered = bool(tiered and quota and schools and with_common and COMMON_SCHOOL not in schools)

    if not layered:
        pool = top_k if not rerank else max(top_k * 4, 16)
        merged = await _fused_search(query, pool, schools, roles, with_common)
        if not rerank:
            return merged[:top_k]
        return await llm_rerank(query, merged, top_k)

    own_pool = await _fused_search(query, max(top_k * 4, 16), schools, roles, with_common=False)
    shared_pool = await _fused_search(query, max(quota * 4, 8), [COMMON_SCHOOL], roles, with_common=False)
    candidates = own_pool + shared_pool
    if rerank:
        candidates = await _rerank_all(query, candidates)

    own_items = [item for item in candidates if item.school != COMMON_SCHOOL]
    shared_items = [item for item in candidates if item.school == COMMON_SCHOOL]
    result = own_items[: top_k - quota] + shared_items[:quota]
    if len(result) < top_k:
        leftover = own_items[top_k - quota :] + shared_items[quota:]
        result.extend(leftover[: top_k - len(result)])
    return result[:top_k]
