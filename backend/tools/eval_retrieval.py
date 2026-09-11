"""知识库检索质量 Golden Set 评测。

面试考察点：**你的 RAG 系统怎么证明比直接问 LLM 好？怎么调优的？**
不能拿"我觉得挺好"搪塞，必须有量化指标：
- **Recall@K**（召回率）：Golden 答案所在的块是否进入了 Top-K。
  工业界标准：Top-4 召回率 ≥ 70%，Top-10 ≥ 85%。
- **MRR**（Mean Reciprocal Rank，平均倒数排名）：第一个命中项排名的倒数均值。
  越接近 1.0 说明真正相关的块越靠前，大模型越容易看到。
- **学派隔离率**：指定学派的检索，结果里是否只有「本派 ∪ common」，
  绝不能出现它派脏数据（出现即 0 分，临床严禁串派方）。

用法：
    python -m tools.eval_retrieval
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# 确保能 import backend/app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.store import hybrid_search, keyword_search, vector_search


@dataclass(frozen=True)
class GoldenQuery:
    query: str
    target_school: str
    target_role: str
    # 期望命中的关键词/短语，命中任一个就算有效召回
    expected_phrases: tuple[str, ...]
    description: str


# 六大门派 + 常见临床辨证问题的评测集
GOLDEN_SET: tuple[GoldenQuery, ...] = (
    GoldenQuery(
        query="太阳病头痛发热汗出恶风用什么方子",
        target_school="shanghan",
        target_role="classic",
        expected_phrases=("桂枝汤", "啬啬恶寒", "翕翕发热", "阳浮而阴弱"),
        description="伤寒派核心方证：桂枝汤条文",
    ),
    GoldenQuery(
        query="少阴病脉微细但欲寐四肢厥逆手足冷",
        target_school="huoshen",
        target_role="classic",
        expected_phrases=("四逆汤", "附子", "通脉四逆", "回阳", "干姜"),
        description="火神/少阴回阳重剂辨证",
    ),
    GoldenQuery(
        query="脾胃虚弱气虚下陷脱肛便溏饮食不进",
        target_school="piwei",
        target_role="classic",
        expected_phrases=("补中益气", "升阳", "黄芪", "柴胡", "升麻"),
        description="脾胃派东垣升阳益胃法",
    ),
    GoldenQuery(
        query="温热病初起邪在卫分身热微恶风寒口微渴",
        target_school="wenbing",
        target_role="classic",
        expected_phrases=("银翘散", "桑菊饮", "卫分", "清络", "辛凉"),
        description="温病派卫分证治",
    ),
    GoldenQuery(
        query="气虚血瘀半身不遂口眼喎斜张锡纯治法",
        target_school="integrative",
        target_role="case",
        expected_phrases=("黄芪", "水蛭", "活络效灵", "衷中参西", "当归"),
        description="衷中参西中西汇通治脑中风",
    ),
    GoldenQuery(
        query="倪海厦治疗乳腺癌淋巴结转移核心思路与辨证",
        target_school="nihaixia",
        target_role="lecture",
        expected_phrases=("奶水", "阳气", "少阴", "附子", "经方", "石膏"),
        description="倪海厦经方辨治乳岩思路",
    ),
    GoldenQuery(
        query="阴阳应象大论清阳出上窍浊阴出下窍",
        target_school="shanghan",  # 查询走伤寒，但典籍在 common
        target_role="classic",
        expected_phrases=("清阳", "浊阴", "上窍", "下窍", "阴阳应象"),
        description="六派共用 common 典籍（素问）穿透测试",
    ),
    GoldenQuery(
        query="心下痞按之濡关上脉浮大黄黄连泻心汤",
        target_school="shanghan",
        target_role="classic",
        expected_phrases=("大黄黄连泻心", "心下痞", "按之濡", "关上脉浮"),
        description="伤寒心下痞经典方证",
    ),
)


async def evaluate_method(name: str, searcher, top_k: int = 4) -> dict:
    hits_at_k = 0
    rr_sum = 0.0
    school_isolated = 0
    total = len(GOLDEN_SET)
    latencies: list[float] = []

    for item in GOLDEN_SET:
        t0 = time.time()
        results = await searcher(
            item.query,
            top_k=top_k,
            schools=[item.target_school],
            roles=[item.target_role] if item.target_role else None,
        )
        latencies.append(time.time() - t0)

        # 1. 学派隔离检查：结果必须来自本派或 common，绝不允许窜派
        allowed_schools = {item.target_school, "common"}
        bad_school = any(r.school not in allowed_schools for r in results)
        if not bad_school:
            school_isolated += 1

        # 2. Recall@K 与 MRR
        matched_rank: int | None = None
        for rank, r in enumerate(results, start=1):
            if any(p in r.content for p in item.expected_phrases):
                matched_rank = rank
                break

        if matched_rank is not None:
            hits_at_k += 1
            rr_sum += 1.0 / matched_rank

    return {
        "method": name,
        f"recall@{top_k}": round(hits_at_k / total, 3),
        "mrr": round(rr_sum / total, 3),
        "isolation_rate": round(school_isolated / total, 3),
        "avg_latency_ms": round(sum(latencies) / len(latencies) * 1000, 1),
    }


async def main() -> None:
    print("=" * 70)
    print("中医知识库检索质量评测 (Golden Set = 8 cases)")
    print("=" * 70)

    for top_k in (4, 8):
        print(f"\n--- Top-K = {top_k} ---")
        dense = await evaluate_method(
            "稠密向量 (Dense)",
            lambda q, top_k, schools, roles: vector_search(q, top_k, schools, roles),
            top_k,
        )
        sparse = await evaluate_method(
            "稀疏 BM25 (Sparse)",
            lambda q, top_k, schools, roles: keyword_search(q, top_k, schools, roles),
            top_k,
        )
        hybrid = await evaluate_method(
            "双路融合 RRF (Hybrid)",
            lambda q, top_k, schools, roles: hybrid_search(q, top_k, schools, roles, rerank=False),
            top_k,
        )

        for report in (dense, sparse, hybrid):
            print(
                f"{report['method']:<24} "
                f"Recall@{top_k}: {report[f'recall@{top_k}'] * 100:>5.1f}%  "
                f"MRR: {report['mrr']:>5.3f}  "
                f"学派隔离: {report['isolation_rate'] * 100:>5.1f}%  "
                f"耗时: {report['avg_latency_ms']:>6.1f}ms"
            )


if __name__ == "__main__":
    asyncio.run(main())
