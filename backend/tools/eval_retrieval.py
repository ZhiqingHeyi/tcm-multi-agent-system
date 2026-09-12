"""知识库检索质量评测与「评测有效性」验证。

第一部分是常规 RAG 指标（Recall@K / MRR / 学派隔离率）。
第二部分才是关键：**怎么证明这套评测本身是有效的。**

一个评测只有在「系统退化时必须掉分」的前提下才有意义。恒定 100% 的评测
通常是标的物太松、或者出了数据泄漏。因此本脚本做三类判别力检验：

A. 标签严格性对照：宽松（命中任一关键词）vs 严格（多个关键词必须在同一块共现）。
   同一批查询、同一个系统，若宽松 100% 而严格明显下降，说明宽松标签虚高。
B. 消融与退化：稠密单通道 / 稀疏单通道 / 双路融合 / 关闭共享基础层 / 随机基线。
   若随机基线也能拿高分，评测无效；若关掉 common 后经典类查询不失败，评测无效。
C. 改写查询：把查询改写成与原文**零关键词重叠**的口语表述。
   稀疏通道应当掉分，稠密/融合通道应当守住——这才能证明语义通道有真实增量。

用法：
    python -m tools.eval_retrieval              # 全部实验
    python -m tools.eval_retrieval --detail     # 附逐条明细
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.store import hybrid_search, keyword_search, random_chunks, vector_search


@dataclass(frozen=True)
class Case:
    query: str
    school: str
    role: str
    must_contain: tuple[str, ...]
    note: str


# 原始金标集：查询用词与典籍原文高度一致（术语查询）
GOLDEN: tuple[Case, ...] = (
    Case("太阳病头痛发热汗出恶风用什么方子", "shanghan", "classic",
         ("桂枝汤", "阳浮而阴弱"), "伤寒派核心方证：桂枝汤"),
    Case("少阴病脉微细但欲寐四肢厥逆手足冷", "huoshen", "classic",
         ("四逆汤", "附子"), "火神/少阴回阳重剂"),
    Case("脾胃虚弱气虚下陷脱肛便溏饮食不进", "piwei", "classic",
         ("补中益气", "黄芪"), "脾胃派东垣升阳益胃法"),
    Case("温热病初起邪在卫分身热微恶风寒口微渴", "wenbing", "classic",
         ("银翘散", "辛凉"), "温病派卫分证治"),
    Case("气虚血瘀半身不遂口眼喎斜张锡纯治法", "integrative", "case",
         ("活络效灵丹", "乳香"), "衷中参西中西汇通治中风"),
    Case("倪海厦治疗乳腺癌淋巴结转移核心思路", "nihaixia", "lecture",
         ("乳癌", "奶水"), "倪海厦经方辨治乳岩"),
    Case("阴阳应象大论清阳出上窍浊阴出下窍", "shanghan", "classic",
         ("清阳", "浊阴"), "六派共用 common 典籍穿透测试"),
    Case("心下痞按之濡关上脉浮大黄黄连泻心汤", "shanghan", "classic",
         ("大黄黄连泻心", "心下痞"), "伤寒心下痞经典方证"),
)

# 改写集：刻意避开原文术语与方名，检验语义通道是否真有增量
PARAPHRASE: tuple[Case, ...] = (
    Case("吹了风以后怕冷 身上微微出汗 后脖子发紧 该用什么方", "shanghan", "classic",
         ("桂枝汤",), "桂枝汤·口语化改写（不含方名）"),
    Case("老年人手脚冰凉 精神萎靡 一天到晚总想睡 需要急救回阳", "huoshen", "classic",
         ("四逆汤",), "四逆汤·口语化改写（不含方名）"),
    Case("白天动不动就出汗 说话没力气 整个人提不起劲 中气不足", "piwei", "classic",
         ("补中益气",), "补中益气·口语化改写（不含方名）"),
    Case("春天流行的外感 嗓子痒轻微咳嗽 有点怕风 不算重", "wenbing", "classic",
         ("银翘散",), "银翘散·口语化改写（不含方名）"),
    Case("一侧手脚突然不听使唤 说话含糊 经络被瘀血堵住", "integrative", "case",
         ("活络效灵丹",), "活络效灵丹·口语化改写（不含方名）"),
)


def hit_rank(results, case: Case, strict: bool) -> int | None:
    """返回首个命中块的名次。strict=True 要求所有关键词在同一块内共现。"""
    for rank, item in enumerate(results, start=1):
        if strict:
            if all(phrase in item.content for phrase in case.must_contain):
                return rank
        elif any(phrase in item.content for phrase in case.must_contain):
            return rank
    return None


@dataclass
class Channel:
    name: str
    search: object


def build_channels(top_k: int) -> list[Channel]:
    return [
        Channel("稠密向量 Dense", lambda q, c: vector_search(q, top_k, [c.school], [c.role])),
        Channel("稀疏 BM25 Sparse", lambda q, c: keyword_search(q, top_k, [c.school], [c.role])),
        Channel("融合·旧扁平(含common)",
                lambda q, c: hybrid_search(q, top_k, [c.school], [c.role], rerank=False, tiered=False)),
        Channel("融合·旧扁平(关common)",
                lambda q, c: hybrid_search(q, top_k, [c.school], [c.role], rerank=False, tiered=False, with_common=False)),
        Channel("融合·新分层配额",
                lambda q, c: hybrid_search(q, top_k, [c.school], [c.role], rerank=False, tiered=True)),
        Channel("随机基线 Random", None),
    ]


async def run_channel(channel: Channel, cases: tuple[Case, ...], strict: bool, top_k: int) -> dict:
    hits = 0
    rr = 0.0
    isolated = 0
    latencies: list[float] = []
    ranks: list[int | None] = []

    for case in cases:
        started = time.time()
        if channel.search is None:
            results = await random_chunks(top_k, [case.school], [case.role])
        else:
            results = await channel.search(case.query, case)
        latencies.append(time.time() - started)

        if all(item.school in {case.school, "common"} for item in results):
            isolated += 1

        rank = hit_rank(results, case, strict)
        ranks.append(rank)
        if rank is not None:
            hits += 1
            rr += 1.0 / rank

    total = len(cases)
    return {
        "name": channel.name,
        "recall": round(hits / total, 3),
        "mrr": round(rr / total, 3),
        "isolation": round(isolated / total, 3),
        "latency_ms": round(sum(latencies) / len(latencies) * 1000, 1),
        "ranks": ranks,
    }


def render_table(title: str, reports: list[dict], top_k: int) -> None:
    print(f"\n{title}")
    print(f"{'通道':<22}{'Recall@' + str(top_k):>12}{'MRR':>10}{'学派隔离':>12}{'耗时':>10}")
    print("-" * 68)
    for report in reports:
        print(
            f"{report['name']:<22}{report['recall'] * 100:>10.1f}%"
            f"{report['mrr']:>10.3f}{report['isolation'] * 100:>10.1f}%{report['latency_ms']:>9.1f}ms"
        )


def render_detail(cases: tuple[Case, ...], reports: list[dict]) -> None:
    headers = ["Dense", "Sparse", "旧扁平", "旧·无common", "新分层", "随机"]
    print(f"\n{'查询与失败定位':<42}" + "".join(f"{h:>12}" for h in headers))
    print("-" * 114)
    for index, case in enumerate(cases):
        cells = "".join(
            f"{(report['ranks'][index] if report['ranks'][index] else '✗'):>12}" for report in reports
        )
        print(f"{case.note[:40]:<42}{cells}")
    print("（数字为命中块排名，✗ 表示 Top-K 内未命中）")


async def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量与评测有效性验证")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()
    top_k = args.top_k
    channels = build_channels(top_k)

    print("=" * 68)
    print(f"知识库检索评测 · Top-K = {top_k} · 金标 {len(GOLDEN)} 条 / 改写 {len(PARAPHRASE)} 条")
    print("=" * 68)

    loose = [await run_channel(channel, GOLDEN, False, top_k) for channel in channels]
    render_table("【实验 A-1】宽松标签（命中任一关键词即算召回）", loose, top_k)

    strict = [await run_channel(channel, GOLDEN, True, top_k) for channel in channels]
    render_table("【实验 A-2】严格标签（关键词必须在同一块内共现）", strict, top_k)

    rewritten = [await run_channel(channel, PARAPHRASE, True, top_k) for channel in channels]
    render_table("【实验 C】改写查询（口语化、不含方名，检验语义通道增量）", rewritten, top_k)

    if args.detail:
        render_detail(GOLDEN, strict)
        render_detail(PARAPHRASE, rewritten)

    flat_recall = strict[2]["recall"]
    tiered_recall = strict[4]["recall"]
    random_recall = strict[5]["recall"]
    sparse_rewrite = rewritten[1]["recall"]
    dense_rewrite = rewritten[0]["recall"]

    print("\n" + "=" * 68)
    print("评测有效性判定")
    print("=" * 68)
    checks = [
        ("随机基线必须接近 0（否则评测可被轻易刷分）",
         random_recall <= 0.15, f"真·全库随机 Recall@{top_k} = {random_recall * 100:.1f}%"),
        ("分层配额必须优于旧扁平并集（证明改造成效）",
         tiered_recall > flat_recall, f"新分层 {tiered_recall * 100:.1f}% > 旧扁平 {flat_recall * 100:.1f}%"),
        ("严格标签下分层配额应达到可用水平（≥60%）",
         tiered_recall >= 0.60, f"新分层严格 Recall = {tiered_recall * 100:.1f}%"),
        ("改写查询下稀疏通道不应优于稠密（证明标签无关键词泄漏）",
         sparse_rewrite <= dense_rewrite, f"sparse {sparse_rewrite * 100:.1f}% ≤ dense {dense_rewrite * 100:.1f}%"),
    ]
    for label, passed, evidence in checks:
        print(f"  [{'通过' if passed else '未通过'}] {label}\n        证据：{evidence}")


if __name__ == "__main__":
    asyncio.run(main())
