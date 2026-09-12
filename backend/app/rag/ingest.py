"""知识入库编排：扫描规范化产物 → 角色感知切块 → 向量化 → 幂等写入。

入库环节的四个工程要点（决定这套系统能不能长期维护）：

1. **增量**：以「文档内容 + 切块策略版本」为指纹，未变的文档整篇跳过。
   全量重嵌入既费钱又费时，10 万块规模下每次都跑一遍是不可接受的。
2. **对齐清理**：文档变更后，旧块必须按 source 删除。否则同一段内容会以
   新旧两种切法同时躺在库里，检索时互相抢排名，是最隐蔽的脏数据来源。
3. **可观测**：每篇文档输出块数、丢弃块数（过短/无中文）、耗时。
   切块质量不行必须当场可见，而不是等用户发现答非所问。
4. **可复现**：切块策略变更时用版本号强制重建，不依赖人工记忆。

用法：
    python -m app.rag.ingest                # 增量入库（跳过未变文档）
    python -m app.rag.ingest --force        # 全量重建
    python -m app.rag.ingest --school piwei # 只处理指定学派
    python -m app.rag.ingest --rebuild-indexes
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import document_to_chunks
from .embeddings import embedder_id, is_remote_configured
from .store import count_chunks, prune_source, source_stats, sweep_orphan_sources, upsert_chunks

# 切分策略版本：改动 chunker 的边界规则或长度参数时必须递增，否则增量会漏更新
CHUNKER_VERSION = "3"

def _knowledge_root() -> Path:
    """同时兼容宿主机（backend/data）与容器（/app/data）两种目录布局。"""
    for base in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
        candidate = base / "data" / "knowledge"
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[2] / "data" / "knowledge"


KNOWLEDGE_ROOT = _knowledge_root()
STATE_FILE = KNOWLEDGE_ROOT / "_ingest_state.json"

# 目录名即学派标签，必须与 agents.contracts.School 严格对齐。
# 出现过 "fuyang" 这类想当然的目录名，会让 school 过滤静默失效，因此这里做硬校验。
# "common" 是共享基础层（内经/本草/脉经等六派共祖典籍），检索时始终并入范围。
SCHOOL_DIRS = ("shanghan", "wenbing", "piwei", "huoshen", "integrative", "nihaixia", "common")

# 非知识性文件白名单：人物语气卡、蒸馏索引等属于 Persona 资产，不进检索库
EXCLUDED_FILES = {"expression_style.md", "SKILL.md", "SKILL_core.md", "distilled_cases.md"}

ROLE_ORDER = {"classic": 0, "case": 1, "lecture": 2}


@dataclass
class IngestReport:
    source: str
    school: str = ""
    role: str = ""
    chunks: int = 0
    dropped_short: int = 0
    dropped_low_cjk: int = 0
    sections: int = 0
    pruned: int = 0
    status: str = "pending"
    elapsed: float = 0.0
    note: str = ""

    def row(self) -> str:
        return (
            f"{self.source[:34]:<36} {self.school:<12} {self.role:<8} "
            f"{self.chunks:>6} {self.sections:>6} {self.dropped_short:>7} {self.dropped_low_cjk:>9} "
            f"{self.pruned:>6} {self.elapsed:>6.1f}s  {self.status} {self.note}"
        )


@dataclass
class IngestSummary:
    reports: list[IngestReport] = field(default_factory=list)
    total_chunks: int = 0
    changed: int = 0
    skipped: int = 0
    db_count: int = 0
    elapsed: float = 0.0
    remote_embeddings: bool = False
    by_school: dict[str, int] = field(default_factory=dict)
    by_role: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _fingerprint(text: str) -> str:
    """内容指纹 = 切分策略版本 + 向量化空间标识 + 文本。

    必须把 embedder_id 算进去：换 embedding 模型后块文本本身没变，
    若指纹不含模型标识，增量入库会把全部文档判为"未变更"而静默跳过，
    结果库里全是旧向量空间的向量，检索质量劣化且极难察觉。
    """
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(CHUNKER_VERSION.encode())
    hasher.update(embedder_id().encode())
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def _load_state() -> dict[str, str]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8")).get("fingerprints", {})
        except ValueError:
            return {}
    return {}


def _save_state(fingerprints: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(
            {"chunker_version": CHUNKER_VERSION, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "fingerprints": fingerprints},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )


def _scan(schools: list[str] | None) -> tuple[list[Path], list[str]]:
    """返回待入库文件与告警。非法目录不会被静默忽略，而是显式报出。"""
    if not KNOWLEDGE_ROOT.exists():
        return [], []
    warnings: list[str] = []
    if schools:
        invalid = [name for name in schools if name not in SCHOOL_DIRS]
        if invalid:
            warnings.append(f"忽略非法学派参数：{invalid}（合法值：{list(SCHOOL_DIRS)}）")
    files: list[Path] = []
    for path in sorted(KNOWLEDGE_ROOT.glob("*/*.md")):
        school = path.parent.name
        if school not in SCHOOL_DIRS:
            warnings.append(f"跳过非法学派目录 {school}/（{path.name}）")
            continue
        if path.name in EXCLUDED_FILES:
            warnings.append(f"跳过非知识文件 {path.name}")
            continue
        if schools and school not in schools:
            continue
        files.append(path)
    return files, warnings


async def ingest(
    schools: list[str] | None = None,
    force: bool = False,
    previous: dict[str, str] | None = None,
) -> tuple[IngestSummary, dict[str, str]]:
    summary = IngestSummary(remote_embeddings=is_remote_configured())
    fingerprints: dict[str, str] = dict(previous or {})
    started = time.time()

    files, warnings = _scan(schools)
    summary.warnings = warnings

    for path in files:
        school = path.parent.name
        raw = path.read_text(encoding="utf-8", errors="ignore")
        digest = _fingerprint(raw)
        report = IngestReport(source=path.name, school=school)

        if not force and fingerprints.get(path.name) == digest:
            report.status = "skipped"
            report.note = "内容与切块策略均未变"
            summary.skipped += 1
            summary.reports.append(report)
            continue

        doc_started = time.time()
        meta_chunks = document_to_chunks(raw, school=school, source=path.name)
        chunks, stats = meta_chunks
        if not chunks:
            report.status = "empty"
            report.note = "切块后无有效内容"
            summary.reports.append(report)
            continue

        report.role = chunks[0].role
        report.sections = stats.sections
        report.dropped_short = stats.dropped_short
        report.dropped_low_cjk = stats.dropped_low_cjk

        await upsert_chunks(chunks)
        report.pruned = await prune_source(path.name, [chunk.token_key[:640] for chunk in chunks])
        report.chunks = len(chunks)
        report.elapsed = time.time() - doc_started
        report.status = "ok"
        fingerprints[path.name] = digest

        summary.total_chunks += len(chunks)
        summary.changed += 1
        summary.by_school[school] = summary.by_school.get(school, 0) + len(chunks)
        summary.by_role[report.role] = summary.by_role.get(report.role, 0) + len(chunks)
        summary.reports.append(report)

    summary.db_count = await count_chunks()
    summary.elapsed = time.time() - started
    if not schools:
        orphans = await sweep_orphan_sources([p.name for p in files])
        if orphans:
            summary.warnings.append(f"清理已淘汰旧文档残留块：{orphans} 块")
    return summary, fingerprints


def _print_summary(summary: IngestSummary) -> None:
    for warning in summary.warnings:
        print(f"[告警] {warning}")
    print(
        f"{'文档':<36} {'学派':<12} {'角色':<8} {'块数':>6} {'章节':>6} "
        f"{'弃短':>7} {'弃非中':>9} {'清旧':>6} {'耗时':>7}  状态"
    )
    print("-" * 132)
    for report in sorted(summary.reports, key=lambda item: (item.school, ROLE_ORDER.get(item.role, 9), item.source)):
        print(report.row())
    print("-" * 132)
    print(
        f"本次入库 {summary.total_chunks} 块（变更 {summary.changed} 篇，跳过 {summary.skipped} 篇），"
        f"耗时 {summary.elapsed:.1f}s"
    )
    print("按学派：" + "，".join(f"{k} {v} 块" for k, v in sorted(summary.by_school.items())))
    print("按角色：" + "，".join(f"{k} {v} 块" for k, v in sorted(summary.by_role.items())))
    print(f"向量库现存 {summary.db_count} 块；向量化方式：{'远程 API' if summary.remote_embeddings else '本地哈希向量'}")


async def _dump_stats() -> None:
    rows = await source_stats()
    print(f"{'学派':<12} {'角色':<8} {'块数':>6}  文档")
    print("-" * 100)
    for row in rows:
        print(f"{row['school']:<12} {row['role']:<8} {row['chunks']:>6}  {row['title'] or row['source']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库入库（增量）")
    parser.add_argument("--force", action="store_true", help="忽略指纹，全量重建")
    parser.add_argument("--school", nargs="*", help="只处理指定学派目录")
    parser.add_argument("--stats", action="store_true", help="只打印库内统计")
    args = parser.parse_args()

    if args.stats:
        asyncio.run(_dump_stats())
        return

    summary, fingerprints = asyncio.run(ingest(schools=args.school, force=args.force, previous=_load_state()))
    if not args.school:
        _save_state(fingerprints)
    _print_summary(summary)


if __name__ == "__main__":
    main()
