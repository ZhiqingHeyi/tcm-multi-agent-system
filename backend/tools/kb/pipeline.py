"""知识库构建编排层。

一条生产可用的 ingestion pipeline 必须具备五件事，缺一不可：
1. **幂等**：同一批输入重复执行，产物完全一致（内容哈希驱动）。
2. **可续跑**：中途失败不从头再来，已完成文档跳过。
3. **有卡口**：质量不达标的文档被拦下并记录原因，而不是悄悄污染向量库。
4. **能扇出**：一个源文件可能包含多本书（合订本），必须拆成一源多产物，
   否则元数据必然失真、检索必然偏置。
5. **有清单**：产出 manifest，人可审、程序可消费、CI 可比对。

用法：
    python -m tools.kb.pipeline              # 全量构建（增量跳过）
    python -m tools.kb.pipeline --force      # 忽略缓存，强制重建
    python -m tools.kb.pipeline --dry-run    # 只体检，不落盘
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from kb.compendium import split_books
    from kb.extract import ExtractError, extract, sniff
    from kb.normalize import normalize, quality_report
    from kb.taxonomy import DocMeta, classify, compose_slug, edition_score, is_compendium, resolve_book
else:
    from .compendium import split_books
    from .extract import ExtractError, extract, sniff
    from .normalize import normalize, quality_report
    from .taxonomy import DocMeta, classify, compose_slug, edition_score, is_compendium, resolve_book

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "知识库原始未处理下载文档"
OUTPUT_DIR = PROJECT_ROOT / "backend" / "data" / "knowledge"
MANIFEST = OUTPUT_DIR / "_manifest.json"

SKIP_SUFFIX = {".dmg", ".zip", ".exe", ".pkg"}
SKIP_NAMES = {".DS_Store", "Thumbs.db"}

# 卡口阈值：低于此值判定为扫描件/抽取失败，拒绝入库
MIN_CJK_RATIO = 0.55
MIN_CHARS = 20_000

# 合订本在引用出处里的书名前缀
COMPENDIUM_LABEL = "中医临床必读丛书（典藏版）"

# 熔断阈值。踩过的坑：解释器环境缺 ebooklib/pypdf 时，全部 EPUB/PDF 抽取失败，
# 而"产物对齐清理"会把上一轮的好产物当成淘汰品删掉——一次环境故障清空整个知识库。
# 因此当失败比例异常时直接中止，且**不进入清理阶段**。
ABORT_REJECT_MIN = 3
ABORT_REJECT_RATIO = 0.34


class BuildAborted(RuntimeError):
    """环境级故障导致的大面积失败，中止构建以保护既有产物。"""


@dataclass
class DocReport:
    source: str
    unit: str = ""
    slug: str = ""
    school: str = ""
    role: str = ""
    title: str = ""
    fmt: str = ""
    status: str = "pending"
    reason: str = ""
    raw_chars: int = 0
    clean_chars: int = 0
    cjk_ratio: float = 0.0
    output: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "source": self.source, "unit": self.unit, "slug": self.slug, "school": self.school,
            "role": self.role, "title": self.title, "fmt": self.fmt, "status": self.status,
            "reason": self.reason, "raw_chars": self.raw_chars, "clean_chars": self.clean_chars,
            "cjk_ratio": self.cjk_ratio, "output": self.output, "warnings": self.warnings,
        }


@dataclass
class Candidate:
    """一个待落盘的文档单元。普通书 = 1 个单元；合订本 = N 册各 1 个单元。"""

    source: str
    unit: str
    meta: DocMeta
    slug: str
    text: str
    suffix: str
    report: DocReport
    compendium: bool = False

    @property
    def cited_source(self) -> str:
        return f"{COMPENDIUM_LABEL}·{self.unit}" if self.compendium else self.source

    @property
    def identity(self) -> tuple[str, str]:
        return (self.meta.school, self.slug)

    @property
    def work(self) -> str:
        """作品标识：不含卷次。上下册同属一部作品，用于与专书比对。"""
        return self.meta.slug


def file_digest(path: Path) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def render_document(meta: DocMeta, text: str, source_name: str) -> str:
    """写入带元数据的 Markdown。元数据是后续"按学派检索"和"按角色分块"的依据。"""
    stats = quality_report(text)
    head = [
        "---",
        f"title: {meta.title}",
        f"school: {meta.school}",
        f"role: {meta.role}",
        f"source: {source_name}",
        f"chars: {stats['chars']}",
        "---",
        "",
    ]
    return "\n".join(head) + text + "\n"


def _expand(path: Path, cleaned: str, report: DocReport) -> list[Candidate]:
    """把一个源文件的清洗文本展开成若干文档单元。"""
    if not is_compendium(path.name):
        meta = classify(path.name)
        if meta is None:
            report.status = "skipped"
            report.reason = "未登记且启发式无法归类"
            return []
        report.slug, report.school, report.role, report.title = meta.slug, meta.school, meta.role, meta.title
        report.unit = meta.title
        report.status = "ready"
        return [Candidate(path.name, meta.title, meta, meta.slug, cleaned, path.suffix, report)]

    books = split_books(cleaned)
    if not books:
        report.status = "skipped"
        report.reason = "判定为合订本但未识别出册级边界"
        return []

    report.status = "split"
    report.reason = f"合订本，拆出 {len(books)} 册"
    candidates: list[Candidate] = []
    for book in books:
        meta = resolve_book(book.title)
        child = DocReport(source=path.name, unit=book.key, fmt=report.fmt)
        if meta is None:
            child.status = "skipped"
            child.reason = "分册未登记，跳过（避免错标学派）"
            report.warnings.append(f"未登记分册：{book.key}")
            continue
        slug = compose_slug(meta, book.volume)
        child.slug, child.school, child.role, child.title = slug, meta.school, meta.role, meta.title
        book_stats = quality_report(book.text)
        child.clean_chars = book_stats["chars"]
        child.cjk_ratio = book_stats["cjk_ratio"]
        child.status = "ready"
        candidates.append(Candidate(path.name, book.key, meta, slug, book.text, path.suffix, child, compendium=True))
    return candidates


def _sweep(desired: set[Path]) -> list[str]:
    """删除本管线产出目录里已被淘汰的旧产物。

    不做这一步，文档改名/拆书后旧文件会一直躺在目录里，
    下一次入库把新旧两份都收进向量库——同一内容两份，互相抢排名。
    """
    removed: list[str] = []
    for school_dir in sorted(OUTPUT_DIR.glob("*/")):
        if not school_dir.is_dir():
            continue
        for existing in sorted(school_dir.glob("*.md")):
            if existing not in desired:
                existing.unlink()
                removed.append(str(existing.relative_to(PROJECT_ROOT)))
    return removed


def build(force: bool = False, dry_run: bool = False) -> dict:
    manifest = load_manifest()
    documents: dict[str, dict] = manifest.get("documents", {})
    by_source: dict[str, list[dict]] = {}
    for entry in documents.values():
        by_source.setdefault(entry.get("source", ""), []).append(entry)

    reports: list[DocReport] = []
    candidates: list[Candidate] = []
    cached_slugs: set[str] = set()
    total_sources = 0
    rejected: list[DocReport] = []
    started = time.time()

    for path in sorted(SOURCE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIX or path.name in SKIP_NAMES:
            continue
        total_sources += 1

        fmt = sniff(path)
        digest = file_digest(path)
        previous = by_source.get(path.name, [])
        if previous and not force and all(entry.get("digest") == digest for entry in previous):
            for entry in previous:
                cached_slugs.add(entry["slug"])
                reports.append(
                    DocReport(
                        source=path.name, unit=entry.get("unit", ""), slug=entry["slug"],
                        school=entry["school"], role=entry["role"], title=entry.get("title", ""),
                        fmt=fmt, status="cached", reason="内容未变，复用已有产物",
                        clean_chars=entry.get("clean_chars", 0), cjk_ratio=entry.get("cjk_ratio", 0.0),
                        output=entry.get("output", ""),
                    )
                )
            continue

        report = DocReport(source=path.name, fmt=fmt)
        try:
            extracted = extract(path)
        except ExtractError as exc:
            report.status = "rejected"
            report.reason = f"抽取失败：{exc}"
            reports.append(report)
            rejected.append(report)
            continue
        except Exception as exc:  # noqa: BLE001
            report.status = "rejected"
            report.reason = f"抽取异常：{type(exc).__name__}: {exc}"
            reports.append(report)
            rejected.append(report)
            continue

        report.raw_chars = len(extracted.text)
        report.warnings = list(extracted.warnings)
        cleaned = normalize(extracted.text)
        stats = quality_report(cleaned)
        report.clean_chars = stats["chars"]
        report.cjk_ratio = stats["cjk_ratio"]

        if stats["chars"] < MIN_CHARS or stats["cjk_ratio"] < MIN_CJK_RATIO:
            report.status = "rejected"
            report.reason = (
                f"质量卡口未通过（中文字符占比 {stats['cjk_ratio']:.2f}，清洗后 {stats['chars']} 字），"
                "疑似扫描件，需要 OCR 或换版本"
            )
            reports.append(report)
            rejected.append(report)
            continue

        produced = _expand(path, cleaned, report)
        if report.status == "split":
            reports.append(report)
        for candidate in produced:
            reports.append(candidate.report)
        candidates.extend(produced)

    # ---- 熔断：环境级故障保护 ----
    if rejected and len(rejected) >= ABORT_REJECT_MIN and len(rejected) / max(total_sources, 1) >= ABORT_REJECT_RATIO:
        detail = "；".join(f"{item.source[:36]}←{item.reason[:38]}" for item in rejected[:5])
        raise BuildAborted(
            f"{len(rejected)}/{total_sources} 个源文件处理失败，超过熔断阈值，已中止且未清理任何既有产物。"
            f"通常是解释器环境问题（用 .venv-knowledge/bin/python 运行）。样例：{detail}"
        )

    # ---- 专书优先：合订本与专书同书时，专书胜出 ----
    # 按「作品」而非「册」比对：衷中参西录上下册必须一起让位给专书版本，
    # 只按册比对会漏掉带卷次后缀的合订册，库里就会留下三份同一部书。
    dedicated = {candidate.work for candidate in candidates if not candidate.compendium}
    survivors: list[Candidate] = []
    for candidate in candidates:
        if candidate.compendium and candidate.work in dedicated:
            candidate.report.status = "duplicate"
            candidate.report.reason = "已有该书的专书版本，合订本对应册跳过"
            continue
        survivors.append(candidate)

    # ---- 同书多版本择优 ----
    best: dict[tuple[str, str], Candidate] = {}
    for candidate in sorted(survivors, key=lambda item: -edition_score(item.suffix, len(item.text), item.meta.priority)):
        key = candidate.identity
        winner = best.get(key)
        if winner is None:
            best[key] = candidate
            candidate.report.reason = candidate.report.reason or "选中该书的当前最优版本"
        else:
            candidate.report.status = "duplicate"
            candidate.report.reason = "同书存在更优版本，本次跳过"

    # ---- 落盘 ----
    written: list[Candidate] = []
    desired: set[Path] = set()
    next_docs: dict[str, dict] = {}
    if not dry_run:
        for candidate in best.values():
            if candidate.report.status != "ready":
                continue
            school_dir = OUTPUT_DIR / candidate.meta.school
            school_dir.mkdir(parents=True, exist_ok=True)
            target = school_dir / f"{candidate.slug}.md"
            target.write_text(
                render_document(candidate.meta, candidate.text, candidate.cited_source), encoding="utf-8"
            )
            candidate.report.output = str(target.relative_to(PROJECT_ROOT))
            desired.add(target)
            written.append(candidate)
            next_docs[candidate.slug] = {
                "source": candidate.source,
                "unit": candidate.unit,
                "digest": file_digest(SOURCE_DIR / candidate.source),
                "slug": candidate.slug, "school": candidate.meta.school, "role": candidate.meta.role,
                "title": candidate.meta.title,
                "clean_chars": candidate.report.clean_chars, "cjk_ratio": candidate.report.cjk_ratio,
                "output": candidate.report.output,
            }

        for entry in documents.values():
            if entry["slug"] in cached_slugs:
                next_docs[entry["slug"]] = entry
                desired.add(PROJECT_ROOT / entry.get("output", ""))
        removed = _sweep(desired)

        manifest = {
            "version": 3,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_sec": round(time.time() - started, 1),
            "schools": sorted({entry["school"] for entry in next_docs.values()}),
            "documents": next_docs,
        }
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        removed = []

    return {
        "reports": [r.as_dict() for r in reports],
        "written": [c.report.as_dict() for c in written],
        "removed": removed,
        "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="中医知识库构建 pipeline")
    parser.add_argument("--force", action="store_true", help="忽略缓存强制重建")
    parser.add_argument("--dry-run", action="store_true", help="只体检不落盘")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = parser.parse_args()

    try:
        result = build(force=args.force, dry_run=args.dry_run)
    except BuildAborted as exc:
        print(f"[熔断中止] {exc}")
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"{'源文件/单元':<44} {'学派':<12} {'角色':<8} {'中文字数':>9} {'中文占比':>8}  {'状态':<10} 说明")
    print("-" * 140)
    for row in result["reports"]:
        label = row["unit"] if row["unit"] and row["unit"] != row["source"] else row["source"]
        if row["status"] == "split":
            label = row["source"]
        print(
            f"{label[:42]:<44} {row['school'] or '-':<12} {row['role'] or '-':<8} "
            f"{row['clean_chars']:>9,} {row['cjk_ratio']:>8.2f}  {row['status']:<10} {row['reason'][:30]}"
        )
    print("-" * 140)
    by_school: dict[str, int] = {}
    for row in result["written"]:
        by_school[row["school"]] = by_school.get(row["school"], 0) + 1
    print(f"落盘文档 {len(result['written'])} 篇：" + "，".join(f"{k} {v} 篇" for k, v in sorted(by_school.items())))
    if result["removed"]:
        print(f"清理淘汰产物 {len(result['removed'])} 个：{result['removed']}")
    print(f"清单：{result['manifest']}")


if __name__ == "__main__":
    main()
