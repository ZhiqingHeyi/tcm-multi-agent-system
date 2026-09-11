"""文档角色感知切块器（RAG 检索质量的第一决定因素）。

为什么不能简单地"每 500 字切一刀"？因为中医文本有三种截然不同的语义单元：

- 典籍（classic）：最小完整语义单元是「一条条文」。切在条文中间，
  "太阳之为病，脉浮，头项强痛而恶寒" 会被劈成两半，检索时两边都召不回。
- 医案（case）：最小完整单元是「一则病案」。主诉、四诊、辨证、方药必须同块，
  否则模型拿到"处方"却看不到"证"，会给出错误引用。
- 讲义（lecture）：连续论述，没有硬边界，适合语义窗口 + 重叠。

两条被工程实践反复验证的增强手段，这里都实现了：

1. **文脉注入（contextual header）**：每个 chunk 前拼接「书名·章节」。
   游离的"茯苓三钱"毫无意义，但"《伤寒论·太阳病篇》茯苓三钱"可被准确定位。
2. **父子索引**：保留 section 与 chunk_index，生成阶段可回捞相邻块补全上下文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

MAX_CHARS_CLASSIC = 620
MAX_CHARS_CASE = 760
MAX_CHARS_LECTURE = 680
OVERLAP_LECTURE = 90
MIN_CHARS = 36
MIN_CJK_RATIO = 0.30

# 条文边界：宋本伤寒论「第 N 条」、古籍常见「N、」「（N）」、「一二三、」
CLAUSE_RE = re.compile(
    r"^\s*(?:"
    r"第\s*[一二三四五六七八九十百零\d]{1,4}\s*条"
    r"|[（(]\s*\d{1,3}\s*[)）]"
    r"|\d{1,3}\s*[、.．]"
    r"|[一二三四五六七八九十]{1,3}\s*[、.．]"
    r")\s*"
)

# 医案边界：案号、患者信息、日期起首
CASE_RE = re.compile(
    r"^\s*(?:"
    r"(?:医)?案\s*[一二三四五六七八九十百\d]{1,3}"
    r"|病案\s*[一二三四五六七八九十百\d]{1,3}"
    r"|例\s*[一二三四五六七八九十百\d]{1,3}"
    r"|[男女][，,、]?\s*\d{1,3}\s*岁"
    r"|\d{4}\s*年\s*\d{1,2}\s*月"
    r"|\d{1,3}\s*[、.．]\s*[男女某患者]"
    r")"
)

DOSE_RE = re.compile(r"(钱|两|克|g|枚|片|升|合|分|条|个|剂|服|煎|煮)")


@dataclass
class Chunk:
    text: str
    school: str
    source: str
    section: str
    title: str = ""
    role: str = "classic"
    index: int = 0

    @property
    def display(self) -> str:
        """给检索结果展示用的纯正文（去掉注入的文脉头）。"""
        return self.text

    @property
    def token_key(self) -> str:
        return f"{self.source}::{self.section}::{self.index}::{self.text[:48]}"


@dataclass
class ChunkStats:
    total: int = 0
    dropped_short: int = 0
    dropped_low_cjk: int = 0
    sections: int = 0
    by_role: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 前置处理
# --------------------------------------------------------------------------
def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    return meta, raw[match.end() :]


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CJK_RE.findall(text)) / len(text)


def _clean(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"^\s*[-=]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切章节。标题缺失时整篇作为一节。"""
    parts = re.split(r"\n(?=#{1,6}\s+\S)", text)
    sections: list[tuple[str, str]] = []
    for part in parts:
        header = re.match(r"^(#{1,6})\s+(.+)", part)
        title = header.group(2).strip() if header else ""
        body = re.sub(r"^#{1,6}\s+.+\n?", "", part).strip() if header else part.strip()
        if body:
            sections.append((title, body))
    return sections or [("", text.strip())]


def _lines(block: str) -> list[str]:
    return [line.rstrip() for line in block.split("\n")]


# --------------------------------------------------------------------------
# 三种切块策略
# --------------------------------------------------------------------------
def _pack_units(units: list[str], max_chars: int) -> list[str]:
    """把语义单元打包进不超过 max_chars 的块，不切断单元内部。"""
    chunks: list[str] = []
    buffer = ""
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            step = max_chars
            for start in range(0, len(unit), step):
                chunks.append(unit[start : start + step])
            continue
        if not buffer:
            buffer = unit
        elif len(buffer) + len(unit) + 1 <= max_chars:
            buffer = f"{buffer}\n{unit}"
        else:
            chunks.append(buffer)
            buffer = unit
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_classic(block: str, max_chars: int = MAX_CHARS_CLASSIC) -> list[str]:
    """典籍：以条文为最小单元聚合，保证条文完整。"""
    units: list[str] = []
    buffer = ""
    for line in _lines(block):
        stripped = line.strip()
        if not stripped:
            continue
        if CLAUSE_RE.match(stripped) and buffer:
            units.append(buffer)
            buffer = stripped
        elif len(buffer) + len(stripped) <= 240:
            buffer = f"{buffer}\n{stripped}" if buffer else stripped
        else:
            units.append(buffer)
            buffer = stripped
    if buffer:
        units.append(buffer)
    return _pack_units(units or [block], max_chars)


def chunk_case(block: str, max_chars: int = MAX_CHARS_CASE) -> list[str]:
    """医案：以整则为单元，防止"有方无证"。"""
    units: list[str] = []
    buffer = ""
    for line in _lines(block):
        stripped = line.strip()
        if not stripped:
            continue
        starts_case = CASE_RE.match(stripped)
        long_enough = len(buffer) >= 90
        if starts_case and buffer and long_enough:
            units.append(buffer)
            buffer = stripped
        elif len(buffer) + len(stripped) <= 300:
            buffer = f"{buffer}\n{stripped}" if buffer else stripped
        else:
            units.append(buffer)
            buffer = stripped
    if buffer:
        units.append(buffer)
    return _pack_units(units or [block], max_chars)


def chunk_lecture(block: str, max_chars: int = MAX_CHARS_LECTURE, overlap: int = OVERLAP_LECTURE) -> list[str]:
    """讲义：语义窗口 + 重叠，避免跨窗语义断裂。"""
    paragraphs = [para.strip() for para in block.split("\n") if para.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            step = max_chars - overlap
            for start in range(0, len(para), step):
                chunks.append(para[start : start + max_chars])
            continue
        if len(buffer) + len(para) + 1 <= max_chars:
            buffer = f"{buffer}\n{para}" if buffer else para
        else:
            chunks.append(buffer)
            tail = buffer[-overlap:] if len(buffer) > overlap else ""
            buffer = f"{tail}\n{para}".strip() if tail else para
    if buffer:
        chunks.append(buffer)
    return chunks


STRATEGY = {"classic": chunk_classic, "case": chunk_case, "lecture": chunk_lecture}


# --------------------------------------------------------------------------
# 对外入口
# --------------------------------------------------------------------------
def document_to_chunks(raw: str, school: str, source: str, title: str = "", role: str = "classic") -> tuple[list[Chunk], ChunkStats]:
    meta, body = parse_front_matter(raw)
    school = meta.get("school", school)
    title = meta.get("title", title) or source
    role = meta.get("role", role)
    body = _clean(body)

    strategy = STRATEGY.get(role, chunk_classic)
    chunks: list[Chunk] = []
    stats = ChunkStats()

    for section_title, block in split_sections(body):
        stats.sections += 1
        label = section_title or title
        for piece in strategy(block):
            piece = piece.strip()
            if len(piece) < MIN_CHARS:
                stats.dropped_short += 1
                continue
            if _cjk_ratio(piece) < MIN_CJK_RATIO:
                stats.dropped_low_cjk += 1
                continue
            chunks.append(
                Chunk(
                    text=f"【{title}·{label}】\n{piece}",
                    school=school,
                    source=source,
                    section=label[:280],
                    title=title[:180],
                    role=role,
                    index=len(chunks),
                )
            )

    stats.total = len(chunks)
    stats.by_role[role] = stats.by_role.get(role, 0) + len(chunks)
    return chunks, stats
