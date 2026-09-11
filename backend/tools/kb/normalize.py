"""中医古籍文本规范化层。

抽取层解决"能不能读出来"，规范化层解决"读出来干不干净"。
这一层直接决定向量质量：RAG 检索里最常见的问题不是模型不行，
而是切块里混着页码、目录点和版权广告，把语义向量污染了。

处理顺序（顺序很关键，先粗后细）：
1. 编码与符号归一（全角空格、零宽字符、引号统一）
2. 结构性噪声剔除（目录页、版权页、水印广告、URL）
3. 行级噪声剔除（孤立页码、页眉页脚、装饰线）
4. 断行修复（中文段落内的物理换行合并）
5. 段落去重（同一段重复出现只保留首次）
6. 空白整形（压缩连续空行、去首尾空白）
"""

from __future__ import annotations

import hashlib
import re

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)

# 目录行：正文 + 大量引导点/空格 + 页码，或纯 "方名<空白>数字"
TOC_DOT_RE = re.compile(r"[.·。…\u2026]{3,}\s*\d+\s*$")
TOC_LEADER_RE = re.compile(r"^\s*\S.{0,40}?[\s.]{3,}\d{1,4}\s*$")
TOC_TAIL_NUM_RE = re.compile(r"^.{2,40}?\s{2,}\d{1,4}\s*$")

# 孤立页码 / 页眉页脚
PAGE_NUM_RE = re.compile(r"^\s*[-—–\[【(]*\s*\d{1,4}\s*[-—–\]】)]*\s*$")
RUNNING_HEAD_RE = re.compile(
    r"^\s*(第\s*[一二三四五六七八九十百零\d]+\s*(页|章|节|卷|篇)|"
    r"Page\s*\d+|\d+\s*/\s*\d+|.{0,20}(z-?library|1lib|z-lib).{0,20})\s*$",
    re.I,
)

# 推广/水印/广告
AD_RE = re.compile(
    r"(z-?library|1lib|z-lib|扫码|二维码|关注(公众)?号|微信号|加群|QQ群|"
    r"免费下载|电子书|侵权|删除|仅供(学习|参考|交流)|请支持正版|淘宝|微店|"
    r"更多(电子书|资源)|www\.[a-z0-9\-]+\.[a-z]{2,}|https?://|@gmail|@qq\.com)",
    re.I,
)

DECOR_LINE_RE = re.compile(r"^\s*[-=*~_—·◇◆※☆★]{3,}\s*$")

PUNCT_MAP = str.maketrans({
    "“": "「", "”": "」", "‘": "『", "’": "』",
    "\u3000": " ",
    "﹑": "、", "‧": "·",
})

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SENT_END = "。！？；：」』）】…!?;:"
HEAD_HINT_RE = re.compile(
    r"^\s*(第[一二三四五六七八九十百千\d]+[条章节篇卷回]|"
    r"[一二三四五六七八九十]+[、.．]|"
    r"[（(][一二三四五六七八九十\d]{1,3}[)）]|"
    r"【.{1,20}】|"
    r"[A-Za-z\u4e00-\u9fff]{2,12}(方|汤|散|丸|饮|证|论|篇|法|病|门)\s*$)"
)


def _try_t2s(text: str) -> str:
    """繁体转简体：opencc 可用则用，否则原样保留（不引入硬依赖）。"""
    try:
        from opencc import OpenCC
    except ImportError:
        return text
    global _T2S
    try:
        _T2S
    except NameError:
        _T2S = OpenCC("t2s")
    return _T2S.convert(text)


def normalize(text: str, to_simplified: bool = True) -> str:
    text = text.translate(ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if to_simplified:
        text = _try_t2s(text)

    text = _drop_front_matter(text)
    lines = text.split("\n")
    kept: list[str] = []
    toc_run = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            kept.append("")
            continue
        if AD_RE.search(line) and len(line) < 120:
            continue
        if DECOR_LINE_RE.match(line):
            continue
        if PAGE_NUM_RE.match(line) or RUNNING_HEAD_RE.match(line):
            continue
        if TOC_DOT_RE.search(line) or TOC_LEADER_RE.match(line):
            toc_run += 1
            if toc_run <= 400:
                continue
        elif TOC_TAIL_NUM_RE.match(line) and toc_run >= 3:
            toc_run += 1
            if toc_run <= 400:
                continue
        else:
            toc_run = 0
        kept.append(line)

    text = "\n".join(kept)
    text = _merge_broken_lines(text)
    text = _dedupe_blocks(text)
    text = _normalize_punct(text)
    text = _reshape_blank(text)
    return text


def _drop_front_matter(text: str) -> str:
    """裁掉开头的封面/版权/书名页噪声，从首个正文特征行开始。"""
    lines = text.split("\n")
    for index, line in enumerate(lines[:400]):
        stripped = line.strip()
        if len(stripped) >= 40 and CJK_RE.search(stripped):
            return "\n".join(lines[index:])
    return text


_CHAPTER_HINT = re.compile(r"(第[一二三四五六七八九十百\d]+[章卷篇回]|^\s*[一二三四五六七八九十]+\s*$)")


def _merge_broken_lines(text: str) -> str:
    """修复 PDF/MOBI 抽取产生的物理断行，但不破坏标题与条文边界。"""
    lines = text.split("\n")
    out: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer:
            out.append(buffer)
            buffer = ""

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            out.append("")
            continue

        if stripped.startswith("#"):
            flush()
            out.append(stripped)
            out.append("")
            continue

        if not buffer:
            buffer = stripped
            continue

        prev_char = buffer[-1]
        prev_short = len(buffer) <= 18
        looks_new_block = (
            HEAD_HINT_RE.match(stripped)
            or _CHAPTER_HINT.match(stripped)
            or PAGE_NUM_RE.match(stripped)
            or prev_char in SENT_END
            or prev_short
            or stripped[0] in "「『（(【["
        )

        if looks_new_block:
            flush()
            buffer = stripped
        else:
            glue = "" if CJK_RE.match(prev_char) and CJK_RE.match(stripped[0]) else " "
            buffer = f"{buffer}{glue}{stripped}"

        if len(buffer) > 600:
            flush()

    flush()
    return "\n".join(out)


def _dedupe_blocks(text: str, min_len: int = 60) -> str:
    """段落级去重：电子书常见正文与附录重复、跨页重复。"""
    blocks = text.split("\n\n")
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        key_src = re.sub(r"\s+", "", block)
        if len(key_src) < min_len:
            out.append(block)
            continue
        digest = hashlib.blake2b(key_src.encode("utf-8"), digest_size=16).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(block)
    return "\n\n".join(out)


def _normalize_punct(text: str) -> str:
    text = text.translate(PUNCT_MAP)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"([\u4e00-\u9fff])\s+([，。、；：！？）」』])", r"\1\2", text)
    text = re.sub(r"([（「『])\s+", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _reshape_blank(text: str) -> str:
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(line)
    return "\n".join(out).strip()


def quality_report(text: str) -> dict[str, float]:
    """产出可观测的质量指标，用于入库前卡口（gate）。"""
    total = max(len(text), 1)
    cjk = len(CJK_RE.findall(text))
    lines = [line for line in text.split("\n") if line.strip()]
    return {
        "chars": len(text),
        "cjk_ratio": round(cjk / total, 4),
        "lines": len(lines),
        "avg_line": round(total / max(len(lines), 1), 1),
        "noise_ratio": round(len(AD_RE.findall(text)) / max(len(lines), 1), 4),
    }
