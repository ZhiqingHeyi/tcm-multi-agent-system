"""异构电子书抽取层：把 EPUB / PDF / MOBI(AZW) / TXT 统一抽取为纯文本。

设计要点（工业级 ingestion 的第一层）：
1. 格式嗅探优先于扩展名 —— 用户下载的文件扩展名常常是错的。
2. 每种格式独立实现、互不影响，失败只影响单文件，不中断整批。
3. 返回结构化结果（text + 元信息 + 质量指标），由上层决定是否降级或标记 OCR。
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path


class ExtractError(Exception):
    pass


@dataclass
class ExtractResult:
    text: str
    fmt: str
    title: str = ""
    pages: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def density(self) -> float:
        """文本密度：每页字符数。低于阈值说明是扫描版 PDF，需要 OCR 兜底。"""
        if not self.pages:
            return len(self.text)
        return len(self.text) / self.pages


HTML_BLOCK_RE = re.compile(r"</(p|div|h[1-6]|li|tr|section|blockquote)>", re.I)
HTML_BR_RE = re.compile(r"<br\s*/?>", re.I)
HTML_HEAD_RE = re.compile(r"<h([1-6])[^>]*>(.*?)</h\1>", re.I | re.S)
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(html: str) -> str:
    html = SCRIPT_STYLE_RE.sub("", html)
    # 保留标题层级：章节标题是后续切块的天然边界，也是引据溯源的抓手
    html = HTML_HEAD_RE.sub(
        lambda m: "\n\n" + "#" * int(m.group(1)) + " " + TAG_RE.sub("", m.group(2)).strip() + "\n\n", html
    )
    html = HTML_BR_RE.sub("\n", html)
    html = HTML_BLOCK_RE.sub("\n", html)
    html = TAG_RE.sub("", html)
    from html import unescape

    return unescape(html)


def sniff(path: Path) -> str:
    """按魔数嗅探真实格式，扩展名只作参考。"""
    head = path.read_bytes()[:16]
    if head.startswith(b"PK\x03\x04"):
        return "epub" if path.suffix.lower() == ".epub" else "zip"
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"CR!") or head.startswith(b"BOOKMOBI"):
        return "mobi"
    if head[:78].isascii() and len(head) >= 8:
        if _looks_like_palmdb(path):
            return "mobi"
    return "txt"


def _looks_like_palmdb(path: Path) -> bool:
    try:
        raw = path.read_bytes()[:78]
    except OSError:
        return False
    if len(raw) < 78:
        return False
    count = struct.unpack_from(">H", raw, 76)[0]
    type_creator = raw[60:68]
    return 0 < count < 65535 and type_creator in (b"BOOKMOBI", b"TEXtREAd")


def extract(path: Path) -> ExtractResult:
    fmt = sniff(path)
    if fmt == "epub":
        return extract_epub(path)
    if fmt == "pdf":
        return extract_pdf(path)
    if fmt == "mobi":
        return extract_mobi(path)
    return extract_text_file(path)


# --------------------------------------------------------------------------
# EPUB
# --------------------------------------------------------------------------
def extract_epub(path: Path) -> ExtractResult:
    try:
        import ebooklib
        from ebooklib import epub
    except ImportError as exc:
        raise ExtractError("缺少 ebooklib，请安装 backend/requirements-knowledge.txt") from exc

    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    spine_ids = [item_id for item_id, _ in book.spine]
    warnings: list[str] = []

    docs: dict[str, object] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        docs[item.get_id()] = item
        docs[item.get_name()] = item

    ordered: list[object] = []
    for item_id in spine_ids:
        item = docs.get(item_id)
        if item is not None:
            ordered.append(item)
    if not ordered:
        ordered = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        warnings.append("spine 为空，回退为文档列表顺序")

    pieces: list[str] = []
    for item in ordered:
        name = getattr(item, "get_name", lambda: "")() or ""
        if re.search(r"(nav|toc|cover|copyright|titlepage)\.", name, re.I):
            continue
        html = item.get_content().decode("utf-8", errors="ignore")
        text = html_to_text(html).strip()
        if not text:
            continue
        if len(text) < 80 and _is_boilerplate(text):
            continue
        pieces.append(text)

    title = ""
    try:
        meta = book.get_metadata("DC", "title")
        if meta:
            title = str(meta[0][0]).strip()
    except (IndexError, KeyError, TypeError):
        title = ""

    return ExtractResult(text="\n\n".join(pieces), fmt="epub", title=title, pages=len(ordered), warnings=warnings)


BOILERPLATE_RE = re.compile(r"(z-?library|1lib|z-lib|书友|扫码|关注公众号|版权|仅供学习)")


def _is_boilerplate(text: str) -> bool:
    return bool(BOILERPLATE_RE.search(text))


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def extract_pdf(path: Path) -> ExtractResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractError("缺少 pypdf，请安装 backend/requirements-knowledge.txt") from exc

    reader = PdfReader(str(path))
    warnings: list[str] = []
    pieces: list[str] = []
    for page in reader.pages:
        try:
            pieces.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - 单页失败不应中断整本
            warnings.append(f"页面解析失败：{exc}")

    text = "\n\n".join(pieces)
    result = ExtractResult(text=text, fmt="pdf", pages=len(reader.pages), warnings=warnings)
    if result.pages and result.density < 120:
        warnings.append("文本密度过低，疑似扫描版，需要 OCR 兜底")
    return result


# --------------------------------------------------------------------------
# MOBI / AZW（PalmDOC LZ77 解压 + 尾部数据剥离）
# --------------------------------------------------------------------------
def extract_mobi(path: Path) -> ExtractResult:
    raw = path.read_bytes()
    record_count = struct.unpack_from(">H", raw, 76)[0]
    offsets: list[int] = []
    for i in range(record_count):
        pos = 78 + i * 8
        offsets.append(struct.unpack_from(">I", raw, pos)[0])
    offsets.append(len(raw))

    rec0 = raw[offsets[0] : offsets[1]]
    if len(rec0) < 24:
        raise ExtractError("MOBI 记录 0 过短")
    compression, _, text_length, text_records, _, encryption = struct.unpack_from(">HHIHHH", rec0, 0)
    if encryption:
        raise ExtractError("MOBI 带 DRM，无法离线抽取")

    mobi_header = rec0[16:]
    if mobi_header[:4] != b"MOBI":
        raise ExtractError("缺少 MOBI 标识")
    header_length = struct.unpack_from(">I", mobi_header, 4)[0]
    codepage = struct.unpack_from(">I", mobi_header, 12)[0]
    version = struct.unpack_from(">I", mobi_header, 20)[0]
    encoding = "utf-8" if codepage == 65001 else "cp1252"

    extra_flags = 0
    if header_length >= 0xE4 and len(mobi_header) >= 0xF4:
        extra_flags = struct.unpack_from(">H", mobi_header, 0xF2)[0]

    records = [raw[offsets[i] : offsets[i + 1]] for i in range(1, text_records + 1)]
    decoded, used_flags, warnings = _decode_records(records, compression, text_length, extra_flags)
    text = decoded[:text_length].decode(encoding, errors="ignore")
    text = _mobi_html_to_text(text)
    warnings.append(f"MOBI version {version}，尾部标记自校准 0x{used_flags:02X}")
    return ExtractResult(text=text, fmt="mobi", pages=text_records, warnings=warnings)


def _decode_records(
    records: list[bytes], compression: int, text_length: int, declared_flags: int
) -> tuple[bytes, int, list[str]]:
    """解压文本记录，并用声明的 text_length 反向自校准尾部数据标记。

    不同工具生成的 MOBI 尾部标记并不可靠（实测某批次声明 0x1C 却应为 0x03，
    误剥离会丢掉 85% 正文）。这里以 mobi 头声明的 text_length 为判据穷举小范围
    标记取值，选出解压长度与声明值字节级吻合的那种，比盲信头字段稳健得多。
    """
    warnings: list[str] = []
    candidates = [declared_flags] + [f for f in range(0x40) if f != declared_flags]
    best: tuple[int, int, bytes] | None = None

    for flags in candidates:
        payload = bytearray()
        for record in records:
            size = _strip_trailing(record, flags) if flags else len(record)
            payload.extend(record[:size])
        if compression == 1:
            decoded = bytes(payload)
        elif compression == 2:
            decoded = _palmdoc_decompress(bytes(payload))
        else:
            raise ExtractError(f"不支持的压缩方式 {compression}")
        diff = abs(len(decoded) - text_length) if text_length else 0
        if best is None or diff < best[0]:
            best = (diff, flags, decoded)
        if diff == 0:
            break

    assert best is not None
    diff, flags, decoded = best
    if flags != declared_flags:
        warnings.append(f"尾部标记由 0x{declared_flags:02X} 校准为 0x{flags:02X}")
    if text_length and diff > 64:
        warnings.append(f"解压长度与声明相差 {diff} 字节，文本可能不完整")
    return decoded, flags, warnings


def _strip_trailing(record: bytes, extra_flags: int) -> int:
    """剥离记录尾部附加数据，返回真实压缩数据长度。

    MOBI/AZW 会在每条文本记录尾追加密数据（多字节重叠位、条目计数等），
    不剥离会导致 PalmDOC 解压整体错位。位序参照 calibre 的成熟实现。
    """
    size = len(record)
    if size < 8 or extra_flags == 0:
        return size

    def read_entry(end: int) -> int:
        """从 end 位置向前读取一个变长整数（MOBI 尾部数据为逆序紧凑编码）。"""
        value = 0
        bitpos = 0
        cursor = end - 1
        while cursor >= 0:
            byte = record[cursor]
            value |= (byte & 0x7F) << bitpos
            bitpos += 7
            cursor -= 1
            if byte & 0x80 or bitpos >= 28 or cursor < 0:
                return value
        return value

    num = 0
    flags = extra_flags >> 1
    while flags:
        if flags & 1:
            num += read_entry(size - num)
            if num >= size:
                return 0
        flags >>= 1
    if extra_flags & 1:
        num += (record[size - num - 1] & 0x03) + 1
    return max(size - num, 0)


def _palmdoc_decompress(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    n = len(data)
    while i < n:
        byte = data[i]
        i += 1
        if byte == 0x00:
            out.append(0)
        elif byte <= 0x08:
            out.extend(data[i : i + byte])
            i += byte
        elif byte <= 0x7F:
            out.append(byte)
        elif byte <= 0xBF:
            if i >= n:
                break
            pair = (byte << 8) | data[i]
            i += 1
            distance = (pair >> 3) & 0x07FF
            length = (pair & 0x07) + 3
            if distance == 0 or distance > len(out):
                continue
            start = len(out) - distance
            for k in range(length):
                out.append(out[start + k])
        else:
            out.append(0x20)
            out.append(byte ^ 0x80)
    return bytes(out)


def _mobi_html_to_text(html: str) -> str:
    text = re.sub(r"<mbp:pagebreak[^>]*>", "\n\n", html, flags=re.I)
    return html_to_text(text)


# --------------------------------------------------------------------------
# 纯文本
# --------------------------------------------------------------------------
def extract_text_file(path: Path) -> ExtractResult:
    for encoding in ("utf-8", "gb18030", "utf-16"):
        try:
            return ExtractResult(text=path.read_text(encoding=encoding), fmt="txt")
        except UnicodeDecodeError:
            continue
    return ExtractResult(text=path.read_text(encoding="utf-8", errors="ignore"), fmt="txt", warnings=["编码回退"])
