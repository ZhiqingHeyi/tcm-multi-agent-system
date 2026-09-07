import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    school: str
    source: str
    section: str

    @property
    def token_key(self) -> str:
        return f"{self.source}::{self.section}::{self.text[:40]}"


def _clean(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^\s*[-*+]\s*\[[ xX]\]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"^\s*[-=]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


def _truncate_role(text: str) -> str:
    lower = text.lower()
    cut = len(text)
    for marker in ("---\nname:", "# 倪师表达速查卡", "expression style"):
        idx = lower.find(marker.lower())
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut] if cut < len(text) else text


def split_sections(text: str) -> list[tuple[str, str]]:
    text = _truncate_role(text)
    parts = re.split(r"\n(?=#{1,3}\s+\S)", text)
    sections: list[tuple[str, str]] = []
    for part in parts:
        header_match = re.match(r"^(#{1,3})\s+(.+)", part)
        section = header_match.group(2).strip() if header_match else ""
        body = part.strip()
        if body:
            sections.append((section, body))
    return sections


def split_chunks(text: str, max_chars: int = 640, overlap: int = 80) -> list[str]:
    text = _clean(text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= max_chars:
            buffer = f"{buffer}\n{para}" if buffer else para
            continue
        if buffer:
            chunks.append(buffer)
            buffer = buffer[-overlap:] + ("\n" + para if para else "")
        else:
            while len(para) > max_chars:
                chunks.append(para[:max_chars])
                para = para[max_chars - overlap :]
            buffer = para
    if buffer:
        chunks.append(buffer)
    return [c for c in chunks if len(c.strip()) > 10]


def document_to_chunks(text: str, school: str, source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(text):
        for piece in split_chunks(body):
            chunks.append(Chunk(text=piece, school=school, source=source, section=section or source))
    return chunks
