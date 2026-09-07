import asyncio
import sys
from pathlib import Path

from .chunker import document_to_chunks
from .embeddings import is_remote_configured
from .store import count_chunks, upsert_chunks

SCHOOL_SOURCES = {
    "shanghan": [
        ("shanghan/01_shanghan_sun.md", "伤寒论条文（太阳篇）"),
        ("shanghan/02_shanghan_other.md", "伤寒论别论"),
        ("shanghan/13_shanghan_quebing.md", "伤寒论阙病补篇"),
        ("shanghan/11_zhongjing_xinfa.md", "仲景心法"),
        ("shanghan/12_stanford_jingfang.md", "经方讲义"),
    ],
    "wenbing": [],
    "piwei": [],
    "huoshen": [
        ("fuyang/10_fuyang_luntan.md", "扶阳论坛"),
    ],
    "integrative": [],
    "nihaixia": [
        ("nihaixia/SKILL_core.md", "倪海厦经方知识库"),
        ("nihaixia/distilled_cases.md", "倪海厦医案精粹"),
        ("nihaixia/06_liangdong.md", "梁冬对话倪海厦"),
        ("nihaixia/07_bimen_hantang.md", "汉唐闭门课"),
    ],
}

KNOWLEDGE_ROOT = Path(__file__).resolve().parents[2] / "data" / "knowledge"


async def ingest(schools: list[str] | None = None) -> dict[str, int]:
    schools = schools or [s for s in SCHOOL_SOURCES if SCHOOL_SOURCES[s]]
    total = 0
    per_school: dict[str, int] = {}
    for school in schools:
        school_total = 0
        for rel_path, source_name in SCHOOL_SOURCES.get(school, []):
            file_path = KNOWLEDGE_ROOT / rel_path
            if not file_path.exists():
                continue
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunks = document_to_chunks(text, school=school, source=source_name)
            school_total += await upsert_chunks(chunks)
        per_school[school] = school_total
        total += school_total
    return {"total": total, "per_school": per_school, "db_count": await count_chunks(), "remote_embeddings": is_remote_configured()}


def main() -> None:
    args = sys.argv[1:]
    result = asyncio.run(ingest(args or None))
    print(result)


if __name__ == "__main__":
    main()
