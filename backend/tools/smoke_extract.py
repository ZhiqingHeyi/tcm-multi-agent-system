"""抽取层冒烟测试：对原始下载目录全量文件做格式嗅探与文本抽取体检。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kb.extract import ExtractError, extract, sniff  # noqa: E402

SOURCE = Path("/Users/heyi998/Desktop/GrowVault/多Agent辩证的中医论治系统/知识库原始未处理下载文档")

SKIP_SUFFIX = {".dmg", ".zip"}


def main() -> None:
    rows = []
    for path in sorted(SOURCE.iterdir()):
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIX:
            continue
        fmt = sniff(path)
        try:
            result = extract(path)
            rows.append((path.name, fmt, len(result.text), result.pages, result.density, "；".join(result.warnings)))
        except ExtractError as exc:
            rows.append((path.name, fmt, 0, 0, 0.0, f"FAIL: {exc}"))
        except Exception as exc:  # noqa: BLE001
            rows.append((path.name, fmt, 0, 0, 0.0, f"ERROR: {type(exc).__name__}: {exc}"))

    print(f"{'文件':<44} {'格式':<6} {'字符数':>9} {'页数':>5} {'密度':>8}  备注")
    print("-" * 120)
    for name, fmt, chars, pages, density, note in rows:
        print(f"{name[:42]:<44} {fmt:<6} {chars:>9,} {pages:>5} {density:>8.0f}  {note[:40]}")


if __name__ == "__main__":
    main()
