"""合订本拆书（compendium splitting）。

一套 20 册的电子书合集，抽取后往往就是"一个 4MB 的纯文本"。不拆会同时踩三个坑：

1. **元数据失真**：一个文件只能打一套 school/role 标签，而合集里素问、局方、
   针灸大成、衷中参西录分属不同学派与角色，整体打标必然出错。
   错误标签比没有标签更危险——它会让"按学派隔离检索"这个核心能力静默失效。
2. **检索偏置**：4.3M 字的合集贡献了全库近半向量，任何查询的稠密近邻都被它垄断，
   其余文档永久拿不到曝光。这是向量检索里最隐蔽的"数据霸权"问题。
3. **去重失效**：合集内的单本书与已单独入库的版本无法按 slug 对齐，
   同一部书两份共存、互相抢排名。

拆书锚点：每册书重复出现的 CIP 版权页。

    图书在版编目（CIP）数据
    灵枢经/田代华,刘更生整理.-北京:人民卫生出版社,2017

书名取 CIP 行 "/" 之前的部分；首册（无 CIP）退化为从「标准书号/ISBN」行
向前回溯，跳过出版信息字段行后的第一个纯中文短行即为书名。

拆完还要清理每册重复出现的出版方套话（出版者的话/导读/整理说明），
这些内容 20 册各一份，属于纯噪声。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# CIP 版权页标记，兼容全角/半角括号与「数据」间的空格
CIP_RE = re.compile(r"图书在版编目\s*[（(]\s*CIP\s*[)）]\s*数\s*据")

# 出版信息字段行：这些行出现在书名与标准书号之间，回溯书名时要跳过
FIELD_RE = re.compile(
    r"^\s*(整理|著|撰|编|校|重订|点校|出版发行|地\s*址|邮\s*编|E\s*-?\s*mail|"
    r"制作单位|排\s*版|制作时间|版\s*本\s*号|格\s*式|标准书号|ISBN|策划编辑|"
    r"责任编辑|打击盗版|出版者的话|图书在版编目)"
)

# 出版社/机构行。首册无 CIP，需从标准书号向上回溯找书名，
# 而出版发行块正挡在中间——「人民卫生电子音像出版社有限公司」这种行
# 同样是不含标点的纯中文短行，必须显式排除，否则会被误当成书名。
PUBLISHER_RE = re.compile(r"出版发行|出版社|有限公司|音像|电子音像|出版者的话")

# 卷次：景岳全书.上／（下）等，必须保留，否则上下册会撞成同一本书
VOLUME_RE = re.compile(r"[.．·]\s*([上中下])\s*$|[（(]\s*([上中下])\s*[)）]\s*$")

# 每册头部套话的收尾标记：越过它就进入正文
FRONT_MATTER_END_RE = re.compile(r"^#{1,3}\s*(整理说明|重订说明|校注说明|点校说明|凡例)\s*$")

SEPARATORS = str.maketrans({"／": "/", "．": ".", "—": "-", "－": "-"})


@dataclass
class Book:
    """拆出的一册书。volume 为空表示单册。"""

    title: str
    volume: str
    text: str
    start: int
    end: int
    basis: str  # cip / isbn / whole

    @property
    def key(self) -> str:
        return f"{self.title}·{self.volume}" if self.volume else self.title


def _split_title(raw: str) -> tuple[str, str]:
    line = raw.translate(SEPARATORS).strip().strip(".")
    volume = ""
    match = VOLUME_RE.search(line)
    if match:
        volume = match.group(1) or match.group(2) or ""
        line = VOLUME_RE.sub("", line).strip()
    return line.strip("．.· "), volume


def _parse_cip_title(lines: list[str], index: int) -> tuple[str, str] | None:
    """CIP 行本身不含书名，书名在同段后续行里带「作者/出版」结构的那个。"""
    for line in lines[index + 1 : index + 12]:
        candidate = line.translate(SEPARATORS).strip()
        if not candidate or "ISBN" in candidate or "CIP" in candidate:
            continue
        if "/" in candidate and "出版社" in candidate:
            title, volume = _split_title(candidate.split("/")[0])
            if title:
                return title, volume
    return None


def _parse_isbn_title(lines: list[str], index: int) -> tuple[str, str] | None:
    """无 CIP 的书：从标准书号行向上回溯，第一个纯中文短行即书名。

    回溯路上会先撞到出版发行块（出版社名、地址、排版单位…），
    这些行必须整体跳过，否则会把出版社名当成书名。
    """
    for cursor in range(index - 1, max(index - 60, -1), -1):
        line = lines[cursor].strip()
        if not line or FIELD_RE.match(line) or PUBLISHER_RE.search(line):
            continue
        if "：" in line or ":" in line:
            continue
        if 2 <= len(line) <= 20 and not re.search(r"[A-Za-z0-9]", line):
            return _split_title(line)
    return None


def _trim_front_matter(segment: list[str], limit: int = 3000) -> list[str]:
    """剥掉出版方套话：定位「整理说明/重订说明」小节，从其后的下一个标题开始。"""
    cut = 0
    for index, line in enumerate(segment[:limit]):
        if FRONT_MATTER_END_RE.match(line):
            cut = index
    if not cut:
        return segment
    for index in range(cut + 1, limit):
        if re.match(r"^#{1,3}\s+\S", segment[index]):
            return segment[index:]
    return segment


def split_books(text: str, min_book_chars: int = 3000) -> list[Book]:
    """把合订本文本切成独立书籍。识别不到两册以上时原样返回整篇。"""
    lines = text.splitlines()
    starts: list[tuple[int, str, str, str]] = []

    for index, line in enumerate(lines):
        if not CIP_RE.search(line):
            continue
        parsed = _parse_cip_title(lines, index)
        if parsed:
            title, volume = parsed
            starts.append((index, title, volume, "cip"))

    if starts and starts[0][0] > 0:
        head_isbn = next(
            (i for i, line in enumerate(lines[: starts[0][0]]) if re.match(r"^\s*(标准书号|ISBN)", line)),
            None,
        )
        if head_isbn is not None:
            parsed = _parse_isbn_title(lines, head_isbn)
            if parsed:
                title, volume = parsed
                starts.insert(0, (0, title, volume, "isbn"))

    if len(starts) < 2:
        return []

    books: list[Book] = []
    for order, (start, title, volume, basis) in enumerate(starts):
        end = starts[order + 1][0] if order + 1 < len(starts) else len(lines)
        segment = _trim_front_matter(lines[start:end])
        body = "\n".join(segment).strip()
        if len(body) < min_book_chars:
            continue
        books.append(Book(title=title, volume=volume, text=body, start=start, end=end, basis=basis))
    return books if len(books) >= 2 else []
