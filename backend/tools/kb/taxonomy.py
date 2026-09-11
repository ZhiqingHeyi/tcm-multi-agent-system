"""学派归类与文档角色识别。

知识库最容易被忽视、却最影响检索质量的一步：**元数据打标**。
向量只负责"像不像"，元数据负责"该不该出现"。没有学派标签，
火神派的姜附重剂医案会污染温病派的检索结果。

这里采用「显式登记表 + 启发式兜底」的双轨策略：
- 已确认的语料走登记表，结果确定、可复现、可评审；
- 未知新文件走启发式，保证 pipeline 不会因为新增语料而中断。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SCHOOLS = ("shanghan", "wenbing", "piwei", "huoshen", "integrative", "nihaixia")

# 共享基础层：内经、本草、脉经、局方这类典籍不属于任何一派，
# 而是六派共同的源头。把它们硬塞给某一派，等于让其它五派检索不到自己的根。
# 检索时用「本派 ∪ common」做范围，既保隔离又不丢基础。
COMMON = "common"

ROLE_CLASSIC = "classic"   # 典籍原文：条文式，切块按条文边界
ROLE_CASE = "case"         # 医案实录：一则一案，切块按案例边界
ROLE_LECTURE = "lecture"   # 讲义论著：连续论述，切块按语义窗口


@dataclass(frozen=True)
class DocMeta:
    school: str
    role: str
    title: str
    slug: str
    priority: int = 50  # 同名不同版本时的择优优先级，越大越优先


# 关键词 → 归类。命中顺序从上到下，先精确后宽泛。
REGISTRY: tuple[tuple[str, DocMeta], ...] = (
    ("倪海厦-人纪-伤寒论", DocMeta("nihaixia", ROLE_LECTURE, "倪海厦人纪·伤寒论讲义", "renji_shanghan", 90)),
    ("人纪系列之-金匮要略", DocMeta("nihaixia", ROLE_LECTURE, "倪海厦人纪·金匮要略讲义", "renji_jingui", 90)),
    ("人纪系列之-黄帝内经", DocMeta("nihaixia", ROLE_LECTURE, "倪海厦人纪·黄帝内经讲义", "renji_neijing", 90)),
    ("倪海厦人纪系列之黄帝内经", DocMeta("nihaixia", ROLE_LECTURE, "倪海厦人纪·黄帝内经讲义", "renji_neijing", 90)),
    ("人纪系列之针灸教程", DocMeta("nihaixia", ROLE_LECTURE, "倪海厦人纪·针灸教程", "renji_zhenjiu", 90)),
    ("倪海厦《黄帝内经》视频同步文稿", DocMeta("nihaixia", ROLE_LECTURE, "倪海厦黄帝内经视频同步文稿", "neijing_video", 85)),
    ("倪海厦先生医案", DocMeta("nihaixia", ROLE_CASE, "倪海厦先生医案", "nihaixia_cases", 88)),
    ("天纪", DocMeta("nihaixia", ROLE_LECTURE, "倪海厦天纪", "tianji", 80)),
    ("中医临床必读丛书典藏版", DocMeta(COMMON, ROLE_CLASSIC, "中医临床必读丛书（典藏版）", "bcl_readings", 95)),
    ("金匮要略 (张仲景", DocMeta("shanghan", ROLE_CLASSIC, "金匮要略（宋本）", "jingui_yaolue", 90)),
    ("伤寒贯珠集", DocMeta("shanghan", ROLE_CLASSIC, "伤寒贯珠集", "shanghan_guanzhu", 88)),
    ("新编伤寒论类方", DocMeta("shanghan", ROLE_CLASSIC, "新编伤寒论类方", "shanghan_leifang", 92)),
    ("独立全解《经方实验录》医案", DocMeta("shanghan", ROLE_CASE, "经方实验录医案全解", "jingfang_shiyanlu", 88)),
    ("温病条辨", DocMeta("wenbing", ROLE_CLASSIC, "温病条辨·温热论·湿热病篇", "wenbing_tiaobian", 92)),
    ("脾胃论", DocMeta("piwei", ROLE_CLASSIC, "脾胃论", "piwei_lun", 92)),
    ("内外伤辨惑论", DocMeta("piwei", ROLE_CLASSIC, "内外伤辨惑论", "neishang_bianhuo", 90)),
    ("兰室秘藏", DocMeta("piwei", ROLE_CLASSIC, "兰室秘藏", "lanshi_micang", 88)),
    ("郑钦安中医火神三书", DocMeta("huoshen", ROLE_CLASSIC, "郑钦安火神三书", "huoshen_sanshu", 90)),
    ("医学衷中参西录", DocMeta("integrative", ROLE_CASE, "医学衷中参西录", "zhongzhong_canxi", 92)),
)

# 合订本标记：命中即触发 compendium 拆书，按册分别归类。
COMPENDIUM_MARKERS = ("中医临床必读丛书典藏版", "现代著名老中医名著重刊丛书")

# 《中医临床必读丛书（典藏版）》20 册分册登记表。
# priority 统一为 80（兜底版本）：若同一部书另有专书电子版（priority 88+），
# 专书胜出、合订本对应册自动退让，避免同书多版本在库里抢排名。
COMPENDIUM_BOOKS: dict[str, DocMeta] = {
    "黄帝内经素问": DocMeta(COMMON, ROLE_CLASSIC, "黄帝内经素问", "huangdi_neijing_suwen", 80),
    "灵枢经": DocMeta(COMMON, ROLE_CLASSIC, "灵枢经", "lingshu_jing", 80),
    "金匮要略": DocMeta("shanghan", ROLE_CLASSIC, "金匮要略", "jingui_yaolue", 80),
    "温病条辨": DocMeta("wenbing", ROLE_CLASSIC, "温病条辨", "wenbing_tiaobian", 80),
    "丹溪心法": DocMeta(COMMON, ROLE_CLASSIC, "丹溪心法", "danxi_xinfa", 80),
    "本草备要": DocMeta(COMMON, ROLE_CLASSIC, "本草备要", "bencao_beiyao", 80),
    "兰室秘藏": DocMeta("piwei", ROLE_CLASSIC, "兰室秘藏", "lanshi_micang", 80),
    "太平惠民和剂局方": DocMeta(COMMON, ROLE_CLASSIC, "太平惠民和剂局方", "ju_fang", 80),
    "针灸甲乙经": DocMeta(COMMON, ROLE_CLASSIC, "针灸甲乙经", "zhenjiu_jiayijing", 80),
    # 针灸大成为倪海厦人纪针灸教程所本，归入汉唐门生一脉
    "针灸大成": DocMeta("nihaixia", ROLE_CLASSIC, "针灸大成", "zhenjiu_dacheng", 80),
    "脉经": DocMeta(COMMON, ROLE_CLASSIC, "脉经", "maijing", 80),
    # 赵献可《医贯》专论命门水火，是火神派学说的直接源头
    "医贯": DocMeta("huoshen", ROLE_CLASSIC, "医贯", "yiguan", 80),
    "遵生八笺": DocMeta(COMMON, ROLE_LECTURE, "遵生八笺", "zunsheng_bajian", 80),
    # 唐宗海为中西医汇通四大家之一
    "血证论": DocMeta("integrative", ROLE_CLASSIC, "血证论", "xuezheng_lun", 80),
    "素问病机气宜保命集": DocMeta(COMMON, ROLE_CLASSIC, "素问病机气宜保命集", "baoming_ji", 80),
    "儒门事亲": DocMeta(COMMON, ROLE_CLASSIC, "儒门事亲", "rumen_shiqin", 80),
    "景岳全书": DocMeta(COMMON, ROLE_CLASSIC, "景岳全书", "jingyue_quanshu", 80),
    "重订医学衷中参西录": DocMeta("integrative", ROLE_CLASSIC, "重订医学衷中参西录", "zhongzhong_canxi", 80),
}

# 《现代著名老中医名著重刊丛书·第10辑：刘渡舟医书七种》7 册分册登记表。
# 刘渡舟是当代经方大家，七种均归伤寒一脉；「讲话/十四讲/诠解」是讲义，
# 「类方」是方剂归类条文，「临证指南」是医案，角色必须区分，
# 否则讲义会按条文边界被切碎、医案会被按语义窗口混成一团。
COMPENDIUM_BOOKS.update({
    "伤寒论通俗讲话": DocMeta("shanghan", ROLE_LECTURE, "伤寒论通俗讲话（刘渡舟）", "shanghan_jianghua", 85),
    "伤寒论十四讲": DocMeta("shanghan", ROLE_LECTURE, "伤寒论十四讲（刘渡舟）", "shanghan_shisijiang", 85),
    "伤寒论诠解": DocMeta("shanghan", ROLE_LECTURE, "伤寒论诠解（刘渡舟）", "shanghan_quanjie", 85),
    "新编伤寒论类方": DocMeta("shanghan", ROLE_CLASSIC, "新编伤寒论类方（刘渡舟）", "shanghan_leifang", 88),
    "金匮要略诠解": DocMeta("shanghan", ROLE_LECTURE, "金匮要略诠解（刘渡舟）", "jingui_quanjie", 85),
    "经方临证指南": DocMeta("shanghan", ROLE_CASE, "经方临证指南（刘渡舟）", "jingfang_linzheng", 85),
    "肝病证治概要": DocMeta("shanghan", ROLE_LECTURE, "肝病证治概要（刘渡舟）", "ganbing_zhengzhi", 85),
})

HEURISTICS: tuple[tuple[str, str, str], ...] = (
    ("温病|温热|卫气营血|三焦辨证|银翘|桑菊", "wenbing", ROLE_CLASSIC),
    ("脾胃|补中益气|升阳益胃|东垣", "piwei", ROLE_CLASSIC),
    ("火神|扶阳|钦安|四逆|附子", "huoshen", ROLE_CLASSIC),
    ("参西|汇通|衷中", "integrative", ROLE_LECTURE),
    ("倪海厦|汉唐|人纪|天纪", "nihaixia", ROLE_LECTURE),
    ("医案|验案|临证", "*", ROLE_CASE),
    ("伤寒|金匮|经方|仲景", "shanghan", ROLE_CLASSIC),
)

EDITION_PRIORITY = {".epub": 20, ".md": 25, ".azw": 15, ".mobi": 15, ".pdf": 5}

VOLUME_SLUG = {"上": "shang", "中": "zhong", "下": "xia"}


def _normalize_name(name: str) -> str:
    return unicodedata.normalize("NFKC", name)


def classify(filename: str) -> DocMeta | None:
    name = _normalize_name(filename)
    for keyword, meta in REGISTRY:
        if keyword in name:
            return meta
    for pattern, school, role in HEURISTICS:
        if re.search(pattern, name):
            resolved = school if school != "*" else "shanghan"
            slug = re.sub(r"[^\w]+", "_", name)[:40].strip("_")
            return DocMeta(resolved, role, name.split("(")[0].strip()[:60], slug)
    return None


def edition_score(path_suffix: str, text_length: int, priority: int) -> int:
    """同一部书存在多个版本时择优：优先级 > 格式 > 文本量。"""
    return priority * 1_000_000_000 + EDITION_PRIORITY.get(path_suffix.lower(), 0) * 1_000_000 + min(text_length, 999_999)


def is_compendium(filename: str) -> bool:
    name = _normalize_name(filename)
    return any(marker in name for marker in COMPENDIUM_MARKERS)


def resolve_book(title: str) -> DocMeta | None:
    """按书名（而非文件名）归类合订本中的一册。

    包含匹配只在「唯一命中且书名足够长」时生效。否则《金匮要略诠解》会
    被《金匮要略》抢先匹配，两本不同的书被当成同一部而互相顶掉。
    """
    name = _normalize_name(title).strip()
    meta = COMPENDIUM_BOOKS.get(name)
    if meta:
        return meta
    hits = [value for key, value in COMPENDIUM_BOOKS.items() if len(key) >= 4 and (key in name or name in key)]
    return hits[0] if len(hits) == 1 else None


def compose_slug(meta: DocMeta, volume: str) -> str:
    """上下册必须区分为两本书，否则会在去重阶段互相顶掉一半内容。"""
    suffix = VOLUME_SLUG.get(volume, "")
    return f"{meta.slug}_{suffix}" if suffix else meta.slug
