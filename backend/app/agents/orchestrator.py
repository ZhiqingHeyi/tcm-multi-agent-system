import asyncio
import json
from typing import Any

import httpx

from ..domain.diagnosis.rules import score_candidates
from . import llm
from .contracts import School
from .personas import SCHOOL_PROFILES, SCHOOL_PROMPTS, INTEGRATOR_PROMPT

LLM_ERRORS = (llm.LLMUnavailable, httpx.HTTPError, ValueError, KeyError, IndexError, TimeoutError)

SCHEMA_HINT = (
    "请只输出 JSON，结构为："
    "{\"diagnosis\": 证型名称, \"confidence\": 0到1的小数, "
    "\"evidence\": [\"支持该证型的关键依据\"], \"counter\": [\"不支持或需警惕的依据\"], "
    "\"mechanism\": 病机分析, \"treatment\": 治法, \"formula\": 代表方或None, "
    "\"differentiation\": 鉴别诊断要点}"
)

FOLLOWUP_SCHEMA = (
    "请只输出 JSON：{\"questions\": [{\"title\": 追问, \"key\": 英文字段名, "
    "\"options\": [\"选项1\",\"选项2\",\"选项3\"]}], \"ready\": true或false}。"
    "若信息已足够辨证，返回 {\"questions\": [], \"ready\": true}。"
)



def _load_nihaixia_context(facts: dict[str, Any]) -> str:
    return """
【倪海厦学术背景提示】：
核心纲领：1. 凡病首辨六经传变（太阳-少阳-阳明-太阴-少阴-厥阴）；2. 极度看重手足温凉（脚热则心脏阳气足，脚冷则下焦虚寒）；
3. 饮食二便与睡眠是阳气运行的晴雨表；4. 治法首选汉唐经方，用药直指病机。
表述请带有倪师亲传特色：开宗明义点破六经与寒热，给出确切经方方证。
"""

def format_facts(facts: dict[str, Any]) -> str:
    lines = [f"- {key}: {value}" for key, value in facts.items() if value and value != "不确定"]
    return "\n".join(lines) if lines else "（患者尚未提供有效信息）"


async def analyze_school(school: School, facts: dict[str, Any], risk_flags: list[str]) -> dict[str, Any]:
    if not llm.is_configured():
        return _rule_fallback(school, facts)
    risk_note = f"\n注意：已检测到风险信号 {risk_flags}，如涉及请优先建议就医。" if risk_flags else ""
    prompt = f"患者四诊信息：\n{format_facts(facts)}{risk_note}\n\n请从你所属学派辨证。{SCHEMA_HINT}"
    try:
        data = await llm.chat_json(SCHOOL_PROMPTS[school], prompt, temperature=0.3)
    except LLM_ERRORS:
        return _rule_fallback(school, facts)
    return _normalize_school(school, data)


def _normalize_school(school: School, data: dict[str, Any]) -> dict[str, Any]:
    profile = SCHOOL_PROFILES[school]
    return {
        "school": school.value,
        "name": profile["name"],
        "title": profile["title"],
        "diagnosis": str(data.get("diagnosis") or "信息不足"),
        "confidence": _clamp(data.get("confidence"), 0.5),
        "evidence": _as_list(data.get("evidence")),
        "counter": _as_list(data.get("counter")),
        "mechanism": str(data.get("mechanism") or ""),
        "treatment": str(data.get("treatment") or ""),
        "formula": (str(data["formula"]) if data.get("formula") else None),
        "differentiation": str(data.get("differentiation") or ""),
        "source": "llm",
    }


def _rule_fallback(school: School, facts: dict[str, Any]) -> dict[str, Any]:
    profile = SCHOOL_PROFILES[school]
    top = score_candidates(facts)[0]
    return {
        "school": school.value,
        "name": profile["name"],
        "title": profile["title"],
        "diagnosis": top["name"],
        "confidence": top["score"],
        "evidence": [f"{k}：{v}" for k, v in facts.items() if v and v != "不确定"],
        "counter": [],
        "mechanism": "基于结构化四诊的证素归纳，为规则引擎给出的倾向性判断（未接入大模型）。",
        "treatment": top["treatment"],
        "formula": top["formula"],
        "differentiation": "证据有限，建议补充舌脉与专科信息。",
        "source": "rule",
    }


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        return [value]
    return []


async def run_panel(facts: dict[str, Any], risk_flags: list[str]) -> list[dict[str, Any]]:
    results = await asyncio.gather(*(analyze_school(school, facts, risk_flags) for school in School))
    return list(results)


async def integrate(facts: dict[str, Any], panel: list[dict[str, Any]], risk_flags: list[str]) -> dict[str, Any]:
    if not llm.is_configured():
        return _rule_integrate(facts, panel, risk_flags)
    payload = json.dumps(panel, ensure_ascii=False)
    prompt = (
        f"患者信息：\n{format_facts(facts)}\n\n各学派会诊意见：\n{payload}\n\n"
        + (f"已检测到风险信号 {risk_flags}，最终结论应突出就医建议。" if risk_flags else "")
        + "请只输出 JSON：{\"diagnosis\":主证型, \"confidence\":0到1, \"mechanism\":病机, "
        "\"treatment\":治法, \"formula\":代表方(仅供医师复核)或None, \"modifications\":加减建议, "
        "\"consensus\":[学派共识], \"divergence\":[学派分歧], \"cautions\":[注意与就医建议], "
        "\"followup\":[建议补充的问诊]}"
    )
    try:
        data = await llm.chat_json(INTEGRATOR_PROMPT, prompt, temperature=0.3)
    except LLM_ERRORS:
        return _rule_integrate(facts, panel, risk_flags)
    formula = data.get("formula")
    if risk_flags:
        formula = None
    return {
        "engine": "llm",
        "diagnosis": str(data.get("diagnosis") or panel[0]["diagnosis"]),
        "confidence": _clamp(data.get("confidence"), panel[0]["confidence"]),
        "mechanism": str(data.get("mechanism") or ""),
        "treatment": str(data.get("treatment") or ""),
        "formula": str(formula) if formula else None,
        "modifications": str(data.get("modifications") or ""),
        "consensus": _as_list(data.get("consensus")),
        "divergence": _as_list(data.get("divergence")),
        "cautions": _as_list(data.get("cautions")) or ["本结论仅供健康参考，方药须执业中医师复核"],
        "followup": _as_list(data.get("followup")),
        "risk_flags": risk_flags,
        "evidence": [f"{k}：{v}" for k, v in facts.items() if v and v != "不确定"],
    }


async def plan_followup(facts: dict[str, Any]) -> dict[str, Any]:
    if not llm.is_configured():
        return {"questions": [], "ready": True}
    prompt = f"已收集信息：\n{format_facts(facts)}\n\n{FOLLOWUP_SCHEMA}"
    try:
        data = await llm.chat_json("", prompt, temperature=0.3, role="fast")
    except LLM_ERRORS:
        return {"questions": [], "ready": True}
    questions = []
    for item in data.get("questions", [])[:3]:
        if isinstance(item, dict) and item.get("title"):
            options = _as_list(item.get("options"))
            if "不确定" not in options:
                options.append("不确定")
            questions.append({
                "title": str(item["title"]),
                "key": str(item.get("key") or f"q{len(questions) + 1}"),
                "options": options[:6],
            })
    return {"questions": questions, "ready": bool(data.get("ready")) or not questions}


def _rule_integrate(facts: dict[str, Any], panel: list[dict[str, Any]], risk_flags: list[str]) -> dict[str, Any]:
    top = max(panel, key=lambda item: item["confidence"])
    names = {item["diagnosis"] for item in panel}
    consensus = [f"{item['name']}主张{item['diagnosis']}" for item in panel]
    divergence = [] if len(names) == 1 else [f"诸家证型不完全一致（{len(names)}种），需结合舌脉复核"]
    return {
        "engine": "rule",
        "diagnosis": top["diagnosis"],
        "confidence": top["confidence"],
        "mechanism": top["mechanism"],
        "treatment": top["treatment"],
        "formula": None if risk_flags else top["formula"],
        "modifications": "具体加减须由执业中医师面诊后定夺。",
        "consensus": consensus,
        "divergence": divergence,
        "cautions": ["本结论仅供健康参考", "方剂与剂量须由执业中医师复核", "出现急重症状请立即就医"],
        "followup": ["建议补充舌象、脉象以进一步精确辨证"],
        "risk_flags": risk_flags,
        "evidence": [f"{k}：{v}" for k, v in facts.items() if v and v != "不确定"],
    }


def _clamp(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
