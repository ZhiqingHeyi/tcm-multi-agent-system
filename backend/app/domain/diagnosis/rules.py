from typing import Any

DANGER_TERMS = {"胸痛", "呼吸困难", "意识不清", "大出血", "高热不退", "孕期出血"}


def scan_risk(facts: dict[str, Any], text: str = "") -> list[str]:
    haystack = f"{text} {facts}".lower()
    flags = [term for term in DANGER_TERMS if term.lower() in haystack]
    if facts.get("pregnant") is True:
        flags.append("孕产场景")
    if facts.get("age") is not None:
        try:
            if int(facts["age"]) < 18:
                flags.append("未成年人")
        except (TypeError, ValueError):
            pass
    if facts.get("serious_conditions"):
        flags.append("严重慢病")
    return sorted(set(flags))


_COLD_OPTION = {"经常怕冷"}


def _has(facts: dict[str, Any], key: str, positive: set[str], negative: set[str] | None = None) -> bool:
    value = facts.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value in positive
    return False


def score_candidates(facts: dict[str, Any]) -> list[dict[str, Any]]:
    cold = _has(facts, "cold", _COLD_OPTION)
    fatigue = _has(facts, "fatigue", {"容易疲倦"})
    poor_appetite = _has(facts, "poor_appetite", {"食欲不振"})
    heat = _has(facts, "heat", {"经常怕热", "手足心热"})
    thirst = _has(facts, "thirst", {"口渴喜饮", "经常口渴"})
    candidates = []
    if cold and fatigue:
        candidates.append({"name": "脾胃阳虚倾向", "score": 0.72, "treatment": "温中健脾，调和营卫", "formula": "理中类方（仅供医师参考）"})
    if poor_appetite and fatigue and not cold:
        candidates.append({"name": "脾气不足倾向", "score": 0.66, "treatment": "健脾益气，改善运化", "formula": "四君子类方（仅供医师参考）"})
    if heat and thirst:
        candidates.append({"name": "胃热倾向", "score": 0.68, "treatment": "清胃生津，和中护阴", "formula": "清胃类方（仅供医师参考）"})
    if not candidates:
        candidates.append({"name": "证据不足，待进一步问诊", "score": 0.32, "treatment": "暂不定治，补充四诊资料", "formula": None})
    return sorted(candidates, key=lambda item: item["score"], reverse=True)