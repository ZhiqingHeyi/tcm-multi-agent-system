from collections.abc import AsyncIterator
from typing import Any

from .contracts import AgentEvent, ConsultationState, School
from .personas import persona_for
from ..domain.diagnosis.rules import scan_risk, score_candidates


SCHOOL_NOTES = {
    School.SHANGHAN: "伤寒视角：先分六经与表里寒热，经方法度严谨，勿妄投寒凉。",
    School.WENBING: "温病视角：须先辨卫气营血、排除伏热，慎勿过早温燥。",
    School.PIWEI: "脾胃视角：重中气升降，补后天以养四旁，方药甘温为主。",
    School.HUOSHEN: "扶阳视角：重阳气，然须有确切寒象方谈扶阳，附子类有毒须医师复核。",
    School.INTEGRATIVE: "汇通视角：建议结合现代检查指标，辨病与辨证相互印证。",
    School.NIHAIXIA: "倪师经方视角：万病不离阳气虚衰与阴阳开阖，首辨六经归属，直取经方。",
}

SCHOOL_OVERRIDES: dict[tuple[str, School], tuple[str, str]] = {
    ("脾胃阳虚倾向", School.SHANGHAN): ("温中祛寒，调理中焦", "理中汤类方（须执业中医师复核剂量）"),
    ("脾胃阳虚倾向", School.WENBING): ("温中为先，同时复核有无伏热", "暂不建议寒凉，先温中观察"),
    ("脾胃阳虚倾向", School.PIWEI): ("补中益气，升举清阳", "补中益气汤类方（须执业中医师复核剂量）"),
    ("脾胃阳虚倾向", School.HUOSHEN): ("温补脾肾之阳", "附子理中汤类方（附子有毒，务必医师复核）"),
    ("脾胃阳虚倾向", School.INTEGRATIVE): ("健脾温中，兼顾消化功能评估", "香砂六君子汤类方（须执业中医师复核剂量）"),
    ("脾气不足倾向", School.PIWEI): ("健脾益气，升阳举陷", "四君子汤/补中益气汤类方（医师复核）"),
    ("脾气不足倾向", School.SHANGHAN): ("建中益气，甘温除热", "小建中汤类方（医师复核）"),
    ("胃热倾向", School.WENBING): ("清胃生津，护养阴液", "清胃散类方（医师复核）"),
    ("胃热倾向", School.HUOSHEN): ("确有实火方可清降，须辨真寒假热", "暂以清热和胃为先（医师复核）"),
    ("胃热倾向", School.PIWEI): ("清胃而不伤中，顾护脾胃", "益胃汤类方（医师复核）"),
    ("脾胃阳虚倾向", School.NIHAIXIA): ("温补中焦阳气，建中御外", "理中汤或小建中汤类方（医师复核）"),
    ("脾气不足倾向", School.NIHAIXIA): ("补足中焦，建运化之机", "四君子合小建中意（医师复核）"),
    ("胃热倾向", School.NIHAIXIA): ("清阳明之经热，透热出表", "白虎汤或竹叶石膏汤意（医师复核）"),
}


class HermesRuntime:
    def __init__(self) -> None:
        self.available = False
        try:
            import hermes_agent
            self.hermes = hermes_agent
            self.available = True
        except ImportError:
            self.hermes = None

    async def stream_consultation(self, state: ConsultationState, user_input: str) -> AsyncIterator[AgentEvent]:
        flags = scan_risk(state.facts, user_input)
        state.risk_flags = sorted(set(state.risk_flags + flags))
        yield AgentEvent(type="stage_started", stage="orchestrator", message="主控正在整理本轮信息")
        if flags:
            yield AgentEvent(type="risk_detected", stage="safety", message="检测到需要优先就医评估的信号", data={"flags": flags})
            yield AgentEvent(type="completed", stage="safety", message="当前不生成方药建议，请尽快联系专业医疗机构")
            return
        state.messages.append({"role": "user", "content": user_input})
        candidate = score_candidates(state.facts)[0]
        yield AgentEvent(type="fact_proposed", stage="school_agent", message="已形成可确认的症状线索", data={"school": state.school, "facts": state.facts})
        yield AgentEvent(type="agent_message", stage="school_agent", message=self._reply(state.school, candidate["name"]))
        yield AgentEvent(type="completed", stage="orchestrator", message="请确认右侧症状归纳后进入辨证")

    async def generate_report(self, state: ConsultationState) -> dict[str, Any]:
        state.risk_flags = sorted(set(state.risk_flags + scan_risk(state.facts)))
        top = score_candidates(state.facts)[0]
        adjusted = self._school_adjust(state.school, top["name"], top["treatment"], top["formula"])
        formula = None if state.risk_flags else adjusted["formula"]
        return {
            "school": state.school.value,
            "school_note": SCHOOL_NOTES[state.school],
            "diagnosis": top["name"],
            "confidence": top["score"],
            "evidence": [{"label": key, "value": value} for key, value in state.facts.items() if value],
            "mechanism": "以上结论基于你确认的寒热、精神、饮食等线索归纳，仅为倾向性判断，仍需结合面诊、舌象、脉象与必要检查综合确认。",
            "treatment": adjusted["treatment"],
            "formula": formula,
            "modifications": "如兼见腹胀加枳壳、陈皮；失眠多梦加酸枣仁；具体加减须由医师定夺。",
            "cautions": ["本报告仅供健康参考", "方剂与剂量须由执业中医师复核", "出现急重症状请立即就医"],
            "risk_flags": state.risk_flags,
        }

    @staticmethod
    def _school_adjust(school: School, name: str, treatment: str, formula: str | None) -> dict[str, str]:
        override = SCHOOL_OVERRIDES.get((name, school))
        if override:
            treatment, formula = override
        return {"treatment": treatment, "formula": formula or "证据不足，暂不推荐具体方剂", }

    @staticmethod
    def _reply(school: School, diagnosis: str) -> str:
        return f"我已从{persona_for(school).split('。')[0]}的角度，捕捉到“{diagnosis}”这一线索。请核对症状簿后进入辨证推演。"
