import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..agents import llm, orchestrator
from ..agents.contracts import School
from ..agents.personas import SCHOOL_PROFILES
from ..domain.diagnosis.rules import scan_risk
from ..domain.questionnaire import QUESTIONNAIRE, TOTAL_QUESTIONS
from ..security import create_access_token, hash_password, verify_password
from .. import settings_store

router = APIRouter(prefix="/api")
sessions: dict[str, dict[str, Any]] = {}
users: dict[str, dict[str, str]] = {}

class AuthInput(BaseModel):
    email: str
    password: str = Field(min_length=8, max_length=128)

class CreateConsultation(BaseModel):
    school: School = School.PIWEI

class FactsInput(BaseModel):
    facts: dict[str, Any] = Field(default_factory=dict)

class LLMConfigInput(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    model_fast: str | None = None
    model_pro: str | None = None

class TestInput(BaseModel):
    role: str = "both"

def _require_admin(x_admin_token: str | None) -> None:
    if x_admin_token != settings_store.settings.admin_token:
        raise HTTPException(status_code=401, detail="管理员令牌无效")

@router.post("/auth/register")
async def register(payload: AuthInput) -> dict[str, str]:
    email = payload.email.strip().lower()
    if email in users:
        raise HTTPException(status_code=409, detail="邮箱已注册")
    users[email] = {"email": email, "password_hash": hash_password(payload.password)}
    return {"access_token": create_access_token(email), "token_type": "bearer"}

@router.post("/auth/login")
async def login(payload: AuthInput) -> dict[str, str]:
    email = payload.email.strip().lower()
    user = users.get(email)
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="邮箱或密码错误")
    return {"access_token": create_access_token(email), "token_type": "bearer"}

@router.get("/agents")
async def list_agents() -> list[dict[str, str]]:
    return [{"id": school.value, **profile} for school, profile in SCHOOL_PROFILES.items()]

@router.get("/questionnaire")
async def questionnaire() -> dict[str, Any]:
    return {"modules": QUESTIONNAIRE, "total": TOTAL_QUESTIONS}

@router.post("/consultations")
async def create_consultation(payload: CreateConsultation) -> dict[str, Any]:
    session_id = str(uuid4())
    sessions[session_id] = {"school": payload.school.value, "facts": {}, "risk_flags": []}
    return {"id": session_id, "school": payload.school.value}

@router.get("/consultations/{consultation_id}")
async def get_consultation(consultation_id: str) -> dict[str, Any]:
    return sessions.setdefault(consultation_id, {"school": School.PIWEI.value, "facts": {}, "risk_flags": []})

@router.post("/consultations/{consultation_id}/answers")
async def save_answers(consultation_id: str, payload: FactsInput) -> dict[str, Any]:
    session = sessions.setdefault(consultation_id, {"school": School.PIWEI.value, "facts": {}, "risk_flags": []})
    session["facts"].update(payload.facts)
    session["risk_flags"] = scan_risk(session["facts"])
    return {"saved": len(session["facts"]), "risk_flags": session["risk_flags"]}

@router.post("/consultations/{consultation_id}/followup")
async def followup(consultation_id: str) -> dict[str, Any]:
    session = sessions.setdefault(consultation_id, {"school": School.PIWEI.value, "facts": {}, "risk_flags": []})
    return await orchestrator.plan_followup(session["facts"])

@router.post("/consultations/{consultation_id}/report")
async def report_stream(consultation_id: str, payload: FactsInput | None = None) -> StreamingResponse:
    session = sessions.setdefault(consultation_id, {"school": School.PIWEI.value, "facts": {}, "risk_flags": []})
    if payload and payload.facts:
        session["facts"].update(payload.facts)
    facts = session["facts"]
    risk_flags = scan_risk(facts)
    session["risk_flags"] = risk_flags

    async def events():
        def sse(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield sse("stage_started", {"stage": "orchestrator", "message": "主控收集四诊信息，准备会诊"})
        if risk_flags:
            yield sse("risk_detected", {"stage": "safety", "message": "检测到需优先就医的信号", "flags": risk_flags})
        yield sse("stage_started", {"stage": "panel", "message": "五位学派医家正在并行辨证…"})
        panel = await orchestrator.run_panel(facts, risk_flags)
        for result in panel:
            yield sse("agent_result", {"stage": "panel", **result})
        yield sse("stage_started", {"stage": "integrator", "message": "主控整合各家意见…"})
        final = await orchestrator.integrate(facts, panel, risk_flags)
        yield sse("report", {"stage": "integrator", **final, "panel": panel})
        yield sse("completed", {"stage": "done", "message": "辨证完成"})

    return StreamingResponse(events(), media_type="text/event-stream")

@router.get("/llm/status")
async def llm_status() -> dict[str, Any]:
    return settings_store.public_settings()

@router.post("/admin/llm")
async def update_llm(payload: LLMConfigInput, x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    _require_admin(x_admin_token)
    return settings_store.update_runtime(payload.base_url, payload.api_key, payload.model_fast, payload.model_pro)

@router.post("/admin/llm/test")
async def test_llm(payload: TestInput = TestInput(), x_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    _require_admin(x_admin_token)
    if not llm.is_configured():
        raise HTTPException(status_code=400, detail="尚未完整配置接口地址、Key 与模型")
    roles = ["fast", "pro"] if payload.role == "both" else [payload.role]
    replies: dict[str, str] = {}
    for role in roles:
        try:
            replies[role] = await llm.test_connection(role)
        except llm.LLMUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{role} 模型调用失败：{exc}") from exc
    return {"ok": True, "replies": replies}
