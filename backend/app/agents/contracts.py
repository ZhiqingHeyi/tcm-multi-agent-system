from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class School(StrEnum):
    SHANGHAN = "shanghan"
    WENBING = "wenbing"
    PIWEI = "piwei"
    HUOSHEN = "huoshen"
    INTEGRATIVE = "integrative"
    NIHAIXIA = "nihaixia"


class AgentEvent(BaseModel):
    type: str
    stage: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    label: str
    value: str
    source: str = "user_confirmed"
    polarity: str = "positive"


class DiagnosisCandidate(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence] = Field(default_factory=list)
    counter_evidence: list[Evidence] = Field(default_factory=list)
    treatment: str
    formula: str | None = None


class ConsultationState(BaseModel):
    phase: str = "basic"
    school: School = School.INTEGRATIVE
    facts: dict[str, Any] = Field(default_factory=dict)
    confirmed_facts: set[str] = Field(default_factory=set)
    risk_flags: list[str] = Field(default_factory=list)
    messages: list[dict[str, str]] = Field(default_factory=list)
