from app.domain.diagnosis.rules import scan_risk, score_candidates


def test_danger_signal_blocks():
    assert "胸痛" in scan_risk({}, "最近有胸痛")


def test_minor_is_risk_flag():
    assert "未成年人" in scan_risk({"age": 12})


def test_rule_engine_returns_explainable_candidate():
    result = score_candidates({"cold": "经常怕冷", "fatigue": "容易疲倦"})
    assert result[0]["name"] == "脾胃阳虚倾向"
    assert result[0]["score"] <= 1


def test_negative_option_does_not_match():
    result = score_candidates({"cold": "没有怕冷", "fatigue": "容易疲倦", "poor_appetite": "食欲不振"})
    assert result[0]["name"] == "脾气不足倾向"


def test_unsure_option_keeps_low_confidence():
    result = score_candidates({"cold": "暂不确定", "fatigue": "暂不确定"})
    assert result[0]["name"].startswith("证据不足")