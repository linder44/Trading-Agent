"""Tests for scalping-mode brain (system prompt, build_prompt)."""

import pytest
from agent.brain import SYSTEM_PROMPT, _compact_json, _strip_empty, _repair_truncated_json


def test_system_prompt_is_scalping():
    """SYSTEM_PROMPT should mention scalping, not long-term."""
    assert "скальпинг" in SYSTEM_PROMPT.lower() or "scalping" in SYSTEM_PROMPT.lower()
    assert "2 час" in SYSTEM_PROMPT or "2 hours" in SYSTEM_PROMPT
    assert "1m" in SYSTEM_PROMPT
    assert "5m" in SYSTEM_PROMPT
    assert "15m" in SYSTEM_PROMPT


def test_system_prompt_has_scalp_signal():
    """Prompt should reference scalp_signal."""
    assert "scalp_signal" in SYSTEM_PROMPT


def test_system_prompt_has_order_flow():
    """Prompt should reference order flow."""
    assert "order_flow" in SYSTEM_PROMPT.lower() or "Order Flow" in SYSTEM_PROMPT


def test_system_prompt_has_risk_rules():
    """Prompt should have risk rules."""
    assert "стоп-лосс" in SYSTEM_PROMPT.lower() or "stop" in SYSTEM_PROMPT.lower()
    assert "HOLD" in SYSTEM_PROMPT or "hold" in SYSTEM_PROMPT


def test_system_prompt_json_format():
    """Prompt should define JSON output format."""
    assert '"decisions"' in SYSTEM_PROMPT
    assert '"action"' in SYSTEM_PROMPT
    assert '"confidence"' in SYSTEM_PROMPT


def test_compact_json_removes_none():
    data = {"a": 1, "b": None, "c": ""}
    result = _compact_json(data)
    assert '"b"' not in result
    assert '"c"' not in result
    assert '"a"' in result


def test_compact_json_rounds_floats():
    data = {"price": 83412.12345}
    result = _compact_json(data)
    # Should be rounded (no excessive decimals)
    assert "83412" in result
    assert "12345" not in result


def test_strip_empty_nan():
    import math
    result = _strip_empty(float("nan"))
    assert result is None


def test_strip_empty_nested():
    data = {"a": {"b": None, "c": 1}, "d": []}
    result = _strip_empty(data)
    assert result == {"a": {"c": 1}}


def test_repair_truncated_json_valid():
    raw = '{"decisions": [{"symbol": "BTC/USDT", "action": "hold", "confidence": 0.5, "reason": "test", "params": {}}], "market_outlook": "ok", "risk_level": "low"'
    result = _repair_truncated_json(raw)
    assert result is not None
    assert len(result["decisions"]) == 1
    assert result["decisions"][0]["symbol"] == "BTC/USDT"


def test_repair_truncated_json_empty():
    result = _repair_truncated_json("broken garbage")
    assert result is None
