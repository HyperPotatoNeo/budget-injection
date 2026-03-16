"""Unit tests for BudgetInjectionEnv (no GPU needed)."""

import pytest
from budget_injection_env.env import _format_budget_message


def test_absolute_format():
    msg = _format_budget_message("absolute", used=2048, total=8192, remaining=6144)
    assert "2048" in msg
    assert "8192" in msg
    assert "6144" in msg


def test_ratio_format():
    msg = _format_budget_message("ratio", used=2048, total=8192, remaining=6144)
    assert "25%" in msg
    assert "75%" in msg


def test_urgency_early():
    msg = _format_budget_message("urgency", used=1024, total=8192, remaining=7168)
    assert "Continue" in msg


def test_urgency_mid():
    msg = _format_budget_message("urgency", used=5000, total=8192, remaining=3192)
    assert "concluding" in msg


def test_urgency_late():
    msg = _format_budget_message("urgency", used=7000, total=8192, remaining=1192)
    assert "Wrap up" in msg


def test_minimal_format():
    msg = _format_budget_message("minimal", used=2048, total=8192, remaining=6144)
    assert "<budget>6144</budget>" == msg


def test_custom_template():
    msg = _format_budget_message(
        "absolute", used=2048, total=8192, remaining=6144,
        template="Tokens left: {remaining}")
    assert msg == "Tokens left: 6144"


def test_zero_remaining():
    msg = _format_budget_message("absolute", used=8192, total=8192, remaining=0)
    assert "0" in msg


def test_ratio_rounding():
    msg = _format_budget_message("ratio", used=2730, total=8192, remaining=5462)
    assert "33%" in msg
    assert "67%" in msg
