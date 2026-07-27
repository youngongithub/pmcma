"""Unit tests for the governed narrative/document steps.

The LLM call is mocked (`_call_model`) so tests need neither network nor an API
key; the emphasis is the human-in-the-loop GOVERNANCE (principle #3) — output is
never auto-accepted."""
import pandas as pd
import pytest

import pmcma.narrative as narr
from pmcma.narrative import generate_narrative, review_gate, ReviewRequiredError, ReviewRejectedError


def _table() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "expected_return": [0.079, 0.114],
            "annual_volatility": [0.162, 0.212],
            "corr_to_public": [1.0, 0.902],
        },
        index=["Public Equity", "Private Equity (proxy)"],
    )
    df.attrs["title"] = "LPX proxy / synthetic — methodology demonstrator"
    df.attrs["notes"] = "expected_return is ARITHMETIC annualized."
    return df


def test_build_prompt_carries_table_and_labeling():
    system, user = narr._build_prompt(_table())
    assert "methodology demonstrator" in system.lower()
    assert "not investment advice" in system.lower()
    assert "Private Equity (proxy)" in user            # table content present
    assert "0.902" in user                             # a real number from the table
    assert "methodology demonstrator" in user          # title carried in


def test_generate_narrative_without_review_returns_model_draft(monkeypatch):
    monkeypatch.setattr(narr, "_call_model", lambda s, u, m, **k: "DRAFT NOTE")
    out = generate_narrative(_table(), "claude-sonnet-5", require_review=False)
    assert out == "DRAFT NOTE"


def test_review_pauses_and_returns_edited_text(monkeypatch):
    monkeypatch.setattr(narr, "_call_model", lambda s, u, m, **k: "MODEL DRAFT")
    seen = {}

    def fake_review(text, kind="research note"):
        seen["text"] = text
        return "HUMAN-EDITED NOTE"

    monkeypatch.setattr(narr, "_human_review", fake_review)
    out = generate_narrative(_table(), "claude-sonnet-5", require_review=True)
    assert out == "HUMAN-EDITED NOTE"       # the reviewed text wins
    assert seen["text"] == "MODEL DRAFT"    # review saw the raw model draft


def test_review_required_raises_without_tty(monkeypatch):
    monkeypatch.setattr(narr, "_call_model", lambda s, u, m, **k: "DRAFT")
    monkeypatch.setattr(narr, "_is_interactive", lambda: False)
    with pytest.raises(ReviewRequiredError):
        generate_narrative(_table(), "claude-sonnet-5", require_review=True)


def test_review_gate_confirms_on_accept(monkeypatch):
    monkeypatch.setattr(narr, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "y")
    extracted = {"commitment": 100, "called": 60, "distributed": 40, "nav": 55,
                 "_status": "UNCONFIRMED"}
    out = review_gate(dict(extracted))
    assert out["nav"] == 55
    assert out["_status"] == "confirmed"


def test_review_gate_aborts_on_reject(monkeypatch):
    monkeypatch.setattr(narr, "_is_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "a")
    with pytest.raises(ReviewRejectedError):
        review_gate({"commitment": 100, "called": 60, "distributed": 40, "nav": 55})


def test_review_gate_required_raises_without_tty(monkeypatch):
    monkeypatch.setattr(narr, "_is_interactive", lambda: False)
    with pytest.raises(ReviewRequiredError):
        review_gate({"commitment": 100, "called": 60, "distributed": 40, "nav": 55})
