"""Two governed LLM steps: (1) optional GP-report ingestion (document analysis),
(2) narrative generation over the finished table. Both pause for human review."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Model used for GP-report extraction (accuracy-sensitive → most capable Opus).
# The narrative model is passed in from config (config/default.yaml: narrative.model).
_EXTRACTION_MODEL = "claude-opus-4-8"
_DIV = "─" * 70


class ReviewRequiredError(RuntimeError):
    """Raised when human review is required (principle #3) but no interactive
    terminal is available — we never auto-accept LLM output."""


class ReviewRejectedError(RuntimeError):
    """Raised when the human reviewer rejects the extracted values / narrative."""


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _call_model(system: str, user: str, model: str, max_tokens: int = 4096) -> str:
    """Single Claude call via the official SDK. Credentials resolve from the
    environment (ANTHROPIC_API_KEY or an `ant auth login` profile)."""
    import anthropic  # lazy: only needed when narrative actually runs

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _loads_json(raw: str) -> dict:
    """Parse a JSON object from model output, tolerating ```json fences."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        s = s[4:] if s.lstrip().lower().startswith("json") else s
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON:\n{raw}") from e


def _coerce_number(text: str):
    try:
        return float(text) if ("." in text or "e" in text.lower()) else int(text)
    except ValueError:
        return text


def parse_gp_report(pdf_path: str) -> dict:
    """Extract cash-flow inputs (commitment, called, distributed, NAV) from a GP
    quarterly report PDF via pdfplumber + an LLM extraction pass. Returns a dict
    for HUMAN REVIEW — never feeds the model until confirmed."""
    import pdfplumber  # lazy

    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    text = "\n".join(parts)

    system = (
        "You extract private-fund cash-flow inputs from a GP quarterly report. "
        "Return ONLY a JSON object with numeric fields: commitment, called, "
        "distributed, nav (in the report's reporting units). Use null for any value "
        "not stated. Do NOT guess or infer beyond what the text states."
    )
    user = f"GP report text:\n\n{text[:20000]}\n\nReturn the JSON object."
    extracted = _loads_json(_call_model(system, user, _EXTRACTION_MODEL))

    # tag as unconfirmed — downstream must route through review_gate()
    extracted["_source"] = str(pdf_path)
    extracted["_status"] = "UNCONFIRMED — must pass review_gate before use"
    return extracted


def review_gate(extracted: dict) -> dict:
    """Print extracted values, require explicit user confirmation/edit before
    returning. This is the governance step the JD cares about."""
    fields = ["commitment", "called", "distributed", "nav"]

    def _show():
        print("\n=== GP-report extraction — HUMAN REVIEW REQUIRED (principle #3) ===")
        for k in fields:
            print(f"  {k:12s}: {extracted.get(k)}")

    _show()
    if not _is_interactive():
        raise ReviewRequiredError(
            "GP-report extraction needs human confirmation but there is no interactive "
            "terminal. Run in a TTY — extracted numbers are never auto-accepted."
        )

    while True:
        resp = input("Accept these values? [y]es / [e]dit / [a]bort: ").strip().lower()
        if resp in ("y", ""):
            confirmed = {k: extracted.get(k) for k in fields}
            confirmed["_status"] = "confirmed"
            confirmed["_source"] = extracted.get("_source")
            return confirmed
        if resp == "a":
            raise ReviewRejectedError("GP-report extraction rejected by reviewer.")
        if resp == "e":
            for k in fields:
                v = input(f"  {k} [{extracted.get(k)}] = ").strip()
                if v != "":
                    extracted[k] = _coerce_number(v)
            _show()


def _edit_in_editor(text: str) -> str:
    editor = os.environ.get("EDITOR", "nano")
    with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        subprocess.call([editor, path])
        return Path(path).read_text()
    finally:
        os.unlink(path)


def _human_review(text: str, kind: str = "research note") -> str:
    print(f"\n{_DIV}\n{text}\n{_DIV}")
    if not _is_interactive():
        raise ReviewRequiredError(
            f"Human review of the {kind} is required (principle #3) but there is no "
            "interactive terminal. Run in a TTY, or set narrative.require_review: false "
            "only if you accept ungoverned output."
        )
    while True:
        resp = input(f"Review this {kind} — [a]ccept / [e]dit / [r]eject: ").strip().lower()
        if resp in ("a", ""):
            return text
        if resp == "e":
            text = _edit_in_editor(text)
            print(f"\n{_DIV}\n{text}\n{_DIV}")
        elif resp == "r":
            raise ReviewRejectedError(f"{kind.capitalize()} rejected by reviewer.")


def _build_prompt(cma_table: pd.DataFrame) -> tuple[str, str]:
    title = cma_table.attrs.get("title", "Comparable public-vs-private CMA table")
    notes = cma_table.attrs.get("notes", "")
    system = (
        "You are a buy-side capital-markets research analyst writing a short, honest "
        "note for an investment committee. You explain, in plain language, what a "
        "comparable public-vs-private CMA table implies for asset allocation, and you "
        "are precise about methodology without overstating certainty.\n\n"
        "Hard rules:\n"
        "- This is a METHODOLOGY DEMONSTRATOR on synthetic/transparent inputs — NOT a "
        "proprietary CMA and NOT investment advice. Say so in a one-line disclaimer at the top.\n"
        "- Explain WHY de-smoothed private volatility and correlation exceed the reported "
        "figures (appraisal smoothing understates risk), what the pacing invested-% implies "
        "for commitment sizing, and — where present — what max_sustainable_pace and the "
        "dispersion band mean for the allocation decision.\n"
        "- Use only numbers present in the table; round sensibly; invent nothing.\n"
        "- 200–350 words, markdown."
    )
    user = (
        f"Write the research note for this table.\n\n"
        f"Title: {title}\n\n{cma_table.to_string()}\n\nMethodology notes: {notes}"
    )
    return system, user


def generate_narrative(cma_table: pd.DataFrame, model: str, require_review: bool) -> str:
    """Draft a short research note explaining the table (why de-smoothed vol is
    higher, what the pacing/liquidity result implies for allocation). If
    require_review, present for edit before saving. Combines data+model+narrative
    into a repeatable process."""
    system, user = _build_prompt(cma_table)
    draft = _call_model(system, user, model)
    if require_review:
        draft = _human_review(draft, kind="research note")
    return draft
