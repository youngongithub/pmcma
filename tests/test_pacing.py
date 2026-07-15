"""Unit tests for Module 2 (Takahashi-Alexander pacing).

No smoothing round-trip here; instead we pin the economic invariants of the
deterministic model: non-negativity, the J-curve sign pattern, linearity in
commitment, bow-factor back-loading, and correct multi-vintage aggregation.
"""
import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from pmcma.pacing import PacingParams, project_fund, project_program


def base_params(**kw) -> PacingParams:
    d = dict(
        commitment=100.0, fund_life=12, bow_factor=2.5, growth_rate=0.13,
        contribution_rate=0.25, yield_rate=0.03, periods_per_year=4,
    )
    d.update(kw)
    return PacingParams(**d)


def test_project_fund_shape_and_columns():
    p = base_params()
    df = project_fund(p)
    assert len(df) == p.fund_life * p.periods_per_year
    assert list(df.columns) == [
        "call", "distribution", "nav", "net_cashflow", "cumulative_paid_in"
    ]


def test_calls_and_distributions_nonneg_and_bounded():
    p = base_params()
    df = project_fund(p)
    assert (df["call"] >= 0).all()
    assert (df["distribution"] >= 0).all()
    # never call more than committed, and eventually call almost all of it
    assert df["cumulative_paid_in"].iloc[-1] <= p.commitment + 1e-9
    assert df["cumulative_paid_in"].iloc[-1] > 0.9 * p.commitment


def test_jcurve_sign_pattern():
    """Early periods are net outflows (calls); late periods net inflows (distributions)."""
    df = project_fund(base_params())
    assert df["net_cashflow"].iloc[0] < 0
    assert df["net_cashflow"].iloc[-1] > 0


def test_linear_in_commitment():
    a = project_fund(base_params(commitment=100.0))
    b = project_fund(base_params(commitment=200.0))
    np.testing.assert_allclose(2 * a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12)


def test_bow_factor_backloads_distributions():
    def early_frac(df: pd.DataFrame) -> float:
        half = len(df) // 2
        return df["distribution"].iloc[:half].sum() / df["distribution"].sum()

    lo = project_fund(base_params(bow_factor=1.0))
    hi = project_fund(base_params(bow_factor=4.0))
    assert early_frac(hi) < early_frac(lo)  # higher bow => fewer early distributions


def test_positive_growth_gives_tvpi_gt_1():
    df = project_fund(base_params())
    tvpi = (df["distribution"].sum() + df["nav"].iloc[-1]) / df["cumulative_paid_in"].iloc[-1]
    assert tvpi > 1.0


def test_project_program_aggregates_and_invested_pct():
    p = base_params()
    commits = [100, 100, 100, 100, 100]
    prog = project_program(commits, p)

    ppy, T = p.periods_per_year, p.fund_life * p.periods_per_year
    assert len(prog) == (len(commits) - 1) * ppy + T

    # aggregate calls must equal the sum of per-vintage calls (linear overlay)
    total_call_indiv = sum(
        project_fund(replace(p, commitment=float(c)))["call"].sum() for c in commits
    )
    assert prog["call"].sum() == pytest.approx(total_call_indiv, abs=1e-6)

    assert prog["cumulative_committed"].iloc[-1] == sum(commits)
    assert 0.0 < prog.attrs["peak_invested_pct"] < 5.0
