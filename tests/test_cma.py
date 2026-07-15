"""Unit tests for synthesis (cma.assemble_cma_table)."""
import numpy as np
import pandas as pd

from pmcma.desmooth import geltner_unsmooth
from pmcma.pacing import PacingParams, project_program
from pmcma.cma import assemble_cma_table


def _fixtures():
    rng = np.random.default_rng(11)
    n = 80
    idx = pd.period_range("2005Q1", periods=n, freq="Q").to_timestamp()
    public = pd.Series(rng.normal(0.02, 0.085, n), index=idx, name="Public Equity")
    true_priv = 1.2 * public.to_numpy() + rng.normal(0.01, 0.06, n)
    reported = np.empty(n); reported[0] = true_priv[0]
    for t in range(1, n):
        reported[t] = 0.4 * true_priv[t] + 0.6 * reported[t - 1]
    reported = pd.Series(reported, index=idx, name="Private Equity (proxy)")
    desmoothed = geltner_unsmooth(reported)
    pacing = project_program([100, 100, 100, 100, 100],
                             PacingParams(100.0, 12, 2.5, 0.13, 0.25, 0.03, 4))
    return public, desmoothed, pacing


def test_table_shape_and_labeling():
    public, priv, pacing = _fixtures()
    t = assemble_cma_table(public, priv, pacing, None, None, 4, "LPX proxy / synthetic")
    assert list(t.index) == ["Public Equity", "Private Equity (proxy)"]
    for col in ["expected_return", "annual_volatility", "corr_to_public", "pacing_invested_pct"]:
        assert col in t.columns
    # principle #5: labeling present
    assert "methodology demonstrator" in t.attrs["title"]
    assert "LPX proxy / synthetic" in t.attrs["title"]


def test_public_corr_is_one_and_pacing_private_only():
    public, priv, pacing = _fixtures()
    t = assemble_cma_table(public, priv, pacing, None, None, 4, "demo")
    assert t.loc["Public Equity", "corr_to_public"] == 1.0
    assert np.isnan(t.loc["Public Equity", "pacing_invested_pct"])   # public has no pacing
    assert t.loc["Private Equity (proxy)", "pacing_invested_pct"] > 0


def test_optional_liquidity_metrics_become_private_only_columns():
    public, priv, pacing = _fixtures()
    liq = pd.Series({"max_sustainable_pace": 0.42})
    t = assemble_cma_table(public, priv, pacing, liq, None, 4, "demo")
    assert t.loc["Private Equity (proxy)", "max_sustainable_pace"] == 0.42
    assert np.isnan(t.loc["Public Equity", "max_sustainable_pace"])   # public has no pacing/liquidity
    # omitted entirely when not provided
    t2 = assemble_cma_table(public, priv, pacing, None, None, 4, "demo")
    assert "max_sustainable_pace" not in t2.columns


def test_optional_dispersion_band_renders_when_provided():
    public, priv, pacing = _fixtures()
    band = pd.Series({"p25": 0.020, "p50": 0.048, "p75": 0.077})  # annualized returns
    t = assemble_cma_table(public, priv, pacing, None, band, 4, "demo")
    assert t.loc["Private Equity (proxy)", "dispersion_band"] == "2.0%–7.7%"
    # omitted entirely when not provided
    t2 = assemble_cma_table(public, priv, pacing, None, None, 4, "demo")
    assert "dispersion_band" not in t2.columns
