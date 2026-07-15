"""Unit tests for Module 1 (de-smoothing).

The centerpiece is the synthetic round-trip test required by the project's
design principle #4: impose a *known* smoothing on a true series, then confirm
the de-smoother recovers it.
"""
import numpy as np
import pandas as pd
import pytest

from pmcma.desmooth import fit_ar1, geltner_unsmooth, glm_unsmooth, summarize_risk


def geltner_smooth(true: np.ndarray, theta: float) -> np.ndarray:
    """Impose Geltner AR(1) smoothing: r_obs(t) = (1-theta)*r_true(t) + theta*r_obs(t-1)."""
    obs = np.empty_like(true, dtype=float)
    obs[0] = true[0]
    for t in range(1, len(true)):
        obs[t] = (1.0 - theta) * true[t] + theta * obs[t - 1]
    return obs


def ma_smooth(true: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Impose MA smoothing: r_obs(t) = sum_j b_j * r_true(t-j)."""
    obs = np.zeros_like(true, dtype=float)
    for t in range(len(true)):
        for j, bj in enumerate(b):
            if t - j >= 0:
                obs[t] += bj * true[t - j]
    return obs


# ---------------------------------------------------------------------------
# The required synthetic round-trip test
# ---------------------------------------------------------------------------

def test_geltner_roundtrip_is_exact():
    """With the true theta supplied, unsmoothing exactly inverts smoothing."""
    rng = np.random.default_rng(0)
    true = rng.normal(0.02, 0.08, 500)
    theta = 0.4
    obs = geltner_smooth(true, theta)

    recovered = geltner_unsmooth(pd.Series(obs), theta=theta)
    # geltner_unsmooth drops the first (lag-less) observation
    np.testing.assert_allclose(recovered.to_numpy(), true[1:], rtol=1e-9, atol=1e-9)


def test_geltner_recovers_imposed_smoothing():
    """Vol-recovery form of the round-trip: de-smoothed vol ~ true vol, >> smoothed."""
    rng = np.random.default_rng(0)
    true = pd.Series(rng.normal(0.03, 0.09, 200))          # true quarterly returns
    theta = 0.6
    smoothed = true.copy()
    for t in range(1, len(true)):                          # impose AR(1) smoothing
        smoothed.iloc[t] = (1 - theta) * true.iloc[t] + theta * smoothed.iloc[t - 1]

    recovered = geltner_unsmooth(smoothed, theta=theta)
    assert recovered.std() > smoothed.std()
    assert abs(recovered.std() - true.std()) / true.std() < 0.15


def test_fit_ar1_recovers_known_theta():
    """The AR(1) coefficient of a Geltner-smoothed series equals theta."""
    rng = np.random.default_rng(1)
    true = rng.normal(0.0, 0.08, 20_000)
    theta = 0.5
    obs = geltner_smooth(true, theta)
    assert abs(fit_ar1(pd.Series(obs)) - theta) < 0.03


def test_geltner_auto_theta_inflates_volatility():
    """De-smoothing recovers risk: unsmoothed vol > reported vol."""
    rng = np.random.default_rng(2)
    true = rng.normal(0.02, 0.08, 1_000)
    obs = geltner_smooth(true, 0.45)
    un = geltner_unsmooth(pd.Series(obs))  # theta estimated internally
    assert un.std() > pd.Series(obs).std()


# ---------------------------------------------------------------------------
# Getmansky-Lo-Makarov MA(q)
# ---------------------------------------------------------------------------

def test_glm_preserves_mean_and_inflates_vol():
    rng = np.random.default_rng(3)
    true = rng.normal(0.02, 0.08, 800)
    b = np.array([0.5, 0.3, 0.2])  # sums to 1
    obs = ma_smooth(true, b)

    un = glm_unsmooth(pd.Series(obs), q=2)
    assert abs(un.mean() - true.mean()) < 0.02
    assert un.std() > pd.Series(obs).std()


# ---------------------------------------------------------------------------
# summarize_risk + guardrails
# ---------------------------------------------------------------------------

def test_summarize_risk_shape_and_vol_ordering():
    rng = np.random.default_rng(4)
    public = pd.Series(rng.normal(0.02, 0.09, 300))
    true = 0.6 * public.to_numpy() + rng.normal(0.0, 0.06, 300)
    obs = geltner_smooth(true, 0.45)
    un = geltner_unsmooth(pd.Series(obs))

    table = summarize_risk(pd.Series(obs), un, public, periods_per_year=4)
    assert list(table.index) == ["annualized_volatility", "correlation_to_public"]
    assert list(table.columns) == ["reported", "de-smoothed"]
    # reported vol is artificially low relative to de-smoothed
    assert table.loc["annualized_volatility", "de-smoothed"] > \
        table.loc["annualized_volatility", "reported"]


def test_fit_ar1_rejects_short_series():
    with pytest.raises(ValueError):
        fit_ar1(pd.Series([0.01, 0.02]))


def test_geltner_rejects_theta_one():
    with pytest.raises(ValueError):
        geltner_unsmooth(pd.Series([0.01, 0.02, 0.03]), theta=1.0)
