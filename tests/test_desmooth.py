"""Unit tests for Module 1 (de-smoothing).

The centerpiece is the synthetic round-trip test required by the project's
design principle #4: impose a *known* smoothing on a true series, then confirm
the de-smoother recovers it.

There are two round-trips, and the second matters more than the first:

  * volatility — smoothing halves reported vol; de-smoothing restores it
  * correlation — smoothing suppresses correlation to public markets, which is
    the more consequential CMA error. Understated vol makes a private asset look
    calm; understated correlation makes it look like a *diversifier*, and that is
    what drives an optimizer into corner solutions.

The correlation tests run against `scripts/make_samples.py`, which links the true
private and public series with a known beta, so the target correlation is
available in closed form rather than being an artifact of independent draws.
"""
import numpy as np
import pandas as pd
import pytest

import make_samples
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
# The correlation round-trip
#
# Understated correlation is the costlier CMA error, so it gets its own
# round-trip against the closed-form target imposed by the sample generator.
# ---------------------------------------------------------------------------

def _linked_series(seed: int, n: int):
    """Public + true-private + reported, using the sample generator's parameters."""
    dates, public, true_priv, reported = make_samples.generate(seed=seed, n=n)
    return (pd.Series(public, index=dates),
            pd.Series(true_priv, index=dates),
            pd.Series(reported, index=dates))


def _corr_to(series: pd.Series, public: pd.Series) -> float:
    a, b = series.align(public, join="inner")
    return float(a.corr(b))


def test_imposed_correlation_matches_closed_form():
    """The generator's beta implies rho ~ 0.8869; a large sample realizes it."""
    rho = make_samples.imposed_correlation()
    assert rho == pytest.approx(0.8869, abs=1e-4)

    public, true_priv, _ = _linked_series(seed=0, n=20_000)
    assert _corr_to(true_priv, public) == pytest.approx(rho, abs=0.01)


def test_geltner_recovers_imposed_correlation():
    """The correlation round-trip: de-smoothing returns the imposed rho.

    This is the counterpart to test_geltner_recovers_imposed_smoothing. The
    reported series correlates ~0.70 with public; the truth is ~0.887.
    """
    rho = make_samples.imposed_correlation()
    public, true_priv, reported = _linked_series(seed=0, n=2_000)

    # smoothing suppresses correlation well below the truth
    assert _corr_to(reported, public) < rho - 0.15

    # de-smoothing recovers it, with theta pinned and with theta estimated
    pinned = geltner_unsmooth(reported, theta=make_samples.THETA)
    estimated = geltner_unsmooth(reported)
    assert _corr_to(pinned, public) == pytest.approx(rho, abs=0.02)
    assert _corr_to(estimated, public) == pytest.approx(rho, abs=0.02)


def test_correlation_roundtrip_on_the_committed_sample():
    """The demo's own 100-quarter series, not a large-n idealization.

    At n=100 sampling error is real, so the tolerance is wider — but the headline
    the artifact reports (~0.90 de-smoothed vs ~0.69 reported) has to hold here or
    the demo is quoting a number it cannot defend.
    """
    rho = make_samples.imposed_correlation()
    public, true_priv, reported = _linked_series(seed=make_samples.SEED, n=make_samples.N)

    reported_corr = _corr_to(reported, public)
    recovered = _corr_to(geltner_unsmooth(reported, theta=make_samples.THETA), public)

    assert reported_corr == pytest.approx(0.685, abs=0.02)
    assert recovered == pytest.approx(0.903, abs=0.02)
    # the recovered figure tracks the imposed truth; the reported one does not
    assert abs(recovered - rho) < abs(reported_corr - rho)


def test_glm_understates_imposed_correlation():
    """A truncated MA(q) is misspecified against geometric smoothing — and it shows.

    The generator's implied weights are b_j = (1-theta)*theta^j, an infinite-order
    geometric MA. GLM's finite MA(q) recovers most of the suppressed correlation
    but consistently stops short of the truth, whereas Geltner (the exact inverse)
    lands on it. This is the honest limitation of the module: an estimator that
    does not match the smoothing process still leaves the asset looking like more
    of a diversifier than it is.
    """
    rho = make_samples.imposed_correlation()
    public, _, reported = _linked_series(seed=0, n=2_000)

    reported_corr = _corr_to(reported, public)
    glm_corr = _corr_to(glm_unsmooth(reported, q=2), public)
    geltner_corr = _corr_to(geltner_unsmooth(reported, theta=make_samples.THETA), public)

    # GLM improves on the reported figure but does not reach either the imposed
    # truth or the correctly-specified estimator
    assert reported_corr < glm_corr < geltner_corr
    assert glm_corr < rho


def test_smoothing_halves_volatility_by_construction():
    """Var ratio for AR(1) smoothing of iid truth is sum(b_j^2) = (1-th)^2/(1-th^2).

    At theta=0.60 that is 0.25, so reported vol is half the truth. Pinning this
    keeps the vol and correlation round-trips reading off the same generator.
    """
    theta = make_samples.THETA
    expected_ratio = np.sqrt((1 - theta) ** 2 / (1 - theta**2))
    assert expected_ratio == pytest.approx(0.5, abs=1e-9)

    _, true_priv, reported = _linked_series(seed=0, n=20_000)
    assert reported.std(ddof=1) / true_priv.std(ddof=1) == pytest.approx(expected_ratio, abs=0.02)


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
