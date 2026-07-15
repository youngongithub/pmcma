"""Recover true (de-smoothed) private-market risk from appraisal-based returns.

Appraisal series are serially correlated (stale valuations), which understates
volatility and correlation. We reverse the smoothing to recover economic risk.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def fit_ar1(returns: pd.Series) -> float:
    """Estimate first-order autocorrelation theta of the reported series.

    theta is the AR(1) coefficient of r_obs. High theta => heavy smoothing.
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 3:
        raise ValueError("Need at least 3 observations to estimate AR(1).")
    x = r.to_numpy()
    xbar = x.mean()
    num = np.sum((x[1:] - xbar) * (x[:-1] - xbar))
    den = np.sum((x - xbar) ** 2)
    if den == 0:
        raise ValueError("Return series has zero variance; theta undefined.")
    return float(num / den)


def geltner_unsmooth(returns: pd.Series, theta: float | None = None) -> pd.Series:
    """Geltner (1991) first-order unsmoothing.

    r_true(t) = [r_obs(t) - theta * r_obs(t-1)] / (1 - theta)

    If theta is None, estimate it via fit_ar1. Returns the de-smoothed series
    (same index, first obs dropped).
    """
    r = pd.Series(returns).dropna().astype(float)
    if theta is None:
        theta = fit_ar1(r)
    if np.isclose(theta, 1.0):
        raise ValueError("theta ~ 1: unsmoothing is undefined (division by zero).")
    true = (r - theta * r.shift(1)) / (1.0 - theta)
    return true.dropna()  # drops the first observation, which has no lag


def glm_unsmooth(returns: pd.Series, q: int = 2) -> pd.Series:
    """Getmansky-Lo-Makarov (2004) MA(q) unsmoothing.

    Fit r_obs(t) = sum_{j=0..q} b_j * r_true(t-j), with b_j >= 0, sum b_j = 1.
    Recover r_true by deconvolving the estimated MA weights. More general than
    Geltner (handles multi-period smoothing). Return de-smoothed series.

    Implementation: if true returns are ~iid, observed returns follow an MA(q)
    process whose reduced-form weights [1, ma_1, ..., ma_q] are proportional to
    the smoothing weights b_j. With S = 1 + sum(ma_j) and innovations e_t from
    the fitted MA(q), the de-smoothed true return is r_true(t) = mu + S * e_t.
    Mean is preserved; variance is inflated by 1 / sum(b_j^2).
    """
    from statsmodels.tsa.arima.model import ARIMA

    if q < 1:
        raise ValueError("q must be >= 1 for MA(q) unsmoothing.")
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < q + 2:
        raise ValueError(f"Need at least {q + 2} observations for MA({q}).")

    res = ARIMA(r.to_numpy(), order=(0, 0, q), trend="c").fit()
    ma = np.asarray(res.maparams, dtype=float)          # [ma_1, ..., ma_q]
    S = 1.0 + ma.sum()                                  # sum of reduced-form weights
    if np.isclose(S, 0.0):
        raise ValueError("MA weights sum to ~0; deconvolution undefined.")

    b = np.concatenate(([1.0], ma)) / S                 # implied smoothing weights
    if (b < -1e-6).any():
        # GLM assumes b_j >= 0; an unconstrained fit can violate it. Flag, don't fail.
        import warnings
        warnings.warn(
            f"Estimated GLM smoothing weights are not all non-negative: {b}. "
            "De-smoothing still returned, but interpret with care.",
            stacklevel=2,
        )

    true = float(r.mean()) + S * np.asarray(res.resid, dtype=float)
    return pd.Series(true, index=r.index)


def summarize_risk(reported: pd.Series,
                   unsmoothed: pd.Series,
                   public: pd.Series,
                   periods_per_year: int = 4) -> pd.DataFrame:
    """The money-shot table: before/after de-smoothing.

    Rows: annualized volatility, correlation-to-public.
    Cols: 'reported', 'de-smoothed'. This is the core insight of the artifact —
    reported vol/corr are artificially low; de-smoothing recovers true risk.
    """
    def ann_vol(s: pd.Series) -> float:
        return float(pd.Series(s).dropna().std(ddof=1) * np.sqrt(periods_per_year))

    def corr_to_public(s: pd.Series) -> float:
        a, b = pd.Series(s).align(pd.Series(public), join="inner")
        return float(a.corr(b))

    return pd.DataFrame(
        {
            "reported": [ann_vol(reported), corr_to_public(reported)],
            "de-smoothed": [ann_vol(unsmoothed), corr_to_public(unsmoothed)],
        },
        index=["annualized_volatility", "correlation_to_public"],
    )
