"""Manager dispersion: 'the asset-class return' is a fiction without a dispersion
assumption. Simulate the return distribution an allocator actually faces."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_lognormal_from_quartiles(top_q: float, median: float, bottom_q: float) -> tuple[float, float]:
    """Back out lognormal (mu, sigma) for the multiple (TVPI/IRR) distribution
    from top-quartile / median / bottom-quartile inputs.

    Three inputs, two parameters: we pin mu to the median (mu = log(median)) and
    sigma to the inter-quartile range in log space (exact), so the fitted IQR and
    median match the inputs; the individual quartiles match up to the small gap
    between the stated median and the geometric mean of the quartiles.
    """
    if not (bottom_q > 0 and top_q > median > bottom_q):
        raise ValueError("require top_q > median > bottom_q > 0")
    z75 = norm.ppf(0.75)
    mu = float(np.log(median))
    sigma = float((np.log(top_q) - np.log(bottom_q)) / (2.0 * z75))
    return mu, sigma


def simulate_selection(mu: float, sigma: float, n_sims: int,
                       access_quantile: float = 0.5, seed: int = 12345) -> np.ndarray:
    """Monte Carlo the realized multiple given selection skill = access_quantile
    (0.5 = median access, 0.75 = top-quartile access). Returns sampled multiples.

    Skill tilts the central tendency: the median accessed manager sits at the
    access_quantile of the base distribution (shift = sigma * z(access_quantile)),
    while full manager-level dispersion (sigma) remains around it. `seed` fixes
    the draw for a reproducible pipeline (added to the handed signature)."""
    if not (0.0 < access_quantile < 1.0):
        raise ValueError("access_quantile must be in (0, 1)")
    rng = np.random.default_rng(seed)
    shift = sigma * norm.ppf(access_quantile)
    draws = rng.normal(mu + shift, sigma, int(n_sims))
    return np.exp(draws)


def dispersion_band(sims: np.ndarray, horizon_years: int) -> pd.Series:
    """Convert simulated multiples to an annualized expected-return BAND
    (e.g., p25/p50/p75). Shows the CMA point estimate is incomplete without it."""
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive")
    sims = np.asarray(sims, dtype=float)
    ann = sims ** (1.0 / horizon_years) - 1.0
    return pd.Series(
        {
            "p25": float(np.percentile(ann, 25)),
            "p50": float(np.percentile(ann, 50)),
            "p75": float(np.percentile(ann, 75)),
        },
        name="annualized_return_band",
    )
