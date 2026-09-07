"""Regenerate the committed synthetic sample series (data/samples/).

Deterministic (fixed seed) so the demo is reproducible. These are SYNTHETIC,
transparent inputs — NOT vendor data. The private series is stored in its
*reported* (appraisal-smoothed) form; the pipeline de-smooths it in Module 1.

The generator imposes two known quantities, both of which the de-smoother is
tested against in tests/test_desmooth.py:

  1. theta = THETA          — the appraisal smoothing applied to the true series
  2. rho   = imposed_correlation()  — the true private/public correlation, which
                                      follows in closed form from BETA and the
                                      two noise scales (see the docstring there)

Run from the repo root:  python scripts/make_samples.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20240101
N = 100                      # quarters (25 years)
THETA = 0.60                 # imposed appraisal smoothing on the private series

# The true private series is a levered function of the public series plus
# idiosyncratic noise: r_priv = ALPHA + BETA * r_pub + eps, eps ~ N(0, SD_IDIO).
# This is what makes the correlation round-trip possible — the two true series
# are linked by a known beta, not drawn independently.
MU_PUBLIC = 0.018            # ~7.2%/yr
SD_PUBLIC = 0.080            # ~16% annualized vol
ALPHA = 0.006
BETA = 1.2
SD_IDIO = 0.050


def imposed_correlation(beta: float = BETA,
                        sd_public: float = SD_PUBLIC,
                        sd_idio: float = SD_IDIO) -> float:
    """Closed-form correlation between the true private and public series.

    With r_priv = alpha + beta * r_pub + eps and eps independent of r_pub:

        cov  = beta * sd_public^2
        var  = beta^2 * sd_public^2 + sd_idio^2
        rho  = beta * sd_public / sqrt(beta^2 * sd_public^2 + sd_idio^2)

    For the committed defaults this is ~0.8869. It is the target the de-smoother
    has to recover; the *reported* series correlates ~0.70 because smoothing
    suppresses it.
    """
    return float(beta * sd_public / np.sqrt(beta**2 * sd_public**2 + sd_idio**2))


def smooth(true: np.ndarray, theta: float = THETA) -> np.ndarray:
    """Impose AR(1) appraisal smoothing: r_obs(t) = (1-theta)*r_true(t) + theta*r_obs(t-1).

    Note this is an *infinite*-order geometric moving average of the true series,
    b_j = (1-theta) * theta^j, not a finite MA(q). Geltner inverts it exactly;
    a truncated MA(q) fit (GLM) is misspecified against it by construction.
    """
    obs = np.empty_like(true, dtype=float)
    obs[0] = true[0]
    for t in range(1, len(true)):
        obs[t] = (1.0 - theta) * true[t] + theta * obs[t - 1]
    return obs


def generate(seed: int = SEED, n: int = N, theta: float = THETA):
    """Build the sample series. Returns (dates, public, true_private, reported).

    `true_private` is never written to disk — it is the unobservable the
    de-smoother is supposed to recover, and the tests use it as ground truth.
    """
    rng = np.random.default_rng(seed)
    dates = pd.period_range("2000Q1", periods=n, freq="Q").to_timestamp()

    public = rng.normal(MU_PUBLIC, SD_PUBLIC, n)
    true_priv = ALPHA + BETA * public + rng.normal(0.0, SD_IDIO, n)
    reported = smooth(true_priv, theta)

    return dates, public, true_priv, reported


def main() -> None:
    dates, public, true_priv, reported = generate()

    out = Path("data/samples")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "ret": public}).to_csv(out / "public_quarterly.csv", index=False)
    pd.DataFrame({"date": dates, "ret": reported}).to_csv(out / "private_proxy_quarterly.csv", index=False)

    print(f"wrote {N} rows each -> {out}/")
    print(f"  public   : {public.mean()*4:6.2%}/yr, {public.std(ddof=1)*2:6.2%} vol")
    print(f"  reported : {reported.mean()*4:6.2%}/yr, {reported.std(ddof=1)*2:6.2%} vol (smoothed)")
    print(f"  imposed  : theta={THETA}, true corr-to-public={imposed_correlation():.4f}")


if __name__ == "__main__":
    main()
