"""Regenerate the committed synthetic sample series (data/samples/).

Deterministic (fixed seed) so the demo is reproducible. These are SYNTHETIC,
transparent inputs — NOT vendor data. The private series is stored in its
*reported* (appraisal-smoothed) form; the pipeline de-smooths it in Module 1.

Run from the repo root:  python scripts/make_samples.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20240101
N = 100                      # quarters (25 years)
THETA = 0.60                 # imposed appraisal smoothing on the private series


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.period_range("2000Q1", periods=N, freq="Q").to_timestamp()

    public = rng.normal(0.018, 0.080, N)                 # ~7.2%/yr, ~16% vol
    alpha, beta = 0.006, 1.2
    true_priv = alpha + beta * public + rng.normal(0.0, 0.050, N)

    reported = np.empty(N)                                # impose AR(1) smoothing
    reported[0] = true_priv[0]
    for t in range(1, N):
        reported[t] = (1 - THETA) * true_priv[t] + THETA * reported[t - 1]

    out = Path("data/samples")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "ret": public}).to_csv(out / "public_quarterly.csv", index=False)
    pd.DataFrame({"date": dates, "ret": reported}).to_csv(out / "private_proxy_quarterly.csv", index=False)

    print(f"wrote {N} rows each -> {out}/")
    print(f"  public   : {public.mean()*4:6.2%}/yr, {public.std(ddof=1)*2:6.2%} vol")
    print(f"  reported : {reported.mean()*4:6.2%}/yr, {reported.std(ddof=1)*2:6.2%} vol (smoothed)")


if __name__ == "__main__":
    main()
