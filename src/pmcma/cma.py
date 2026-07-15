"""Assemble the comparable public-vs-private CMA table — the whole point."""
from __future__ import annotations
import warnings
import pandas as pd


def _ann_return(s: pd.Series, ppy: int) -> float:
    """Arithmetic annualized return of a periodic series.

    Deliberately arithmetic, NOT geometric: Geltner de-smoothing preserves the
    arithmetic mean (E[r_true] = E[r_obs]) but inflates volatility, so a
    *compound* return computed on the de-smoothed series would absorb spurious
    variance drag and understate the return level. De-smoothing must change the
    RISK estimate (vol/corr), never the return estimate. A geometric CMA figure,
    if needed, is arithmetic - 0.5 * (de-smoothed annual vol)**2.
    """
    r = pd.Series(s).dropna().astype(float)
    if len(r) == 0:
        return float("nan")
    return float(r.mean() * ppy)


def _ann_vol(s: pd.Series, ppy: int) -> float:
    r = pd.Series(s).dropna().astype(float)
    return float(r.std(ddof=1) * (ppy ** 0.5))


def _corr_to(a: pd.Series, b: pd.Series) -> float:
    x, y = pd.Series(a).align(pd.Series(b), join="inner")
    return float(x.corr(y))


def _pacing_metric(pacing_summary: pd.DataFrame) -> float:
    """Steady-state (peak) invested capital as % of committed, from M2 output."""
    if pacing_summary is None:
        return float("nan")
    if "peak_invested_pct" in getattr(pacing_summary, "attrs", {}):
        return float(pacing_summary.attrs["peak_invested_pct"])
    if "invested_pct" in getattr(pacing_summary, "columns", []):
        return float(pacing_summary["invested_pct"].max())
    warnings.warn("pacing_summary has no recognizable invested-% metric.", stacklevel=2)
    return float("nan")


def assemble_cma_table(public: pd.Series,
                       private_desmoothed: pd.Series,
                       pacing_summary: pd.DataFrame,
                       liquidity_summary: pd.Series | None,
                       dispersion_band: pd.Series | None,
                       periods_per_year: int,
                       source_label: str) -> pd.DataFrame:
    """One table, public and private side by side, consistent basis:
      expected_return | volatility (de-smoothed) | corr_to_public |
      liquidity/pacing-adjusted metric | dispersion band (if enabled)
    Header must carry source_label + 'methodology demonstrator'.
    """
    ppy = periods_per_year
    pub_name = public.name if getattr(public, "name", None) else "Public"
    priv_name = private_desmoothed.name if getattr(private_desmoothed, "name", None) \
        else "Private (de-smoothed)"
    if pub_name == priv_name:
        raise ValueError(
            f"public and private series share the name {pub_name!r}; assign distinct "
            "Series.name (e.g. from the config asset names) so table rows don't collide."
        )

    df = pd.DataFrame(
        {
            "expected_return": {
                pub_name: _ann_return(public, ppy),
                priv_name: _ann_return(private_desmoothed, ppy),
            },
            "annual_volatility": {   # private is on a de-smoothed basis (see attrs['notes'])
                pub_name: _ann_vol(public, ppy),
                priv_name: _ann_vol(private_desmoothed, ppy),
            },
            "corr_to_public": {
                pub_name: 1.0,
                priv_name: _corr_to(private_desmoothed, public),
            },
        }
    )

    # M2 pacing metric — private only (public has no pacing dimension)
    df["pacing_invested_pct"] = pd.Series(
        {pub_name: float("nan"), priv_name: _pacing_metric(pacing_summary)}
    )

    # M3 liquidity (optional) — private only. Contract: a pd.Series of named
    # liquidity metrics for the private asset (e.g. {"max_sustainable_pace": 0.35}).
    # Each metric becomes its own column; public rows are N/A.
    if liquidity_summary is not None:
        for metric, val in pd.Series(liquidity_summary).dropna().items():
            df[str(metric)] = pd.Series({pub_name: float("nan"), priv_name: float(val)})

    # M4 dispersion band (optional) — private only, rendered "low–high"
    if dispersion_band is not None:
        band = pd.Series(dispersion_band).dropna().astype(float)
        if len(band):
            df["dispersion_band"] = pd.Series(
                {pub_name: None, priv_name: f"{band.min():.1%}–{band.max():.1%}"}
            )

    notes = [
        "expected_return is ARITHMETIC annualized (invariant to de-smoothing).",
        "Private volatility and correlation are on a DE-SMOOTHED basis (Module 1);",
        "pacing_invested_pct is the steady-state peak from Module 2.",
    ]
    if liquidity_summary is not None:
        notes.append("max_sustainable_pace is the largest commitment pace (x base program) "
                     "that avoids a liquidity breach under the config shock (Module 3).")
    if dispersion_band is not None:
        notes.append("dispersion_band is the p25–p75 annualized-return band across manager "
                     "selection (Module 4).")
    notes.append("Not a vendor CMA — synthetic/transparent inputs only.")

    df.attrs["title"] = f"{source_label} — methodology demonstrator"
    df.attrs["source_label"] = source_label
    df.attrs["notes"] = " ".join(notes)
    return df
