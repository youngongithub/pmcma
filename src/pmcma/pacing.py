"""Takahashi-Alexander (Yale) deterministic cash-flow model.

Projects capital calls, distributions, and NAV for a commitment, and aggregates
multi-vintage programs to a steady-state invested %.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, replace


@dataclass
class PacingParams:
    """Units matter here — a per-period rate used as annual (or vice versa) passes
    every mechanical test and silently moves the headline invested-% figure.

      fund_life          years
      growth_rate        ANNUAL NAV growth (G); compounded down to per-period
      yield_rate         ANNUAL minimum distribution yield (Y); compounded down like G
      contribution_rate  PER-PERIOD fraction of uncalled commitment called (RC)
      bow_factor         unitless back-loading exponent (B)
    """
    commitment: float
    fund_life: int
    bow_factor: float
    growth_rate: float
    contribution_rate: float
    yield_rate: float
    periods_per_year: int = 4


def project_fund(p: PacingParams) -> pd.DataFrame:
    """Single-fund projection. Returns per-period DataFrame with columns:
    ['call', 'distribution', 'nav', 'net_cashflow', 'cumulative_paid_in'].

    Per period t (of T = fund_life * periods_per_year):
      call(t)  = contribution_rate * uncalled_commitment(t)
      rate_of_dist(t) = max(yield_per_period, (t/T) ** bow_factor)   # bow back-loads
      distribution(t) = rate_of_dist(t) * nav(t-1) * (1 + growth_per_period)
      nav(t)   = nav(t-1) * (1 + growth_per_period) + call(t) - distribution(t)
    """
    ppy = p.periods_per_year
    T = int(round(p.fund_life * ppy))
    # annual NAV growth and annual yield floor, both compounded down to per-period
    g = (1.0 + p.growth_rate) ** (1.0 / ppy) - 1.0
    y = (1.0 + p.yield_rate) ** (1.0 / ppy) - 1.0

    rows = []
    nav_prev = 0.0
    cum_paid = 0.0
    for t in range(1, T + 1):
        uncalled = p.commitment - cum_paid
        call = p.contribution_rate * uncalled
        rate = max(y, (t / T) ** p.bow_factor)
        grown = nav_prev * (1.0 + g)
        dist = rate * grown
        nav = grown + call - dist
        cum_paid += call
        rows.append(
            {
                "call": call,
                "distribution": dist,
                "nav": nav,
                "net_cashflow": dist - call,   # LP perspective: + = cash returned
                "cumulative_paid_in": cum_paid,
            }
        )
        nav_prev = nav

    return pd.DataFrame(rows, index=pd.RangeIndex(1, T + 1, name="period"))


def project_program(commitments: list[float], p: PacingParams) -> pd.DataFrame:
    """Overlay staggered annual vintages (one commitment per year) and sum
    their cash flows. Returns aggregate calls/distributions/NAV over time and
    the steady-state invested capital as % of cumulative commitments.

    Answers: 'to hold X% invested, how much must I COMMIT and when?'
    """
    ppy = p.periods_per_year
    T = int(round(p.fund_life * ppy))
    n_vint = len(commitments)
    total = (n_vint - 1) * ppy + T

    call = np.zeros(total)
    dist = np.zeros(total)
    nav = np.zeros(total)
    committed = np.zeros(total)

    for k, c in enumerate(commitments):
        if c == 0:
            continue
        fund = project_fund(replace(p, commitment=float(c)))
        off = k * ppy  # each vintage starts one year (ppy periods) after the last
        call[off:off + T] += fund["call"].to_numpy()
        dist[off:off + T] += fund["distribution"].to_numpy()
        nav[off:off + T] += fund["nav"].to_numpy()
        committed[off:] += c  # commitment stands from its vintage start onward

    with np.errstate(divide="ignore", invalid="ignore"):
        invested_pct = np.where(committed > 0, nav / committed, np.nan)

    df = pd.DataFrame(
        {
            "call": call,
            "distribution": dist,
            "nav": nav,
            "net_cashflow": dist - call,
            "cumulative_paid_in": np.cumsum(call),
            "cumulative_committed": committed,
            "invested_pct": invested_pct,
        },
        index=pd.RangeIndex(1, total + 1, name="period"),
    )
    # steady-state ≈ the plateau of NAV / committed capital
    df.attrs["peak_invested_pct"] = float(np.nanmax(invested_pct))
    return df
