"""Couple the pacing engine to a public-market shock to stress portfolio liquidity.

Demonstrates the denominator effect and the max commitment pace that survives.
This is equity/FI/FoF liquidity-stress logic applied to a private allocation.

MODEL / KEY ASSUMPTIONS (the signatures under-specify the portfolio anchor, so
these are stated explicitly):
  * Private value each period = the program's appraisal NAV. It is NOT marked
    down in the shock — that staleness IS the denominator effect.
  * `apply_shock` anchors the funding pool to the program's committed capital
    (the only scale available in its signature), applies a one-off public
    drawdown, then draws stressed calls and adds haircut distributions.
  * Pace-finding (`coverage_at_pace` / `max_sustainable_pace`) instead holds a
    FIXED liquid pool while scaling commitments — because project_program is
    linear in commitment (proven in M2), a pool that scaled with pace would
    leave coverage invariant and make the bisection degenerate.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def apply_shock(program: pd.DataFrame,
                public_drawdown: float,
                distribution_haircut: float,
                call_acceleration: float = 1.0) -> pd.DataFrame:
    """Overlay a stress scenario on a pacing program.

    - liquid (public) sleeve falls by public_drawdown
    - distributions cut by distribution_haircut (GPs stop exiting in weakness)
    - calls continue (or accelerate by call_acceleration)
    Returns per-period frame with liquid_value, private_value, private_weight.
    """
    committed = float(program["cumulative_committed"].max())
    call_s = program["call"].to_numpy() * call_acceleration
    dist_s = program["distribution"].to_numpy() * (1.0 - distribution_haircut)

    # funding pool: committed capital, marked down once by the public drawdown,
    # then drawn by stressed calls and topped up by (haircut) distributions
    liquid = committed * (1.0 + public_drawdown) + np.cumsum(dist_s - call_s)
    private = program["nav"].to_numpy()  # appraisal NAV — deliberately not shocked
    denom = private + liquid
    with np.errstate(divide="ignore", invalid="ignore"):
        private_weight = np.where(denom != 0, private / denom, np.nan)

    out = pd.DataFrame(
        {
            "call_stressed": call_s,
            "distribution_stressed": dist_s,
            "net_need": call_s - dist_s,        # +ve = cash needed this period
            "liquid_value": liquid,
            "private_value": private,
            "private_weight": private_weight,
        },
        index=program.index,
    )
    out.attrs["committed_pool"] = committed
    out.attrs["public_drawdown"] = public_drawdown
    return out


def denominator_effect(stressed: pd.DataFrame, target_private_weight: float) -> pd.DataFrame:
    """Track private_weight vs. target through the shock. Flag periods where the
    private allocation breaches target purely because the public denominator fell.
    """
    pw = stressed["private_weight"]
    return pd.DataFrame(
        {
            "private_weight": pw,
            "target": target_private_weight,
            "excess_over_target": pw - target_private_weight,
            "breach": pw > target_private_weight,
        },
        index=stressed.index,
    )


def liquidity_coverage(stressed: pd.DataFrame,
                       liquid_sleeve_weight: float,
                       redemption_demand: float = 0.0) -> pd.DataFrame:
    """Per period: can the liquid sleeve cover (net calls + redemptions +
    rebalancing need)? Returns coverage_ratio and a boolean breach flag.
    """
    # sleeve available BEFORE funding this period's need: liquid_value(t-1) is
    # liquid_value(t) + net_need(t) by construction in apply_shock
    pre_sleeve = liquid_sleeve_weight * (stressed["liquid_value"] + stressed["net_need"])
    need = stressed["net_need"] + redemption_demand

    ratio = np.full(len(need), np.inf)
    mask = need.to_numpy() > 0
    ratio[mask] = pre_sleeve.to_numpy()[mask] / need.to_numpy()[mask]

    return pd.DataFrame(
        {
            "available_liquid": pre_sleeve,
            "funding_need": need,
            "coverage_ratio": ratio,
            "breach": mask & (pre_sleeve.to_numpy() < need.to_numpy()),
        },
        index=stressed.index,
    )


def coverage_at_pace(params: dict, shock: dict, pace: float) -> float:
    """Minimum coverage ratio over the horizon when the base commitment schedule
    is scaled by `pace`, against a FIXED liquid pool. Exposed (and used by
    max_sustainable_pace) so the bisection boundary is directly testable.

    params keys: 'pacing' (PacingParams), 'schedule' (base list),
    'liquid_sleeve_weight' (float), optional 'liquid_pool' (float; defaults to
    the base schedule's committed capital) and 'redemption_demand' (float).
    shock keys: apply_shock kwargs (public_drawdown, distribution_haircut,
    call_acceleration).
    """
    from .pacing import project_program

    base = project_program(params["schedule"], params["pacing"])
    pool = float(params.get("liquid_pool", base["cumulative_committed"].max()))
    lsw = params["liquid_sleeve_weight"]
    red = params.get("redemption_demand", 0.0)
    accel = shock.get("call_acceleration", 1.0)
    hc = shock["distribution_haircut"]
    dd = shock["public_drawdown"]

    # linearity of project_program: program(pace) = pace * program(1)
    call_s = base["call"].to_numpy() * pace * accel
    dist_s = base["distribution"].to_numpy() * pace * (1.0 - hc)
    net_need = call_s - dist_s
    liquid = pool * (1.0 + dd) + np.cumsum(dist_s - call_s)
    pre_sleeve = lsw * (liquid + net_need)
    need = net_need + red

    ratio = np.full(len(need), np.inf)
    mask = need > 0
    ratio[mask] = pre_sleeve[mask] / need[mask]
    return float(ratio.min())


def max_sustainable_pace(params: dict, shock: dict, tol: float = 1e-3) -> float:
    """Bisection over commitment pace: the largest program pace whose
    liquidity_coverage never breaches under the shock. The allocator-relevant output.
    """
    if coverage_at_pace(params, shock, 0.0) < 1.0:
        return 0.0  # can't even survive redemptions at zero commitment pace

    lo, hi = 0.0, 1.0  # lo is always a PROVEN-safe pace (pace 0 => infinite coverage)
    while coverage_at_pace(params, shock, hi) >= 1.0:
        lo = hi        # hi confirmed safe; raise the safe floor before probing higher
        hi *= 2.0
        if hi > 1e6:   # pool so large the shock never bites; report the cap
            return hi

    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if coverage_at_pace(params, shock, mid) >= 1.0:
            lo = mid
        else:
            hi = mid
    return lo
