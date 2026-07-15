"""Unit tests for Module 3 (liquidity stress).

Pins the shock mechanics, the denominator effect, coverage monotonicity, and
the max-sustainable-pace bisection boundary."""
import numpy as np
import pandas as pd
import pytest

from pmcma.pacing import PacingParams, project_program
from pmcma.liquidity import (
    apply_shock, denominator_effect, liquidity_coverage,
    coverage_at_pace, max_sustainable_pace,
)

PP = PacingParams(100.0, 12, 2.5, 0.13, 0.25, 0.03, 4)
PROG = project_program([100, 100, 100, 100, 100], PP)


def test_apply_shock_mechanics():
    s = apply_shock(PROG, public_drawdown=-0.35, distribution_haircut=0.5, call_acceleration=1.2)
    np.testing.assert_allclose(s["private_value"], PROG["nav"])
    np.testing.assert_allclose(s["call_stressed"], PROG["call"] * 1.2)
    np.testing.assert_allclose(s["distribution_stressed"], PROG["distribution"] * 0.5)
    # a public drawdown reduces the funding pool below committed capital
    assert s["liquid_value"].iloc[0] < PROG["cumulative_committed"].max()


def test_denominator_effect_flags_breach():
    s = apply_shock(PROG, public_drawdown=-0.35, distribution_haircut=0.5)
    de = denominator_effect(s, target_private_weight=0.15)
    assert de["breach"].any()                       # private weight blows past 15%
    assert (de["excess_over_target"][de["breach"]] > 0).all()


def test_coverage_more_breaches_with_bigger_haircut():
    mild = liquidity_coverage(apply_shock(PROG, -0.35, 0.2), 0.85)
    harsh = liquidity_coverage(apply_shock(PROG, -0.35, 0.8), 0.85)
    assert harsh["breach"].sum() >= mild["breach"].sum()


def test_coverage_at_pace_decreases_with_pace():
    params = dict(pacing=PP, schedule=[100] * 5, liquid_sleeve_weight=0.85, liquid_pool=150.0)
    shock = dict(public_drawdown=-0.3, distribution_haircut=0.5, call_acceleration=1.0)
    assert coverage_at_pace(params, shock, 0.5) > coverage_at_pace(params, shock, 2.0)


def test_max_sustainable_pace_is_the_boundary():
    params = dict(pacing=PP, schedule=[100] * 5, liquid_sleeve_weight=0.85, liquid_pool=150.0)
    shock = dict(public_drawdown=-0.30, distribution_haircut=0.5, call_acceleration=1.0)
    m = max_sustainable_pace(params, shock)
    assert m > 0.0
    assert coverage_at_pace(params, shock, m * 0.98) >= 1.0 - 1e-6   # just under: safe
    assert coverage_at_pace(params, shock, m * 1.05) < 1.0           # just over: breaches


def test_max_pace_decreases_with_shock_severity():
    params = dict(pacing=PP, schedule=[100] * 5, liquid_sleeve_weight=0.85, liquid_pool=200.0)
    mild = max_sustainable_pace(params, dict(public_drawdown=-0.10, distribution_haircut=0.20))
    severe = max_sustainable_pace(params, dict(public_drawdown=-0.40, distribution_haircut=0.60,
                                               call_acceleration=1.2))
    assert mild > severe > 0.0
