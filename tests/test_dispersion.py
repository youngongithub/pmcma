"""Unit tests for Module 4 (manager dispersion)."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pmcma.dispersion import fit_lognormal_from_quartiles, simulate_selection, dispersion_band

TOP, MED, BOT = 2.1, 1.6, 1.2


def test_fit_pins_median_and_iqr():
    mu, sigma = fit_lognormal_from_quartiles(TOP, MED, BOT)
    z = norm.ppf(0.75)
    assert np.isclose(np.exp(mu), MED)                       # median exact
    assert np.isclose(np.exp(2 * sigma * z), TOP / BOT)      # log-IQR exact
    assert sigma > 0


def test_fit_rejects_misordered_inputs():
    with pytest.raises(ValueError):
        fit_lognormal_from_quartiles(1.2, 1.6, 2.1)          # top < bottom


def test_selection_skill_shifts_realized_multiple():
    mu, sigma = fit_lognormal_from_quartiles(TOP, MED, BOT)
    med_access = np.median(simulate_selection(mu, sigma, 200_000, 0.50))
    top_access = np.median(simulate_selection(mu, sigma, 200_000, 0.75))
    assert np.isclose(med_access, MED, rtol=0.02)            # median access ~ median mgr
    assert top_access > med_access                           # skill lifts the multiple
    assert np.isclose(top_access, TOP, rtol=0.05)            # ~ top-quartile manager


def test_band_is_ordered_and_reproduces_inputs():
    mu, sigma = fit_lognormal_from_quartiles(TOP, MED, BOT)
    band = dispersion_band(simulate_selection(mu, sigma, 200_000, 0.50), horizon_years=10)
    assert band["p25"] < band["p50"] < band["p75"]
    assert np.isclose(band["p50"], MED ** (1 / 10) - 1, atol=0.004)
    assert np.isclose(band["p25"], BOT ** (1 / 10) - 1, atol=0.005)


def test_simulate_is_reproducible():
    mu, sigma = fit_lognormal_from_quartiles(TOP, MED, BOT)
    a = simulate_selection(mu, sigma, 1000)
    b = simulate_selection(mu, sigma, 1000)
    np.testing.assert_array_equal(a, b)                      # fixed seed => deterministic
