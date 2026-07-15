"""End-to-end smoke test for the config-driven pipeline (principle #1).

Runs pmcma.pipeline.run on the committed default config + sample data and checks
the artifact is produced with BOTH asset rows (regression guard for the
Series-name collision that silently dropped a row)."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def run_outputs(tmp_path_factory):
    # run in-process but from the repo root so relative config/data/outputs resolve
    import os
    from pmcma.pipeline import run
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        run("config/default.yaml")
    finally:
        os.chdir(cwd)
    return REPO / "outputs"


def test_cma_output_has_both_asset_rows(run_outputs):
    csv = run_outputs / "pe_vs_public_equity_demo_cma.csv"
    assert csv.exists()
    df = pd.read_csv(csv, comment="#", index_col=0)
    # the collision bug produced a single "ret" row; require two named assets
    assert len(df) == 2
    assert "Public Equity (MSCI ACWI)" in df.index
    assert "Private Equity (proxy)" in df.index
    # principle #5: labeling persisted into the file header
    header = csv.read_text().splitlines()[0]
    assert "methodology demonstrator" in header.lower()


def test_public_row_has_no_pacing_and_corr_one(run_outputs):
    df = pd.read_csv(run_outputs / "pe_vs_public_equity_demo_cma.csv",
                     comment="#", index_col=0)
    pub = df.loc["Public Equity (MSCI ACWI)"]
    assert pub["corr_to_public"] == pytest.approx(1.0)
    assert pd.isna(pub["pacing_invested_pct"])


def test_desmoothing_recovers_higher_private_risk(run_outputs):
    df = pd.read_csv(run_outputs / "pe_vs_public_equity_demo_desmooth.csv",
                     comment="#", index_col=0)
    assert df.loc["annualized_volatility", "de-smoothed"] > \
        df.loc["annualized_volatility", "reported"]
    assert df.loc["correlation_to_public", "de-smoothed"] > \
        df.loc["correlation_to_public", "reported"]
