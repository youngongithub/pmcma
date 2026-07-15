"""Data loading. Minimal surface for M1; extend as later modules need it."""
from __future__ import annotations
import pandas as pd


def load_returns(path: str, date_col: str = "date", ret_col: str = "ret") -> pd.Series:
    """Load a periodic return series from CSV into a float Series indexed by date.

    Expects columns [date, ret] with ret as decimal (0.03 = 3%), one row per
    period. Sorts by date and drops NaNs. This is the only IO M1 needs.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    s = df.sort_values(date_col).set_index(date_col)[ret_col].astype(float)
    return s.dropna()
