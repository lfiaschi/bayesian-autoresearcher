"""NHEFS problem runner.
Usage: uv run python problems/nhefs/prepare.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from runner import run_experiment

DATA_DIR = Path(__file__).parent / "data"

CONFOUNDER_COLS = [
    "sex", "race", "age", "school", "smokeintensity", "smokeyrs",
    "exercise", "active", "wt71",
]
CONTINUOUS_COLS = ["age", "school", "smokeintensity", "smokeyrs", "wt71"]


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load NHEFS dataset, drop NAs on key columns, standardize continuous confounders."""
    df = pd.read_csv(DATA_DIR / "nhefs.csv")

    # Keep only required columns
    key_cols = ["qsmk", "wt82_71"] + CONFOUNDER_COLS
    df = df[key_cols].dropna().reset_index(drop=True)

    # Standardize continuous confounders
    for col in CONTINUOUS_COLS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # No true_ate for this observational dataset
    metadata = {
        "treatment_col": "qsmk",
        "outcome_col": "wt82_71",
        "confounder_cols": CONFOUNDER_COLS,
    }
    return df, metadata


if __name__ == "__main__":
    run_experiment(
        problem_dir=Path(__file__).parent,
        load_data_fn=load_data,
    )
