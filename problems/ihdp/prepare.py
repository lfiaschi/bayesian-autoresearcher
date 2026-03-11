"""IHDP problem runner.
Usage: uv run python problems/ihdp/prepare.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from runner import run_experiment

DATA_DIR = Path(__file__).parent / "data"
CONFOUNDER_COLS = [f"x{i}" for i in range(1, 26)]
CONTINUOUS_COLS = [f"x{i}" for i in range(1, 7)]


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load IHDP dataset with metadata."""
    df = pd.read_csv(DATA_DIR / "ihdp.csv")

    # Standardize continuous confounders
    for col in CONTINUOUS_COLS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    true_ate = float((df["mu1"] - df["mu0"]).mean())

    metadata = {
        "treatment_col": "treatment",
        "outcome_col": "y_factual",
        "confounder_cols": CONFOUNDER_COLS,
        "true_ate": true_ate,
    }
    return df, metadata


if __name__ == "__main__":
    run_experiment(
        problem_dir=Path(__file__).parent,
        load_data_fn=load_data,
    )
