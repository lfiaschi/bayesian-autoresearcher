"""LaLonde problem runner.
Usage: uv run python problems/lalonde/prepare.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from runner import run_experiment

DATA_DIR = Path(__file__).parent / "data"

CONFOUNDER_COLS = ["age", "education", "black", "hispanic", "married", "nodegree", "re75"]
CONTINUOUS_COLS = ["age", "education", "re75"]


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load LaLonde dataset and standardize continuous confounders."""
    df = pd.read_csv(DATA_DIR / "lalonde.csv")

    # Standardize continuous confounders
    for col in CONTINUOUS_COLS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # True ATE: difference in mean re78 between treated and control
    # (treated group is from a randomized experiment, so this is valid)
    treated_mean = df.loc[df["treatment"] == 1, "re78"].mean()
    control_mean = df.loc[df["treatment"] == 0, "re78"].mean()
    true_ate = float(treated_mean - control_mean)

    metadata = {
        "treatment_col": "treatment",
        "outcome_col": "re78",
        "confounder_cols": CONFOUNDER_COLS,
        "true_ate": true_ate,
    }
    return df, metadata


if __name__ == "__main__":
    run_experiment(
        problem_dir=Path(__file__).parent,
        load_data_fn=load_data,
    )
