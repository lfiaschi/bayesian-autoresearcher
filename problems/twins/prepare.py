"""Twins problem runner.
Usage: uv run python problems/twins/prepare.py

Dataset note: The raw CSV contains only heavier twins (treatment=1) with both
factual (y_factual = Y(1)) and counterfactual (y_cfactual = Y(0)) outcomes.
We reconstruct a balanced dataset with both treatment arms using counterfactuals.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from runner import run_experiment

DATA_DIR = Path(__file__).parent / "data"

# Use 10 key confounders to keep sampling tractable
CONFOUNDER_COLS = [
    "pldel", "birattnd", "mager8", "ormoth", "mrace",
    "meduc6", "dmar", "adequacy", "gestat10", "csex",
]

# Columns with >5 unique values are treated as continuous and standardized
CONTINUOUS_COLS = ["mager8", "mrace", "gestat10"]

SUBSAMPLE_N = 5000
RANDOM_SEED = 42


def load_data() -> tuple[pd.DataFrame, dict]:
    """Load Twins dataset.

    Constructs a balanced dataset by stacking:
    - treatment=1 rows: heavier twin outcomes (y_factual)
    - treatment=0 rows: lighter twin outcomes (y_cfactual)
    then subsamples for tractable inference.
    """
    df_raw = pd.read_csv(DATA_DIR / "twins.csv")

    use_cols = CONFOUNDER_COLS + ["y_factual", "y_cfactual"]
    df_raw = df_raw[use_cols].dropna().reset_index(drop=True)

    # Construct treatment=1 rows (heavier twin, factual outcome)
    df_treated = df_raw[CONFOUNDER_COLS].copy()
    df_treated["treatment"] = 1
    df_treated["outcome"] = df_raw["y_factual"].values

    # Construct treatment=0 rows (lighter twin, counterfactual outcome)
    df_control = df_raw[CONFOUNDER_COLS].copy()
    df_control["treatment"] = 0
    df_control["outcome"] = df_raw["y_cfactual"].values

    df = pd.concat([df_treated, df_control], ignore_index=True)

    # Subsample to keep sampling fast (balanced subsample)
    half_n = SUBSAMPLE_N // 2
    df_t = df[df["treatment"] == 1].sample(n=half_n, random_state=RANDOM_SEED)
    df_c = df[df["treatment"] == 0].sample(n=half_n, random_state=RANDOM_SEED)
    df = pd.concat([df_t, df_c], ignore_index=True)

    # Standardize continuous confounders
    for col in CONTINUOUS_COLS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # True ATE: heavier twin has lower mortality
    true_ate = float(df_raw["y_factual"].mean() - df_raw["y_cfactual"].mean())

    metadata = {
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "confounder_cols": CONFOUNDER_COLS,
        "true_ate": true_ate,
    }
    return df, metadata


if __name__ == "__main__":
    run_experiment(
        problem_dir=Path(__file__).parent,
        load_data_fn=load_data,
    )
