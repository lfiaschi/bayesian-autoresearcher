"""Shared experiment runner for Bayesian autoresearcher.

Handles: config parsing, data splitting, model import, sampling,
scoring, and result printing. Each problem's prepare.py calls
run_experiment() with a custom load_data function.
"""
import importlib.util
import signal
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import yaml
from sklearn.model_selection import train_test_split

from scoring import (
    check_convergence,
    compute_ate_bias,
    compute_crps,
    compute_elpd,
    compute_mae,
    compute_rmse,
)


def parse_problem_config(problem_md_path: Path) -> dict:
    """Parse YAML frontmatter from problem.md.
    Expects file to start with '---', YAML content, then '---'.
    """
    text = problem_md_path.read_text()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    return yaml.safe_load(parts[1]) or {}


def split_data_random(
    df: pd.DataFrame,
    treatment_col: str,
    ratios: list[float],
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data randomly, stratified by treatment variable."""
    train_ratio, val_ratio, _test_ratio = ratios
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, stratify=df[treatment_col], random_state=seed
    )
    val_fraction_of_remaining = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_fraction_of_remaining,
        stratify=temp_df[treatment_col], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_data_temporal(
    df: pd.DataFrame,
    temporal_col: str,
    ratios: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically by temporal column."""
    df_sorted = df.sort_values(temporal_col).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    return (
        df_sorted.iloc[:train_end].reset_index(drop=True),
        df_sorted.iloc[train_end:val_end].reset_index(drop=True),
        df_sorted.iloc[val_end:].reset_index(drop=True),
    )


def make_data_dict(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounder_cols: list[str],
) -> dict:
    """Convert a DataFrame into the dict format expected by model.py."""
    X = df[confounder_cols].values.astype(np.float64)
    treatment = df[treatment_col].values.astype(np.float64)
    outcome = df[outcome_col].values.astype(np.float64)
    coords = {
        "obs": np.arange(len(df)),
        "features": confounder_cols,
    }
    return {"X": X, "treatment": treatment, "outcome": outcome, "coords": coords}


def import_model_module(model_path: Path):
    """Dynamically import model.py from a problem directory."""
    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SamplingTimeout(Exception):
    """Raised when sampling exceeds the time budget."""
    pass


def _timeout_handler(signum, frame):
    raise SamplingTimeout("Sampling exceeded time budget")


def sample_model(
    model: pm.Model, time_budget: int = 300, random_seed: int = 42
) -> az.InferenceData:
    """Sample from a PyMC model using nutpie with a time budget.

    Raises:
        SamplingTimeout: If sampling exceeds time_budget.
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(time_budget)
    try:
        with model:
            idata = pm.sample(
                draws=1000, tune=1000, chains=4,
                nuts_sampler="nutpie",
                random_seed=random_seed,
            )
            pm.compute_log_likelihood(idata, model=model)
            pm.sample_posterior_predictive(idata, extend_inferencedata=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    return idata


def _format_line(key: str, val) -> str:
    """Format a single key-value line with consistent alignment."""
    label = f"{key}:"
    if isinstance(val, bool):
        return f"{label:<20s}{val}"
    if isinstance(val, float):
        return f"{label:<20s}{val:.4f}"
    if isinstance(val, int):
        return f"{label:<20s}{val}"
    return f"{label:<20s}{val}"


def print_results(
    scores: dict,
    convergence: dict,
    causal: dict,
    timing: dict,
    n_params: int = 0,
) -> None:
    """Print standardized results block for log parsing."""
    print("---")
    for key, val in scores.items():
        if val is not None:
            print(_format_line(key, val))
    print(_format_line("convergence_ok", convergence["ok"]))
    print(_format_line("r_hat_max", convergence["r_hat_max"]))
    print(_format_line("ess_min", convergence["ess_min"]))
    print(_format_line("divergences", convergence["divergences"]))
    print(_format_line("n_params", n_params))
    for key, val in timing.items():
        print(_format_line(key, val))
    for key, val in causal.items():
        if val is not None and not isinstance(val, np.ndarray):
            print(_format_line(key, val))


def run_experiment(
    problem_dir: Path,
    load_data_fn: Callable[[], tuple[pd.DataFrame, dict]],
) -> None:
    """Run a single Bayesian modeling experiment.

    Args:
        problem_dir: Path to the problem directory.
        load_data_fn: Returns (dataframe, metadata_dict).
            metadata must have: treatment_col, outcome_col, confounder_cols.
            Optionally: true_ate.
    """
    t_start = time.time()

    # 1. Parse config
    config = parse_problem_config(problem_dir / "problem.md")
    split_strategy = config.get("split_strategy", "random")
    split_ratios = config.get("split_ratios", [0.6, 0.2, 0.2])
    time_budget = config.get("time_budget", 300)

    # 2. Load data
    df, metadata = load_data_fn()
    treatment_col = metadata["treatment_col"]
    outcome_col = metadata["outcome_col"]
    confounder_cols = metadata["confounder_cols"]
    true_ate = metadata.get("true_ate")

    # 3. Split
    if split_strategy == "temporal":
        temporal_col = config["temporal_column"]
        train_df, val_df, test_df = split_data_temporal(df, temporal_col, split_ratios)
    else:
        train_df, val_df, test_df = split_data_random(df, treatment_col, split_ratios)

    # 4. Make data dicts
    train_data = make_data_dict(train_df, treatment_col, outcome_col, confounder_cols)
    val_data = make_data_dict(val_df, treatment_col, outcome_col, confounder_cols)

    # 5. Import and build model
    model_module = import_model_module(problem_dir / "model.py")
    pm_model = model_module.build_model(train_data)
    n_params = sum(1 for _ in pm_model.free_RVs)

    # 6. Sample with time budget
    t_sample_start = time.time()
    idata = sample_model(pm_model, time_budget=time_budget)
    sampling_seconds = time.time() - t_sample_start

    # Save immediately
    runs_dir = problem_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    idata.to_netcdf(str(runs_dir / "latest.nc"))

    # 7. Convergence gate
    convergence = check_convergence(idata)

    scores = {}
    causal = {}

    if convergence["ok"]:
        # 8. Score on validation set
        val_pred = model_module.predict(idata, pm_model, val_data)
        scores["val_crps"] = compute_crps(val_data["outcome"], val_pred)
        scores["val_mae"] = compute_mae(val_data["outcome"], np.mean(val_pred, axis=0))
        scores["val_rmse"] = compute_rmse(val_data["outcome"], np.mean(val_pred, axis=0))
        scores["val_elpd"] = compute_elpd(idata)

        # 9. Causal effects
        causal_result = model_module.estimate_causal_effect(idata, pm_model, train_data)
        if causal_result:
            causal["ate_estimate"] = causal_result.get("ate")
            ate_samples = causal_result.get("ate_samples")
            if ate_samples is not None:
                causal["ate_hdi_3"] = float(np.percentile(ate_samples, 3))
                causal["ate_hdi_97"] = float(np.percentile(ate_samples, 97))
            if true_ate is not None:
                scores["val_ate_bias"] = compute_ate_bias(causal_result["ate"], true_ate)

    total_seconds = time.time() - t_start
    timing = {"sampling_seconds": sampling_seconds, "total_seconds": total_seconds}
    print_results(scores, convergence, causal, timing, n_params=n_params)
