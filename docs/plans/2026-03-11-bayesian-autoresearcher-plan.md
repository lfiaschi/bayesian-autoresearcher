# Bayesian Autoresearcher Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous research loop where Claude iterates on PyMC causal models, scored by CRPS and convergence diagnostics, across 4 causal inference datasets.

**Architecture:** Orchestrator agent runs a Karpathy-style loop forever. Per experiment, it spawns a coding sub-agent (with pymc-modeling skill) that edits only `model.py`. The runner (`prepare.py`) handles data loading, sampling, scoring, and result printing. Shared `scoring.py` and `runner.py` provide reusable infrastructure.

**Tech Stack:** Python 3.11+, uv, PyMC 5+, nutpie, ArviZ, properscoring, pandas, scikit-learn

---

## File Map

```
bayesian-autoresearcher/
├── pyproject.toml              # uv project definition
├── .python-version             # Pin Python 3.11
├── .gitignore
├── CLAUDE.md                   # Project-specific Claude instructions
├── program.md                  # Orchestrator loop instructions
├── scoring.py                  # Shared: CRPS, ELPD, MAE, convergence checks
├── runner.py                   # Shared: experiment runner (sample, score, print)
├── download_datasets.py        # One-time dataset fetcher
├── tests/
│   ├── test_scoring.py
│   └── test_runner.py
├── problems/
│   ├── ihdp/
│   │   ├── problem.md          # YAML frontmatter config + problem description
│   │   ├── data/               # CSV files (gitignored)
│   │   ├── prepare.py          # load_data() + call runner
│   │   ├── model.py            # Agent-editable baseline
│   │   └── runs/               # Saved .nc files (gitignored)
│   ├── twins/
│   │   ├── problem.md
│   │   ├── data/
│   │   ├── prepare.py
│   │   ├── model.py
│   │   └── runs/
│   ├── lalonde/
│   │   ├── problem.md
│   │   ├── data/
│   │   ├── prepare.py
│   │   ├── model.py
│   │   └── runs/
│   └── nhefs/
│       ├── problem.md
│       ├── data/
│       ├── prepare.py
│       ├── model.py
│       └── runs/
```

---

## Chunk 1: Project Foundation

### Task 1: Initialize project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `CLAUDE.md`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "bayesian-autoresearcher"
version = "0.1.0"
description = "Autonomous Bayesian causal model research loop"
requires-python = ">=3.11"
dependencies = [
    "pymc>=5.10",
    "nutpie>=0.13",
    "arviz>=0.18",
    "numpy>=1.26",
    "pandas>=2.1",
    "scipy>=1.12",
    "scikit-learn>=1.4",
    "requests>=2.31",
    "properscoring>=0.1",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create .python-version**

```
3.11
```

- [ ] **Step 3: Create .gitignore**

```gitignore
# Data and results (large files, downloaded per-machine)
problems/*/data/
problems/*/runs/
*.nc

# Python
__pycache__/
*.pyc
.venv/

# Experiment logs
run.log
results.tsv

# uv — lock file is committed for reproducibility

# OS
.DS_Store
```

- [ ] **Step 4: Create CLAUDE.md**

```markdown
# Bayesian Autoresearcher

## Project Overview
Autonomous Bayesian causal model research loop. Claude iterates on PyMC models
to estimate causal effects, scored by proper scoring rules.

## Key Files
- `program.md` — Read this first. Orchestrator loop instructions.
- `scoring.py` — Shared scoring functions (CRPS, ELPD, convergence)
- `runner.py` — Shared experiment runner
- `problems/<name>/prepare.py` — Per-problem runner (DO NOT MODIFY)
- `problems/<name>/model.py` — The ONLY file the coding sub-agent edits
- `problems/<name>/problem.md` — Problem statement and config

## Rules
- Sub-agents that write model.py MUST load the pymc-modeling skill first
- model.py must expose: build_model(), predict(), estimate_causal_effect()
- Never modify prepare.py, scoring.py, or runner.py during experiments
- Use functional programming style — pure functions, no classes (except PyMC context)
- Do not catch generic exceptions

## Running an experiment
```bash
uv run python problems/<name>/prepare.py
```
```

- [ ] **Step 5: Init git repo and install deps**

Run: `git init && uv sync`

- [ ] **Step 6: Commit scaffold**

```bash
git add pyproject.toml .python-version .gitignore CLAUDE.md
git commit -m "chore: init project scaffold with uv"
```

---

### Task 2: Write scoring.py

**Files:**
- Create: `scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write failing tests for scoring functions**

Create `tests/test_scoring.py`:

```python
"""Tests for scoring functions."""
import numpy as np
import pytest
from scoring import (
    compute_crps,
    compute_mae,
    compute_rmse,
    check_convergence,
    compute_ate_bias,
    compute_elpd,
)


def test_crps_perfect_prediction():
    """CRPS should be near 0 when samples match observations."""
    observed = np.array([1.0, 2.0, 3.0])
    # Samples tightly centered on observed values
    samples = np.column_stack([
        np.random.normal(loc=observed, scale=0.01, size=len(observed))
        for _ in range(1000)
    ]).T  # shape: (1000, 3)
    crps = compute_crps(observed, samples)
    assert crps < 0.05


def test_crps_bad_prediction():
    """CRPS should be large when predictions are far off."""
    observed = np.array([1.0, 2.0, 3.0])
    samples = np.random.normal(loc=100.0, scale=1.0, size=(1000, 3))
    crps = compute_crps(observed, samples)
    assert crps > 10.0


def test_crps_returns_float():
    observed = np.array([1.0, 2.0])
    samples = np.random.normal(size=(100, 2))
    result = compute_crps(observed, samples)
    assert isinstance(result, float)


def test_mae():
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.1, 2.2, 2.7])
    mae = compute_mae(observed, predicted)
    assert abs(mae - 0.2) < 1e-10


def test_rmse():
    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 3.0])
    assert compute_rmse(observed, predicted) == 0.0


def test_ate_bias():
    estimated = 3.5
    true_ate = 4.0
    bias = compute_ate_bias(estimated, true_ate)
    assert abs(bias - 0.5) < 1e-10


def test_ate_bias_none_true():
    """When true ATE is None, return None."""
    assert compute_ate_bias(3.5, None) is None


def test_convergence_good(good_idata):
    result = check_convergence(good_idata)
    assert result["ok"] is True
    assert result["r_hat_max"] < 1.01
    assert result["ess_min"] > 400
    assert result["divergences"] == 0


def test_convergence_bad_rhat(bad_rhat_idata):
    result = check_convergence(bad_rhat_idata)
    assert result["ok"] is False


def test_elpd_returns_none_without_log_likelihood():
    """ELPD should return None when log_likelihood is missing."""
    import arviz as az
    rng = np.random.default_rng(42)
    data = {"mu": rng.normal(size=(4, 1000))}
    idata = az.from_dict(posterior=data)
    assert compute_elpd(idata) is None


@pytest.fixture
def good_idata():
    """Create a mock InferenceData with good convergence."""
    import arviz as az
    rng = np.random.default_rng(42)
    # 4 chains, 1000 draws, well-mixed
    data = {"mu": rng.normal(size=(4, 1000)), "sigma": rng.exponential(size=(4, 1000))}
    return az.from_dict(posterior=data)


@pytest.fixture
def bad_rhat_idata():
    """Create InferenceData with poor convergence (chains disagree)."""
    import arviz as az
    rng = np.random.default_rng(42)
    # Chains have very different means → high r_hat
    data = {
        "mu": np.array([
            rng.normal(loc=0, size=1000),
            rng.normal(loc=10, size=1000),
            rng.normal(loc=20, size=1000),
            rng.normal(loc=30, size=1000),
        ]),
    }
    return az.from_dict(posterior=data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scoring'`

- [ ] **Step 3: Write scoring.py**

Create `scoring.py`:

```python
"""Shared scoring functions for Bayesian model evaluation.

Provides proper scoring rules (CRPS), point metrics (MAE, RMSE),
causal-specific metrics (ATE bias), and MCMC convergence checking.
"""
from typing import Optional

import arviz as az
import numpy as np
from properscoring import crps_ensemble


def compute_crps(observed: np.ndarray, samples: np.ndarray) -> float:
    """Compute mean CRPS across observations.

    Args:
        observed: Shape (n_obs,). Actual values.
        samples: Shape (n_samples, n_obs). Posterior predictive draws.

    Returns:
        Mean CRPS (lower is better).
    """
    scores = np.array([
        crps_ensemble(observed[i], samples[:, i])
        for i in range(len(observed))
    ])
    return float(np.mean(scores))


def compute_mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(observed - predicted)))


def compute_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def compute_ate_bias(estimated_ate: float, true_ate: Optional[float]) -> Optional[float]:
    """Absolute bias of ATE estimate vs ground truth.

    Returns None if true_ate is not available.
    """
    if true_ate is None:
        return None
    return float(abs(estimated_ate - true_ate))


def check_convergence(idata: az.InferenceData) -> dict:
    """Check MCMC convergence diagnostics.

    Returns dict with:
        ok: bool — all diagnostics pass
        r_hat_max: float — worst r_hat across parameters
        ess_min: float — minimum ESS across parameters
        divergences: int — total divergent transitions
    """
    summary = az.summary(idata)
    r_hat_max = float(summary["r_hat"].max())
    ess_bulk_min = float(summary["ess_bulk"].min())
    ess_tail_min = float(summary["ess_tail"].min())
    ess_min = min(ess_bulk_min, ess_tail_min)

    divergences = 0
    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divergences = int(idata.sample_stats["diverging"].sum().item())

    total_samples = 0
    if hasattr(idata, "posterior"):
        shape = list(idata.posterior.data_vars.values())[0].shape
        total_samples = shape[0] * shape[1]  # chains * draws

    div_rate = divergences / max(total_samples, 1)

    ok = (r_hat_max < 1.01) and (ess_min > 400) and (div_rate < 0.001)

    return {
        "ok": ok,
        "r_hat_max": r_hat_max,
        "ess_min": ess_min,
        "divergences": divergences,
    }


def compute_elpd(idata: az.InferenceData) -> Optional[float]:
    """Compute ELPD via LOO-CV. Returns None if log_likelihood not available."""
    if not hasattr(idata, "log_likelihood"):
        return None
    loo = az.loo(idata, pointwise=False)
    return float(loo.elpd_loo)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scoring.py tests/test_scoring.py
git commit -m "feat: add scoring module with CRPS, MAE, RMSE, ATE bias, convergence checks"
```

---

### Task 3: Write runner.py

**Files:**
- Create: `runner.py`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for runner utilities**

Create `tests/test_runner.py`:

```python
"""Tests for experiment runner utilities."""
import numpy as np
import pandas as pd
import pytest
import yaml
from runner import (
    parse_problem_config,
    split_data_random,
    split_data_temporal,
    make_data_dict,
    print_results,
)


def test_parse_problem_config(tmp_path):
    """Parse YAML frontmatter from problem.md."""
    problem_md = tmp_path / "problem.md"
    problem_md.write_text(
        "---\n"
        "primary_metric: crps\n"
        "secondary_metrics: [elpd, mae]\n"
        "split_strategy: random\n"
        "split_ratios: [0.6, 0.2, 0.2]\n"
        "---\n"
        "# Test Problem\n"
        "Some description.\n"
    )
    config = parse_problem_config(problem_md)
    assert config["primary_metric"] == "crps"
    assert config["split_strategy"] == "random"
    assert config["split_ratios"] == [0.6, 0.2, 0.2]


def test_split_data_random():
    """Random split stratified by treatment."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "treatment": rng.choice([0, 1], size=200),
        "outcome": rng.normal(size=200),
        "x1": rng.normal(size=200),
    })
    train, val, test = split_data_random(
        df, treatment_col="treatment", ratios=[0.6, 0.2, 0.2], seed=42
    )
    assert len(train) + len(val) + len(test) == 200
    assert abs(len(train) - 120) < 10  # ~60%
    assert abs(len(val) - 40) < 10     # ~20%
    # Stratification: treatment proportions roughly preserved
    orig_rate = df["treatment"].mean()
    train_rate = train["treatment"].mean()
    assert abs(train_rate - orig_rate) < 0.1


def test_split_data_temporal():
    """Temporal split by date column."""
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "treatment": np.random.choice([0, 1], size=100),
        "outcome": np.random.normal(size=100),
    })
    train, val, test = split_data_temporal(
        df, temporal_col="date", ratios=[0.6, 0.2, 0.2]
    )
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    # Temporal ordering preserved
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()


def test_make_data_dict():
    """Convert DataFrame to model-ready dict."""
    df = pd.DataFrame({
        "treatment": [0, 1, 0, 1],
        "outcome": [1.0, 2.0, 1.5, 2.5],
        "x1": [0.1, 0.2, 0.3, 0.4],
        "x2": [0.5, 0.6, 0.7, 0.8],
    })
    result = make_data_dict(
        df,
        treatment_col="treatment",
        outcome_col="outcome",
        confounder_cols=["x1", "x2"],
    )
    assert result["X"].shape == (4, 2)
    assert result["treatment"].shape == (4,)
    assert result["outcome"].shape == (4,)
    assert "obs" in result["coords"]
    assert result["coords"]["features"] == ["x1", "x2"]


def test_print_results_format(capsys):
    """Output format matches expected pattern."""
    scores = {"val_crps": 0.45, "val_mae": 1.23}
    convergence = {"ok": True, "r_hat_max": 1.002, "ess_min": 856, "divergences": 0}
    causal = {"ate_estimate": 3.45, "ate_hdi_3": 2.1, "ate_hdi_97": 4.8}
    timing = {"sampling_seconds": 142.3, "total_seconds": 180.1}

    print_results(scores, convergence, causal, timing, n_params=5)

    captured = capsys.readouterr()
    assert "---" in captured.out
    assert "val_crps:" in captured.out
    assert "convergence_ok:" in captured.out
    assert "True" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'runner'`

- [ ] **Step 3: Write runner.py**

Create `runner.py`:

```python
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
    Returns the parsed YAML as a dict.
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
    """Split data randomly, stratified by treatment variable.

    Args:
        df: Full dataset.
        treatment_col: Column name for stratification.
        ratios: [train, val, test] fractions summing to 1.0.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
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
    """Split data chronologically by temporal column.

    Sorts by temporal_col, then takes first train_ratio as train,
    next val_ratio as validation, rest as test.
    """
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
    """Convert a DataFrame into the dict format expected by model.py.

    Returns:
        Dict with keys: X, treatment, outcome, coords.
    """
    X = df[confounder_cols].values.astype(np.float64)
    treatment = df[treatment_col].values.astype(np.float64)
    outcome = df[outcome_col].values.astype(np.float64)
    coords = {
        "obs": np.arange(len(df)),
        "features": confounder_cols,
    }
    return {
        "X": X,
        "treatment": treatment,
        "outcome": outcome,
        "coords": coords,
    }


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

    Args:
        model: PyMC model to sample.
        time_budget: Maximum seconds for sampling (default 300 = 5 min).
        random_seed: Random seed for reproducibility.

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
    """Print standardized results block for log parsing.

    Every line starts with 'key:' left-aligned in 20 chars, then value.
    This format is parseable by grep '^key:'.
    """
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

    This is the main entry point called by each problem's prepare.py.

    Args:
        problem_dir: Path to the problem directory.
        load_data_fn: Returns (dataframe, metadata_dict).
            metadata_dict must contain: treatment_col, outcome_col,
            confounder_cols. Optionally: true_ate.
    """
    t_start = time.time()

    # 1. Parse config
    config = parse_problem_config(problem_dir / "problem.md")
    split_strategy = config.get("split_strategy", "random")
    split_ratios = config.get("split_ratios", [0.6, 0.2, 0.2])
    primary_metric = config.get("primary_metric", "crps")
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
        scores["val_mae"] = compute_mae(
            val_data["outcome"], np.mean(val_pred, axis=0)
        )
        scores["val_rmse"] = compute_rmse(
            val_data["outcome"], np.mean(val_pred, axis=0)
        )
        scores["val_elpd"] = compute_elpd(idata)

        # 9. Causal effects
        causal_result = model_module.estimate_causal_effect(
            idata, pm_model, train_data
        )
        if causal_result:
            causal["ate_estimate"] = causal_result.get("ate")
            ate_samples = causal_result.get("ate_samples")
            if ate_samples is not None:
                causal["ate_hdi_3"] = float(np.percentile(ate_samples, 3))
                causal["ate_hdi_97"] = float(np.percentile(ate_samples, 97))
            if true_ate is not None:
                scores["val_ate_bias"] = compute_ate_bias(
                    causal_result["ate"], true_ate
                )

    total_seconds = time.time() - t_start
    timing = {
        "sampling_seconds": sampling_seconds,
        "total_seconds": total_seconds,
    }

    print_results(scores, convergence, causal, timing, n_params=n_params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add runner.py tests/test_runner.py
git commit -m "feat: add experiment runner with config parsing, splitting, and scoring pipeline"
```

---

## Chunk 2: Dataset Download & IHDP Problem

### Task 4: Write download_datasets.py

**Files:**
- Create: `download_datasets.py`

- [ ] **Step 1: Write download script**

Create `download_datasets.py`:

```python
"""Download all causal inference datasets.

Usage:
    uv run python download_datasets.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent / "problems"


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination path."""
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"  Saved to {dest}")


def download_ihdp() -> None:
    """Download IHDP dataset from CEVAE repository."""
    print("\n=== IHDP ===")
    data_dir = BASE_DIR / "ihdp" / "data"
    dest = data_dir / "ihdp.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # IHDP CSV has no header; columns are: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25
    col_names = (
        ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]
        + [f"x{i}" for i in range(1, 26)]
    )
    df = pd.read_csv(io.StringIO(response.text), header=None, names=col_names)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  IHDP: {len(df)} rows, {len(df.columns)} columns")
    print(f"  True ATE: {(df['mu1'] - df['mu0']).mean():.4f}")


def download_twins() -> None:
    """Download Twins dataset from CEVAE repository."""
    print("\n=== Twins ===")
    data_dir = BASE_DIR / "twins" / "data"
    dest = data_dir / "twins.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    # Twin data: treatment = heavier twin, outcome = mortality
    twin_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/twin_pairs_T_3years_sGA.csv"
    covar_url = "https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/TWINS/covar_type.csv"

    twin_resp = requests.get(twin_url, timeout=60)
    twin_resp.raise_for_status()
    covar_resp = requests.get(covar_url, timeout=60)
    covar_resp.raise_for_status()

    twins_df = pd.read_csv(io.StringIO(twin_resp.text))
    covar_df = pd.read_csv(io.StringIO(covar_resp.text))

    # Build dataset: outcome_0, outcome_1 are mortality for lighter/heavier twin
    # Treatment: randomly assigned (binary)
    rng = np.random.default_rng(42)
    n = len(twins_df)
    treatment = rng.choice([0, 1], size=n)

    outcome_0 = twins_df["dbirwt_0"].values  # birth weight twin 0
    outcome_1 = twins_df["dbirwt_1"].values  # birth weight twin 1
    y_factual = np.where(treatment == 0, outcome_0, outcome_1)
    y_cfactual = np.where(treatment == 0, outcome_1, outcome_0)

    # Select confounder columns from covariate types
    confounder_cols = [c for c in covar_df.columns if c not in ["Unnamed: 0"]]

    result_df = pd.DataFrame({
        "treatment": treatment,
        "y_factual": y_factual,
        "y_cfactual": y_cfactual,
    })

    # Add covariates
    for col in confounder_cols:
        if col in covar_df.columns:
            result_df[col] = covar_df[col].values[:n]

    data_dir.mkdir(parents=True, exist_ok=True)
    result_df.dropna().to_csv(dest, index=False)
    print(f"  Twins: {len(result_df)} rows")


def download_lalonde() -> None:
    """Download LaLonde dataset (NSW experimental data)."""
    print("\n=== LaLonde ===")
    data_dir = BASE_DIR / "lalonde" / "data"
    dest = data_dir / "lalonde.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    # Use Dehejia-Wahba sample
    treat_url = "https://users.nber.org/~rdehejia/data/nsw_treated.txt"
    control_url = "https://users.nber.org/~rdehejia/data/nsw_control.txt"

    col_names = [
        "treatment", "age", "education", "black", "hispanic",
        "married", "nodegree", "re74", "re75", "re78"
    ]

    treat_resp = requests.get(treat_url, timeout=60)
    treat_resp.raise_for_status()
    control_resp = requests.get(control_url, timeout=60)
    control_resp.raise_for_status()

    treat_df = pd.read_csv(
        io.StringIO(treat_resp.text), sep=r"\s+", header=None, names=col_names
    )
    control_df = pd.read_csv(
        io.StringIO(control_resp.text), sep=r"\s+", header=None, names=col_names
    )

    df = pd.concat([treat_df, control_df], ignore_index=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  LaLonde: {len(df)} rows")
    # Experimental ATE is the difference in mean re78
    ate = treat_df["re78"].mean() - control_df["re78"].mean()
    print(f"  Experimental ATE (re78): ${ate:.2f}")


def download_nhefs() -> None:
    """Download NHEFS dataset for smoking cessation study."""
    print("\n=== NHEFS ===")
    data_dir = BASE_DIR / "nhefs" / "data"
    dest = data_dir / "nhefs.csv"
    if dest.exists():
        print("  Already downloaded.")
        return

    url = "https://raw.githubusercontent.com/causalinfbook/causalinfbook2024/main/nhefs.csv"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text))
    # Keep complete cases for key variables
    key_cols = [
        "qsmk", "sex", "race", "age", "school", "smokeintensity",
        "smokeyrs", "exercise", "active", "wt71", "wt82", "wt82_71"
    ]
    df = df.dropna(subset=key_cols)
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"  NHEFS: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    print("Downloading causal inference datasets...")
    download_ihdp()
    download_twins()
    download_lalonde()
    download_nhefs()
    print("\nDone! All datasets ready.")
```

- [ ] **Step 2: Run the download**

Run: `uv run python download_datasets.py`
Expected: All 4 datasets downloaded to their respective `data/` directories.

- [ ] **Step 3: Commit**

```bash
git add download_datasets.py
git commit -m "feat: add dataset download script for IHDP, Twins, LaLonde, NHEFS"
```

---

### Task 5: Set up IHDP problem

**Files:**
- Create: `problems/ihdp/problem.md`
- Create: `problems/ihdp/prepare.py`
- Create: `problems/ihdp/model.py`

- [ ] **Step 1: Create problem.md**

Create `problems/ihdp/problem.md`:

```markdown
---
primary_metric: crps
secondary_metrics: [elpd, mae, ate_bias]
split_strategy: random
split_ratios: [0.6, 0.2, 0.2]
temporal_column: null
time_budget: 300
---

# IHDP — Infant Health and Development Program

## Problem Statement

Estimate the causal effect of home visits by specialist doctors (treatment) on
infant cognitive test scores (outcome) using observational data from the Infant
Health and Development Program.

## Dataset

Semi-synthetic benchmark from Hill (2011). 747 observations, 25 covariates.
Treatment group was non-randomly subsampled to create confounding.
Both factual and counterfactual outcomes are available, enabling ground truth
ATE computation.

## Variables

- **treatment**: `treatment` (binary: 0/1, home visits)
- **outcome**: `y_factual` (continuous: cognitive test score)
- **confounders**: `x1` through `x25` (mix of continuous and binary)
  - x1-x6: continuous (birth weight, head circumference, weeks preterm, etc.)
  - x7-x25: binary (demographics, maternal characteristics)

## Causal Estimand

- **ATE**: Average Treatment Effect = E[Y(1)] - E[Y(0)]
- **Ground truth available**: Computed from `mu1 - mu0` columns.
  True ATE ≈ 4.0 (varies by realization).
- **CATE**: Conditional effects across subgroups (optional stretch goal)

## Modeling Guidance

- Start with a simple linear model with treatment + confounders
- Consider hierarchical structure on confounder effects
- Treatment-confounder interactions may improve CATE estimation
- Non-centered parameterization if divergences appear
- Standardize continuous confounders (x1-x6)
```

- [ ] **Step 2: Create prepare.py**

Create `problems/ihdp/prepare.py`:

```python
"""IHDP problem runner.

Loads the IHDP dataset and runs the experiment pipeline.
Usage: uv run python problems/ihdp/prepare.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
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

    # Compute ground truth ATE from potential outcomes
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
```

- [ ] **Step 3: Create baseline model.py**

Create `problems/ihdp/model.py`:

```python
"""IHDP Bayesian causal model — baseline.

Simple linear regression with treatment and confounders.
This is the starting point; the autoresearcher will iterate from here.
"""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    """Build a linear Bayesian model for IHDP.

    Baseline: outcome ~ Normal(alpha + beta_t * treatment + X @ beta_x, sigma)
    """
    coords = train_data["coords"]

    with pm.Model(coords=coords) as model:
        # Data containers for out-of-sample prediction
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")
        outcome = train_data["outcome"]

        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=5)

        # Linear model
        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)

        # Likelihood
        pm.Normal("y", mu=mu, sigma=sigma, observed=outcome, dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Generate posterior predictive samples for new data.

    Returns array of shape (n_samples, n_obs).
    """
    with model:
        pm.set_data({
            "X": new_data["X"],
            "treatment": new_data["treatment"],
        })
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"])

    # Shape: (chain, draw, obs) -> (chain*draw, obs)
    samples = ppc.posterior_predictive["y"].values
    n_chains, n_draws, n_obs = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Estimate ATE via posterior predictive intervention.

    Sets treatment=1 for all, predicts. Sets treatment=0 for all, predicts.
    ATE = mean(Y(1) - Y(0)) across posterior samples.
    """
    n_obs = len(train_data["outcome"])

    # Predict under treatment = 1
    with model:
        pm.set_data({
            "X": train_data["X"],
            "treatment": np.ones(n_obs),
        })
        ppc_t1 = pm.sample_posterior_predictive(idata, var_names=["y"])

    # Predict under treatment = 0
    with model:
        pm.set_data({
            "X": train_data["X"],
            "treatment": np.zeros(n_obs),
        })
        ppc_t0 = pm.sample_posterior_predictive(idata, var_names=["y"])

    y1 = ppc_t1.posterior_predictive["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.posterior_predictive["y"].values.reshape(-1, n_obs)

    ate_samples = (y1 - y0).mean(axis=1)

    return {
        "ate": float(ate_samples.mean()),
        "ate_samples": ate_samples,
    }
```

- [ ] **Step 4: Run end-to-end test on IHDP**

Run: `uv run python problems/ihdp/prepare.py`
Expected: Model samples, convergence checks pass, prints results block with `---` header, `val_crps`, `convergence_ok: True`, `ate_estimate` near 4.0.

- [ ] **Step 5: Commit**

```bash
git add problems/ihdp/
git commit -m "feat: add IHDP problem with baseline linear Bayesian model"
```

---

## Chunk 3: Remaining Problems

### Task 6: Set up Twins problem

**Files:**
- Create: `problems/twins/problem.md`
- Create: `problems/twins/prepare.py`
- Create: `problems/twins/model.py`

- [ ] **Step 1: Create problem.md**

Create `problems/twins/problem.md`:

```markdown
---
primary_metric: crps
secondary_metrics: [mae, ate_bias]
split_strategy: random
split_ratios: [0.6, 0.2, 0.2]
temporal_column: null
time_budget: 300
---

# Twins — Twin Birth Weight Study

## Problem Statement

Estimate the causal effect of being the heavier twin (treatment) on birth weight
outcome using twin pair data. Natural counterfactual: each twin pair provides
both potential outcomes.

## Variables

- **treatment**: `treatment` (binary: randomly assigned heavier/lighter)
- **outcome**: `y_factual` (continuous: birth weight in grams)
- **confounders**: All columns except treatment, y_factual, y_cfactual

## Causal Estimand

- **ATE**: E[Y(1)] - E[Y(0)], ground truth from counterfactual column.
```

- [ ] **Step 2: Create prepare.py**

Create `problems/twins/prepare.py`:

```python
"""Twins problem runner."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from runner import run_experiment

DATA_DIR = Path(__file__).parent / "data"


def load_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_DIR / "twins.csv")

    confounder_cols = [
        c for c in df.columns
        if c not in ["treatment", "y_factual", "y_cfactual"]
    ]

    # Standardize continuous confounders
    for col in confounder_cols:
        if df[col].nunique() > 5:
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - df[col].mean()) / std

    # Ground truth ATE from counterfactuals
    true_ate = float(
        (df["y_factual"][df["treatment"] == 1].mean()
         - df["y_cfactual"][df["treatment"] == 1].mean()
         + df["y_cfactual"][df["treatment"] == 0].mean()
         - df["y_factual"][df["treatment"] == 0].mean()) / 2
    )

    metadata = {
        "treatment_col": "treatment",
        "outcome_col": "y_factual",
        "confounder_cols": confounder_cols,
        "true_ate": true_ate,
    }
    return df, metadata


if __name__ == "__main__":
    run_experiment(
        problem_dir=Path(__file__).parent,
        load_data_fn=load_data,
    )
```

- [ ] **Step 3: Create baseline model.py**

Create `problems/twins/model.py`:

```python
"""Twins Bayesian causal model — baseline."""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    coords = train_data["coords"]
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=100)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=50)
        beta_x = pm.Normal("beta_x", mu=0, sigma=10, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=100)

        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)
        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    with model:
        pm.set_data({"X": new_data["X"], "treatment": new_data["treatment"]})
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"])
    samples = ppc.posterior_predictive["y"].values
    n_chains, n_draws, n_obs = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    n_obs = len(train_data["outcome"])
    with model:
        pm.set_data({"X": train_data["X"], "treatment": np.ones(n_obs)})
        ppc_t1 = pm.sample_posterior_predictive(idata, var_names=["y"])
    with model:
        pm.set_data({"X": train_data["X"], "treatment": np.zeros(n_obs)})
        ppc_t0 = pm.sample_posterior_predictive(idata, var_names=["y"])

    y1 = ppc_t1.posterior_predictive["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.posterior_predictive["y"].values.reshape(-1, n_obs)
    ate_samples = (y1 - y0).mean(axis=1)
    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
```

- [ ] **Step 4: Run end-to-end test on Twins**

Run: `uv run python problems/twins/prepare.py`
Expected: Results block printed with `convergence_ok: True`.

- [ ] **Step 5: Commit**

```bash
git add problems/twins/
git commit -m "feat: add Twins problem with baseline model"
```

---

### Task 7: Set up LaLonde problem

**Files:**
- Create: `problems/lalonde/problem.md`
- Create: `problems/lalonde/prepare.py`
- Create: `problems/lalonde/model.py`

- [ ] **Step 1: Create problem.md**

Create `problems/lalonde/problem.md`:

```markdown
---
primary_metric: crps
secondary_metrics: [mae, ate_bias]
split_strategy: random
split_ratios: [0.6, 0.2, 0.2]
temporal_column: null
time_budget: 300
---

# LaLonde — Job Training Program

## Problem Statement

Estimate the causal effect of a job training program (NSW) on post-intervention
earnings (re78). Classic Dehejia-Wahba sample with experimental benchmark.

## Variables

- **treatment**: `treatment` (binary: 1=NSW program participant)
- **outcome**: `re78` (continuous: real earnings in 1978, USD)
- **confounders**: `age`, `education`, `black`, `hispanic`, `married`, `nodegree`, `re74`, `re75`

## Causal Estimand

- **ATE**: Difference in mean re78 between treated and control.
  Experimental benchmark ATE ≈ $1794.
```

- [ ] **Step 2: Create prepare.py**

Create `problems/lalonde/prepare.py`:

```python
"""LaLonde problem runner."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from runner import run_experiment

DATA_DIR = Path(__file__).parent / "data"

CONFOUNDER_COLS = [
    "age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"
]
CONTINUOUS_COLS = ["age", "education", "re74", "re75"]


def load_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_DIR / "lalonde.csv")

    for col in CONTINUOUS_COLS:
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - df[col].mean()) / std

    # Experimental ATE
    true_ate = float(
        df.loc[df["treatment"] == 1, "re78"].mean()
        - df.loc[df["treatment"] == 0, "re78"].mean()
    )

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
```

- [ ] **Step 3: Create baseline model.py**

Create `problems/lalonde/model.py`:

```python
"""LaLonde Bayesian causal model — baseline."""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    coords = train_data["coords"]
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=5000)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=2000)
        beta_x = pm.Normal("beta_x", mu=0, sigma=1000, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=5000)

        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)
        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    with model:
        pm.set_data({"X": new_data["X"], "treatment": new_data["treatment"]})
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"])
    samples = ppc.posterior_predictive["y"].values
    n_chains, n_draws, n_obs = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    n_obs = len(train_data["outcome"])
    with model:
        pm.set_data({"X": train_data["X"], "treatment": np.ones(n_obs)})
        ppc_t1 = pm.sample_posterior_predictive(idata, var_names=["y"])
    with model:
        pm.set_data({"X": train_data["X"], "treatment": np.zeros(n_obs)})
        ppc_t0 = pm.sample_posterior_predictive(idata, var_names=["y"])

    y1 = ppc_t1.posterior_predictive["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.posterior_predictive["y"].values.reshape(-1, n_obs)
    ate_samples = (y1 - y0).mean(axis=1)
    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
```

- [ ] **Step 4: Run end-to-end test on LaLonde**

Run: `uv run python problems/lalonde/prepare.py`
Expected: Results block printed with `convergence_ok: True`.

- [ ] **Step 5: Commit**

```bash
git add problems/lalonde/
git commit -m "feat: add LaLonde problem with baseline model"
```

---

### Task 8: Set up NHEFS problem

**Files:**
- Create: `problems/nhefs/problem.md`
- Create: `problems/nhefs/prepare.py`
- Create: `problems/nhefs/model.py`

- [ ] **Step 1: Create problem.md**

Create `problems/nhefs/problem.md`:

```markdown
---
primary_metric: crps
secondary_metrics: [mae]
split_strategy: temporal
split_ratios: [0.6, 0.2, 0.2]
temporal_column: age
time_budget: 300
---

# NHEFS — Smoking Cessation and Weight Gain

## Problem Statement

Estimate the causal effect of smoking cessation (treatment) on weight gain
(outcome) using the NHEFS observational cohort. This dataset has no ground
truth ATE — evaluation relies on predictive performance (CRPS) and
convergence quality.

## Variables

- **treatment**: `qsmk` (binary: 1=quit smoking between visits)
- **outcome**: `wt82_71` (continuous: weight change in kg, 1971-1982)
- **confounders**: `sex`, `race`, `age`, `school`, `smokeintensity`, `smokeyrs`, `exercise`, `active`, `wt71`

## Splitting

Temporal split by `age` (proxy for cohort effects). Younger subjects
in train, older in test. This tests the model's ability to generalize
across age groups.

## Modeling Guidance

- age is used for splitting, but also a confounder — include it in the model
- smokeintensity and smokeyrs are key confounders (smoking history)
- Consider interaction between treatment and smoking history
- Standardize continuous variables
```

- [ ] **Step 2: Create prepare.py**

Create `problems/nhefs/prepare.py`:

```python
"""NHEFS problem runner."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from runner import run_experiment

DATA_DIR = Path(__file__).parent / "data"

CONFOUNDER_COLS = [
    "sex", "race", "age", "school", "smokeintensity",
    "smokeyrs", "exercise", "active", "wt71"
]
CONTINUOUS_COLS = ["age", "school", "smokeintensity", "smokeyrs", "wt71"]


def load_data() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA_DIR / "nhefs.csv")

    # Keep only complete cases for key variables
    key_cols = ["qsmk", "wt82_71"] + CONFOUNDER_COLS
    df = df.dropna(subset=key_cols).reset_index(drop=True)

    for col in CONTINUOUS_COLS:
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - df[col].mean()) / std

    metadata = {
        "treatment_col": "qsmk",
        "outcome_col": "wt82_71",
        "confounder_cols": CONFOUNDER_COLS,
        "true_ate": None,  # No ground truth for observational data
    }
    return df, metadata


if __name__ == "__main__":
    run_experiment(
        problem_dir=Path(__file__).parent,
        load_data_fn=load_data,
    )
```

- [ ] **Step 3: Create baseline model.py**

Create `problems/nhefs/model.py`:

```python
"""NHEFS Bayesian causal model — baseline."""
import numpy as np
import pymc as pm


def build_model(train_data: dict) -> pm.Model:
    coords = train_data["coords"]
    with pm.Model(coords=coords) as model:
        X = pm.Data("X", train_data["X"], dims=("obs", "features"))
        treatment = pm.Data("treatment", train_data["treatment"], dims="obs")

        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta_t = pm.Normal("beta_treatment", mu=0, sigma=5)
        beta_x = pm.Normal("beta_x", mu=0, sigma=2, dims="features")
        sigma = pm.HalfNormal("sigma", sigma=10)

        mu = alpha + beta_t * treatment + pm.math.dot(X, beta_x)
        pm.Normal("y", mu=mu, sigma=sigma, observed=train_data["outcome"], dims="obs")

    return model


def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    with model:
        pm.set_data({"X": new_data["X"], "treatment": new_data["treatment"]})
        ppc = pm.sample_posterior_predictive(idata, var_names=["y"])
    samples = ppc.posterior_predictive["y"].values
    n_chains, n_draws, n_obs = samples.shape
    return samples.reshape(n_chains * n_draws, n_obs)


def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    n_obs = len(train_data["outcome"])
    with model:
        pm.set_data({"X": train_data["X"], "treatment": np.ones(n_obs)})
        ppc_t1 = pm.sample_posterior_predictive(idata, var_names=["y"])
    with model:
        pm.set_data({"X": train_data["X"], "treatment": np.zeros(n_obs)})
        ppc_t0 = pm.sample_posterior_predictive(idata, var_names=["y"])

    y1 = ppc_t1.posterior_predictive["y"].values.reshape(-1, n_obs)
    y0 = ppc_t0.posterior_predictive["y"].values.reshape(-1, n_obs)
    ate_samples = (y1 - y0).mean(axis=1)
    return {"ate": float(ate_samples.mean()), "ate_samples": ate_samples}
```

- [ ] **Step 4: Run end-to-end test on NHEFS**

Run: `uv run python problems/nhefs/prepare.py`
Expected: Results block printed with `convergence_ok: True`.

- [ ] **Step 5: Commit**

```bash
git add problems/nhefs/
git commit -m "feat: add NHEFS problem with baseline model and temporal split"
```

---

## Chunk 4: Orchestrator

### Task 9: Write program.md

**Files:**
- Create: `program.md`

- [ ] **Step 1: Create program.md**

Create `program.md`:

````markdown
# Bayesian Autoresearcher

This is an experiment to have the LLM autonomously build and iterate on Bayesian causal models.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a problem**: Which problem directory to work on (ihdp, twins, lalonde, nhefs).
2. **Agree on a run tag**: Propose a tag based on today's date (e.g. `mar11`). The branch `autoresearch/<problem>/<tag>` must not already exist.
3. **Create the branch**: `git checkout -b autoresearch/<problem>/<tag>` from current main.
4. **Read the in-scope files**:
   - This file (`program.md`) — your operating instructions
   - `problems/<name>/problem.md` — the problem statement, variables, scoring config
   - `problems/<name>/prepare.py` — fixed runner, DO NOT MODIFY
   - `problems/<name>/model.py` — the file you modify (via sub-agent)
   - `scoring.py` — shared scoring functions, DO NOT MODIFY
   - `runner.py` — shared experiment runner, DO NOT MODIFY
5. **Verify data exists**: Check that `problems/<name>/data/` has CSV files. If not, run `uv run python download_datasets.py`.
6. **Initialize results.tsv**: Create `problems/<name>/results.tsv` with header row.
7. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Experimentation

Each experiment runs a Bayesian model. The prepare.py runner handles sampling, convergence checking, and scoring. You launch it as:

```
uv run python problems/<name>/prepare.py > run.log 2>&1
```

**What you CAN do:**
- Modify `problems/<name>/model.py` — this is the only file you edit (via sub-agent).
- Everything is fair game: model structure, priors, likelihood, parameterization, interactions, hierarchical structure, GP components, BART, mixture models.

**What you CANNOT do:**
- Modify `prepare.py`, `runner.py`, or `scoring.py`. They are read-only.
- Install new packages or add dependencies.
- Modify the scoring or evaluation harness.

**The goal: minimize the primary_metric on the validation set** (as defined in problem.md). For most problems this is CRPS (lower is better). The model must also pass convergence checks (r_hat < 1.01, ESS > 400, no divergences).

**Sub-agent usage**: For each experiment, spawn a sub-agent using the Agent tool:

```
Agent(
    description="Bayesian model iteration N",
    prompt="""
    You are a Bayesian modeler. First, load the pymc-modeling skill using the Skill tool:
    Skill(skill="pymc-modeling")

    Then read these files:
    - problems/<name>/problem.md
    - problems/<name>/prepare.py
    - problems/<name>/model.py (current version)
    - scoring.py
    - problems/<name>/results.tsv (past experiments)

    Your task: <SPECIFIC EXPERIMENT IDEA>

    Edit ONLY problems/<name>/model.py. The model must:
    - Define build_model(train_data) returning a pm.Model
    - Define predict(idata, model, new_data) returning shape (n_samples, n_obs)
    - Define estimate_causal_effect(idata, model, train_data) returning dict
    - Use pm.Data containers for X and treatment (required for out-of-sample prediction)
    - Use nutpie sampler (via runner.py) — keep model complexity reasonable for 5-min budget
    """,
    subagent_type="general-purpose",
)
```

**The first run**: Always establish the baseline first. Run the existing model.py as-is.

## Output format

The prepare.py runner prints a summary block:

```
---
val_crps:             0.4523
val_mae:              1.2340
val_rmse:             1.5670
val_elpd:          -234.5000
val_ate_bias:         0.5600
convergence_ok:     True
r_hat_max:          1.0020
ess_min:            856
divergences:        0
n_params:           5
sampling_seconds:     142.3
total_seconds:        180.1
ate_estimate:         3.4500
ate_hdi_3:            2.1000
ate_hdi_97:           4.8200
```

Extract the primary metric:
```
grep "^val_crps:" run.log
```

## Logging results

When an experiment is done, log it to `problems/<name>/results.tsv` (tab-separated).

Header and columns:

```
commit	val_crps	val_mae	convergence	status	description
```

1. git commit hash (short, 7 chars)
2. primary metric value (e.g. val_crps) — use 999.0 for crashes
3. val_mae — use 999.0 for crashes
4. convergence: `pass` or `fail`
5. status: `keep`, `discard`, or `crash`
6. short description of what this experiment tried

Example:
```
commit	val_crps	val_mae	convergence	status	description
a1b2c3d	0.4523	1.234	pass	keep	baseline linear model
b2c3d4e	0.4210	1.198	pass	keep	add treatment-age interaction
c3d4e5f	999.0	999.0	fail	crash	hierarchical model (divergences)
```

## The experiment loop

LOOP FOREVER:

1. Look at git state: current branch/commit
2. Review results.tsv: what has been tried, what worked, what didn't
3. Formulate a hypothesis for the next experiment (use insights from past results)
4. Spawn a sub-agent to implement the idea in model.py
5. git commit the change with a descriptive message
6. Run the experiment: `uv run python problems/<name>/prepare.py > run.log 2>&1`
   - Use a 10-minute timeout. If exceeded, treat as crash.
7. Read results: `grep "^val_crps:\|^convergence_ok:\|^val_mae:\|^ate_estimate:" run.log`
8. If grep is empty, the run crashed. Run `tail -n 50 run.log` for the stack trace.
9. Record results in results.tsv (do NOT commit results.tsv — keep it untracked)
10. If primary metric improved → keep the commit ("advance" the branch)
11. If primary metric is worse or convergence failed → `git reset --hard HEAD~1`
12. Go to step 1

## Experiment strategy

Build complexity incrementally:

1. **Baseline**: Simple linear model (already provided)
2. **Better priors**: Informative priors based on domain knowledge
3. **Interactions**: Treatment-confounder interactions
4. **Non-linearity**: Splines, polynomial terms
5. **Hierarchical**: Group-level effects if subgroups exist
6. **Flexible models**: BART, Gaussian processes (HSGP)
7. **Robustness**: Student-t likelihood, heteroscedastic noise
8. **Causal**: Propensity score inclusion, doubly robust estimation

If stuck, try:
- Combining elements from previous near-misses
- Simplifying (removing complexity that didn't help)
- Re-reading problem.md for new angles
- Trying a fundamentally different model family

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. Run until manually interrupted.

**Crashes**: If a run crashes with something easy to fix (typo, import), have the sub-agent fix it and re-run. If fundamentally broken, log "crash", revert, and move on.
````

- [ ] **Step 2: Add results.tsv initialization to program.md setup**

The orchestrator creates `results.tsv` during setup with this header:
```
printf 'commit\tval_crps\tval_mae\tconvergence\tstatus\tdescription\n' > problems/<name>/results.tsv
```

- [ ] **Step 3: Commit**

```bash
git add program.md
git commit -m "feat: add orchestrator instructions (program.md)"
```

---

### Task 10: End-to-end validation

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass.

- [ ] **Step 2: Run IHDP baseline end-to-end**

Run: `uv run python problems/ihdp/prepare.py`
Expected: Complete results block printed. Convergence passes. CRPS and ATE reported.

- [ ] **Step 3: Verify results format is parseable**

Run: `uv run python problems/ihdp/prepare.py > run.log 2>&1 && grep "^val_crps:" run.log`
Expected: Single line like `val_crps:             0.4523`

- [ ] **Step 4: Verify git workflow**

```bash
# Simulate the keep/discard loop
git checkout -b autoresearch/ihdp/test
# ... run experiment, check results ...
git checkout main
git branch -d autoresearch/ihdp/test
```

- [ ] **Step 5: Final commit**

```bash
git add program.md CLAUDE.md
git commit -m "chore: complete bayesian autoresearcher setup"
```
