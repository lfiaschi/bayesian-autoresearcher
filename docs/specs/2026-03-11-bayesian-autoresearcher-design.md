# Bayesian Autoresearcher — Design Spec

## Overview

An autonomous research loop where Claude builds and iterates on PyMC causal models. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Given a dataset and problem statement, the system autonomously builds Bayesian models to estimate causal effects, scoring them with proper scoring rules and convergence diagnostics.

## Architecture

**Orchestrator agent** (main Claude Code session): Runs the loop forever. Reads results, decides strategy, spawns sub-agents. Never writes model code itself.

**Coding sub-agent** (spawned per experiment via Agent tool): Loads `pymc-modeling` skill first. Reads problem context. Edits only `model.py`. Gets fresh context each time.

```
Human starts Claude Code → Claude reads program.md → LOOP:
  1. Analyze results.tsv, decide next experiment
  2. Spawn sub-agent (loads pymc-modeling skill, edits model.py)
  3. Git commit
  4. Run: uv run problems/<name>/prepare.py > run.log 2>&1
  5. Parse results from run.log
  6. Keep (improved) or discard (git reset)
  7. Log to results.tsv
  8. Never stop
```

## File Structure

```
bayesian-autoresearcher/
├── pyproject.toml
├── program.md                  # Orchestrator instructions
├── scoring.py                  # Shared scoring functions (CRPS, ELPD, convergence)
├── download_datasets.py        # One-time dataset fetcher
├── problems/
│   ├── ihdp/
│   │   ├── problem.md          # Problem statement, variables, estimand, scoring config
│   │   ├── data/               # Raw CSV(s)
│   │   ├── prepare.py          # Fixed: load data, split, run model, score, print results
│   │   ├── model.py            # Agent-editable: build_model(), predict(), estimate_causal_effect()
│   │   ├── results.tsv         # Experiment log
│   │   └── runs/               # Saved InferenceData (.nc files)
│   ├── twins/
│   ├── lalonde/
│   └── nhefs/
```

## model.py Contract

Sub-agent edits only `model.py`. Must expose:

```python
def build_model(train_data: dict) -> pm.Model:
    """Build and return a PyMC model. Do NOT sample here.
    train_data keys: "X" (confounders), "treatment", "outcome", "coords"
    """

def predict(idata, model: pm.Model, new_data: dict) -> np.ndarray:
    """Posterior predictive samples for new_data. Shape: (n_samples, n_obs)."""

def estimate_causal_effect(idata, model: pm.Model, train_data: dict) -> dict:
    """Return {"ate": float, "ate_samples": np.ndarray, ...}. Optional."""
```

## prepare.py (per-problem runner)

Fixed entry point. Handles:
1. Load data and split (train/val/test)
2. Import model.py functions
3. Build model, sample with 5-min timeout
4. Convergence gate (r_hat, ESS, divergences)
5. Score on validation set
6. Causal effect estimation
7. Print standardized results block

## Scoring

Configurable per problem via problem.md:

```markdown
## Scoring
primary_metric: crps
secondary_metrics: [elpd, mae, ate_bias]
split_strategy: random       # or: temporal
split_ratios: [0.6, 0.2, 0.2]
temporal_column: null         # or: "date", "year"
```

Available metrics in scoring.py:
- **CRPS**: Posterior predictive calibration (properscoring library)
- **ELPD (LOO-CV)**: Bayesian model comparison via ArviZ
- **ATE bias**: |estimated_ATE - true_ATE| for datasets with known ground truth
- **MAE/RMSE**: Point prediction reference

**Convergence is a gate**: r_hat > 1.01, ESS < 400, or divergences > 0.1% → run scored as "crash".

**Output format from prepare.py:**
```
---
val_crps:           0.4523
val_elpd:          -234.5
val_mae:            1.234
convergence_ok:     True
r_hat_max:          1.002
ess_min:            856
divergences:        0
n_params:           5
sampling_seconds:   142.3
total_seconds:      180.1
ate_estimate:       3.45
ate_hdi_3:          2.10
ate_hdi_97:         4.82
```

## Splitting Strategies

- **Random**: Stratified by treatment variable. Default for cross-sectional data.
- **Temporal**: Sort by temporal_column, split chronologically. Train=earliest 60%, val=next 20%, test=final 20%.

## Datasets

| Dataset | Estimand | Ground Truth? | Split | Description |
|---------|----------|---------------|-------|-------------|
| IHDP | ATE + CATE | Yes (semi-synthetic) | Random | Infant health, binary treatment, continuous outcome |
| Twins | ATE | Yes (counterfactual) | Random | Twin birth weight, binary treatment, continuous outcome |
| LaLonde | ATE | Yes (experimental) | Random | Job training program, earnings outcome |
| NHEFS | ATE | No (observational) | Temporal | Smoking cessation → weight gain |

Sources: IHDP/Twins from CEVAE GitHub repo, LaLonde from NBER, NHEFS from Harvard.

## Dependencies

```toml
[project]
name = "bayesian-autoresearcher"
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
]
```

## Future Extension: Modal GPU

Architecture supports offloading experiment execution to Modal with a thin wrapper. Not needed for current tabular datasets (CPU nutpie is fast enough), but useful for scaling to large datasets or GP/BART models.
