# Bayesian Autoresearcher

An autonomous research loop where an LLM iterates on Bayesian causal models,
scored by proper scoring rules. The system builds PyMC models to estimate causal
effects from observational data and improves them through guided experimentation
— without human intervention.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator (LLM)                    │
│  Reads results.tsv → formulates hypothesis → spawns     │
│  sub-agent to edit model.py → runs experiment → scores  │
│  → keep/discard → repeat                                │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
       ┌─────▼─────┐              ┌───────▼────────┐
       │  model.py  │              │   prepare.py   │
       │  (edited   │              │   (read-only   │
       │  each iter)│              │   data loader) │
       └─────┬──────┘              └───────┬────────┘
             │                             │
       ┌─────▼─────────────────────────────▼──────┐
       │              runner.py                    │
       │  parse config → split data → build model │
       │  → sample (nutpie) → score → compare     │
       └──────────────────┬───────────────────────┘
                          │
                    ┌─────▼──────┐
                    │ scoring.py │
                    │ ELPD, CRPS │
                    │ convergence│
                    └────────────┘
```

### Core loop

1. **Hypothesize** — review past results in `results.tsv` and decide what
   modeling change to try next (stronger priors, interactions, splines, BART, …).
2. **Implement** — a sub-agent edits `problems/<name>/model.py`, the only
   mutable file.
3. **Run** — `uv run python problems/<name>/prepare.py` executes the experiment
   within a 5-minute time budget.
4. **Score** — the runner computes ELPD (PSIS-LOO), CRPS, MAE, ATE bias, and
   MCMC convergence diagnostics.
5. **Keep or discard** — the new model is compared against the current best:
   - **Better** (dELPD > 2·dSE): keep.
   - **Equivalent** (|dELPD| < dSE): keep only if the ATE HDI is narrower.
   - **Worse**: discard (git reset).
6. **Log** — append a row to `results.tsv` and loop back to step 1.

The loop runs indefinitely until manually stopped.

## Adaptation to Bayesian causal modeling

### Why Bayesian?

Causal effect estimation from observational data demands more than point
predictions. A Bayesian framework provides:

- **Full posterior distributions** over treatment effects, enabling uncertainty
  quantification via HDI (Highest Density Interval) rather than fragile
  frequentist confidence intervals.
- **Prior encoding** — domain knowledge about plausible effect sizes, outcome
  scales, and confounder relationships can be injected directly into the model.
- **PSIS-LOO ELPD** as the primary model comparison metric — a proper scoring
  rule that penalizes overfitting without requiring a held-out validation set,
  critical when sample sizes are small (as in most causal benchmarks).
- **Posterior predictive interventions** — treatment effects are estimated by
  setting treatment to 1 for all units, then to 0, and differencing the
  posterior predictions. This naturally propagates uncertainty from all model
  parameters into the causal estimate.

### The model contract

Every `model.py` exposes three functions that together form the Bayesian causal
estimation pipeline:

| Function | Role |
|---|---|
| `build_model(train_data) → pm.Model` | Define the PyMC DAG: priors, likelihood, and `pm.Data` containers for treatment and covariates. |
| `predict(idata, model, new_data) → ndarray` | Generate posterior predictive samples for new observations by swapping `pm.Data` values. |
| `estimate_causal_effect(idata, model, train_data) → dict` | Intervene: predict under do(T=1) and do(T=0), return ATE and its posterior samples. |

The `pm.Data` container pattern is central — it allows the same compiled model
to be reused for prediction on test data and for counterfactual interventions
without rebuilding or resampling.

### Experiment strategy

Models are built incrementally, guided by causal modeling best practices:

1. **Baseline** — simple linear model with default priors.
2. **Informative priors** — tighten priors using domain knowledge about outcome
   scale and plausible effect sizes.
3. **Treatment–confounder interactions** — allow heterogeneous treatment effects.
4. **Non-linearity** — splines or polynomials for continuous confounders.
5. **Hierarchical structure** — partial pooling where subgroups exist.
6. **Flexible models** — BART or Gaussian processes for complex response
   surfaces.
7. **Robust likelihoods** — Student-t errors, heteroscedasticity.
8. **Causal enhancements** — propensity score adjustment, doubly robust
   estimators.

Each step is a hypothesis tested against the ELPD criterion. The loop
naturally discovers which modeling choices matter for a given dataset and
discards those that don't help.

### Convergence diagnostics

Before any model comparison, MCMC quality is verified:

| Diagnostic | Threshold |
|---|---|
| R-hat | < 1.01 |
| ESS (bulk and tail) | > 400 |
| Divergence rate | < 0.1% |

Non-converged runs are never kept as the best model.

## Datasets

| Problem | Causal question | N | Outcome | Ground truth ATE |
|---|---|---|---|---|
| **IHDP** | Do home visits improve infant cognitive scores? | 747 | Continuous | ~4.0 (semi-synthetic) |
| **LaLonde** | Does job training increase earnings? | 722 | Continuous ($) | ~$886 (experimental) |
| **NHEFS** | Does quitting smoking increase weight? | 1,566 | Continuous (kg) | None (observational) |
| **Twins** | Does heavier birth weight reduce mortality? | ~5,000 | Binary | Available (semi-synthetic) |

These span the canonical challenges in causal inference: small samples (IHDP),
selection bias (LaLonde), purely observational data (NHEFS), and binary outcomes
with rare events (Twins).

## Project structure

```
bayesian-autoresearcher/
├── program.md              # Orchestrator loop instructions
├── CLAUDE.md               # Sub-agent rules
├── runner.py               # Experiment runner (read-only)
├── scoring.py              # Scoring functions (read-only)
├── download_datasets.py    # Dataset downloader
├── pyproject.toml          # Dependencies
├── tests/
│   ├── test_runner.py
│   └── test_scoring.py
└── problems/
    └── <name>/
        ├── problem.md      # Problem statement + YAML config
        ├── prepare.py      # Data loader (read-only)
        ├── model.py        # The ONLY file edited per iteration
        ├── data/           # Downloaded CSV
        └── runs/           # latest.nc, best.nc (InferenceData)
```

## Getting started

```bash
# Install dependencies
uv sync

# Download all datasets
uv run python download_datasets.py

# Run a single experiment
uv run python problems/ihdp/prepare.py

# Run tests
uv run pytest
```

## Dependencies

- **PyMC ≥ 5.10** — probabilistic programming
- **nutpie ≥ 0.13** — fast NUTS sampler (Rust backend)
- **ArviZ ≥ 0.18** — diagnostics, LOO-CV, model comparison
- **NumPy, Pandas, SciPy, scikit-learn** — data handling
- **properscoring** — CRPS computation

Python ≥ 3.11 required. Managed with [uv](https://github.com/astral-sh/uv).
