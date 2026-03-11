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
- Use type hints on all function signatures (parameters and return types)

## Running an experiment
```bash
uv run python problems/<name>/prepare.py
```
