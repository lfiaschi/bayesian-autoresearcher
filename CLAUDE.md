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

## Visualization & Publication Quality

All experiment output — figures, tables, printed summaries — must be publication
quality. Use `plotting.py` (shared library) for consistent style across problems.

- `plotting.py` — Shared visualization: pub-style config, color palette,
  results.tsv parsing, ELPD trajectory + waterfall plots
- `problems/<name>/visualize.py` — Problem-specific plots (causal estimates,
  model diagrams) that import from `plotting.py`

### Standards
- Call `set_pub_style()` from `plotting.py` before any matplotlib figure
- Use the shared `COLORS` dict for consistent palette across all problems
- Use `parse_results()` / `split_by_status()` from `plotting.py` to read results.tsv
- Use `plot_iterations()` from `plotting.py` for the ELPD trajectory + waterfall
  (pass problem-specific `kept_labels` dict as parameter)
- 300 DPI, white background, tight bounding box for all saved figures
- Clean spines (remove top/right), subtle grid, consistent font sizes
- Problem-specific plots (causal forest plots, model diagrams) stay in
  `problems/<name>/visualize.py` but follow the same style conventions

## Running an experiment
```bash
uv run python problems/<name>/prepare.py
```
