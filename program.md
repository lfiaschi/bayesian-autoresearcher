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
6. **Initialize results.tsv**: Create with header row:
   ```
   printf 'commit\tval_crps\tval_mae\tconvergence\tstatus\tdescription\n' > problems/<name>/results.tsv
   ```
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
    - IMPORTANT: In predict() and estimate_causal_effect(), use:
      pm.set_data({...}, coords={"obs": np.arange(n_obs)})
      pm.sample_posterior_predictive(idata, var_names=["y"], predictions=True)
      Then access results via ppc.predictions["y"] (NOT ppc.posterior_predictive["y"])
    """,
    subagent_type="general-purpose",
)
```

**The first run**: Always establish the baseline first. Run the existing model.py as-is.

## Output format

The prepare.py runner prints a summary block:

```
---
val_crps:           0.4523
val_mae:            1.2340
val_rmse:           1.5670
val_elpd:           -234.5000
val_ate_bias:       0.5600
convergence_ok:     True
r_hat_max:          1.0020
ess_min:            856
divergences:        0
n_params:           5
sampling_seconds:   142.3
total_seconds:      180.1
ate_estimate:       3.4500
ate_hdi_3:          2.1000
ate_hdi_97:         4.8200
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
