---
primary_metric: crps
secondary_metrics: [elpd, mae, ate_bias]
split_strategy: random
split_ratios: [0.6, 0.2, 0.2]
temporal_column: null
time_budget: 300
---

# Twins — Twin Birth Causal Inference

## Problem Statement

Estimate the causal effect of being the heavier twin at birth (treatment) on one-year
mortality (outcome) using data from US twin births. The goal is to assess how birth
weight difference between twins affects survival probability.

## Dataset

Semi-synthetic benchmark based on US twin birth records (Louizos et al., 2017).
~32,000 observations with 52 covariates recording maternal and birth characteristics.
Both factual and counterfactual outcomes are available for ground truth ATE computation.
This prepare.py subsamples to ~5000 rows and uses 10 key confounders for tractable inference.

## Variables

- **treatment**: `treatment` (binary: 0/1, heavier twin indicator)
- **outcome**: `y_factual` (binary: mortality at 1 year)
- **confounders**: subset of birth and maternal characteristics
  - pldel, birattnd, mager8, ormoth, mrace, meduc6, dmar, adequacy, gestat10, csex

## Causal Estimand

- **ATE**: Average Treatment Effect = E[Y(1)] - E[Y(0)]
- **Ground truth available**: Computed from `y_factual` and `y_cfactual` columns.

## Modeling Guidance

- Start with a linear model (logistic would be more appropriate, but linear is the baseline)
- Binary outcome with ~1-4% mortality rate
- Treatment effect is expected to be negative (heavier twin has lower mortality)
- Adjust prior scales for binary outcome range [0, 1]
