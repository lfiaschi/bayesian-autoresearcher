---
primary_metric: elpd
secondary_metrics: [crps, mae, ate_bias]
split_strategy: random
split_ratios: [0.8, 0.2]
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
- **Ground truth available**: Computed from `mu1 - mu0` columns. True ATE ≈ 4.0
- **CATE**: Conditional effects across subgroups (optional stretch goal)

## Modeling Guidance

- Start with a simple linear model with treatment + confounders
- Consider hierarchical structure on confounder effects
- Treatment-confounder interactions may improve CATE estimation
- Non-centered parameterization if divergences appear
- Standardize continuous confounders (x1-x6)
