---
primary_metric: elpd
secondary_metrics: [crps, mae, ate_bias]
split_strategy: random
split_ratios: [0.8, 0.2]
temporal_column: null
time_budget: 300
---

# LaLonde — Job Training Program Earnings

## Problem Statement

Estimate the causal effect of participation in a job training program (treatment) on
1978 earnings (outcome) using the LaLonde (1986) dataset. This is a classic benchmark
for causal inference methods comparing experimental and observational estimates.

## Dataset

722 observations combining experimental treatment group (297 participants) and
comparison group (425 controls). Covariates include demographic and pre-treatment
earnings variables.

## Variables

- **treatment**: `treatment` (binary: 0/1, job training participation)
- **outcome**: `re78` (continuous: 1978 earnings in USD)
- **confounders**: age, education, black, hispanic, married, nodegree, re75
  - continuous: age, education, re75
  - binary: black, hispanic, married, nodegree

## Causal Estimand

- **ATE**: Average Treatment Effect = E[re78(1)] - E[re78(0)]
- **Ground truth approximation**: Difference in mean re78 between treated and control
  (valid because the treated group is from a randomized experiment).
  True ATE ≈ $886 (LaLonde 1986 experimental estimate).

## Modeling Guidance

- Outcome is earnings in USD (range 0 to ~60,000, mean ~5,000)
- Adjust prior scales for dollar-valued outcome (alpha ~ Normal(0, 5000))
- Heavy zero-inflation present: many individuals have zero earnings
- Standardize continuous confounders (age, education, re75)
