---
primary_metric: crps
secondary_metrics: [elpd, mae]
split_strategy: temporal
split_ratios: [0.6, 0.2, 0.2]
temporal_column: age
time_budget: 300
---

# NHEFS — Smoking Cessation and Weight Change

## Problem Statement

Estimate the causal effect of smoking cessation (treatment: qsmk) on weight change
from 1971 to 1982 (outcome: wt82_71) using the NHANES Epidemiologic Follow-up Study.
This is a widely-used observational study benchmark for causal inference.

## Dataset

1,566 observations from the NHANES I Epidemiologic Follow-up Study (NHEFS).
Contains baseline (1971) measurements and follow-up (1982) outcomes.
No ground truth ATE is available — this is purely observational data.

## Variables

- **treatment**: `qsmk` (binary: 0/1, quit smoking by 1982)
- **outcome**: `wt82_71` (continuous: weight change in kg from 1971 to 1982)
- **confounders**: sex, race, age, school, smokeintensity, smokeyrs, exercise, active, wt71
  - continuous: age, school, smokeintensity, smokeyrs, wt71
  - categorical/binary: sex, race, exercise, active

## Causal Estimand

- **ATE**: Average Treatment Effect = E[wt82_71(qsmk=1)] - E[wt82_71(qsmk=0)]
- **No ground truth**: Observational study, true ATE unknown.
  Typical estimates in the literature: ~3-5 kg weight gain from quitting.

## Modeling Guidance

- Weight change ranges roughly from -40 to +50 kg (mean ~2.6 kg)
- Adjust prior scales for kg-scale outcome
- Temporal split by age (as a proxy for cohort ordering)
- Smoking intensity and duration are important confounders
