# Phase 2 Conclusions

## Objective

Identify the minimum set of variables required to reproduce the performance of the full model.

---

## Model Comparison

Full Model

83 Features

AUC

0.8739

---

Top 10 Features

10 Features

AUC

0.8704

Performance Retained

99.60%

---

Top 5 Features

5 Features

AUC

0.8678

Performance Retained

99.30%

---

## Top 5 Variables

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

---

## Main Finding

Only five variables reproduce virtually all predictive power of the complete 83-feature model.

---

## Dominant Signal Families

Readiness

- delta_surface_inactivity_days
- delta_inactivity_days
- days_inactive_a
- days_inactive_b

Strength

- delta_elo

---

## Interpretation

The majority of model performance appears to be driven by:

Competitive Readiness
+
Player Strength

Most additional variables provide refinement rather than core predictive information.

---

## Next Phase

Evaluate whether the Top 5 model retains:

- ROI
- Yield
- Hit Rate
- Calibration

relative to the full production model.
