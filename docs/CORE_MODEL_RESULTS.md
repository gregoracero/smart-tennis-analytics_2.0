# Core Model Results

## AUC Results

Full Model

0.8739

Top 10 Features

0.8704

Difference vs Full Model

0.0035

Top 5 Features

0.8678

Difference vs Full Model

0.0061

---

## Top 5 Variables

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

---

## Top 10 Variables

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b
6. delta_surface_elo
7. surface_days_inactive_a
8. delta_rank_points
9. surface_days_inactive_b
10. delta_rank

---

## Importance Ranking

delta_surface_inactivity_days    0.2408
delta_elo                        0.0876
delta_inactivity_days            0.0770
days_inactive_a                  0.0687
days_inactive_b                  0.0687
delta_surface_elo                0.0427
surface_days_inactive_a          0.0376
delta_rank_points                0.0267
surface_days_inactive_b          0.0266
delta_rank                       0.0100

---

## Key Findings

Top 5 Features

AUC

0.8678

Percentage of Full Model AUC

0.8678 / 0.8739

= 99.30%

---

Top 10 Features

AUC

0.8704

Percentage of Full Model AUC

0.8704 / 0.8739

= 99.60%

---

## Interpretation

The vast majority of predictive power is concentrated in a very small number of variables.

The dominant signal family is:

Readiness

represented primarily by:

- delta_surface_inactivity_days
- delta_inactivity_days
- days_inactive_a
- days_inactive_b

The second most important signal family is:

Strength

represented primarily by:

- delta_elo
- delta_surface_elo

Ranking contributes some additional information but appears significantly less important than readiness and Elo.

---

## Practical Implication

A model using only the Top 5 variables retains approximately 99.3% of the predictive performance of the full 83-feature model.

This suggests that model complexity can potentially be reduced dramatically while maintaining almost all predictive power.

---

## Working Hypothesis

The strongest prediction engine appears to be:

Surface Readiness
+
Overall Readiness
+
Elo Strength

Most remaining variables provide incremental refinement rather than core predictive value.
