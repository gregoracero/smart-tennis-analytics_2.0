# Top 5 vs Full Model

## Models

### Full Model

Features

83

AUC

0.8739

---

### Top 5 Model

Features

5

AUC

0.8678

---

## Performance Retained

0.8678 / 0.8739

= 99.3%

---

## Top 5 Variables

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

---

## Main Discovery

A five-feature model reproduces approximately 99% of the predictive performance of the complete model.

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

The evidence suggests that the majority of predictive value is concentrated in:

Surface Readiness
+
Overall Readiness
+
Player Strength

while the remaining variables provide mostly incremental refinement.

---

## Business Implication

A dramatically simpler model may be capable of delivering nearly identical predictive performance while being:

- Easier to maintain
- Easier to explain
- Faster to train
- More robust to feature drift

---

## Next Validation Step

Compare:

- ROI
- Yield
- Hit Rate
- Drawdown

between:

- Full Model
- Top 5 Model

using identical betting rules.
