# Top 5 Model Results

## Model Definition

Features

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

---

## Performance

AUC

0.8678

Accuracy

0.7747

---

## Comparison Against Full Model

Full Model

Features

83

AUC

0.8739

Top 5 Model

Features

5

AUC

0.8678

Difference

0.0061

---

## Predictive Power Retained

0.8678 / 0.8739

= 99.30%

---

## Main Interpretation

A model containing only five features retains approximately 99.3% of the predictive performance of the full model.

Most predictive value is concentrated in:

- Surface Readiness
- Overall Readiness
- Elo Strength

The remaining 78 variables provide incremental refinement but contribute relatively little to total predictive performance.

---

## Key Insight

The strongest predictive variable discovered during the investigation is:

delta_surface_inactivity_days

which exceeds the importance of:

- delta_elo
- delta_rank
- delta_rank_points
- rolling performance metrics

---

## Working Hypothesis

Recent competitive readiness is systematically undervalued relative to historical player quality.

The model appears to derive most of its edge from identifying players with superior recent activity levels, especially on the relevant playing surface.
