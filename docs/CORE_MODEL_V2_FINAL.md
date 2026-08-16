# CORE MODEL V2 FINAL

## Objective

Determine whether the predictive power and economic value of the full 83-feature model can be reproduced using a dramatically simplified feature set.

---

# Executive Summary

The investigation demonstrates that a five-feature model captures nearly all predictive performance across ATP and Challenger competitions.

Core Model V1 consistently retains between 98% and 99% of Full Model AUC across all evaluated tennis segments.

The evidence suggests that the majority of predictive value comes from:

Surface Readiness
+
Overall Readiness
+
Player Strength

rather than from the complete collection of 83 features.

---

# Core Model V1

Variables

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

Total Features

5

---

# ATP Validation

## ATP Hard

Full Model

Features

83

AUC

0.8739

Core Model V1

Features

5

AUC

0.8678

Retention

99.30%

---

## ATP Clay

Full Model

AUC

0.8546

Core Model V1

AUC

0.8390

Retention

98.17%

---

## ATP Grass

Full Model

AUC

0.7999

Core Model V1

AUC

0.7902

Retention

98.79%

---

# Challenger Validation

## Challenger Hard

Full Model

AUC

0.8557

Core Model V1

AUC

0.8490

Retention

99.22%

---

## Challenger Clay

Full Model

AUC

0.8597

Core Model V1

AUC

0.8523

Retention

99.14%

---

## Challenger Grass

Full Model

AUC

0.8105

Core Model V1

AUC

0.8401

Retention

103.65%

Note:

Grass Challenger sample size is very small:

Test Rows

332

The apparent improvement should therefore be interpreted cautiously.

---

# Validation Summary

| Segment | Full Model | Core Model | Retained |
|----------|----------:|----------:|----------:|
| ATP Hard | 0.8739 | 0.8678 | 99.30% |
| ATP Clay | 0.8546 | 0.8390 | 98.17% |
| ATP Grass | 0.7999 | 0.7902 | 98.79% |
| CH Hard | 0.8557 | 0.8490 | 99.22% |
| CH Clay | 0.8597 | 0.8523 | 99.14% |
| CH Grass | 0.8105 | 0.8401 | 103.65% |

Average Retention

Approximately 99%

---

# Feature Importance Analysis

Top Features from the Full Model

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

These five variables became the foundation of Core Model V1.

---

# ATP Hard Economic Validation

Core Model V1

EDGE > 0.05

BETS

4882

HIT RATE

68.95%

ROI

37.67%

---

EDGE > 0.10

BETS

3603

HIT RATE

77.19%

ROI

55.35%

---

EDGE > 0.15

BETS

2839

HIT RATE

84.61%

ROI

70.43%

---

EDGE > 0.20

BETS

2398

HIT RATE

89.49%

ROI

84.67%

---

EDGE > 0.25

BETS

2065

HIT RATE

92.06%

ROI

96.34%

---

EDGE > 0.30

BETS

1770

HIT RATE

92.54%

ROI

107.45%

---

# ATP Clay Economic Validation

EDGE > 0.05

BETS

1459

ROI

39.40%

---

EDGE > 0.10

BETS

1147

ROI

52.80%

---

EDGE > 0.15

BETS

902

ROI

66.64%

---

EDGE > 0.20

BETS

739

ROI

80.64%

---

EDGE > 0.25

BETS

635

ROI

94.70%

---

EDGE > 0.30

BETS

552

ROI

103.47%

---

# ATP Grass Economic Validation

EDGE > 0.05

BETS

465

ROI

27.83%

---

EDGE > 0.10

BETS

349

ROI

43.05%

---

EDGE > 0.15

BETS

272

ROI

56.68%

---

EDGE > 0.20

BETS

217

ROI

67.69%

---

EDGE > 0.25

BETS

174

ROI

81.28%

---

EDGE > 0.30

BETS

144

ROI

95.42%

---

# Main Discovery

The strongest predictive variable in the entire project is:

delta_surface_inactivity_days

This variable consistently outranked:

- Elo
- Ranking
- Rank Points
- Rolling Statistics
- Experience Metrics

across multiple experiments.

---

# Interpretation

The evidence strongly supports the following model:

Predictive Power

=

Surface Readiness
+
Overall Readiness
+
Player Strength

where

Surface Readiness

is primarily captured by:

- delta_surface_inactivity_days

Overall Readiness

is primarily captured by:

- delta_inactivity_days
- days_inactive_a
- days_inactive_b

Player Strength

is primarily captured by:

- delta_elo

---

# Working Hypothesis

Betting markets appear to price:

Historical Player Quality

reasonably efficiently.

Betting markets appear less efficient at pricing:

- Recent Activity
- Surface Activity
- Competitive Readiness

The majority of model edge appears when:

Readiness Advantage
+
Positive Elo Profile
+
Market Underestimation

occur at the same time.

---

# Production Recommendation

Recommended Baseline Model

CORE_MODEL_V1

Variables

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

Advantages

- 94% feature reduction
- Approximately 99% AUC retention
- Strong economic performance
- Easier explainability
- Faster training
- Lower maintenance cost
- Lower feature drift risk

---

# Final Conclusion

After validation across:

- ATP Hard
- ATP Clay
- ATP Grass
- Challenger Hard
- Challenger Clay
- Challenger Grass

the evidence indicates that the true engine of the project is:

Surface Readiness
+
Overall Readiness
+
Elo Strength

Core Model V1 reproduces nearly all predictive performance of the full 83-feature system while using only five variables.

Core Model V1 should be considered the benchmark and primary production candidate for future development.
