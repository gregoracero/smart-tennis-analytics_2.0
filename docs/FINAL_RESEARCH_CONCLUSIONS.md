# FINAL RESEARCH CONCLUSIONS

## Project Objective

Determine whether the ATP and Challenger prediction system contains genuine predictive signal, identify the primary sources of predictive power, and isolate the variables responsible for the majority of economic value.

---

# Phase 1 - Technical Audit

## Objective

Verify that model performance is not the result of implementation issues.

Investigations Performed

- Walk-forward validation
- Time-based validation
- Calibration analysis
- Feature audits
- Elo generation review
- Inactivity feature review
- Dataset duplication checks
- Symmetric vs real-match analysis

Conclusion

No clear evidence was found of:

- Data leakage
- Player swaps
- Target inversion
- Temporal contamination
- Major duplication issues

The system appears technically valid.

---

# Phase 2 - Core Model Discovery

## Goal

Identify the smallest subset of variables capable of reproducing Full Model performance.

---

## Full Model

Features

83

AUC

0.8739

---

## Core Model V1

Features

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

Features

5

AUC

0.8678

Performance Retained

99.30%

Feature Reduction

83 -> 5

Reduction

93.98%

---

# Feature Importance Findings

Most Important Features

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

Finding

The strongest predictive variable in the entire project is:

delta_surface_inactivity_days

The dominant signal family is:

Readiness

not ranking.

---

# Phase 3 - Surface Validation

## ATP

Hard

Full

0.8739

Core

0.8678

Retention

99.30%

---

Clay

Full

0.8546

Core

0.8390

Retention

98.17%

---

Grass

Full

0.7999

Core

0.7902

Retention

98.79%

---

# Phase 4 - Challenger Validation

Hard

Full

0.8557

Core

0.8490

Retention

99.22%

---

Clay

Full

0.8597

Core

0.8523

Retention

99.14%

---

Grass

Full

0.8105

Core

0.8401

Retention

103.65%

Note

Grass Challenger contains only 332 test rows and should be interpreted cautiously.

---

# Validation Summary

| Segment | Full Model | Core Model | Retention |
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

# Phase 5 - ROI Zone Discovery

## Objective

Identify where economic value is generated.

Main Discovery

Maximum ROI occurs when:

delta_elo < 0

and

delta_surface_inactivity_days < -14

Interpretation

- Elo suggests the player is weaker.
- Surface readiness suggests the player is better prepared.
- Market continues to price the player according to Elo.

This creates the largest market inefficiencies identified in the project.

---

# Premium ROI Zone

Conditions

delta_elo < 0

delta_surface_inactivity_days < -14

---

Results

BETS

471

Hit Rate

74.31%

ROI

90.96%

Average Odds

3.29

---

# Phase 6 - Strategy Development

## Strategy A

Conditions

top5_edge > 0.05

Results

Bets

4882

Hit Rate

68.95%

ROI

37.67%

Average Odds

2.48

---

## Strategy B

Conditions

top5_edge > 0.05

delta_elo < 0

delta_surface_inactivity_days < -14

Results

Bets

471

Hit Rate

74.31%

ROI

90.96%

Average Odds

3.29

---

## Strategy C

Conditions

top5_edge > 0.10

delta_elo < 0

delta_surface_inactivity_days < -14

Results

Bets

412

Hit Rate

81.31%

ROI

109.02%

Average Odds

3.06

---

## Strategy D

Conditions

top5_edge > 0.15

delta_elo < 0

delta_surface_inactivity_days < -14

Results

Bets

377

Hit Rate

86.21%

ROI

121.08%

Average Odds

2.90

---

# Strategy V3 Validation

Selected Strategy

Strategy C

Conditions

top5_edge > 0.10

delta_elo < 0

delta_surface_inactivity_days < -14

---

Results

Bets

412

Hit Rate

81.31%

ROI

109.02%

Average Odds

3.06

Total Profit

449.16 units

Maximum Drawdown

-11.00 units

Longest Winning Streak

37

Longest Losing Streak

11

---

# Year-by-Year Results

2023

Bets

113

Hit Rate

82.30%

ROI

99.11%

Profit

111.99 units

---

2024

Bets

148

Hit Rate

82.43%

ROI

112.84%

Profit

167.01 units

---

2025

Bets

141

Hit Rate

85.11%

ROI

127.77%

Profit

180.16 units

---

2026

Only 10 bets

Not statistically significant.

---

# Key Discoveries

Discovery 1

No major methodological flaw was identified.

---

Discovery 2

Readiness is the dominant signal family.

---

Discovery 3

Surface readiness is the strongest individual predictor discovered.

---

Discovery 4

Five variables reproduce approximately 99% of Full Model performance.

---

Discovery 5

The greatest market inefficiency occurs when:

Negative Elo
+
Strong Surface Readiness Advantage

occur simultaneously.

---

Discovery 6

The most profitable strategy discovered is:

top5_edge > 0.10

delta_elo < 0

delta_surface_inactivity_days < -14

---

# Final Thesis

Predictive Power

=

Surface Readiness
+
Overall Readiness
+
Player Strength

Economic Edge

=

Market Underestimation of Competitive Readiness

---

# Production Recommendation

Baseline Model

CORE_MODEL_V1

Features

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

---

Baseline Strategy

STRATEGY_V3

Conditions

top5_edge > 0.10

delta_elo < 0

delta_surface_inactivity_days < -14

---

# Final Conclusion

The investigation indicates that the true engine of the system is not the complete 83-feature model.

The majority of predictive and economic value can be explained by a small readiness-driven framework built around:

Surface Readiness
+
Overall Readiness
+
Elo Strength

Core Model V1 and Strategy V3 should be considered the primary benchmark and production candidates for future development, forward testing and paper trading.
