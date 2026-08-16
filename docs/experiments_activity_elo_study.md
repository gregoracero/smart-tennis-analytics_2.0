# Activity vs Elo Study

## Objective

Understand the real source of predictive power behind the ATP prediction system and determine whether the observed ROI could be explained by data leakage, implementation issues or by genuine predictive signals.

---

# Dataset Audit

## Training Dataset

File:

training_matches.parquet

Properties:

- Symmetric dataset
- Each real match generates two rows
- Winner vs Loser -> target = 1
- Loser vs Winner -> target = 0

Purpose:

- Model training

---

## Elo History Dataset

File:

player_elo_history.parquet

Properties:

- One row per real ATP match
- Stores Elo state before each match
- Stores inactivity information
- Stores surface inactivity information

Rows:

366,065

Unique Matches:

366,035

Duplicates:

30

Duplicate rate:

0.008%

Conclusion:

- Effectively a real-match dataset
- Suitable for causal analysis

---

# Walk Forward Results

Experience Only
AUC = 0.570

Rank Only
AUC = 0.687

No Elo
AUC = 0.714

Inactivity Only
AUC = 0.764

Readiness Only
AUC = 0.817

Elo Only
AUC = 0.840

Elo + Readiness
AUC = 0.877

Full Model
AUC = 0.884

Main finding:

- Elo and Readiness explain nearly all predictive power

---

# ROI Results

Experience Only
ROI = -9%

Readiness Only
ROI = 31%

Elo Only
ROI = 41%

Elo + Readiness
ROI = 43%

Full Model
ROI = 46%

Main finding:

- Most economic value comes from Elo and Readiness

---

# Calibration

0.95 - 0.98

Predicted = 96.77%
Actual    = 97.42%

0.98 - 0.99

Predicted = 98.53%
Actual    = 99.07%

0.99 - 1.00

Predicted = 99.43%
Actual    = 99.05%

Conclusion:

- Excellent calibration in extreme regions

---

# Stress Tests

ODDS x 1.00

ROI = 40.95%

ODDS x 0.98

ROI = 38.13%

ODDS x 0.95

ROI = 33.90%

ODDS x 0.90

ROI = 26.86%

ODDS x 0.85

ROI = 19.81%

ODDS x 0.80

ROI = 12.76%

Conclusion:

- Edge survives significant price degradation

---

# ROI by Edge

EDGE > 0.02

ROI = 32.45%

EDGE > 0.05

ROI = 40.95%

EDGE > 0.10

ROI = 58.74%

EDGE > 0.15

ROI = 76.49%

EDGE > 0.20

ROI = 87.74%

EDGE > 0.25

ROI = 95.06%

EDGE > 0.30

ROI = 102.73%

EDGE > 0.35

ROI = 112.02%

EDGE > 0.40

ROI = 125.15%

Conclusion:

- Edge ranking appears highly informative

---

# Favorites vs Underdogs

Favorites

BETS     2599
HIT      87.80%
ROI      25.84%

Underdogs

BETS     2229
HIT      57.65%
ROI      58.69%

Conclusion:

- Largest economic edge comes from selected underdogs

---

# Activity Signal Investigation

## Correlation Analysis

After isolating low Elo differences:

corr(delta_inactivity_days, delta_elo)

=
-0.035

corr(delta_inactivity_days, delta_rank_points)

=
-0.023

Conclusion:

- Activity appears largely independent of Elo and ranking strength

---

# Real Match Analysis

Dataset:

player_elo_history.parquet

Only matches with valid inactivity information were used.

---

## Activity Advantage

Real ATP Matches

Matches = 220,417

When players had different activity levels:

Winner was the more active player

=
81.52%

---

## ATP Modern Era

2020+

Matches = 32,509

Winner More Active

=
79.83%

---

## Betting Period

2021+

Matches = 30,290

Winner More Active

=
79.73%

---

## By Surface

Hard

81.04%

Clay

81.10%

Grass

82.28%

Carpet

81.79%

Conclusion:

- Signal remains stable across eras and surfaces

---

# Activity and ROI

All Bets

Delta Inactivity < -30

ROI       = 48.89%
Hit Rate  = 85.09%

Delta Inactivity -30 to -14

ROI       = 83.01%
Hit Rate  = 87.99%

Delta Inactivity -14 to -7

ROI       = 64.75%
Hit Rate  = 86.87%

Delta Inactivity > 30

ROI      = -71.85%
Hit Rate = 13.04%

Conclusion:

- Activity differential strongly affects profitability

---

# Underdog Activity Study

Delta Inactivity -30 to -14

Hit Rate   = 79.75%
ROI        = 132.13%
Avg Odds   = 3.22
Avg Elo    = -118

Delta Inactivity -14 to -7

Hit Rate   = 74.86%
ROI        = 100.61%
Avg Odds   = 3.01
Avg Elo    = -95

Conclusion:

- Active underdogs generate the strongest returns

---

# Elo vs Activity Conflict

Conditions:

edge > 5%
delta_elo < 0
delta_inactivity_days < -14

Results:

BETS               = 315
WIN RATE           = 77.78%
ROI                = 106.32%

AVG DELTA ELO      = -137.36

AVG DELTA INACTIVITY

= -32.83 days

Interpretation:

- Activity can overcome substantial Elo disadvantages

---

# Final Conclusions

Finding 1

Elo is a strong signal.

---

Finding 2

Readiness is also a strong signal.

---

Finding 3

Readiness contains information not captured by Elo.

---

Finding 4

Elo + Readiness explains almost all predictive power.

Elo + Readiness
=
0.877 AUC

Full Model
=
0.884 AUC

---

Finding 5

Selected active underdogs represent the most profitable segment.

---

Finding 6

No clear evidence of:

- Data leakage
- Label inversion
- Player swapping
- Major duplication issues
- Sign errors

---

# Working Hypothesis

The market appears to price historical player quality more efficiently than current competitive readiness.

The system gains most of its advantage when:

- Elo and market disagree
- Recent competitive activity strongly favors one player
- The market underestimates the impact of match readiness

# Executive Summary

Most predictive power appears to come from:

Quality
+
Readiness

where:

Quality

=
Elo
+
Surface Elo

and

Readiness

=
Days Inactive
+
Surface Days Inactive

The difference between:

Elo + Readiness = 0.877 AUC

and

Full Model = 0.884 AUC

is very small, suggesting that the majority of model skill is concentrated in these two components.

The strongest economic signal discovered during the investigation is:

Underdog
+
Much More Active
+
Opponent Recently Inactive

which consistently generates the highest ROI across multiple robustness tests.
