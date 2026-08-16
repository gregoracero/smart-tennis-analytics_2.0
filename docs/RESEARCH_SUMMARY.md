# Research Summary

## Main Findings

### 1. No evidence of data leakage found

Audits performed:

- Temporal validation
- Walk-forward validation
- Feature construction review
- Elo generation review
- Inactivity generation review
- Dataset duplication checks

Result:

No clear evidence of:

- Data leakage
- Target inversion
- Player swapping
- Temporal contamination

---

### 2. Elo is the strongest standalone signal

Walk Forward AUC

Elo Only

0.840

Interpretation:

Elo captures most of the player-strength information and remains the strongest individual predictor.

---

### 3. Readiness is the second strongest standalone signal

Walk Forward AUC

Readiness Only

0.817

Interpretation:

Competitive activity and inactivity carry substantial predictive information independent from Elo.

---

### 4. Elo + Readiness explains almost all predictive power

Walk Forward Results

Elo + Readiness

0.877

Full Model

0.884

Difference

0.007

Interpretation:

Most predictive power is concentrated in:

- Elo
- Surface Elo
- Days Inactive
- Surface Days Inactive

All other features add only a small incremental contribution.

---

### 5. ROI remains robust under stress testing

ROI by odds degradation

Base

40.95%

Odds x 0.98

38.13%

Odds x 0.95

33.90%

Odds x 0.90

26.86%

Odds x 0.85

19.81%

Odds x 0.80

12.76%

Interpretation:

The betting edge survives substantial deterioration in execution prices.

---

### 6. Calibration remains strong even in extreme probability regions

0.95 - 0.98

Predicted

96.77%

Actual

97.42%

0.98 - 0.99

Predicted

98.53%

Actual

99.07%

0.99 - 1.00

Predicted

99.43%

Actual

99.05%

Interpretation:

The model remains well calibrated even for its highest confidence predictions.

---

### 7. Underdogs generate the strongest ROI

Favorites

BETS

2599

HIT RATE

87.80%

ROI

25.84%

Underdogs

BETS

2229

HIT RATE

57.65%

ROI

58.69%

Interpretation:

The largest economic value comes from correctly identifying underdogs.

---

### 8. Activity appears independent from Elo

Correlation Analysis

corr(delta_inactivity_days, delta_elo)

-0.035

corr(delta_inactivity_days, delta_rank_points)

-0.023

corr(delta_inactivity_days, delta_surface_elo)

0.005

Interpretation:

Activity is largely independent from:

- Elo
- Surface Elo
- Ranking Points

and therefore contributes unique information.

---

### 9. Real match audit confirms an activity signal

Dataset:

player_elo_history.parquet

Rows

366,065

Unique Matches

366,035

Duplicates

30

Duplicate Rate

0.008%

Interpretation:

Effectively a real-match dataset.

---

### 10. Activity signal persists in real ATP matches

Matches with different activity levels

220,417

Winner was the more active player

81.52%

Interpretation:

Activity signal remains visible outside the symmetric training dataset.

---

### 11. Activity signal remains stable in modern ATP

ATP 2020+

Matches

32,509

Winner More Active

79.83%

ATP 2021+

Matches

30,290

Winner More Active

79.73%

Interpretation:

The relationship persists in both the modern ATP era and the betting evaluation period.

---

### 12. Activity signal persists across surfaces

Hard

81.04%

Clay

81.10%

Grass

82.28%

Carpet

81.79%

Interpretation:

The signal is not specific to a single surface.

---

### 13. Activity strongly impacts ROI

All Bets

Delta Inactivity < -30

ROI

48.89%

Hit Rate

85.09%

Delta Inactivity -30 to -14

ROI

83.01%

Hit Rate

87.99%

Delta Inactivity -14 to -7

ROI

64.75%

Hit Rate

86.87%

Delta Inactivity > 30

ROI

-71.85%

Hit Rate

13.04%

Interpretation:

Profitability changes dramatically depending on relative competitive activity.

---

### 14. Active underdogs are the most profitable subgroup

Delta Inactivity -30 to -14

Hit Rate

79.75%

ROI

132.13%

Average Odds

3.22

Average Elo Difference

-118

Delta Inactivity -14 to -7

Hit Rate

74.86%

ROI

100.61%

Average Odds

3.01

Average Elo Difference

-95

Interpretation:

The strongest betting opportunities occur when:

- Player has worse Elo
- Player has much greater recent activity
- Market still prices the player as an underdog

---

### 15. Elo vs Activity Conflict Study

Conditions

edge > 5%

delta_elo < 0

delta_inactivity_days < -14

Results

BETS

315

WIN RATE

77.78%

ROI

106.32%

AVG DELTA ELO

-137.36

AVG DELTA INACTIVITY

-32.83

Interpretation:

Competitive readiness can overcome substantial Elo disadvantages.

---

# Working Hypothesis

The market appears to price historical player quality more efficiently than current competitive readiness.

Most of the observed edge emerges when:

- Elo and market disagree
- Recent competitive activity strongly favors one player
- Market underestimates the value of match readiness

---

# Executive Summary

Main Drivers

Quality

- Elo
- Surface Elo

Readiness

- Days Inactive
- Surface Days Inactive

Most predictive performance can be explained by the interaction of these two dimensions.

Quality determines how strong a player is.

Readiness determines how prepared a player is to compete today.

The strongest economic signal discovered during the investigation is:

Underdog
+
Much More Active Recently
+
Opponent Less Active
+
Negative Elo Difference

This profile consistently generated the highest ROI across:

- Walk-forward validation
- Calibration analysis
- Surface analysis
- Year-by-year analysis
- Stress testing
- Activity studies
- Underdog studies

---

# Final Verdict

Current evidence supports the following conclusions:

1. No major methodological flaw has been identified.

2. Elo is a powerful predictive signal.

3. Readiness is a powerful predictive signal.

4. Readiness contains information not captured by Elo.

5. Elo + Readiness captures nearly all model skill.

6. Active underdogs appear to be the most mispriced segment of the market.

7. The primary hypothesis is that betting markets price historical quality better than current competitive readiness.

8. The next stage of validation should be real-time paper trading.
