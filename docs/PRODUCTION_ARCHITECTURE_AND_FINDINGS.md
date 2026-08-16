# Smart Tennis Analytics 2.0
# Production Architecture and Research Findings

## Executive Summary

After a complete audit of the project, the original architecture and the production architecture have been reconstructed.

The investigation revealed that most predictive power comes from:

- delta_surface_inactivity_days
- delta_elo
- delta_inactivity_days
- days_inactive_a
- days_inactive_b

These features originate from the Elo Engine and not from the Rolling Metrics Engine.

The project should therefore be considered as two separate systems:

1. Analytics Platform
2. Prediction Platform

---

# Analytics Platform

Purpose:

- Player analysis
- ATP-style comparisons
- Dashboarding
- Feature discovery
- Research

Pipeline:

update_tml_data.py

?

build_all_matches.py

?

build_master_matches.py

?

build_analytics_matches.py

?

enrich_tiebreaks.py

?

build_analytics_player_matches.py

?

rolling_metrics.parquet

?

signal_engine.parquet

?

player_metrics.parquet

?

Dashboard

---

# Prediction Platform

Purpose:

- Match prediction
- Probability generation
- Betting edge detection

Pipeline:

master_matches.parquet

?

build_player_lookup.py

?

player_lookup.parquet

?

build_player_elo.py

?

player_elo_history.parquet

?

build_training_with_player_ids.py

?

training_matches_with_ids.parquet

?

build_training_with_elo.py

?

training_matches_with_elo.parquet

?

XGBoost

?

Probabilities

?

Strategy V3

---

# Reconstructed Historical Pipeline

update_tml_data.py
    -> ongoing_tourneys.csv

build_all_matches.py
    -> all_matches.parquet

build_master_matches.py
    -> master_matches.parquet

build_analytics_matches.py
    -> analytics_matches.parquet

enrich_tiebreaks.py
    -> analytics_matches_tiebreaks.parquet

build_analytics_player_matches.py
    -> analytics_player_matches.parquet

build_training_matches.py
    -> training_matches.parquet

build_training_with_player_ids.py
    -> training_matches_with_ids.parquet

build_player_elo.py
    -> player_elo_history.parquet

build_training_with_elo.py
    -> training_matches_with_elo.parquet

train_xgboost*.py
    -> trained model

---

# Recovered Missing Scripts

Two missing scripts were reconstructed and validated.

## build_player_lookup.py

Input

- master_matches.parquet

Output

- player_lookup.parquet

Validation

Original dataset:
(11682, 2)

Rebuilt dataset:
(11682, 2)

Result

IDENTICAL DATASET = TRUE

---

## build_training_with_player_ids.py

Input

- training_matches.parquet
- player_lookup.parquet

Output

- training_matches_with_ids.parquet

Validation

Original dataset:
(200520, 72)

Rebuilt dataset:
(200520, 72)

Result

IDENTICAL DATASET = TRUE

---

# Elo Engine Findings

player_elo_history.parquet stores a historical snapshot before every match.

Columns include:

- winner_elo_before
- loser_elo_before
- winner_surface_elo_before
- loser_surface_elo_before
- winner_days_inactive
- loser_days_inactive
- winner_surface_days_inactive
- loser_surface_days_inactive
- winner_matches_played
- loser_matches_played

Rows:

366065

Players:

11683

Last Match Date:

2026-08-13

---

# Production Artifacts Created

## player_state.parquet

Purpose

Current state per player.

One row per player.

Columns

- player_id
- elo
- matches_played
- match_date

Rows

11683

---

## player_surface_last_match.parquet

Purpose

Latest state per surface.

Columns

- player_id
- surface
- surface_elo
- surface_matches_played
- last_surface_match_date

Used to calculate future readiness.

---

## prediction_features.parquet

Purpose

Generate Core Model features for future matches.

Features:

- elo_a
- elo_b
- delta_elo

- days_inactive_a
- days_inactive_b
- delta_inactivity_days

- surface_elo_a
- surface_elo_b

- surface_days_inactive_a
- surface_days_inactive_b
- delta_surface_inactivity_days

---

# Core Model V1

Most important features:

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

Performance retained:

Approximately 99% of Full Model performance.

---

# Strategy V3

Conditions:

delta_elo < 0

delta_surface_inactivity_days < -14

top5_edge > 0.10

Observed Results:

- Hit Rate ~81%
- ROI ~109%
- Max Drawdown -11

---

# Recommended Daily Production Pipeline

update_tml_data.py

?

build_master_matches.py

?

build_player_lookup.py

?

build_player_elo.py

?

player_state.parquet

+

player_surface_last_match.parquet

?

prediction_features.parquet

?

Core Model V1

?

Strategy V3

?

daily_picks.csv

---

# Future Work

1. Rebuild complete datasets using latest ongoing data.

2. Create train_xgboost_production.py

Outputs:

- core_model_v1.joblib
- imputer.joblib
- features.json

3. Create Streamlit prediction application.

Inputs:

- Player A
- Player B
- Surface
- Date

Outputs:

- Probability
- Delta Elo
- Readiness Metrics
- Strategy V3 Signal

4. Automate daily pipeline.

5. Generate daily picks.

---

# Final Conclusion

The project originally evolved around rolling metrics and player analytics.

The audit demonstrated that production value is primarily explained by:

- Elo
- Surface Elo
- Inactivity
- Surface Inactivity

The recommended architecture for production is therefore centered on the Elo Engine and Core Model V1 rather than the Analytics branch.
