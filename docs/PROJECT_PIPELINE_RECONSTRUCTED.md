# SMART TENNIS ANALYTICS 2.0
# PIPELINE DOCUMENTATION
# Version: Reconstructed after Core Model V1 Research

# ============================================================
# OVERVIEW
# ============================================================

The project currently contains two major pipelines:

1. Analytics Pipeline
   - Dashboards
   - H2H
   - Player Form
   - Surface Statistics
   - Signal Engine

2. Prediction Pipeline
   - Elo Engine
   - Inactivity Engine
   - Core Model V1
   - Strategy V3

Research findings suggest that the Prediction Pipeline
provides most of the predictive and economic value.

# ============================================================
# COMPLETE PIPELINE
# ============================================================

| Order | Script | Input | Output | Description |
|--------|--------|--------|--------|--------|
| 1 | ingestion/update_tml_data.py | TennisMyLife source | YYYY.csv, YYYY_challenger.csv, ongoing_tourneys.csv | Downloads ATP, Challenger and ongoing tournaments data |
| 2 | ingestion/build_all_matches.py | ATP historical CSVs, Challenger historical CSVs | all_matches.parquet | Consolidates ATP and Challenger historical matches |
| 3 | ingestion/build_master_matches.py | all_matches.parquet, ongoing_tourneys.csv | master_matches.parquet | Adds new matches and removes duplicates |
| 4 | ingestion/build_analytics_matches.py | master_matches.parquet | analytics_matches.parquet | Creates analytics-ready dataset |
| 5 | ingestion/enrich_tiebreaks.py | analytics_matches.parquet | analytics_matches_tiebreaks.parquet | Adds tie-break information |
| 6 | ingestion/build_analytics_player_matches.py | analytics_matches.parquet | analytics_player_matches.parquet | Creates winner-row and loser-row dataset |
| 7 | build_player_matches.py | all_matches.parquet | player_matches.parquet | Creates player-centric match history |
| 8 | Surface Metrics Engine | player_matches.parquet | player_surface_metrics.parquet | Aggregated surface statistics |
| 9 | Rolling Metrics Engine | analytics_player_matches.parquet | rolling_metrics.parquet | Last 10 matches, Last 3 months, Last 6 months metrics |
| 10 | Signal Engine | rolling_metrics.parquet | signal_engine.parquet | Surface Form, Fatigue, Break Risk and Confidence signals |
| 11 | build_player_metrics.py | analytics_player_matches.parquet | player_metrics.parquet | Metrics dataset used by dashboards |
| 12 | H2H Engine | analytics_player_matches.parquet | h2h_engine.parquet | Head-to-head calculations |
| 13 | H2H Builder | h2h_engine.parquet | h2h_overall.parquet | Overall H2H statistics |
| 14 | H2H Builder | h2h_engine.parquet | h2h_surface.parquet | Surface H2H statistics |
| 15 | H2H Builder | h2h_engine.parquet | h2h_matches.parquet | Match-level H2H history |
| 16 | build_player_lookup.py | master_matches.parquet | player_lookup.parquet | ATP player lookup table |
| 17 | build_player_elo.py | master_matches.parquet | player_elo_history.parquet | Elo, Surface Elo, Inactivity and Experience engine |
| 18 | build_training_with_player_ids.py | training_matches.parquet, player_lookup.parquet | training_matches_with_ids.parquet | Adds ATP player IDs |
| 19 | ml/build_training_with_elo.py | training_matches_with_ids.parquet, player_elo_history.parquet | training_matches_with_elo.parquet | Adds Elo and readiness features |
| 20 | train_xgboost*.py | training_matches_with_elo.parquet | Trained ML models | Model training |
| 21 | Betting Pipeline | training_matches_with_elo.parquet, atp_odds_2020_2026.parquet | atp_matches_with_odds*.parquet | Joins match data with betting odds |
| 22 | Prediction Pipeline | atp_matches_with_odds*.parquet | atp_matches_with_predictions*.parquet | Generates model probabilities |
| 23 | Validation Layer | prediction datasets | temporal_predictions_2023_plus.parquet, temporal_predictions_unique.parquet | Research and ROI validation datasets |

# ============================================================
# ANALYTICS BRANCH
# ============================================================

update_tml_data.py

?

build_all_matches.py

?

build_master_matches.py

?

build_analytics_matches.py

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
Players Form
H2H Analysis

Purpose:

- Dashboards
- ATP-style comparison
- Player analysis
- Signal exploration
- Research

# ============================================================
# ELO ENGINE BRANCH
# ============================================================

master_matches.parquet

?

build_player_lookup.py

?

player_lookup.parquet

?

build_player_elo.py

?

player_elo_history.parquet

Generated Features:

Elo

- elo_a
- elo_b
- delta_elo

Surface Elo

- surface_elo_a
- surface_elo_b
- delta_surface_elo

Inactivity

- days_inactive_a
- days_inactive_b
- delta_inactivity_days

Surface Inactivity

- surface_days_inactive_a
- surface_days_inactive_b
- delta_surface_inactivity_days

Experience

- matches_played_a
- matches_played_b
- delta_matches_played

Surface Experience

- surface_matches_played_a
- surface_matches_played_b
- delta_surface_matches_played

# ============================================================
# MACHINE LEARNING BRANCH
# ============================================================

training_matches.parquet

?

build_training_with_player_ids.py

?

training_matches_with_ids.parquet

?

build_training_with_elo.py

+

player_elo_history.parquet

?

training_matches_with_elo.parquet

?

XGBoost

?

Predictions

# ============================================================
# CORE MODEL V1
# ============================================================

Research concluded these variables explain almost all predictive power:

1. delta_surface_inactivity_days
2. delta_elo
3. delta_inactivity_days
4. days_inactive_a
5. days_inactive_b

Performance Retention:

~99% of Full Model performance.

# ============================================================
# STRATEGY V3
# ============================================================

Conditions

top5_edge > 0.10

delta_elo < 0

delta_surface_inactivity_days < -14

Results

Bets: 412

Hit Rate: 81.31%

ROI: 109.02%

Max Drawdown: -11

Total Profit: 449.16 units

# ============================================================
# PRODUCTION PIPELINE (RECOMMENDED)
# ============================================================

Step 1

Script

ingestion/update_tml_data.py

Input

TennisMyLife

Output

ongoing_tourneys.csv

Purpose

Download latest data.

---

Step 2

Script

ingestion/build_master_matches.py

Input

all_matches.parquet
ongoing_tourneys.csv

Output

master_matches.parquet

Purpose

Consolidate historical and new matches.

---

Step 3

Script

build_player_elo.py

Input

master_matches.parquet

Output

player_elo_history.parquet

Purpose

Update Elo and readiness information.

---

Step 4

Prediction Feature Generator

Input

player_elo_history.parquet

Output

Core Model V1 features

- delta_elo
- delta_inactivity_days
- delta_surface_inactivity_days

---

Step 5

Core Model V1

Output

Win probabilities

---

Step 6

Strategy V3

Output

Final betting opportunities

# ============================================================
# MOST IMPORTANT DATASETS
# ============================================================

1. ongoing_tourneys.csv

Daily source of new matches.

2. master_matches.parquet

Official match repository.

3. player_elo_history.parquet

Core player state.

Contains:

- Elo
- Surface Elo
- Inactivity
- Surface Inactivity
- Experience

4. training_matches_with_elo.parquet

Final ML dataset.

5. temporal_predictions_2023_plus.parquet

Research validation dataset.

# ============================================================
# ARCHITECTURAL CONCLUSION
# ============================================================

Original System

Analytics Platform

master_matches

?

analytics_matches

?

analytics_player_matches

?

rolling_metrics

?

signal_engine

?

player_metrics

?

dashboard

---

Current Discovery

Prediction Engine

master_matches

?

player_elo_history

?

Core Model V1

?

Strategy V3

The strongest predictive and economic signals discovered in the project originate from:

- Elo
- Surface Readiness
- General Readiness

represented by:

- delta_surface_inactivity_days
- delta_elo
- delta_inactivity_days
- days_inactive_a
- days_inactive_b

The recommended production path is:

ongoing_tourneys.csv

?

master_matches.parquet

?

player_elo_history.parquet

?

Core Model V1

?

Strategy V3
