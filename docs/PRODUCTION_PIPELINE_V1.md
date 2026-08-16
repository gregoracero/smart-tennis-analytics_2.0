# Smart Tennis Analytics 2.0
# Production Pipeline V1

## Status

WORK IN PROGRESS

## Goal

Transform the research platform into a production prediction platform capable of generating future match predictions.

---

# Original Research Pipeline

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

build_training_matches.py

?

training_matches.parquet

?

build_training_with_player_ids.py

?

training_matches_with_ids.parquet

?

build_player_elo.py

?

player_elo_history.parquet

?

build_training_with_elo.py

?

training_matches_with_elo.parquet

?

train_xgboost*.py

---

# Reconstructed Missing Scripts

The following scripts were reconstructed and validated against existing datasets.

## build_player_lookup.py

Input

master_matches.parquet

Output

player_lookup.parquet

Validation

IDENTICAL DATASET = TRUE

---

## build_training_with_player_ids.py

Input

training_matches.parquet

player_lookup.parquet

Output

training_matches_with_ids.parquet

Validation

IDENTICAL DATASET = TRUE

---

# Elo State Layer

The Elo Engine generates:

player_elo_history.parquet

This dataset contains historical snapshots before every match.

Includes:

- Elo
- Surface Elo
- Days Inactive
- Surface Days Inactive
- Matches Played
- Surface Matches Played

---

# New Production Artifacts

## player_state.parquet

Generator

build_player_state.py

Input

player_elo_history.parquet

Output

player_state.parquet

Purpose

Current player state.

One row per player.

Columns

- player_id
- elo
- days_inactive
- matches_played
- match_date

---

## player_surface_last_match.parquet

Generator

build_player_surface_last_match.py

Input

player_elo_history.parquet

Output

player_surface_last_match.parquet

Purpose

Latest known state by surface.

Columns

- player_id
- surface
- surface_elo
- surface_matches_played
- last_surface_match_date

Used to calculate:

- surface_days_inactive
- delta_surface_inactivity_days

for future predictions.

---

# Production Training Pipeline

update_tml_data.py

?

build_master_matches.py

?

build_player_lookup.py

?

player_lookup.parquet

?

build_analytics_matches.py

?

analytics_matches.parquet

?

enrich_tiebreaks.py

?

analytics_matches_tiebreaks.parquet

?

build_analytics_player_matches.py

?

analytics_player_matches.parquet

?

build_training_matches.py

?

training_matches.parquet

?

build_training_with_player_ids.py

?

training_matches_with_ids.parquet

?

build_player_elo.py

?

player_elo_history.parquet

?

build_player_state.py

?

player_state.parquet

?

build_player_surface_last_match.py

?

player_surface_last_match.parquet

?

build_training_with_elo.py

?

training_matches_with_elo.parquet

?

train_model.py

---

# Model Registry

Models are stored per:

- Tour
- Surface
- Algorithm
- Version

Example:

atp_hard_xgboost_v1

Contents:

- model.joblib
- imputer.joblib
- features.json
- metrics.json
- metadata.json

---

# Available Models

ATP

- atp_hard_xgboost_v1
- atp_clay_xgboost_v1
- atp_grass_xgboost_v1

Challenger

- challenger_hard_xgboost_v1
- challenger_clay_xgboost_v1
- challenger_grass_xgboost_v1

---

# Production Metrics

ATP

- Hard: AUC 0.8739 | ACC 0.7785
- Clay: AUC 0.8546 | ACC 0.7554
- Grass: AUC 0.7999 | ACC 0.7102

Challenger

- Hard: AUC 0.8557 | ACC 0.7567
- Clay: AUC 0.8597 | ACC 0.7603
- Grass: AUC 0.8105 | ACC 0.7259

---

# Prediction Architecture

User Input

- Tour
- Surface
- Player A
- Player B
- Match Date

?

feature_builder.py

?

Features generated in memory

?

run_prediction.py

?

Load model

?

Load imputer

?

Predict probability

?

Return result

No intermediate CSV required.

No intermediate prediction parquet required.

---

# Core Model Features

Primary production signals

- delta_elo
- delta_inactivity_days
- delta_surface_inactivity_days

Secondary supporting features

- elo_a
- elo_b
- surface_elo_a
- surface_elo_b
- days_inactive_a
- days_inactive_b
- surface_days_inactive_a
- surface_days_inactive_b

---

# Strategy V3

Conditions

- top5_edge > 0.10
- delta_elo < 0
- delta_surface_inactivity_days < -14

Historical Results

- Hit Rate ~81%
- ROI ~109%
- Max Drawdown -11

---

# Future Streamlit Application

Inputs

- Tournament Type
- Surface
- Player A
- Player B
- Match Date

Outputs

- Win Probability
- Delta Elo
- Delta Inactivity
- Delta Surface Inactivity
- Strategy V3 Signal

---

# Long Term Architecture

User

?

Prediction App

?

feature_builder.py

?

run_prediction.py

?

Model Registry

?

Probability

?

Strategy V3

?

Daily Picks

---

# Current State

Completed

? Pipeline reconstruction

? Elo Engine reconstruction

? Missing scripts reconstruction

? Training pipeline reproduction

? Player State Layer

? Model Registry

? Six production models

Next Steps

1. feature_builder.py
2. run_prediction.py
3. Streamlit prediction UI
4. Automated daily picks generation
