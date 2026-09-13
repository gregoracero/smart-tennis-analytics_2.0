# ATP + QUALY + CHALLENGER PIPELINE

## Objetivo

Pipeline completo para ATP, ATP Qualifying y Challenger.

## Pipeline

```text
tml_v2
¦
+-- tml_player_dimension_v2
+-- tml_tournament_dimension_v2
¦
+-- master_matches_atp_qualy_challenger
        ¦
        +-- player_per_match_statistics_atp_qualy_challenger
        ¦       ¦
        ¦       +-- player_statistics_atp_qualy_challenger
        ¦       +-- elo_history_atp_qualy_challenger
        ¦       +-- player_snapshots_atp_qualy_challenger
        ¦
        +-- tournament_statistics_atp_qualy_challenger
                ¦
                +-- training_matches_atp_qualy_challenger
                        ¦
                        +-- model_dataset_atp_qualy_challenger
```

---

# master_matches_atp_qualy_challenger

## Script

build_master_matches_atp_qualy_challenger.py

## Objetivo

Dataset maestro de partidos ATP, ATP Qualifying y Challenger.

## Key

master_match_id

## Campos principales

competition_type

canonical_tournament_id

tourney_id

tourney_name

surface

season

winner_player_id

winner_name

winner_rank

winner_rank_points

loser_player_id

loser_name

loser_rank

loser_rank_points

w_ace

w_df

w_svpt

w_1stIn

w_1stWon

w_2ndWon

l_ace

l_df

l_svpt

l_1stIn

l_1stWon

l_2ndWon

---

# player_per_match_statistics_atp_qualy_challenger

## Script

build_player_per_match_statistics_atp_qualy_challenger.py

## Objetivo

Representación jugador-partido.

Cada partido genera:

Winner Row

Loser Row

## Campos principales

player_id

player_name

opponent_player_id

opponent_name

win

loss

player_rank

player_rank_points

opponent_rank

opponent_rank_points

aces

double_faults

first_serve_in

first_serve_points_won

second_serve_points_won

service_points_played

service_games_played

service_games_won

break_points_faced

break_points_saved

break_points_chances

break_points_converted

minutes

---

# player_statistics_atp_qualy_challenger

## Script

build_player_statistics_atp_qualy_challenger.py

## Objetivo

Estadísticas agregadas por jugador.

## Scopes

CAREER

SEASON

## Dimensiones

competition_type

surface

tournament_level

season

## Features

matches

wins

losses

win_pct

loss_pct

aces_per_match

double_faults_per_match

first_serve_in_pct

first_serve_won_pct

second_serve_won_pct

service_points_won_pct

service_games_won_pct

break_points_saved_pct

break_points_converted_pct

pressure_index

avg_opponent_rank

median_opponent_rank

top10_matches

top20_matches

top50_matches

top100_matches

---

# tournament_statistics_atp_qualy_challenger

## Script

build_tournament_statistics_atp_qualy_challenger.py

## Objetivo

Estadísticas agregadas por torneo.

## Features

matches

players

avg_rank

median_rank

top10_players

top20_players

top50_players

top100_players

total_aces

aces_per_match

total_double_faults

double_faults_per_match

total_minutes

avg_minutes

---

# elo_history_atp_qualy_challenger

## Script

build_elo_history_atp_qualy_challenger.py

## Objetivo

Histórico Elo pre-partido.

## Features

elo_overall_before

elo_overall_after

elo_surface_before

elo_surface_after

elo_mix_before

expected_win_probability

career_matches_before

days_since_last_match

inactivity_adjusted

---

# player_snapshots_atp_qualy_challenger

## Script

build_player_snapshots_atp_qualy_challenger.py

## Objetivo

Snapshot histórico previo al partido.

## Bloques

Career

Surface

Season

Recent Form

Activity

H2H

Elo

Performance

---

# training_matches_atp_qualy_challenger

## Script

build_training_matches_atp_qualy_challenger.py

## Objetivo

Unir snapshot ganador y perdedor.

## Estructura

winner_*

loser_*

diff_*

target

---

# model_dataset_atp_qualy_challenger

## Script

build_model_dataset_atp_qualy_challenger.py

## Objetivo

Dataset final para Machine Learning.

## Balanceado

target = 1

target = 0

## Modelos soportados

Logistic Regression

Random Forest

XGBoost

LightGBM

CatBoost

---

# Estado actual

? player_dimension

? tournament_dimension

? master_matches

? player_per_match_statistics

? player_statistics

? tournament_statistics

? elo_history

? player_snapshots

? training_matches

? model_dataset
