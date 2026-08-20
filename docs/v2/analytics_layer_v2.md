# Smart Tennis Analytics 2.0

## Objetivo

Construir una plataforma de analítica avanzada y predicción para tenis ATP basada en una capa maestra canónica de partidos.

La plataforma se divide en dos grandes dominios:

- Analytics Layer
- Prediction Layer

Fuente única de verdad:

master_matches_v2.parquet

---

# Arquitectura

Raw Layer
+-- tennis_data_co_uk_v2.parquet
+-- tml_v2.parquet

Identity Layer
+-- player_mapping_v2.parquet
+-- tournament_mapping_v2.parquet

Matching Layer
+-- match_link_audit_v2.parquet

Master Layer
+-- master_matches_v2.parquet

Analytics Layer
+-- player_statistics_v2.parquet
+-- player_surface_metrics_v2.parquet
+-- tournament_statistics_v2.parquet
+-- betting_statistics_v2.parquet

Prediction Layer
+-- elo_history_v2.parquet
+-- training_matches_v2.parquet
+-- prediction_features_v2.parquet

---

# Estado actual

## Master Layer

Dataset:

master_matches_v2.parquet

Características:

Rows              : 68.220
Columns           : 124
Unique Match IDs  : 68.220

Single Point Of Truth del proyecto.

---

# Analytics Layer

## Objetivo

Generar métricas históricas agregadas para:

- análisis deportivo
- scouting
- betting analytics
- feature engineering

Fuente:

master_matches_v2.parquet

---

# Contextos soportados

## Scope

CAREER

SEASON

TOURNAMENT

---

## Surface

ALL

HARD

CLAY

GRASS

INDOOR

---

## Tournament Level

ALL

GRAND_SLAM

ATP_1000

ATP_500

ATP_250

---

# Grano

player_key

surface

competition_level

scope

season

---

# Player Dimensions

player_key

player_name

age

height

hand

backhand

country

---

# Match Statistics

matches

wins

losses

win_pct

loss_pct

---

# Sets

sets_won

sets_lost

set_win_pct

---

# Games

games_won

games_lost

game_win_pct

---

# Match Dominance

wins_to_love

losses_to_love

Definición:

2-0
3-0

y

0-2
0-3

---

# First Set

first_sets_won

first_sets_lost

first_set_win_pct

---

# Service Metrics

## Aces

aces

aces_per_match

---

## Double Faults

double_faults

double_faults_per_match

---

## First Serve In %

first_serve_in_pct

---

## First Serve Won %

first_serve_won_pct

Métrica principal para medir el daño que genera un jugador cuando entra el primer saque.

---

## Second Serve Won %

second_serve_won_pct

Métrica principal para medir la vulnerabilidad bajo presión.

---

## Service Points Won %

service_points_won_pct

Una de las variables más estables y predictivas del tenis.

---

## Service Games Won %

service_games_won_pct

---

# Return Metrics

## First Return Points Won

first_return_points_won

first_return_points_won_per_match

---

## Second Return Points Won

second_return_points_won

second_return_points_won_per_match

---

## Return Points Won %

return_points_won_pct

Variable clave para identificar grandes restadores.

---

## Return Games Won

return_games_won

Equivalente práctico:

breaks

---

# Break Points

## Break Chances

break_points_chances

break_points_chances_per_match

---

## Breaks Converted

break_points_converted

break_points_converted_per_match

---

## Break Conversion %

break_points_converted_pct

---

## Break Points Faced

break_points_faced

---

## Break Points Saved

break_points_saved

---

## Break Points Saved %

break_points_saved_pct

---

# Tie Breaks

tiebreaks_won

tiebreaks_lost

tiebreak_win_pct

---

# Pressure Metrics

pressure_points_won_pct

Basada en:

- break_points_saved_pct
- break_points_converted_pct
- tiebreak_win_pct

---

# Time On Court

minutes_played

minutes_per_match

---

# Opponent Strength

avg_opponent_rank

median_opponent_rank

---

## Ranking Buckets

top10_matches
top10_wins

top20_matches
top20_wins

top50_matches
top50_wins

top100_matches
top100_wins

---

# Betting Analytics

## Yield General

yield_total

Definición:

Apostar una unidad a todos los partidos históricos del jugador.

---

## Favorite Performance

matches_as_favorite

wins_as_favorite

favorite_win_pct

yield_as_favorite

---

## Underdog Performance

matches_as_underdog

wins_as_underdog

underdog_win_pct

yield_as_underdog

---

# Principios de Contexto

Las estadísticas nunca deben analizarse sin contexto.

Toda métrica debe poder filtrarse por:

- superficie
- nivel del rival
- periodo temporal

Ejemplo:

LAST_3_MONTHS

CLAY

VS_TOP_50

tiene mucho más valor que:

CAREER

ALL

---

# Prediction Layer

## Objetivo

Crear datasets para Machine Learning y modelos predictivos.

---

# Training Dataset

training_matches_v2.parquet

Grano:

1 fila = 1 partido

---

# Regla Fundamental

Todas las variables deben calcularse usando únicamente información anterior a la fecha del partido.

No puede existir data leakage.

---

# Snapshot Features

## Actividad

days_since_last_match

surface_days_since_last_match

---

## Fatiga

minutes_last_7d

minutes_last_30d

minutes_current_tournament

sets_last_30d

---

## Forma Reciente

last5_win_pct

last10_win_pct

surface_last10_win_pct

---

## ELO

elo_overall

elo_hard

elo_clay

elo_grass

elo_indoor

---

## Ranking

current_rank

best_rank

rank_change_90d

---

## Head To Head

h2h_matches

h2h_wins

surface_h2h_wins

---

# Variables Diferenciales

delta_elo_surface

delta_rank

delta_first_serve_won_pct

delta_second_serve_won_pct

delta_return_points_won_pct

delta_break_conversion_pct

delta_days_inactive

delta_minutes_last_30d

---

# Modelos Objetivo

Logistic Regression

XGBoost

LightGBM

CatBoost

---

# Roadmap

## Fase 1

? tennis_data_co_uk_v2

? tml_v2

? player_mapping_v2

? tournament_mapping_v2

? match_link_audit_v2

? master_matches_v2

---

## Fase 2

? player_statistics_v2

? tournament_statistics_v2

? betting_statistics_v2

---

## Fase 3

? elo_history_v2

---

## Fase 4

? training_matches_v2

? prediction_features_v2

? first predictive models

