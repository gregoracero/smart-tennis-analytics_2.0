# TRAINING MATCHES V2 - DOCUMENTACION TECNICA

## Objetivo

Construir el dataset de entrenamiento final para modelos de Machine Learning.

Granularidad:

master_match_id

Resultado:

1 fila por partido

---

# Posición en el Pipeline

ingestion
↓
master_matches_v2

analytics
↓
player_per_match_statistics_v2
↓
player_statistics_v2
↓
tournament_statistics_v2
↓
elo_history_v2

prediction
↓
player_snapshots_v2
↓
training_matches_v2

models
↓
LightGBM
XGBoost
CatBoost

---

# Fuente de Datos

Input:

player_snapshots_v2.parquet

Granularidad de entrada:

2 filas por partido

Una por cada jugador.

---

# Objetivo Funcional

Transformar:

Jugador A Snapshot

+

Jugador B Snapshot

↓

1 fila de entrenamiento

con variable objetivo:

target

---

# Balanceo de Clases

Para evitar sesgo:

50%

A = ganador
B = perdedor

target = 1

50%

A = perdedor
B = ganador

target = 0

Implementado mediante:

numpy.random.default_rng()

Resultado observado:

target

0 = 50.0008%

1 = 49.9992%

Dataset perfectamente balanceado.

---

# Variables de Identidad

- master_match_id
- match_date
- season
- surface
- tournament_level

---

# Variables de Ranking

- a_rank
- b_rank
- rank_delta

- a_rank_points
- b_rank_points
- rank_points_delta

---

# Variables Biográficas

- a_age
- b_age
- age_delta

- a_height
- b_height
- height_delta

---

# Variables Elo

## General Elo

- a_elo
- b_elo
- elo_delta

---

## Surface Elo

- a_surface_elo
- b_surface_elo
- surface_elo_delta

---

## Mixed Elo

- a_elo_mix
- b_elo_mix
- elo_mix_delta

---

## Elo Edge

- elo_mix_expected_edge

Definición:

elo_mix_delta
/
400

---

# Variables Career

- a_career_win_pct
- b_career_win_pct
- career_win_pct_delta

- a_career_matches
- b_career_matches
- career_matches_delta

---

# Variables Surface

- a_surface_win_pct
- b_surface_win_pct
- surface_win_pct_delta

- a_surface_matches
- b_surface_matches
- surface_matches_delta

---

# Surface Specialization

Representa la ventaja específica en la superficie actual.

Definición:

surface_elo
-
overall_elo

Variables:

- a_surface_specialization
- b_surface_specialization
- surface_specialization_delta

---

# Variables Season

- a_season_win_pct
- b_season_win_pct
- season_win_pct_delta

- season_matches_delta

---

# Actividad

- days_since_last_match_delta
- career_matches_before_delta

---

# Inactividad

Variables:

- a_inactivity_adjusted
- b_inactivity_adjusted

Procedentes de:

elo_history_v2

---

# Experience Ratios

## Career Experience

experience_ratio

Definición:

a_career_matches
/
(b_career_matches + 1)

---

## Surface Experience

surface_experience_ratio

Definición:

a_surface_matches
/
(b_surface_matches + 1)

---

# Elo Probability

Variables:

- a_expected_win_probability
- b_expected_win_probability
- expected_win_probability_delta

---

# Variable Objetivo

target

Definición:

1 = gana A

0 = pierde A

---

# Dataset Resultante

training_matches_v2.parquet

Filas:

66573

Columnas:

59

Target:

50 / 50

---

# Validaciones Realizadas

✅ 1 fila por partido

✅ Dataset balanceado

✅ Sin leakage Winner/Loser

✅ Variables Elo presentes

✅ Variables Ranking presentes

✅ Variables Surface presentes

✅ Variables Season presentes

✅ Variables Activity presentes

---

# Futuras Mejoras

## Recent Form

Variables previstas:

- last_5_matches_win_pct
- last_10_matches_win_pct
- last_20_matches_win_pct

- last_10_surface_win_pct
- last_20_surface_win_pct

Calculadas únicamente utilizando partidos anteriores al encuentro.

---

## Head-To-Head

Variables previstas:

- h2h_matches
- h2h_wins
- h2h_losses
- h2h_win_pct

- h2h_surface_matches
- h2h_surface_win_pct

Sin leakage temporal.

Usando exclusivamente partidos anteriores a la fecha del encuentro.

---

## Serve Strength

Variables previstas:

- service_games_won_pct
- service_points_won_pct
- first_serve_won_pct
- second_serve_won_pct

Career

Surface

Recent Form

---

## Return Strength

Variables previstas:

- break_points_converted_pct
- return_games_won_pct

Career

Surface

Recent Form

---

# Conclusión

training_matches_v2 constituye el dataset final de Machine Learning.

Integra:

✅ Ranking

✅ Rank Points

✅ Edad

✅ Altura

✅ Elo

✅ Surface Elo

✅ Elo Mix

✅ Career Stats

✅ Surface Stats

✅ Season Stats

✅ Actividad

✅ Experiencia

y está preparado para entrenar modelos LightGBM, XGBoost y CatBoost.

