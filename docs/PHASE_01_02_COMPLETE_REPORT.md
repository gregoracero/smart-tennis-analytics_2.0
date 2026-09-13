# SMART TENNIS ANALYTICS 2.0

# PHASE 01

Leakage Free Baseline

---

## Objetivo

Construcción de un modelo predictivo ATP completamente libre de leakage utilizando únicamente información disponible antes del partido.

---

# Arquitectura Final

master_matches_v2

?

player_per_match_statistics_v2

?

elo_history_v2

?

player_snapshots_v2

?

training_matches_v2

?

Models

---

# Leakage Audit

## Problema Detectado

player_statistics_v2 generaba:

- career_win_pct
- surface_win_pct
- season_win_pct

mediante agregación final de temporada.

Estas variables incluían información futura.

---

## Solución

Snapshot reconstruido desde:

player_per_match_statistics_v2

mediante:

- cumcount()
- cumsum()
- shift(1)
- rolling()

---

# Variables Temporales

## Career

- career_matches_played_before
- career_wins_before
- career_win_pct_before

## Surface

- surface_matches_before
- surface_wins_before
- surface_win_pct_before

## Season

- season_matches_before
- season_wins_before
- season_win_pct_before

---

# Recent Form

## General

- last_5_matches_win_pct
- last_10_matches_win_pct
- last_20_matches_win_pct

## Surface

- last_5_surface_win_pct
- last_10_surface_win_pct
- last_20_surface_win_pct

---

# Activity

- days_since_last_match
- inactivity_adjusted

---

# Fatigue

- matches_last_30_days
- matches_last_90_days

- minutes_last_30_days
- minutes_last_90_days

---

# Quality Of Opposition

- avg_opponent_elo_last_5
- avg_opponent_elo_last_10
- avg_opponent_elo_last_20

- avg_opponent_surface_elo_last_5
- avg_opponent_surface_elo_last_10
- avg_opponent_surface_elo_last_20

- avg_opponent_mix_elo_last_5
- avg_opponent_mix_elo_last_10
- avg_opponent_mix_elo_last_20

---

# Recent Performance

performance_score

Definición:

performance_score =

win

-

expected_win_probability

Rolling:

- performance_last_5
- performance_last_10
- performance_last_20

---

# Head To Head

- h2h_matches_before
- h2h_wins_before
- h2h_win_pct_before

- h2h_surface_matches_before
- h2h_surface_wins_before
- h2h_surface_win_pct_before

---

# Auditoría Temporal

Validaciones realizadas:

## career_matches_before

Correctamente incremental.

Ejemplo:

0
1
2
3
4
...

---

## Elo

Validado:

- elo_before
- elo_after

Actualización temporal correcta.

---

## Expected Win Probability

Distribución:

MIN = 0.0022

MAX = 0.9978

Media = 0.50

Sin valores degenerados.

---

## Shuffle Test

TEST_10_SHUFFLE_TARGET

Accuracy ˜ 50%

AUC ˜ 0.50

Conclusión:

No existe evidencia de leakage significativo.

---

# Baseline Oficial

TEST_09_ALL_FEATURES

Accuracy = 75.46%

AUC = 0.8379

LogLoss = 0.5052

---

# Análisis de Ablación

## AUC Drop

REMOVE_RECENT_FORM

AUC DROP = 0.1068

---

REMOVE_ACTIVITY

AUC DROP = 0.0552

---

REMOVE_FATIGUE

AUC DROP = 0.0151

---

REMOVE_SEASON

AUC DROP = 0.0014

---

REMOVE_H2H

AUC DROP = 0.0011

---

REMOVE_ELO

AUC DROP = 0.0010

---

REMOVE_SURFACE

AUC DROP = 0.0009

---

REMOVE_PERFORMANCE

AUC DROP = 0.0006

---

REMOVE_QUALITY

AUC DROP = -0.0001

---

# Ranking Real de Familias

Tier 1

1. Recent Form
2. Activity

Tier 2

3. Fatigue

Tier 3

4. Season
5. Elo
6. Surface
7. Performance
8. H2H
9. Quality Of Opposition

---

# Feature Importance LightGBM

Top Variables

1. days_since_last_match_delta

2. avg_opponent_elo_last_10_delta

3. a_minutes_last_30_days

4. avg_opponent_surface_elo_last_10_delta

5. minutes_last_30_days_delta

6. age_delta

7. performance_last_5_delta

8. performance_last_10_delta

9. season_win_pct_before_delta

10. rank_points_delta

---

# Model Benchmark

## LightGBM

Accuracy = 75.46%

AUC = 0.8379

LogLoss = 0.5052

---

## CatBoost

Accuracy = 76.01%

AUC = 0.8411

LogLoss = 0.4953

---

## XGBoost

Accuracy = 75.67%

AUC = 0.8384

LogLoss = 0.5133

---

# Ranking Final

?? CatBoost

Accuracy = 76.01%

AUC = 0.8411

LogLoss = 0.4953

---

?? XGBoost

Accuracy = 75.67%

AUC = 0.8384

LogLoss = 0.5133

---

?? LightGBM

Accuracy = 75.46%

AUC = 0.8379

LogLoss = 0.5052

---

# Conclusiones

Confirmado:

- Pipeline libre de leakage.
- Recent Form es la señal dominante.
- Activity es la segunda señal dominante.
- Fatigue aporta valor incremental.
- CatBoost supera a LightGBM y XGBoost.

No confirmado:

- H2H.
- Quality Of Opposition.
- Recent Performance.

---

# Estado del Proyecto

PHASE 01

? COMPLETADA

Leakage-Free Baseline

---

PHASE 02

? Benchmarking completado

Winner:

CatBoost

Accuracy = 76.01%

AUC = 0.8411

---

# Próxima Fase

PHASE 03

Model Optimization

Objetivos:

1. CatBoost Feature Importance.

2. CatBoost Hyperparameter Search.

3. Calibration Analysis.

4. Probability Reliability.

5. Betting Simulation Engine.

6. Value Betting Framework.

7. ROI Backtesting.

