# Smart Tennis Analytics 2.0

# Phase 01 - Leakage Free Baseline

## Objetivo

Construcción de un baseline predictivo libre de data leakage para predicción de partidos ATP utilizando:

- Elo
- Surface Elo
- Ranking
- Career Statistics
- Surface Statistics
- Season Statistics
- Activity
- Fatigue
- Recent Form
- Quality Of Opposition
- Head To Head

---

# Arquitectura

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
LightGBM

---

# Hallazgo crítico

## Data Leakage detectado

Se detectó que:

player_statistics_v2

generaba estadísticas agregadas finales:

- career_win_pct
- surface_win_pct
- season_win_pct

que posteriormente eran incorporadas al snapshot.

Esto permitía que partidos antiguos vieran información futura.

---

## Solución aplicada

El snapshot dejó de utilizar:

player_statistics_v2

y pasó a construir métricas temporales directamente desde:

player_per_match_statistics_v2

mediante:

- shift(1)
- cumcount()
- cumsum()
- rolling()

---

# Features temporales incorporadas

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

# Performance

performance_score

Definición:

performance_score =
win - expected_win_probability

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

# Validación Leakage

## Shuffle Target Test

Resultados:

Accuracy ˜ 50%
AUC ˜ 0.50

Conclusión:

No existe evidencia de leakage significativo.

---

# Resultados finales

## Mejor modelo

TEST_09_ALL_FEATURES

Resultados:

Accuracy = 75.46%

AUC = 0.838

LogLoss = 0.505

---

# Resultados de experimentación

## Elo

TEST_01_ELO_ONLY

Accuracy = 62.90%
AUC = 0.690

---

## Ranking

TEST_02_ELO_RANK

Accuracy = 63.88%
AUC = 0.700

Aporta valor marginal.

---

## Career

TEST_04_CAREER

Accuracy = 64.46%
AUC = 0.704

---

## Surface

TEST_05_SURFACE

Accuracy = 64.87%
AUC = 0.709

---

## Season

TEST_06_SEASON

Accuracy = 66.06%
AUC = 0.725

---

## Activity

TEST_07_ACTIVITY

Accuracy = 74.18%
AUC = 0.814

Feature family más potente del proyecto.

---

## Fatigue

TEST_06C_FATIGUE_ONLY

Accuracy = 65.10%
AUC = 0.704

Buena señal pero muy correlacionada con Activity.

---

## Recent Form

TEST_06B_RECENT_FORM

Accuracy = 65.21%
AUC = 0.711

Aporta menos de lo esperado.

---

## Quality Of Opposition

TEST_06G_QUALITY_OF_OPPOSITION

Accuracy = 64.27%
AUC = 0.702

Poca señal incremental.

---

## Recent Performance

TEST_06H_RECENT_PERFORMANCE

Accuracy = 63.46%
AUC = 0.692

Poca señal incremental.

---

## Head To Head

TEST_06J_H2H

Accuracy = 63.59%
AUC = 0.693

Valor predictivo limitado.

---

# Ranking aproximado de importancia

1. Activity

2. Elo Surface

3. Elo Mix

4. Season Before

5. Ranking

6. Surface Statistics

7. Fatigue

8. Recent Form

9. Quality Of Opposition

10. H2H

11. Recent Performance

---

# Conclusiones

## Confirmado

- Elo aporta señal real.
- Surface Elo aporta señal real.
- Activity es extremadamente predictiva.
- El leakage fue eliminado correctamente.
- El modelo mantiene rendimiento elevado tras la corrección.

## No confirmado

- H2H aporta poco.
- Quality Of Opposition aporta poco.
- Recent Performance aporta poco.

---

# Baseline Oficial

Accuracy = 75.46%

AUC = 0.838

LogLoss = 0.505

---

# Phase 02

Objetivos prioritarios:

1. Ablation Analysis.
2. Feature Importance.
3. Hyperparameter Optimization.
4. CatBoost Benchmark.
5. XGBoost Benchmark.
6. Ensemble Comparison.
7. Calibration Analysis.
8. Betting Simulation Engine.

