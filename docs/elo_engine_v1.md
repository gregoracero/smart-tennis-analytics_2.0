
# ATP Elo Engine V1

## Estado

Production Ready

Version:
ATP Elo Engine V1

Status:
Validated

---

# Objetivo

Construir un motor Elo hist?rico ATP para generar variables predictivas destinadas a modelos de Machine Learning.

---

# Artefactos Generados

## player_lookup.parquet

Shape:

(11682, 2)

Columnas:

- player_id
- player_name

---

## player_elo_history.parquet

Shape:

(366065, 23)

---

## training_matches_with_ids.parquet

Dataset de entrenamiento enriquecido con ATP IDs.

---

## training_matches_with_elo.parquet

Shape:

(199940, 91)

Dataset final utilizado para entrenamiento.

---

# Elo General

INITIAL_ELO = 1500

BASE_ELO = 1500

HALF_LIFE_DAYS = 730

---

# Surface Elo

Superficies soportadas:

- Hard
- Clay
- Grass
- Carpet

SURFACE_HALF_LIFE_DAYS = 3650

---

# Dynamic K

Grand Slam: 40

Masters: 32

ATP 500: 28

ATP 250 y resto: 24

Reglas:

if inactivity > 180:
    k *= 2

elif inactivity > 70:
    k *= 1.5

---

# Resultados Elo

Partidos procesados:

366065

Jugadores ?nicos:

11683

---

# Overall Elo

Winner Mean:
1733.33

Loser Mean:
1666.45

Winner Std:
195.36

Loser Std:
165.16

Max Elo:
2639.43

---

# Distribuci?n Final Elo

Mean:
1500

Std:
81.47

Min:
1240.82

Max:
2471.47

---

# Surface Elo

Hard

Std:
73.47

Max:
2341.40

Clay

Std:
71.87

Max:
2177.86

Grass

Std:
65.73

Max:
2131.02

Carpet

Std:
55.27

Max:
1967.64

---

# Cobertura

## Player Lookup

Player A Missing IDs:
32

Player B Missing IDs:
32

Coverage:
99.97%

---

## Elo Merge

Training Rows:
200520

Covered Rows:
199940

Coverage:
99.71%

---

# Features Elo

## Elo

- elo_a
- elo_b
- delta_elo

## Surface Elo

- surface_elo_a
- surface_elo_b
- delta_surface_elo

## Inactivity

- days_inactive_a
- days_inactive_b
- delta_inactivity_days

## Surface Inactivity

- surface_days_inactive_a
- surface_days_inactive_b
- delta_surface_inactivity_days

## Experience

- matches_played_a
- matches_played_b
- delta_matches_played

## Surface Experience

- surface_matches_played_a
- surface_matches_played_b
- delta_surface_matches_played

---

# Se?al Predictiva Detectada

## delta_elo

target=0 -> -58.96

target=1 -> +58.96

Gap:
117.92

---

## delta_surface_elo

target=0 -> -51.89

target=1 -> +51.89

Gap:
103.78

---

## delta_matches_played

target=0 -> -35.64

target=1 -> +35.64

Gap:
71.28

---

## delta_surface_matches_played

target=0 -> -23.56

target=1 -> +23.56

Gap:
47.12

---

## delta_inactivity_days

target=0 -> +12.51

target=1 -> -12.51

---

## delta_surface_inactivity_days

target=0 -> +22.98

target=1 -> -22.98

---

# Benchmarking

## Logistic Regression Baseline

Dataset:
training_matches.parquet

Features:
64

Accuracy:
0.6679

ROC AUC:
0.7317

---

## Logistic Regression + Elo

Dataset:
training_matches_with_elo.parquet

Features:
83

Accuracy:
0.7420

ROC AUC:
0.8161

Improvement:

+0.0844 ROC AUC

---

## XGBoost Baseline

Dataset:
training_matches.parquet

ROC AUC:
0.7362

---

## XGBoost + Elo (Random Split)

Dataset:
training_matches_with_elo.parquet

Accuracy:
0.7988

ROC AUC:
0.8926

Improvement:

+0.1564 ROC AUC

---

## XGBoost + Elo (Temporal Validation)

Split Date:

2023-01-01

Train Rows:
35808

Test Rows:
10566

Accuracy:
0.7785

ROC AUC:
0.8739

---

# Conclusiones

## Elo

Validado.

Aporta se?al predictiva independiente.

---

## Surface Elo

Validado.

Aporta informaci?n complementaria al Elo general.

---

## Inactivity

Validado.

Es una de las variables m?s importantes del sistema.

---

## Temporal Validation

La ca?da entre validaci?n aleatoria y temporal es m?nima:

0.8926
?
0.8739

Delta:
-0.0187

Por tanto el modelo generaliza correctamente y no depende de fugas de informaci?n significativas.

---

# Pipeline Final

master_matches.parquet
    ->
build_player_lookup.py
    ->
player_lookup.parquet

master_matches.parquet
    ->
build_player_elo.py
    ->
player_elo_history.parquet

training_matches.parquet
    ->
build_training_with_player_ids.py
    ->
training_matches_with_ids.parquet

training_matches_with_ids.parquet
+
player_elo_history.parquet
    ->
build_training_with_elo.py
    ->
training_matches_with_elo.parquet

training_matches_with_elo.parquet
    ->
train_logistic.py

training_matches_with_elo.parquet
    ->
train_xgboost.py

training_matches_with_elo.parquet
    ->
train_xgboost_time_split.py

---

# Estado del Proyecto

Data Quality:
DONE

Elo Engine:
DONE

Feature Engineering:
DONE

Integration:
DONE

Model Validation:
DONE

Temporal Validation:
DONE

---

# Resultado Actual

Best Model:

XGBoost + Elo

Temporal Validation ROC AUC:

0.8739

Status:

Ready for Hyperparameter Optimization
Ready for Ensemble Models
Ready for Probability Calibration
