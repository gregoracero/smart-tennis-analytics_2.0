
# ATP Elo Engine V1

## Estado

Production Ready

Versi?n: ATP Elo Engine V1

Estado actual:

- Frozen
- Ready for Feature Engineering
- Ready for Model Training

---

# Objetivo

Construir un motor Elo hist?rico ATP para generar variables predictivas para modelos de Machine Learning.

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

Dataset de entrenamiento enriquecido con IDs ATP.

---

## training_matches_with_elo.parquet

Shape:

(199940, 112)

Dataset final utilizado para entrenamiento con variables Elo.

---

# Elo General

Configuraci?n:

INITIAL_ELO = 1500

BASE_ELO = 1500

HALF_LIFE_DAYS = 730

---

# Surface Elo

Superficies:

- Hard
- Clay
- Grass
- Carpet

Configuraci?n:

SURFACE_HALF_LIFE_DAYS = 3650

---

# Dynamic K

Grand Slam:
40

Masters:
32

ATP 500:
28

ATP 250 y resto:
24

Incrementos:

if inactivity > 180:
    k *= 2

elif inactivity > 70:
    k *= 1.5

---

# Variables Elo

## Overall Elo

- winner_elo_before
- loser_elo_before

## Surface Elo

- winner_surface_elo_before
- loser_surface_elo_before

## Actividad

- winner_days_inactive
- loser_days_inactive

- winner_surface_days_inactive
- loser_surface_days_inactive

## Experiencia

- winner_matches_played
- loser_matches_played

- winner_surface_matches_played
- loser_surface_matches_played

---

# Resultados Elo

## Partidos Procesados

366065

## Jugadores ?nicos

11683

---

# Overall Elo

Winner Mean: 1733.33

Loser Mean: 1666.45

Winner Std: 195.36

Loser Std: 165.16

Max Elo: 2639.43

---

# Surface Elo

Hard

Std: 73.47

Max: 2341.40

Clay

Std: 71.87

Max: 2177.86

Grass

Std: 65.73

Max: 2131.02

Carpet

Std: 55.27

Max: 1967.64

---

# Cobertura Lookup

Missing Player A IDs: 32

Missing Player B IDs: 32

Cobertura: 99.97%

---

# Cobertura Elo Merge

Training Rows:
200520

Direct Matches:
99970

Reverse Matches:
99970

Total Covered:
199940

Coverage:
99.71%

---

# Nuevas Features

## Elo

- elo_a
- elo_b
- delta_elo

## Surface Elo

- surface_elo_a
- surface_elo_b
- delta_surface_elo

## Experience

- matches_played_a
- matches_played_b
- delta_matches_played

## Surface Experience

- surface_matches_played_a
- surface_matches_played_b
- delta_surface_matches_played

## Inactivity

- days_inactive_a
- days_inactive_b
- delta_inactivity_days

## Surface Inactivity

- surface_days_inactive_a
- surface_days_inactive_b
- delta_surface_inactivity_days

---

# Se?al Inicial Detectada

delta_elo

target=0 -> -58.96

target=1 -> +58.96

Gap = 117.92

---

delta_surface_elo

target=0 -> -51.89

target=1 -> +51.89

Gap = 103.78

---

delta_matches_played

target=0 -> -35.64

target=1 -> +35.64

Gap = 71.28

---

delta_surface_matches_played

target=0 -> -23.56

target=1 -> +23.56

Gap = 47.12

---

delta_inactivity_days

target=0 -> +12.51

target=1 -> -12.51

---

delta_surface_inactivity_days

target=0 -> +22.98

target=1 -> -22.98

---

# Pipeline

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

---

# Estado del Proyecto

Elo Engine: READY

Integration: READY

Training Dataset: READY

Training Models: PENDING

---

# Pr?ximo Sprint

Modelos:

- Logistic Regression
- XGBoost

Comparativas:

- Baseline
- Baseline + Elo

Objetivo:

Superar ROC AUC = 0.7362
