# PHASE 04

# IMPLEMENTATION PLAN

## Objetivo

Convertir Smart Tennis Analytics en una plataforma operacional preparada para actualización diaria y generación de predicciones.

---

# Estado Actual

## Model Leader

catboost_with_odds_v1

Accuracy = 76.01%

AUC = 0.8411

LogLoss = 0.4953

---

# Arquitectura Actual

INGESTION

TML
+
TennisData UK

?

player_mapping_v2

?

tournament_mapping_v2

?

match_link_audit_v2

?

master_matches_v2

---

ANALYTICS

?

player_per_match_statistics_v2

?

player_statistics_v2

?

tournament_statistics_v2

?

elo_history_v2

---

PREDICTION

?

player_snapshots_v2

?

training_matches_v2

?

models

---

# Nueva Arquitectura

Separación completa entre:

1. TRAINING PIPELINE
2. DAILY OPERATIONS PIPELINE

---

# TRAINING PIPELINE

Frecuencia:

Mensual o Trimestral

Objetivo:

Entrenamiento y backtesting.

Fuentes:

ATP

ATP Qualifying

Challenger

+

TennisData UK

+

Odds

---

Outputs

training_matches_with_odds_v1

training_matches_without_odds_v1

catboost_with_odds_v1

catboost_without_odds_v1

---

# DAILY OPERATIONS PIPELINE

Frecuencia:

Diaria

Objetivo:

Actualizar estado de jugadores y generar predicciones.

Fuentes:

TML ATP Ongoing

TML ATP Qualifying Ongoing

TML Challenger Ongoing

---

Pipeline

TML Ongoing

?

master_matches_v2

?

player_per_match_statistics_v2

?

player_statistics_v2

?

tournament_statistics_v2

?

elo_history_v2

?

player_snapshots_v2

?

daily_predictions

---

# MODEL WITH ODDS

Dataset

training_matches_with_odds_v1

Entrenamiento

ATP Main Draw

+

Odds

Uso

ATP Main Draw

Dependencias

TML

+

TennisData UK

---

# MODEL WITHOUT ODDS

Dataset

training_matches_without_odds_v1

Entrenamiento

ATP Main Draw

ATP Qualifying

Challenger

Uso

ATP Main Draw

ATP Qualifying

Challenger

Dependencias

Solo TML

---

# Hallazgos de Modelado

## Variables Más Importantes

1. Recent Form

2. Activity

3. Fatigue

---

Variables secundarias

- Elo

- Surface

- Season

- H2H

- Performance

- Quality Of Opposition

---

# Prioridad Operacional

Mantener actualizado diariamente:

- Recent Form

- Activity

- Fatigue

- Elo

porque son las familias más predictivas del sistema.

---

# Sprint 4.1

ATP Qualifying Historical Load

Objetivos

- Ingestar histórico ATP Qualifying
- Actualizar mappings
- Actualizar linking
- Integrar en master_matches_v3

Entregable

master_matches_v3

---

# Sprint 4.2

Challenger Historical Load

Objetivos

- Ingestar histórico Challenger
- Actualizar mappings
- Actualizar linking
- Integrar en master_matches_v3

Entregable

master_matches_v3 completo

---

# Sprint 4.3

Universal Dataset

Objetivos

Generar:

training_matches_without_odds_v1

Modelo:

catboost_without_odds_v1

---

# Sprint 4.4

Daily Update Pipeline

Scripts objetivo

update_master_matches.py

update_player_per_match_statistics.py

update_player_statistics.py

update_tournament_statistics.py

update_elo_history.py

update_player_snapshots.py

---

# Sprint 4.5

Prediction Layer

generate_predictions.py

Outputs

match_date

player_a

player_b

win_probability_a

win_probability_b

predicted_winner

model_used

---

# Dashboard

Métricas disponibles

- Current Elo

- Surface Elo

- Recent Form

- Activity

- Fatigue

- Surface Performance

- Season Performance

- Career Performance

---

# Success Criteria

1. Actualización diaria mediante TML.

2. Sin dependencia operacional de TennisData UK.

3. Modelo without_odds operativo.

4. Modelo with_odds operativo.

5. Predicciones automáticas diarias.

6. Base preparada para Dashboard.

7. Base preparada para Betting Engine.

---

# Visión Final

DAILY SOURCES

ATP Ongoing

ATP Qualifying Ongoing

Challenger Ongoing

?

master_matches_v3

?

player_per_match_statistics_v3

?

elo_history_v3

?

player_snapshots_v3

?

model_without_odds

?

daily_predictions

---

TRAINING SOURCES

ATP

ATP Qualifying

Challenger

+

TennisData UK

+

Odds

?

training_matches_with_odds_v1

?

model_with_odds

