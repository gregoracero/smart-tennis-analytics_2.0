# PHASE 03A

# DAILY OPERATIONS

## Objetivo

Transformar Smart Tennis Analytics desde una plataforma de entrenamiento histórico hacia una plataforma operacional diaria.

---

# Daily Sources

## TML ATP Ongoing

Actualización diaria ATP Main Draw.

---

## TML ATP Qualifying Ongoing

Actualización diaria ATP Qualifying.

---

## TML Challenger Ongoing

Actualización diaria Challenger Tour.

---

# Daily Pipeline

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

# Objetivos Principales

Actualizar diariamente:

- Recent Form
- Activity
- Fatigue
- Elo

---

# Outputs

## Dashboard

- Rankings
- Current Elo
- Recent Form
- Activity
- Fatigue
- Surface Statistics

## Prediction

- Match Winner Probability
- Expected Edge
- Confidence Score

---

# Dependencias

Fuente crítica:

TML

Fuente opcional:

Tennis Data UK

---

# Decisiones

No se recalculará training_matches_v2 diariamente.

No se reentrenará el modelo diariamente.

