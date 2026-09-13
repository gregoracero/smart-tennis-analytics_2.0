# PHASE 03B

# MODEL WITHOUT ODDS

## Objetivo

Modelo generalista independiente de cuotas.

---

# Entradas

## ATP Main Draw

TML ATP

## ATP Qualifying

TML ATP Qualifying

## Challenger

TML Challenger

---

# Variables

## Recent Form

- last_5_matches_win_pct
- last_10_matches_win_pct
- last_20_matches_win_pct

## Activity

- days_since_last_match
- inactivity_adjusted

## Fatigue

- matches_last_30_days
- matches_last_90_days

- minutes_last_30_days
- minutes_last_90_days

## Elo

- elo_overall_before
- elo_surface_before
- elo_mix_before

## Ranking

- rank
- rank_points

---

# Excluido

- Odds
- Bet365
- Tennis Data UK

---

# Dataset

training_matches_without_odds_v1

---

# Modelo

catboost_without_odds_v1

---

# Cobertura

- ATP Main Draw
- ATP Qualifying
- Challenger

---

# Ventajas

- Independiente de cuotas.
- Actualización diaria mediante TML.
- Cobertura completa ATP.
- Reutilizable para futuros circuitos.

