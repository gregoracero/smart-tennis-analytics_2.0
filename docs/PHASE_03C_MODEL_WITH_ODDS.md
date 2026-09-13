# PHASE 03C

# MODEL WITH ODDS

## Objetivo

Modelo especializado para ATP Main Draw utilizando cuotas históricas.

---

# Entradas

## ATP Main Draw

TML ATP

## Odds

Tennis Data UK

Bet365

---

# Variables

Todas las variables del modelo without odds.

+

Variables derivadas de cuotas.

---

# Dataset

training_matches_with_odds_v1

---

# Modelo

catboost_with_odds_v1

---

# Benchmark Actual

Accuracy = 76.01%

AUC = 0.8411

LogLoss = 0.4953

---

# Cobertura

ATP Main Draw

---

# Dependencias

TML

+

Tennis Data UK

---

# Objetivo Futuro

Superar:

AUC > 0.845

mediante:

- Hyperparameter Search
- Calibration
- Betting Simulation
- ROI Backtesting

---

# Uso Esperado

Cuando existan cuotas disponibles para el partido.

