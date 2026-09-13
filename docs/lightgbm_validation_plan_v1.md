# LIGHTGBM V1 - PLAN DE VALIDACION

## Objetivo

Validar:

- Calidad real del modelo
- Posible leakage
- Valor incremental de cada grupo de features

---

# TEST 1

BASELINE ELO ONLY

## Features

elo_delta

surface_elo_delta

elo_mix_delta

## Objetivo

Medir el poder predictivo puro del Elo.

## Resultado Esperado

Accuracy:

65% - 72%

AUC:

0.70 - 0.80

---

# TEST 2

ELO + RANKING

## Features

elo_delta

surface_elo_delta

elo_mix_delta

rank_delta

rank_points_delta

## Objetivo

Medir el valor incremental del ranking.

## Resultado Esperado

Mejora moderada frente a Elo Only.

---

# TEST 3

BIOGRAPHICS

## Features

Test 2 +

age_delta

height_delta

## Objetivo

Comprobar si edad y altura aportan señal.

## Resultado Esperado

Mejora pequeña.

---

# TEST 4

CAREER STATS

## Features

Test 3 +

career_win_pct_delta

career_matches_delta

experience_ratio

## Objetivo

Validar experiencia y rendimiento histórico.

## Resultado Esperado

Mejora moderada.

---

# TEST 5

SURFACE STATS

## Features

Test 4 +

surface_win_pct_delta

surface_matches_delta

surface_experience_ratio

surface_specialization_delta

## Objetivo

Validar especialización por superficie.

## Resultado Esperado

Mejora importante.

---

# TEST 6

SEASON STATS

## Features

Test 5 +

season_win_pct_delta

season_matches_delta

a_season_win_pct

b_season_win_pct

## Objetivo

Detectar posible leakage.

## Interpretación

Si el salto es enorme:

Ejemplo

Accuracy

69%
↓
78%

Puede existir leakage temporal.

---

# TEST 7

INACTIVITY

## Features

Test 6 +

days_since_last_match_delta

a_inactivity_adjusted

b_inactivity_adjusted

## Objetivo

Medir impacto de lesiones y parones.

---

# TEST 8

EXPECTED WIN PROBABILITY

## Features

Test 7 +

expected_win_probability_delta

## Objetivo

Comprobar si la probabilidad Elo aporta valor adicional.

---

# TEST 9

REMOVER SEASON STATS

## Features

Todos menos:

season_win_pct_delta

season_matches_delta

a_season_win_pct

b_season_win_pct

## Objetivo

Buscar leakage.

## Interpretación

Si Accuracy cae mucho:

78%
↓
69%

Probable contaminación.

---

# TEST 10

REMOVER CAREER STATS

## Features

Todos menos:

career_win_pct_delta

career_matches_delta

experience_ratio

## Objetivo

Medir valor real del histórico.

---

# TEST 11

REMOVER SURFACE STATS

## Features

Todos menos:

surface_win_pct_delta

surface_matches_delta

surface_experience_ratio

surface_specialization_delta

## Objetivo

Validar la importancia de la superficie.

---

# TEST 12

SHUFFLE TEST

## Procedimiento

Randomizar target.

Ejemplo:

target = np.random.randint(
    0,
    2,
    size=len(df)
)

## Resultado Esperado

Accuracy ≈ 50%

AUC ≈ 0.50

## Objetivo

Detectar leakage extremo.

---

# TEST 13

TEMPORAL SPLIT STRICT

## Train

2001-2022

## Test

2023-2026

## Objetivo

Simular producción real.

## Resultado Esperado

Accuracy ligeramente inferior.

---

# TEST 14

FEATURE IMPORTANCE AUDIT

## Objetivo

Revisar Top 20 variables.

Variables esperadas:

elo_delta

surface_elo_delta

elo_mix_delta

rank_delta

rank_points_delta

surface_win_pct_delta

days_since_last_match_delta

## Señal de alerta

Variables inesperadas dominan:

season_win_pct_delta

season_matches_delta

career_win_pct_delta

con ventaja excesiva.

---

# TEST 15

LEAKAGE AUDIT

Revisar:

build_player_statistics_v2

Confirmar:

career_win_pct

surface_win_pct

season_win_pct

se calculan exclusivamente con datos anteriores al partido.

Nunca con resultados futuros.

---

# CRITERIO DE APROBACION

Modelo aceptado si:

Accuracy > 68%

AUC > 0.75

LogLoss < 0.60

Sin evidencia de leakage.

---

# FASE SIGUIENTE

Una vez validado:

✅ Recent Form

- last_5_matches_win_pct
- last_10_matches_win_pct
- last_20_matches_win_pct

✅ H2H

- h2h_matches
- h2h_win_pct
- h2h_surface_win_pct

✅ Serve Metrics

✅ Return Metrics

