
# ATP Elo Engine V1 - Experimental Analysis

## Objetivo

Validar el impacto real de:

- Elo
- Surface Elo
- Inactividad
- Ranking
- Estad?sticas ATP

sobre la capacidad predictiva del modelo.

Todos los experimentos se ejecutaron utilizando:

- Surface = Hard
- Time Split = 2023-01-01
- XGBoost
- Validaci?n temporal

---

# Benchmark Principal

## XGBoost Baseline

Dataset:

training_matches.parquet

ROC AUC:

0.7362

---

## XGBoost + Elo + Inactivity

Dataset:

training_matches_with_elo.parquet

ROC AUC:

0.8739

Mejora:

+0.1377

---

# Ablation Study

## Elo Only

Features:

- elo_a
- elo_b
- surface_elo_a
- surface_elo_b
- delta_elo
- delta_surface_elo

Resultado:

ROC AUC = 0.7211

---

## Inactivity Only

Features:

- days_inactive_a
- days_inactive_b
- delta_inactivity_days

- surface_days_inactive_a
- surface_days_inactive_b
- delta_surface_inactivity_days

Resultado:

ROC AUC = 0.7814

---

## Ranking Only

Features:

- rank_a
- rank_b
- delta_rank

- rank_points_a
- rank_points_b
- delta_rank_points

Resultado:

ROC AUC = 0.6920

---

## Win Rate Only

Resultado:

ROC AUC = 0.6494

---

## Service Only

Resultado:

ROC AUC = 0.6299

---

## Return Only

Resultado:

ROC AUC = 0.5600

---

## All Without Elo

ROC AUC = 0.8708

P?rdida respecto al modelo final:

-0.0031

---

## All Without Inactivity

ROC AUC = 0.8335

P?rdida respecto al modelo final:

-0.0404

---

## All Features

ROC AUC = 0.8739

---

# Conclusiones Ablation Study

Ranking de capacidad predictiva individual:

1. Inactivity Only     0.7814
2. Elo Only            0.7211
3. Ranking Only        0.6920
4. Win Rate Only       0.6494
5. Service Only        0.6299
6. Return Only         0.5600

Observaciones:

- Inactivity supera al baseline completo original.
- Elo aporta se?al independiente.
- Elo e Inactivity son complementarios.
- Ranking explica parte de la se?al Elo.
- Service y Return son m?s ?tiles en combinaci?n que individualmente.

---

# Inactivity Analysis

## Distribuci?n General

days_inactive

P50:

1 d?a

P75:

13 d?as

P90:

28 d?as

P95:

50 d?as

P99:

176 d?as

M?ximo:

2457 d?as

---

## Distribuci?n Surface

surface_days_inactive

P50:

1 d?a

P75:

14 d?as

P90:

55 d?as

P95:

135 d?as

P99:

357 d?as

M?ximo:

2646 d?as

---

# Removal Test

Filtro:

days_inactive <= 540

Resultado:

Original:

199940

Filtrado:

199388

Eliminados:

552

Porcentaje:

0.28%

Conclusi?n:

La se?al de inactividad NO proviene de jugadores retirados o casos extremos.

---

# General vs Surface Inactivity

## General Inactivity Only

Features:

- days_inactive_a
- days_inactive_b
- delta_inactivity_days

ROC AUC:

0.7980

---

## Surface Inactivity Only

Features:

- surface_days_inactive_a
- surface_days_inactive_b
- delta_surface_inactivity_days

ROC AUC:

0.8004

---

## Conclusi?n

Ambas se?ales tienen pr?cticamente el mismo valor predictivo.

La importancia de la inactividad no est? limitada a la superficie.

---

# Elo Plus Inactivity

Features:

- Elo
- Surface Elo
- Inactivity
- Surface Inactivity

Total:

12 variables

Resultado:

ROC AUC = 0.8576

Comparaci?n:

Modelo Completo:

0.8739

Diferencia:

0.0163

Conclusi?n:

S?lo 12 variables explican aproximadamente el 98% del rendimiento del modelo.

---

# Inactivity Capping Test

L?mites:

days_inactive:

90

surface_days_inactive:

180

Resultado:

ROC AUC = 0.7989

Comparaci?n:

Inactivity Original:

0.7814

Conclusi?n:

Los valores extremos no son responsables de la capacidad predictiva observada.

---

# Delta Inactivity Buckets

Variable:

delta_inactivity_days

Interpretaci?n:

delta < 0

player_a m?s activo

delta > 0

player_b m?s activo

Resultados:

< -90

count = 3869

win rate player_a = 83.38%

---

-90 a -30

count = 8576

win rate player_a = 81.44%

---

-30 a -7

count = 41080

win rate player_a = 81.65%

---

-7 a 7

count = 107194

win rate player_a = 45.50%

---

7 a 30

count = 26873

win rate player_a = 19.44%

---

30 a 90

count = 8556

win rate player_a = 18.52%

---

> 90

count = 3792

win rate player_a = 16.64%

---

Conclusi?n

La diferencia de actividad competitiva es extremadamente predictiva.

Existe una relaci?n monot?nica clara:

M?s activo -> mayor probabilidad de victoria.

---

# Correlation Analysis

Variables:

- delta_elo
- delta_surface_elo
- delta_inactivity_days
- delta_surface_inactivity_days
- delta_rank

Resultados clave:

delta_elo ? delta_surface_elo

0.877

---

delta_inactivity ? delta_surface_inactivity

0.678

---

delta_elo ? delta_inactivity

-0.119

---

delta_surface_elo ? delta_surface_inactivity

-0.108

---

Conclusi?n

Elo e Inactivity son pr?cticamente independientes.

Miden dimensiones diferentes:

Elo:

- Fuerza hist?rica

Inactivity:

- Estado competitivo actual

---

# Active Players Validation

Objetivo:

Comprobar que el modelo no depende de lesiones o inactividad extrema.

---

## Ambos jugadores activos <= 14 d?as

Rows:

70682

ROC AUC:

0.8529

---

## Ambos jugadores activos <= 7 d?as

Rows:

53252

ROC AUC:

0.8328

---

# Conclusi?n Principal

Incluso cuando ambos jugadores est?n activos y compitiendo regularmente:

ROC AUC:

0.83 - 0.85

La capacidad predictiva sigue siendo muy elevada.

Por tanto:

- El modelo no depende de detectar lesionados.
- El modelo no depende de retirados.
- Elo mantiene su valor predictivo.
- Las estad?sticas ATP mantienen valor predictivo.
- Inactividad representa "match readiness" m?s que lesi?n.

---

# Modelo Conceptual Final

El sistema parece organizarse en tres capas:

## 1. Strength

Variables:

- Elo
- Surface Elo

Representan:

Fuerza hist?rica.

---

## 2. Readiness

Variables:

- days_inactive
- surface_days_inactive

Representan:

Estado competitivo actual.

---

## 3. Tennis Skills

Variables:

- Service
- Return
- Win Rate
- Ranking

Representan:

Caracter?sticas t?cnicas y rendimiento reciente.

---

# Conclusi?n Final

La combinaci?n:

Strength
+
Readiness
+
Tennis Skills

produce:

ROC AUC = 0.8739

en validaci?n temporal.

El hallazgo m?s importante del estudio es que la preparaci?n competitiva (Inactivity / Match Readiness) aporta una se?al predictiva tan relevante como la fuerza hist?rica medida mediante Elo.
