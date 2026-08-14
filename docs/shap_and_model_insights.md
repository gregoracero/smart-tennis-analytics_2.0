
# SHAP Analysis & Model Insights

## Objetivo

Comprender qu? ha aprendido realmente el modelo XGBoost despu?s de:

- Integraci?n de Elo
- Integraci?n de Surface Elo
- Integraci?n de Inactivity Features
- Validaci?n temporal
- Ablation Study

---

# Modelo Analizado

Modelo:

XGBoost + Elo + Inactivity + ATP Features

Validaci?n:

Temporal

Train:

< 2023-01-01

Test:

>= 2023-01-01

Resultado:

ROC AUC = 0.8739

Accuracy = 0.7785

---

# Top SHAP Features

| Rank | Feature | Mean Abs SHAP |
|--------|--------|--------:|
| 1 | delta_surface_inactivity_days | 0.5146 |
| 2 | surface_days_inactive_a | 0.4193 |
| 3 | surface_days_inactive_b | 0.3857 |
| 4 | delta_elo | 0.2889 |
| 5 | delta_inactivity_days | 0.2799 |
| 6 | days_inactive_b | 0.1370 |
| 7 | delta_rank_points | 0.1233 |
| 8 | delta_age | 0.1192 |
| 9 | delta_surface_elo | 0.1172 |
| 10 | days_inactive_a | 0.1151 |

---

# Principal Hallazgo

La variable m?s importante del modelo NO es Elo.

La variable m?s importante del modelo es:

delta_surface_inactivity_days

seguida por:

- surface_days_inactive_a
- surface_days_inactive_b

Por tanto:

La preparaci?n competitiva reciente es la dimensi?n predictiva m?s fuerte del sistema.

---

# Elo Analysis

Variables relevantes:

- delta_elo
- delta_surface_elo

Observaciones SHAP:

- Relaci?n extremadamente limpia.
- Comportamiento pr?cticamente mon?tono.
- M?s Elo implica mayor probabilidad de victoria.
- Menos Elo implica menor probabilidad de victoria.

No se observan efectos extra?os ni discontinuidades.

Conclusi?n:

Elo est? siendo utilizado exactamente de la forma esperada por la teor?a Elo.

---

# Surface Elo Analysis

Variables:

- delta_surface_elo

Observaciones:

- Comportamiento similar a delta_elo.
- Relaci?n monot?nica.
- Se?al estable.
- Aporta informaci?n distinta y ?til.

Conclusi?n:

La decisi?n de construir Surface Elo queda completamente validada.

---

# Inactivity Analysis

Variables:

- delta_inactivity_days
- delta_surface_inactivity_days

Observaciones:

Las curvas SHAP muestran una estructura escalonada.

El modelo parece responder m?s a:

?Qui?n est? m?s activo?

que a:

?Cu?ntos d?as exactos de diferencia existen?

Interpretaci?n:

La se?al parece comportarse m?s como una ventaja/desventaja competitiva que como una variable continua.

---

# Match Readiness Hypothesis

Los resultados sugieren que las variables de inactividad no est?n midiendo lesiones.

Est?n midiendo:

- ritmo competitivo
- frecuencia reciente de competici?n
- preparaci?n para competir
- estado competitivo actual

Por tanto se propone el concepto:

Match Readiness

como dimensi?n independiente dentro del modelo.

---

# Correlation Analysis

Correlaciones observadas:

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

Conclusi?n:

Elo e Inactividad son pr?cticamente independientes.

Capturan dimensiones distintas del rendimiento.

---

# Active Players Validation

Objetivo:

Comprobar que el modelo no depende de jugadores lesionados o retirados.

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

Conclusi?n

El modelo conserva gran parte de su capacidad predictiva incluso cuando ambos jugadores est?n plenamente activos.

Esto demuestra que:

- Elo sigue funcionando.
- ATP Stats siguen funcionando.
- El modelo no depende de lesiones extremas.
- El modelo no depende de retirados.

---

# Conceptual Architecture

El modelo parece estructurarse en tres bloques principales.

## Layer 1

Player Strength

Variables:

- Elo
- Surface Elo

Representa:

Fuerza hist?rica acumulada.

---

## Layer 2

Match Readiness

Variables:

- days_inactive
- surface_days_inactive

Representa:

Estado competitivo actual.

---

## Layer 3

Tennis Skill

Variables:

- Service Statistics
- Return Statistics
- Win Rate
- Ranking
- Ranking Points

Representa:

Calidad t?cnica y rendimiento reciente.

---

# Modelo Conceptual Final

Prediction

=

Player Strength

+

Match Readiness

+

Tennis Skill

---

# Key Discovery

La hip?tesis inicial del proyecto era:

Ranking
+
ATP Statistics

=

Predicci?n

La evidencia obtenida muestra una estructura diferente:

Player Strength
+
Match Readiness
+
ATP Statistics

=

Predicci?n significativamente mejor

---

# ATP Elo Engine V1

Estado:

VALIDATED

---

# Match Readiness Engine V1

Estado:

DISCOVERED

Pendiente de investigaci?n espec?fica.

---

# Pr?xima Fase Recomendada

1. Hyperparameter Optimization
2. Binary Readiness Features
3. SHAP Interactions
4. Probability Calibration
5. Match Prediction Service
6. Ensemble Models
7. Readiness Engine V1

---

# Executive Summary

La principal conclusi?n del proyecto es que la capacidad predictiva no proviene ?nicamente del Elo.

El rendimiento surge de la combinaci?n de dos dimensiones complementarias:

- Fuerza hist?rica del jugador (Elo)
- Estado competitivo actual (Match Readiness)

Las estad?sticas ATP refinan la predicci?n, pero no constituyen el n?cleo principal del sistema.

Resultado final:

ROC AUC = 0.8739

Validaci?n temporal.

Estado:

READY FOR PHASE 2
