# Prediction Engine

## Objetivo

Convertir métricas y señales en una estimación
de probabilidad de victoria para cada jugador.

## Inputs

data/parquet/rolling_metrics.parquet

data/parquet/signal_engine.parquet

data/parquet/h2h_surface.parquet

data/parquet/h2h_overall.parquet

## Output

data/parquet/prediction_features.parquet

## Factores considerados

### Forma reciente

Derivada de:

- win_pct
- matches
- recent performance

### Servicio

Derivado de:

- first_serve_won
- second_serve_won

### Resistencia

Derivada de:

- avg_minutes

### Protección al break

Derivada de:

- bp_saved

### Señales

- surface_form
- fatigue_risk
- break_risk
- tiebreak_probability

### Head To Head

- h2h_surface
- h2h_overall

## Prediction Score

Cada jugador recibe un score agregado
a partir de sus indicadores recientes.

Este score servirá como base para:

- Match Comparator
- Prediction Dashboard
- Value Betting Engine
- Model Training

## Dataset generado

prediction_features.parquet

### Campos esperados

- player
- surface

- prediction_score

- win_pct

- first_serve_won
- second_serve_won

- confidence_score

- surface_form

- fatigue_risk

- break_risk

- tiebreak_probability

## Uso

Este dataset será la capa inmediatamente
anterior al motor predictor.

## Próximas fases

### Match Comparator

Comparación directa:

Jugador A vs Jugador B

Incluyendo:

- Forma reciente
- Señales
- H2H
- Prediction Score

### Prediction API

Generación de probabilidades:

- Win Probability A
- Win Probability B

### Value Betting Engine

Comparación:

Model Probability

vs

Bookmaker Probability
