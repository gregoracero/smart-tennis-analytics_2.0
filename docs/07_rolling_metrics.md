# Rolling Metrics

## Objetivo

Calcular métricas recientes por:

- jugador
- superficie

Estas métricas representan la forma actual del jugador y serán la base
del motor de análisis prepartido.

## Dataset de entrada

data/parquet/analytics_player_matches.parquet

## Dataset de salida

data/parquet/rolling_metrics.parquet

## Ventanas

### LAST_10_MATCHES

Últimos 10 partidos del jugador en la misma superficie.

### LAST_3_MONTHS

Partidos disputados durante los últimos 3 meses
en la misma superficie.

### LAST_6_MONTHS

Partidos disputados durante los últimos 6 meses
en la misma superficie.

## Métricas

- matches
- wins
- losses

- win_pct

- avg_minutes
- avg_ace
- avg_double_faults

- first_serve_in_pct
- first_serve_won_pct
- second_serve_won_pct

- bp_saved_pct

## Principio fundamental

Todas las métricas deben calcularse:

1. Por superficie.
2. Sobre una ventana reciente.
3. Utilizando únicamente el dataset analítico (2010+).

## Uso

Estas métricas alimentarán:

- Surface Form Engine
- Fatigue Engine
- Break Risk Engine
- Tie-Break Probability Engine
- Match Prediction Engine
- Value Betting Engine

## Próximas fases

08_signal_engine.md

- Surface Form
- Fatigue Score
- Break Risk
- Tie-Break Probability
- Confidence Score

09_pre_match_analyzer.md

- Comparación jugador vs jugador
- H2H
- Ventajas por superficie
- Señales automáticas
