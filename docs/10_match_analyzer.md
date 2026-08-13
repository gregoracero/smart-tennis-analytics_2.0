# Match Analyzer

## Objetivo

Comparar dos jugadores antes de un partido.

## Inputs

data/parquet/rolling_metrics.parquet

data/parquet/signal_engine.parquet

data/parquet/h2h_overall.parquet

data/parquet/h2h_surface.parquet

## Output

Comparación estructurada entre dos jugadores.

## Métricas mostradas

### Forma reciente

- Win %
- Últimos 10 partidos

### Señales

- Surface Form
- Fatigue Risk
- Break Risk
- Tie-Break Probability

### H2H

- H2H General
- H2H por Superficie

### Score Global

Comparación de confianza entre jugadores.

## Uso

Base del dashboard y del motor predictivo.
