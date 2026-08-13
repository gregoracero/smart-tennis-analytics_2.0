# Player Metrics Engine

## Objetivo

Generar las métricas oficiales del proyecto
para análisis prepartido.

## Dataset de entrada

data/parquet/analytics_player_matches.parquet

## Dataset de salida

data/parquet/player_metrics.parquet

## Clave

player
surface
window

## Ventanas

### LAST_10_MATCHES

Últimos 10 partidos en la superficie.

### LAST_3_MONTHS

Últimos 3 meses en la superficie.

### LAST_6_MONTHS

Últimos 6 meses en la superficie.

### CAREER_SURFACE

Histórico completo en la superficie.

## Métricas

- matches
- wins
- losses

- surface_recent_win_pct

- first_serve_won_pct
- first_serve_in_pct
- second_serve_won_pct

- bp_saved_pct

- avg_minutes

## Uso

Fuente oficial para:

- Match Comparator
- Dashboard
- Prediction Engine
