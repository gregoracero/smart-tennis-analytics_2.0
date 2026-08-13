# Match Comparator V2

## Objetivo

Comparar dos jugadores utilizando métricas reales
agregadas por superficie y ventana temporal.

## Inputs

data/parquet/player_metrics.parquet

data/parquet/h2h_overall.parquet

data/parquet/h2h_surface.parquet

## Parámetros

- player_a
- player_b
- surface
- window

## Ventanas soportadas

- LAST_10_MATCHES
- LAST_3_MONTHS
- LAST_6_MONTHS
- CAREER_SURFACE

## Métricas comparadas

### Forma

- matches
- wins
- losses
- surface_recent_win_pct

### Servicio

- first_serve_in_pct
- first_serve_won_pct
- second_serve_won_pct

### Break Points

- bp_saved_pct
- break_conversion_pct

### Tie-Breaks

- tiebreaks_played
- tiebreaks_won
- tiebreak_win_pct
- tiebreaks_per_match

### Resistencia

- avg_minutes

### Head To Head

- h2h_overall
- h2h_surface

## Salida

Comparativa lado a lado entre ambos jugadores.

## Uso

Base principal del análisis prepartido ATP y Challenger.

