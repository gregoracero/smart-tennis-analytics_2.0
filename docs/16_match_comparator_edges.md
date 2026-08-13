# Match Comparator Edges

## Objetivo

Calcular automáticamente la ventaja relativa entre dos jugadores
para cada métrica principal.

## Inputs

player_metrics.parquet

h2h_overall.parquet

h2h_surface.parquet

## Métricas evaluadas

- surface_recent_win_pct

- first_serve_in_pct
- first_serve_won_pct
- second_serve_won_pct

- bp_saved_pct
- break_conversion_pct

- tiebreak_win_pct

## Salida

Para cada métrica:

- Valor jugador A
- Valor jugador B
- Diferencia
- Jugador con ventaja

## Uso

Capa previa al Insight Engine.
