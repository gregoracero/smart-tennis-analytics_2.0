# Head To Head Engine

## Objetivo

Generar estadísticas históricas entre jugadores.

## Outputs

data/parquet/h2h_overall.parquet

data/parquet/h2h_surface.parquet

## H2H Overall

Agregado sobre todas las superficies.

## H2H Surface

Agregado por superficie.

### Clave

player_a
player_b
surface

## Métricas

- matches
- wins_a
- wins_b
- h2h_pct_a
- h2h_pct_b
- last_match_date

## Uso

Motor de análisis prepartido.

Las consultas por superficie tendrán prioridad sobre
las consultas agregadas.
