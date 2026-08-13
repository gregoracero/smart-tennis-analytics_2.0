# Player Matches

## Objetivo

Normalizar los partidos ATP y Challenger para disponer
de una única perspectiva por jugador.

Cada partido genera dos registros:

- ganador
- perdedor

## Input

data/parquet/all_matches.parquet

## Output

data/parquet/player_matches.parquet

## Uso

Base para:

- métricas por superficie
- forma reciente
- H2H
- fatiga
- señales prepartido
