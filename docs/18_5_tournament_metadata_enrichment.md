# 18.5 Tournament Metadata Enrichment

## Objetivo

Enriquecer analytics_player_matches.parquet con metadatos de torneo.

## Nuevos campos

- tourney_level
- tourney_name
- tourney_id
- round

## Casos de uso

- ATP/Slam vs Challenger
- Recent Results
- Tournament filtering
- Tournament breakdowns
- ATP Head2Head replication

## Pipeline

master_matches
    ?
analytics_player_matches
    ?
player_metrics
    ?
dashboard

## Motivación

Permitir segmentar métricas por categoría de torneo y replicar filtros ATP/Slam.
