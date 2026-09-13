# 6. Datasets para machine learning

## Salidas

```text
data/processed/tennis_ml_no_market.parquet
data/processed/tennis_ml_with_market.parquet
data/processed/tennis_ml_dataset_report.csv
data/processed/tennis_ml_feature_manifest.csv
data/processed/tennis_ml_dataset_metadata.json
```

## Filtros

Por defecto:

- Fecha mínima: 2001-01-01.
- `eligible_for_model == True`.
- Competiciones: ATP, ATP qualifying y Challenger.
- Futures excluidos.
- Mínimo de cinco partidos previos para ambos jugadores.
- Para mercado, al menos una pareja de cuotas válida.

## Split temporal

```text
Train       <= 2024-12-31
Validation  2025-01-01 ... 2025-12-31
Test        >= 2026-01-01
```

## Sin mercado

```text
Train       236.031   target 0,500205
Validation   14.049   target 0,498185
Test         14.864   target 0,510428
Total       264.944
Features         82
```

Cobertura: 2001-01-01 a 2026-08-30.

## Con mercado

```text
Train        52.441   target 0,498103
Validation    2.135   target 0,499297
Test          1.618   target 0,518541
Total        56.194
Features        102
```

Cobertura: 2002-06-10 a 2026-08-14.

## TML en test

Dataset sin mercado:

```text
ATP                 2.108
ATP_QUALIFYING      1.077
CHALLENGER          5.100
Total               8.285
```

Target TML: aproximadamente 0,509958.

## Lista blanca

El manifiesto clasifica las columnas como:

```text
sport_feature
market_feature
context
traceability
identity
target
split
```

Solo `sport_feature` y `market_feature` tienen `allowed_for_training = True`.

## Variables de calidad

```text
min_career_matches_before
min_stat_matches_before
cold_start_any_player
stats_reliable_both
synthetic_player_any
```

## Control de leakage

Se bloquean patrones asociados a:

- Ganador y perdedor.
- Marcador.
- Estadísticas del propio partido.
- Metadatos de fuente.
- IDs TML.
- Indicadores de exclusión.
- Identidades de jugadores.
