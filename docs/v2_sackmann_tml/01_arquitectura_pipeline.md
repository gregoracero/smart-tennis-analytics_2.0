# 1. Arquitectura del pipeline

## Diagrama lógico

```text
data/raw/tennis_atp-master/*.csv
                       \
data/raw/tml_v2/current_season/*.csv
                         +--> build_jeff_sackmann_with_odds_tml.py
                       /
data/raw/tennis_data_co_uk_v2/*.{xls,xlsx}
                                    |
                                    v
               jeff_sackmann_with_odds.parquet
                                    |
                                    v
                   reconcile_tml_integration.py
                                    |
                                    v
               jeff_sackmann_with_odds.parquet
                        reconciliado y validado
                                    |
                                    v
                build_player_match_features_tml.py
                                    |
                                    v
          tennis_matches_with_player_stats.parquet
                                    |
                                    v
              build_clean_tennis_ml_dataset_tml.py
                         /                         \
                        v                           v
       tennis_ml_no_market.parquet     tennis_ml_with_market.parquet
                        \                           /
                         v                         v
                train_professional_tennis_model_tml.py
                                    |
                                    v
                  modeling/artifacts/candidate/*
                                    |
                         validación y promoción
                                    |
                                    v
                 modeling/artifacts/production/*
                                    |
                                    v
                         Streamlit prediction page
```

## Principios de diseño

### Temporalidad estricta

Todas las estadísticas de jugador se capturan antes de actualizar el estado con el partido actual. El orden es:

1. Leer el estado acumulado.
2. Crear el snapshot prepartido.
3. Escribir las features de la fila.
4. Si el partido es elegible, actualizar Elo, forma, H2H y acumulados.

### Trazabilidad

Los Parquets conservan columnas como:

```text
source_origin
source_priority
source_sackmann_file
source_tml_file
winner_id_tml
loser_id_tml
stats_quality_flag
competition_type
```

Estas columnas son auditables, pero no están autorizadas para entrenamiento.

### Separación de responsabilidades

- La integración reúne y normaliza fuentes.
- La reconciliación resuelve identidades y anomalías.
- El constructor de features mantiene estados históricos.
- El limpiador selecciona una lista blanca y crea splits.
- El entrenador usa exclusivamente las features declaradas en metadata.
- Streamlit reproduce el mismo esquema y orientación que el entrenamiento.

### Escritura atómica

Los Parquets principales se escriben primero como `*.new.parquet`, se validan y después reemplazan el archivo final. Cuando corresponde, se conserva un backup `*.before_rebuild.parquet`.
