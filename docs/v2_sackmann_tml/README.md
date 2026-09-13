# Integración Sackmann + TennisMyLife v2

## Propósito

Esta documentación describe el pipeline construido para integrar datos históricos de Jeff Sackmann, datos de temporada actual de TennisMyLife y cuotas de tennis-data.co.uk, generar características prepartido sin fuga de información, construir datasets temporales para machine learning, entrenar modelos CatBoost y servir predicciones en Streamlit.

El alcance documentado llega hasta el **30 de agosto de 2026**.

## Estado validado

- Partidos integrados finales: **971.080**.
- Columnas del Parquet integrado: **119**.
- Claves operativas duplicadas: **0**.
- IDs nulos de ganador o perdedor: **0**.
- Última fecha disponible: **2026-08-30**.
- Partidos TML integrados: **8.469**.
- IDs canónicos reconciliados automáticamente: **20**.
- IDs sintéticos restantes: **37**.
- Walkovers excluidos del estado histórico y del modelado: **4.219**.
- Registros con la misma identidad en ambos lados: **6**.
- Dataset limpio sin mercado: **264.944** filas.
- Dataset limpio con mercado: **56.194** filas.
- Modelo candidato sin mercado: **82** features.
- Modelo candidato con mercado: **102** features.

## Índice

1. [Arquitectura y flujo completo](01_arquitectura_pipeline.md)
2. [Estructura de directorios y artefactos](02_estructura_repositorio.md)
3. [Integración de fuentes](03_integracion_fuentes.md)
4. [Reconciliación de identidades y calidad](04_reconciliacion_calidad.md)
5. [Construcción de features prepartido](05_features_prepartido.md)
6. [Construcción de datasets ML](06_datasets_ml.md)
7. [Entrenamiento y evaluación](07_entrenamiento_evaluacion.md)
8. [Predicción en Streamlit](08_streamlit_prediccion.md)
9. [Validaciones y controles](09_validaciones.md)
10. [Operación del pipeline](10_operacion_pipeline.md)
11. [Limitaciones y trabajo pendiente](11_limitaciones_pendientes.md)
12. [Historial de decisiones](12_decisiones_tecnicas.md)
13. [Referencia de comandos](13_comandos.md)

## Scripts principales

```text
ingestion/v2/build_jeff_sackmann_with_odds_tml.py
ingestion/v2/reconcile_tml_integration.py
ingestion/v2/build_player_match_features_tml.py
ingestion/v2/build_clean_tennis_ml_dataset_tml.py
modeling/train_professional_tennis_model_tml.py
modeling/finalize_tennis_trainer.py
modeling/patch_match_prediction_page_tml.py
```

## Orden recomendado

```text
Fuentes raw
  -> integración Sackmann + TML + cuotas
  -> reconciliación de jugadores y calidad
  -> jeff_sackmann_with_odds.parquet
  -> features prepartido
  -> tennis_matches_with_player_stats.parquet
  -> datasets ML con splits temporales
  -> entrenamiento candidato
  -> validación de artefactos
  -> promoción a producción
  -> Streamlit
```
