# 10. Operación del pipeline

## Actualización de datos

### 1. Actualizar raw

Colocar los CSV TML en:

```text
data/raw/tml_v2/current_season
```

### 2. Integrar y reconciliar

```powershell
python .\ingestion\v2\build_jeff_sackmann_with_odds_tml.py
python .\ingestion\v2\reconcile_tml_integration.py `
  --parquet .\data\processed\jeff_sackmann_with_odds.parquet `
  --report-dir .\data\processed\tml_integration_reports `
  --max-synthetic-players 40 `
  --allow-more-synthetic
```

### 3. Generar features

```powershell
python .\ingestion\v2\build_player_match_features_tml.py
```

### 4. Generar datasets limpios

```powershell
python .\ingestion\v2\build_clean_tennis_ml_dataset_tml.py
```

### 5. Entrenar candidato no market

Primera prueba segura:

```powershell
python .\modeling\train_professional_tennis_model_tml.py `
  --mode no_market `
  --trials 0 `
  --threads 2 `
  --tuning-max-rows 100000
```

### 6. Entrenar candidato with market

```powershell
python .\modeling\train_professional_tennis_model_tml.py `
  --mode with_market `
  --trials 20 `
  --threads 4 `
  --tuning-max-rows 50000
```

### 7. Validar artefactos

Comprobar:

- Número de features.
- Tipo del calibrador.
- `probability_policy.json`.
- Métricas de test.
- Subgrupos TML y sintéticos.

### 8. Promover no market

Hacer backup de producción, copiar el candidato y validar de nuevo.

### 9. Limpiar caché

```powershell
python -m streamlit cache clear
```

### 10. Arrancar aplicación

```powershell
streamlit run .\streamlit_app\Home.py
```

## Política de promoción

Promover solo si:

- El modelo carga.
- El esquema coincide.
- El calibrador carga con módulo estable.
- La política de probabilidad es válida.
- Test mejora baselines relevantes.
- No existe leakage.
- La página supera la prueba de simetría.
