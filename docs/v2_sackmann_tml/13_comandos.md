# 13. Referencia de comandos

## Integración

```powershell
python .\ingestion\v2\build_jeff_sackmann_with_odds_tml.py
```

## Reconciliación

```powershell
python .\ingestion\v2\reconcile_tml_integration.py `
  --parquet .\data\processed\jeff_sackmann_with_odds.parquet `
  --report-dir .\data\processed\tml_integration_reports `
  --max-synthetic-players 40 `
  --allow-more-synthetic
```

## Features

```powershell
python .\ingestion\v2\build_player_match_features_tml.py
```

## Datasets ML

```powershell
python .\ingestion\v2\build_clean_tennis_ml_dataset_tml.py
```

## Entrenamiento no market

```powershell
python .\modeling\train_professional_tennis_model_tml.py `
  --mode no_market `
  --trials 0 `
  --threads 2 `
  --tuning-max-rows 100000
```

## Entrenamiento with market

```powershell
python .\modeling\train_professional_tennis_model_tml.py `
  --mode with_market `
  --trials 20 `
  --threads 4 `
  --tuning-max-rows 50000
```

## Validar calibrador

```powershell
python -c "import joblib; c=joblib.load(r'modeling\artifacts\candidate\no_market\platt_calibrator.joblib'); print(type(c)); print(type(c).__module__)"
```

## Validar número de features

```powershell
python -c "from catboost import CatBoostClassifier; m=CatBoostClassifier(); m.load_model(r'modeling\artifacts\candidate\no_market\model.cbm'); print(len(m.feature_names_))"
```

## Limpiar Streamlit

```powershell
python -m streamlit cache clear
```

## Arrancar Streamlit

```powershell
streamlit run .\streamlit_app\Home.py
```
