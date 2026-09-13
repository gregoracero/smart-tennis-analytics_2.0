# 9. Validaciones y controles

## Parquet integrado

```powershell
python -c "import pandas as pd; p=r'data\processed\jeff_sackmann_with_odds.parquet'; x=pd.read_parquet(p,columns=['tourney_id','competition_type','match_num','winner_id','loser_id','tourney_date']); print(len(x)); print(x.duplicated(['tourney_id','competition_type','match_num']).sum()); print(x[['winner_id','loser_id']].isna().sum()); print(pd.to_numeric(x.tourney_date,errors='coerce').max())"
```

Esperado:

```text
971080 filas
0 duplicados
0 IDs nulos
20260830
```

## Parquet de features

```powershell
python -c "import pandas as pd; p=r'data\processed\tennis_matches_with_player_stats.parquet'; x=pd.read_parquet(p,columns=['tourney_id','competition_type','match_num','player_1_id','player_2_id','match_date','eligible_for_model']); print(len(x)); print(x.duplicated(['tourney_id','competition_type','match_num']).sum()); print(x[['player_1_id','player_2_id']].isna().sum()); print(x.match_date.max()); print(x.eligible_for_model.sum())"
```

Esperado:

```text
971080 filas
0 duplicados
0 IDs nulos
2026-08-30
966855 elegibles
```

## Datasets ML

```powershell
python -c "import pandas as pd; files=[r'data\processed\tennis_ml_no_market.parquet',r'data\processed\tennis_ml_with_market.parquet']; [(print(p,len(x:=pd.read_parquet(p)),x.duplicated(['tourney_id','competition_type','match_num']).sum(),x.target_player_1_win.isna().sum(),x.match_date.max())) for p in files]"
```

## Manifiesto

```powershell
python -c "import pandas as pd; x=pd.read_csv(r'data\processed\tennis_ml_feature_manifest.csv'); bad=x[x.role.isin(['context','traceability','identity']) & x.allowed_for_training.astype(bool)]; print(len(bad)); print(bad.to_string(index=False))"
```

Esperado: `0`.

## Modelos

```powershell
python -c "from catboost import CatBoostClassifier; m=CatBoostClassifier(); m.load_model(r'modeling\artifacts\candidate\no_market\model.cbm'); print(len(m.feature_names_))"
```

Esperado: `82`.

```powershell
python -c "import joblib; c=joblib.load(r'modeling\artifacts\candidate\no_market\platt_calibrator.joblib'); print(type(c).__module__)"
```

Esperado:

```text
modeling.calibration
```

## Producción

Validar modelo, calibrador, política y metadata antes de arrancar Streamlit.
