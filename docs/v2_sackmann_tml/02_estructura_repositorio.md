# 2. Estructura del repositorio

```text
smart-tennis-analytics_2.0/
├── data/
│   ├── raw/
│   │   ├── tennis_atp-master/
│   │   │   ├── atp_players.csv
│   │   │   ├── atp_matches_*.csv
│   │   │   ├── atp_matches_futures_*.csv
│   │   │   └── atp_matches_qual_chall_*.csv
│   │   ├── tennis_data_co_uk_v2/
│   │   │   └── 2001.xls ... 2026.xlsx
│   │   └── tml_v2/
│   │       ├── ATP_Database.csv
│   │       └── current_season/
│   │           ├── 2026.csv
│   │           ├── 2026_atp_quali.csv
│   │           └── 2026_challenger.csv
│   └── processed/
│       ├── jeff_sackmann_with_odds.parquet
│       ├── tennis_matches_with_player_stats.parquet
│       ├── tennis_ml_no_market.parquet
│       ├── tennis_ml_with_market.parquet
│       ├── tennis_ml_dataset_report.csv
│       ├── tennis_ml_feature_manifest.csv
│       ├── tennis_ml_dataset_metadata.json
│       └── tml_integration_reports/
├── ingestion/
│   └── v2/
│       ├── build_jeff_sackmann_with_odds_tml.py
│       ├── reconcile_tml_integration.py
│       ├── build_player_match_features_tml.py
│       └── build_clean_tennis_ml_dataset_tml.py
├── modeling/
│   ├── __init__.py
│   ├── calibration.py
│   ├── train_professional_tennis_model_tml.py
│   ├── finalize_tennis_trainer.py
│   ├── patch_match_prediction_page_tml.py
│   └── artifacts/
│       ├── candidate/
│       │   ├── no_market/
│       │   └── with_market/
│       └── production/
│           └── no_market/
├── streamlit_app/
└── docs/
    └── v2_sackmann_tml/
```

## Informes de integración

```text
tml_integration_reports/
├── integration_summary.json
├── reconciliation_summary.json
├── final_validation.json
├── player_dimension.csv
├── tournament_dimension.csv
├── player_id_canonical_map.csv
├── synthetic_player_candidates.csv
├── remaining_synthetic_players.csv
├── tml_round_reclassification.csv
├── tml_invalid_statistics_postprocess.csv
├── tml_duplicate_matches_postprocess.csv
└── tml_match_num_collisions_postprocess.csv
```

## Artefactos de un modelo candidato

```text
modeling/artifacts/candidate/no_market/
├── model.cbm
├── platt_calibrator.joblib
├── probability_policy.json
├── test_predictions.parquet
├── feature_importance.csv
├── metrics.json
├── metrics.csv
└── run_metadata.json
```
