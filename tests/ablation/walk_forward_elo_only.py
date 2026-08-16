
import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

TARGET = "target"

df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

df["match_date"] = pd.to_datetime(
    df["match_date"]
)

df = df[
    df["tourney_level"] != "C"
].copy()

FEATURES = [
    "delta_elo",
    "delta_surface_elo",
    "delta_inactivity_days",
    "delta_surface_inactivity_days"
]

YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025
]

print()
print("=" * 80)
print("ELO ONLY WALK FORWARD")
print("=" * 80)

for year in YEARS:

    train_df = df[
        df["match_date"].dt.year < year
    ]

    test_df = df[
        df["match_date"].dt.year == year
    ]

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        X_test
    )

    model = CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        ),
        method="isotonic",
        cv=3
    )

    model.fit(
        X_train,
        y_train
    )

    proba = (
        model.predict_proba(
            X_test
        )[:,1]
    )

    auc = roc_auc_score(
        y_test,
        proba
    )

    print()
    print(year)
    print("AUC :", round(auc,4))
