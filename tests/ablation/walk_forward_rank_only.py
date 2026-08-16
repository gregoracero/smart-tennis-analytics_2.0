
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
    "rank_a",
    "rank_b",
    "delta_rank",
    "rank_points_a",
    "rank_points_b",
    "delta_rank_points"
]

YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025
]

results = []

print()
print("=" * 80)
print("RANK ONLY WALK FORWARD")
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

    results.append({
        "year": year,
        "auc": round(auc,4)
    })

    print()
    print(year)
    print("AUC :", round(auc,4))

results_df = pd.DataFrame(
    results
)

output = (
    "ml/reports/"
    "walk_forward_rank_only.csv"
)

results_df.to_csv(
    output,
    index=False
)

print()
print(results_df)

print()
print("MEAN AUC")
print(
    round(
        results_df["auc"].mean(),
        4
    )
)

print()
print("Saved:")
print(output)
