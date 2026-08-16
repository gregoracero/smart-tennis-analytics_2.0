
import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)

TARGET = "target"
EDGE_THRESHOLD = 0.05

TRAINING_PATH = (
    "data/parquet/training_matches_with_elo.parquet"
)

ODDS_PATH = (
    "data/parquet/atp_matches_with_nearest_odds.parquet"
)

OUTPUT_PATH = (
    "ml/reports/walk_forward_validation.csv"
)

print()
print("Loading datasets...")

train_df = pd.read_parquet(
    TRAINING_PATH
)

odds_df = pd.read_parquet(
    ODDS_PATH
)

train_df["match_date"] = pd.to_datetime(
    train_df["match_date"]
)

odds_df["match_date"] = pd.to_datetime(
    odds_df["match_date"]
)

train_df = train_df[
    train_df["tourney_level"] != "C"
].copy()

DROP_COLS = [
    "player_a",
    "player_b",
    "surface",
    "match_date",
    "tourney_level",
    TARGET
]

feature_cols = [
    c
    for c in train_df.columns
    if c not in DROP_COLS
]

feature_cols = (
    train_df[feature_cols]
    .select_dtypes(include=["number"])
    .columns
    .tolist()
)

YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025
]

results = []

for year in YEARS:

    print()
    print("=" * 80)
    print(f"YEAR {year}")
    print("=" * 80)

    train_split = train_df[
        train_df["match_date"].dt.year < year
    ].copy()

    test_training = train_df[
        train_df["match_date"].dt.year == year
    ].copy()

    test_odds = odds_df[
        odds_df["match_date"].dt.year == year
    ].copy()

    if (
        len(train_split) == 0
        or len(test_training) == 0
        or len(test_odds) == 0
    ):
        continue

    X_train = train_split[
        feature_cols
    ]

    y_train = train_split[
        TARGET
    ]

    X_test_auc = test_training[
        feature_cols
    ]

    y_test_auc = test_training[
        TARGET
    ]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test_auc = imputer.transform(
        X_test_auc
    )

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model = CalibratedClassifierCV(
        base_model,
        method="isotonic",
        cv=3
    )

    print("Training...")

    model.fit(
        X_train,
        y_train
    )

    proba_auc = (
        model.predict_proba(
            X_test_auc
        )[:,1]
    )

    auc = roc_auc_score(
        y_test_auc,
        proba_auc
    )

    brier = brier_score_loss(
        y_test_auc,
        proba_auc
    )

    X_test_roi = pd.DataFrame(
        imputer.transform(
            test_odds[feature_cols]
        ),
        columns=feature_cols
    )

    test_odds[
        "model_probability"
    ] = (
        model.predict_proba(
            X_test_roi
        )[:,1]
    )

    test_odds["edge"] = (
        test_odds["model_probability"]
        -
        test_odds["market_probability_a"]
    )

    bets = test_odds[
        test_odds["edge"] > EDGE_THRESHOLD
    ].copy()

    if len(bets) > 0:

        bets["profit"] = bets.apply(
            lambda r:
            r["odds_a"] - 1
            if r["target"] == 1
            else -1,
            axis=1
        )

        roi = (
            bets["profit"].sum()
            /
            len(bets)
        )

        hit_rate = (
            bets["target"].mean()
        )

        bets_count = len(
            bets
        )

    else:

        roi = 0
        hit_rate = 0
        bets_count = 0

    results.append({
        "year": year,
        "train_rows": len(train_split),
        "test_rows": len(test_training),
        "auc": round(auc, 4),
        "brier": round(brier, 6),
        "bets": bets_count,
        "hit_rate": round(hit_rate, 4),
        "roi": round(roi, 4)
    })

    print(
        f"AUC={auc:.4f}"
    )

    print(
        f"ROI={roi:.2%}"
    )

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(results_df)

print()
print("Saved:")
print(OUTPUT_PATH)
