
import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

TARGET = "target"
EDGE = 0.05

train_df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

odds_df = pd.read_parquet(
    "data/parquet/atp_matches_with_nearest_odds.parquet"
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

FEATURES = [
    "days_inactive_a",
    "days_inactive_b",

    "surface_days_inactive_a",
    "surface_days_inactive_b",

    "delta_inactivity_days",
    "delta_surface_inactivity_days",

    "matches_played_a",
    "matches_played_b",

    "surface_matches_played_a",
    "surface_matches_played_b",

    "delta_matches_played",
    "delta_surface_matches_played"
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
print("READINESS ROI")
print("=" * 80)

for year in YEARS:

    train_split = train_df[
        train_df["match_date"].dt.year < year
    ].copy()

    test_split = odds_df[
        odds_df["match_date"].dt.year == year
    ].copy()

    if len(test_split) == 0:
        continue

    X_train = train_split[FEATURES]

    y_train = train_split[
        TARGET
    ]

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train
    )

    X_test = imputer.transform(
        test_split[FEATURES]
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

    test_split["model_probability"] = (
        model.predict_proba(
            X_test
        )[:,1]
    )

    test_split["edge"] = (
        test_split["model_probability"]
        -
        test_split["market_probability_a"]
    )

    bets = test_split[
        test_split["edge"] > EDGE
    ].copy()

    if len(bets) == 0:
        continue

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

    results.append({
        "year": year,
        "bets": len(bets),
        "hit_rate": round(
            hit_rate,
            4
        ),
        "roi": round(
            roi,
            4
        )
    })

    print()
    print(year)
    print("BETS :", len(bets))
    print(
        "HIT  :",
        f"{hit_rate:.2%}"
    )
    print(
        "ROI  :",
        f"{roi:.2%}"
    )

results_df = pd.DataFrame(
    results
)

output = (
    "ml/reports/"
    "walk_forward_roi_readiness_only.csv"
)

results_df.to_csv(
    output,
    index=False
)

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print(results_df)

print()
print(
    "MEAN ROI:",
    round(
        results_df["roi"].mean() * 100,
        2
    ),
    "%"
)

print()
print("Saved:")
print(output)
