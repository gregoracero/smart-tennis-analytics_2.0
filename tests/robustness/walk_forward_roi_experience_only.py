
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

for year in YEARS:

    train_split = train_df[
        train_df["match_date"].dt.year < year
    ]

    test_split = odds_df[
        odds_df["match_date"].dt.year == year
    ].copy()

    X_train = train_split[FEATURES]
    y_train = train_split[TARGET]

    imp = SimpleImputer(
        strategy="median"
    )

    X_train = imp.fit_transform(
        X_train
    )

    X_test = imp.transform(
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
        model.predict_proba(X_test)[:,1]
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

    results.append({
        "year": year,
        "bets": len(bets),
        "roi": round(roi,4)
    })

    print()
    print(year)
    print("BETS :", len(bets))
    print("ROI  :", f"{roi:.2%}")

results_df = pd.DataFrame(
    results
)

print()
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
