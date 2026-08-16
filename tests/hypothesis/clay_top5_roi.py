import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

FEATURES = [

    "delta_surface_inactivity_days",
    "delta_elo",
    "delta_inactivity_days",
    "days_inactive_a",
    "days_inactive_b"

]

train = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

pred = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

train["match_date"] = pd.to_datetime(
    train["match_date"]
)

surface_column = (
    "surface_x"
    if "surface_x" in train.columns
    else "surface"
)

train = train[
    (train[surface_column] == "Clay")
    &
    (train["match_date"] < "2023-01-01")
]

pred = pred[
    pred["surface_x"] == "Clay"
].copy()

X_train = train[FEATURES]
y_train = train["target"]

imp = SimpleImputer(
    strategy="median"
)

X_train = imp.fit_transform(
    X_train
)

X_pred = imp.transform(
    pred[FEATURES]
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train
)

pred["prob"] = (
    model.predict_proba(X_pred)[:,1]
)

pred["edge"] = (
    pred["prob"]
    -
    pred["market_probability_a"]
)

print()

for threshold in [
    0.05,
    0.10,
    0.15,
    0.20
]:

    bets = pred[
        pred["edge"] > threshold
    ].copy()

    bets["profit"] = bets.apply(
        lambda r:
        r["odds_a"] - 1
        if r["target"] == 1
        else -1,
        axis=1
    )

    print()

    print(
        f"EDGE > {threshold}"
    )

    print(
        "BETS:",
        len(bets)
    )

    print(
        "ROI:",
        round(
            bets["profit"].mean() * 100,
            2
        )
    )
