import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

TARGET = "target"
TRAIN_END = "2023-01-01"

print()
print("Loading datasets...")

train_df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

pred_df = pd.read_parquet(
    "data/parquet/atp_matches_with_nearest_odds.parquet"
)

train_df["match_date"] = pd.to_datetime(
    train_df["match_date"]
)

pred_df["match_date"] = pd.to_datetime(
    pred_df["match_date"]
)

#
# ATP only
#
train_df = train_df[
    train_df["tourney_level"] != "C"
].copy()

#
# Temporal split
#
train_split = train_df[
    train_df["match_date"] < TRAIN_END
].copy()

pred_split = pred_df[
    pred_df["match_date"] >= TRAIN_END
].copy()

print()
print("TRAIN ROWS:", len(train_split))
print("TEST ROWS :", len(pred_split))

#
# Features
#
DROP_COLS = [
    "player_a",
    "player_b",
    "surface",
    "match_date",
    "tourney_level",
    TARGET
]

features = [
    c
    for c in train_split.columns
    if c not in DROP_COLS
]

X_train = (
    train_split[features]
    .select_dtypes(include=["number"])
)

feature_cols = X_train.columns.tolist()

y_train = train_split[TARGET]

#
# Imputer
#
imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

#
# Model
#
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

print()
print("Training model...")

model.fit(
    X_train,
    y_train
)

#
# Prediction
#
X_pred = pd.DataFrame(
    imputer.transform(
        pred_split[feature_cols]
    ),
    columns=feature_cols
)

print()
print("Generating probabilities...")

pred_split["model_probability"] = (
    model.predict_proba(X_pred)[:, 1]
)

pred_split["edge"] = (
    pred_split["model_probability"]
    -
    pred_split["market_probability_a"]
)

#
# Save temporal predictions
#
output = (
    "data/parquet/"
    "temporal_predictions_2023_plus.parquet"
)

pred_split.to_parquet(output)

print()
print("Saved:", output)
print("ROWS :", len(pred_split))

#
# Diagnostics
#
print()
print("=" * 80)
print("MODEL PROBABILITY")
print("=" * 80)

print(
    pred_split["model_probability"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print()
print("=" * 80)
print("MARKET PROBABILITY")
print("=" * 80)

print(
    pred_split["market_probability_a"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print()
print("=" * 80)
print("EDGE")
print("=" * 80)

print(
    pred_split["edge"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]
    )
)

print()
print("=" * 80)
print("CORRELATION")
print("=" * 80)

print(
    pred_split[
        [
            "model_probability",
            "market_probability_a"
        ]
    ].corr()
)

#
# Calibration table
#
print()
print("=" * 80)
print("CALIBRATION TABLE")
print("=" * 80)

tmp = pred_split.copy()

tmp["bucket"] = pd.qcut(
    tmp["model_probability"],
    10,
    duplicates="drop"
)

calibration = (
    tmp.groupby("bucket")
    .agg(
        predicted=("model_probability", "mean"),
        actual=("target", "mean"),
        n=("target", "size")
    )
)

print(calibration)

#
# ROI
#
thresholds = [
    0.03,
    0.05,
    0.08,
    0.10
]

print()
print("=" * 80)
print("TEMPORAL ROI BACKTEST")
print("=" * 80)

for threshold in thresholds:

    bets = pred_split[
        pred_split["edge"] > threshold
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

    total_bets = len(bets)

    wins = (
        bets["target"] == 1
    ).sum()

    hit_rate = wins / total_bets

    total_profit = bets["profit"].sum()

    roi = total_profit / total_bets

    print()
    print(f"EDGE > {threshold:.0%}")
    print("-" * 40)
    print("BETS     :", total_bets)
    print("WINS     :", wins)
    print("HIT RATE :", f"{hit_rate:.2%}")
    print("PROFIT   :", round(total_profit, 2))
    print("ROI      :", f"{roi:.2%}")

print()
print("=" * 80)
print("FINISHED")
print("=" * 80)