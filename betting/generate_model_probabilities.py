
import pandas as pd

from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

TARGET = "target"

SOURCE_DATASET = (
    "data/parquet/training_matches_with_elo.parquet"
)

ODDS_DATASET = (
    "data/parquet/atp_matches_with_odds.parquet"
)

OUTPUT_DATASET = (
    "data/parquet/atp_matches_with_odds_predictions.parquet"
)

print("Loading datasets...")

train_df = pd.read_parquet(
    SOURCE_DATASET
)

pred_df = pd.read_parquet(
    ODDS_DATASET
)

train_df["match_date"] = pd.to_datetime(
    train_df["match_date"]
)

#
# ATP ONLY
#

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

features = [

    c

    for c in train_df.columns

    if c not in DROP_COLS
]

X_train_full = (
    train_df[
        features
    ]
    .select_dtypes(
        include=["number"]
    )
)

feature_columns = (
    X_train_full
    .columns
    .tolist()
)

y_train_full = train_df[TARGET]

print()
print("TRAIN ROWS")
print(len(train_df))

imputer = SimpleImputer(
    strategy="median"
)

X_train_full = (
    imputer.fit_transform(
        X_train_full
    )
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

print()
print("Training calibrated model...")

model = CalibratedClassifierCV(

    base_model,

    method="isotonic",

    cv=3
)

model.fit(
    X_train_full,
    y_train_full
)

#
# Prediction dataset
#

prediction_features = (
    pred_df[
        feature_columns
    ]
)

prediction_features = pd.DataFrame(

    imputer.transform(
        prediction_features
    ),

    columns=feature_columns
)

print()
print(
    "Generating probabilities..."
)

pred_df["model_probability"] = (
    model.predict_proba(
        prediction_features
    )[:,1]
)

pred_df["market_probability"] = (
    pred_df[
        "market_prob_winner"
    ]
)

pred_df["edge"] = (

    pred_df[
        "model_probability"
    ]

    -

    pred_df[
        "market_probability"
    ]
)

pred_df.to_parquet(
    OUTPUT_DATASET
)

print()
print(
    "ROWS:",
    len(pred_df)
)

print()
print(
    pred_df[
        [
            "target",
            "model_probability",
            "market_probability",
            "edge"
        ]
    ]
    .head()
)

print()
print(
    f"Saved: {OUTPUT_DATASET}"
)
