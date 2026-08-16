
import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

TARGET = "target"

train_df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

pred_df = pd.read_parquet(
    "data/parquet/atp_matches_with_aligned_odds.parquet"
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

X_train = (
    train_df[
        features
    ]
    .select_dtypes(
        include=["number"]
    )
)

y_train = train_df[TARGET]

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
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

print()
print("Training calibrated model...")

model.fit(
    X_train,
    y_train
)

numeric_features = (
    train_df[features]
    .select_dtypes(include=["number"])
    .columns
    .tolist()
)

for col in numeric_features:
    if col not in pred_df.columns:
        pred_df[col] = pd.NA

X_pred = pred_df[numeric_features]

print("\nTRAIN FEATURES")
print(len(features))

print("\nPRED COLUMNS NOT IN TRAIN")
print(set(pred_df.columns) - set(features))

print("\nTRAIN FEATURES MISSING IN PRED")
print(set(features) - set(pred_df.columns))

numeric_features = list(imputer.feature_names_in_)

X_pred = pd.DataFrame(

    imputer.transform(
        pred_df[numeric_features]
    ),

    columns=numeric_features
)

print()
print("Generating probabilities...")

pred_df["model_probability"] = (
    model.predict_proba(
        X_pred
    )[:,1]
)

pred_df["edge"] = (

    pred_df[
        "model_probability"
    ]

    -

    pred_df[
        "market_probability_a"
    ]
)

output = (
    "data/parquet/"
    "atp_matches_with_predictions_nearest.parquet"
)

pred_df.to_parquet(
    output
)

print()
print("ROWS")
print(len(pred_df))

print()
print(
    pred_df[
        [
            "target",
            "model_probability",
            "market_probability_a",
            "edge"
        ]
    ].head()
)

print()

print(
    pred_df["model_probability"]
    .describe()
)

print()

print(
    pred_df["market_probability_a"]
    .describe()
)

print()

print(
    pred_df[
        ["model_probability","market_probability_a"]
    ].corr()
)

print()
print("Saved:", output)
