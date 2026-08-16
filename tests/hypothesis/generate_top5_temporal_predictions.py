
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

DATASET = (
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

print("Loading dataset...")

df = pd.read_parquet(
    DATASET
)

train = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

train["match_date"] = pd.to_datetime(
    train["match_date"]
)

train = train[
    train["match_date"] < "2023-01-01"
]

X_train = train[FEATURES]
y_train = train["target"]

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_pred = imputer.transform(
    df[FEATURES]
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

df["top5_probability"] = (
    model.predict_proba(
        X_pred
    )[:,1]
)

df["top5_edge"] = (
    df["top5_probability"]
    -
    df["market_probability_a"]
)

output = (
    "tests/reports/"
    "top5_temporal_predictions.parquet"
)

df.to_parquet(
    output,
    index=False
)

print()
print("SAVED")
print(output)

print()
print("ROWS")
print(len(df))
