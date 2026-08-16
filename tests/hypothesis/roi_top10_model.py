
import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

DATASET = (
    "data/parquet/training_matches_with_elo.parquet"
)

SURFACE = "Hard"

SPLIT_DATE = "2023-01-01"

ATP_LEVELS = [
    "G",
    "M",
    "1000",
    "500",
    "250"
]

FEATURES = [

    "delta_surface_inactivity_days",

    "delta_elo",
    "delta_inactivity_days",

    "days_inactive_a",
    "days_inactive_b",

    "delta_surface_elo",

    "surface_days_inactive_a",

    "delta_rank_points",

    "surface_days_inactive_b",

    "delta_rank"
]

df = pd.read_parquet(
    DATASET
)

surface_column = (
    "surface_x"
    if "surface_x" in df.columns
    else "surface"
)

df = df[
    df[surface_column] == SURFACE
]

df = df[
    df["tourney_level"].isin(
        ATP_LEVELS
    )
]

df["match_date"] = pd.to_datetime(
    df["match_date"]
)

train_df = df[
    df["match_date"] < SPLIT_DATE
]

test_df = df[
    df["match_date"] >= SPLIT_DATE
]

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df["target"]
y_test = test_df["target"]

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
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

proba = model.predict_proba(
    X_test
)[:,1]

print()
print("TOP 10 MODEL")
print()

print(
    pd.Series(proba)
    .describe()
)
