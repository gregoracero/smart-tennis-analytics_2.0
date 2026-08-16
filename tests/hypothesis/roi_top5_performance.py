
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

    "days_inactive_b"
]

print("Loading dataset...")

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
].copy()

X_train = train_df[
    FEATURES
]

X_test = test_df[
    FEATURES
]

y_train = train_df["target"]

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

test_df["probability"] = (
    model
    .predict_proba(X_test)[:,1]
)

for threshold in [
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80
]:

    bets = test_df[
        test_df["probability"] > threshold
    ]

    hit_rate = (
        bets["target"]
        .mean()
    )

    print()
    print("=" * 70)
    print(
        f"THRESHOLD {threshold}"
    )
    print("=" * 70)

    print(
        "BETS:",
        len(bets)
    )

    print(
        "HIT RATE:",
        round(
            hit_rate * 100,
            2
        ),
        "%"
    )
