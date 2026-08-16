import pandas as pd

from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

USE_ELO = True

if USE_ELO:

    DATASET = (
        "data/parquet/training_matches_with_elo.parquet"
    )

else:

    DATASET = (
        "data/parquet/training_matches.parquet"
    )

SURFACE = "Clay"

SPLIT_DATE = "2023-01-01"

ATP_LEVELS = [
    "G",
    "M",
    "1000",
    "500",
    "250"
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

print()
print("SHAPE")
print(df.shape)

TARGET = "target"

DROP_COLUMNS = [
    "player_a",
    "player_b",
    "surface",
    "surface_x",
    "surface_y",
    "match_date",
    "tourney_level"
]

feature_columns = [

    c

    for c in df.columns

    if c not in DROP_COLUMNS
    and c != TARGET
]

X = df[
    feature_columns
]

X = X.select_dtypes(
    include=["number"]
)

feature_columns = (
    X.columns.tolist()
)

print()
print("FEATURES")
print(len(feature_columns))

train_df = df[
    df["match_date"] < SPLIT_DATE
]

test_df = df[
    df["match_date"] >= SPLIT_DATE
]

print()
print("TRAIN ROWS")
print(len(train_df))

print()
print("TEST ROWS")
print(len(test_df))

X_train = train_df[
    feature_columns
]

X_test = test_df[
    feature_columns
]

y_train = train_df[
    TARGET
]

y_test = test_df[
    TARGET
]

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

print()
print("Training model...")

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:,1]

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities
)

print()
print("ACCURACY")
print(round(
    accuracy,
    4
))

print()
print("ROC AUC")
print(round(
    auc,
    4
))

print()
print(
    classification_report(
        y_test,
        predictions
    )
)
