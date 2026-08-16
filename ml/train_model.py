
import argparse
import json
import joblib
from pathlib import Path

import pandas as pd

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score
)

DATASET = (
    "data/parquet/training_matches_with_elo.parquet"
)

TARGET = "target"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tour",
    required=True
)

parser.add_argument(
    "--surface",
    required=True
)

parser.add_argument(
    "--engine",
    default="xgboost"
)

parser.add_argument(
    "--version",
    default="v1"
)

args = parser.parse_args()

model_key = (
    f"{args.tour}_{args.surface}_{args.engine}_{args.version}"
).lower()

model_dir = Path(
    f"ml/models/{model_key}"
)

model_dir.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading dataset...")

df = pd.read_parquet(
    DATASET
)

df["match_date"] = pd.to_datetime(
    df["match_date"]
)

surface_name = args.surface.capitalize()

df = df[
    df["surface"] == surface_name
]

if args.tour.lower() == "atp":

    df = df[
        df["tourney_level"].isin(
            [
                "G",
                "M",
                "1000",
                "500",
                "250"
            ]
        )
    ]

elif args.tour.lower() == "challenger":

    df = df[
        df["tourney_level"] == "C"
    ]

drop_cols = [
    "player_a",
    "player_b",
    "surface",
    "match_date",
    "tourney_level",
    TARGET
]

X = (
    df[
        [
            c
            for c in df.columns
            if c not in drop_cols
        ]
    ]
    .select_dtypes(
        include=["number"]
    )
)

features = X.columns.tolist()

y = df[TARGET]

train_mask = (
    df["match_date"] <
    "2023-01-01"
)

X_train = X[train_mask]
X_test = X[~train_mask]

y_train = y[train_mask]
y_test = y[~train_mask]

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

print("Training...")

model.fit(
    X_train,
    y_train
)

proba = model.predict_proba(
    X_test
)[:,1]

pred = (
    proba >= 0.5
).astype(int)

metrics = {

    "roc_auc":
        float(
            roc_auc_score(
                y_test,
                proba
            )
        ),

    "accuracy":
        float(
            accuracy_score(
                y_test,
                pred
            )
        ),

    "training_rows":
        int(len(X_train)),

    "validation_rows":
        int(len(X_test))
}

metadata = {

    "tour":
        args.tour,

    "surface":
        args.surface,

    "engine":
        args.engine,

    "version":
        args.version,

    "rows":
        int(len(df))
}

joblib.dump(
    model,
    model_dir / "model.joblib"
)

joblib.dump(
    imputer,
    model_dir / "imputer.joblib"
)

with open(
    model_dir / "features.json",
    "w"
) as f:
    json.dump(
        features,
        f,
        indent=2
    )

with open(
    model_dir / "metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=2
    )

with open(
    model_dir / "metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

print()
print("MODEL:", model_key)
print("AUC:", round(metrics["roc_auc"],4))
print("ACC:", round(metrics["accuracy"],4))
print("SAVED:", model_dir)
