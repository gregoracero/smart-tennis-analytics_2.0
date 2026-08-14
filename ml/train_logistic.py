import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET = "data/parquet/training_matches.parquet"

SURFACE = "Grass"

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

df = df[
    df["surface"] == SURFACE
]


df = df[
    df["tourney_level"].isin(
        ATP_LEVELS
    )
]

print()
print("SHAPE")
print(df.shape)

TARGET = "target"

DROP_COLUMNS = [
    "player_a",
    "player_b",
    "surface",
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

y = df[
    TARGET
]

print()
print("FEATURES")
print(len(feature_columns))

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(
    X
)

scaler = StandardScaler()

X = scaler.fit_transform(X)

print()

print("SURFACE")
print(SURFACE)

print()

print(
    df["tourney_level"]
    .value_counts()
)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)

print()
print("TRAIN")
print(len(X_train))

print()
print("TEST")
print(len(X_test))

model = LogisticRegression(
    max_iter=2000
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

print()

print("AVG PREDICTION")

print(
    probabilities.mean()
)

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

importance = pd.DataFrame({

    "feature": feature_columns,

    "coefficient": model.coef_[0]

})

importance["abs_coef"] = (
    importance["coefficient"]
    .abs()
)

importance = (
    importance
    .sort_values(
        "abs_coef",
        ascending=False
    )
)

print()
print("TOP FEATURES")

print(
    importance[
        [
            "feature",
            "coefficient"
        ]
    ]
    .head(20)
)

importance.to_csv(
    "ml/reports/logistic_feature_importance.csv",
    index=False
)

print()
print(
    "Saved: ml/reports/logistic_feature_importance.csv"
)

