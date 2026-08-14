import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)
from sklearn.model_selection import train_test_split

SURFACE = "Hard"
USE_ELO = True

if USE_ELO:

    DATASET = (
        "data/parquet/training_matches_with_elo.parquet"
    )

else:

    DATASET = (
        "data/parquet/training_matches.parquet"
    )

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

y = df[
    TARGET
]

print()
print("FEATURES")
print(len(feature_columns))

print()
print("FEATURES AFTER FILTER")
print(len(feature_columns))

imputer = SimpleImputer(
    strategy="median"
)
X = imputer.fit_transform(X)


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

#Modelo inicial
model = XGBClassifier(

   n_estimators=300,

   max_depth=4,

   learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="logloss"
)
#FIN MODELO INCIAL

#MODELO OPTIMIZADO 1
#model = XGBClassifier(

 #   n_estimators=1000,

  #  max_depth=3,

   # learning_rate=0.02,

    #subsample=0.8,

    #colsample_bytree=0.8,

    #min_child_weight=5,

    #random_state=42,

    #eval_metric="logloss"
#)
#FIN MODELO OPTIMIZADO 1

#MODELO OPTIMIZADO 2
# model = XGBClassifier(

#     n_estimators=500,

#     max_depth=5,

#     learning_rate=0.05,

#     subsample=0.9,

#     colsample_bytree=0.9,

#     min_child_weight=3,

#     random_state=42,

#     eval_metric="logloss"
# )
#FIN MODELO OPTIMIZADO 2

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

    "importance": model.feature_importances_

})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print()
print("TOP FEATURES")

print(
    importance[
        [
            "feature",
            "importance"
        ]
    ]
    .head(20)
)

importance.to_csv(
    "ml/reports/xgboost_feature_importance.csv",
    index=False
)

print()
print(
    "Saved: ml/reports/xgboost_feature_importance.csv"
)


