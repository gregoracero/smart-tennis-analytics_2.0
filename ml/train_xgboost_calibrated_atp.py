
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)

DATASET = (
    "data/parquet/training_matches_with_elo.parquet"
)

TARGET = "target"

print("Loading dataset...")

df = pd.read_parquet(
    DATASET
)

df["match_date"] = pd.to_datetime(
    df["match_date"]
)

#
# ATP ONLY
#

df = df[
    df["tourney_level"] != "C"
].copy()

print()
print("ATP MATCHES")

print(len(df))

drop_cols = [

    "player_a",
    "player_b",

    "surface",

    "match_date",

    "tourney_level",

    TARGET
]

features = [

    c

    for c in df.columns

    if c not in drop_cols
]

X = (
    df[features]
    .select_dtypes(
        include=["number"]
    )
)

y = df[TARGET]

train_mask = (
    df["match_date"]
    < "2023-01-01"
)

X_train = X[
    train_mask
]

X_test = X[
    ~train_mask
]

y_train = y[
    train_mask
]

y_test = y[
    ~train_mask
]

print()
print("TRAIN")
print(len(X_train))

print()
print("TEST")
print(len(X_test))

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_test = imputer.transform(
    X_test
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

calibrated = CalibratedClassifierCV(

    base_model,

    method="isotonic",

    cv=3
)

calibrated.fit(
    X_train,
    y_train
)

proba = calibrated.predict_proba(
    X_test
)[:,1]

auc = roc_auc_score(
    y_test,
    proba
)

brier = brier_score_loss(
    y_test,
    proba
)

print()
print("ROC AUC")
print(
    round(
        auc,
        4
    )
)

print()
print("BRIER SCORE")
print(
    round(
        brier,
        6
    )
)

#
# Calibration Curve
#

prob_true, prob_pred = calibration_curve(

    y_test,

    proba,

    n_bins=10
)

bins = pd.DataFrame({

    "predicted": prob_pred,

    "actual": prob_true
})

bins.to_csv(

    "ml/reports/probability_bins_atp.csv",

    index=False
)

plt.figure(
    figsize=(8,6)
)

plt.plot(

    prob_pred,

    prob_true,

    marker="o",

    label="ATP Calibration"
)

plt.plot(
    [0,1],
    [0,1],
    "--",
    label="Perfect Calibration"
)

plt.xlabel(
    "Predicted Probability"
)

plt.ylabel(
    "Observed Probability"
)

plt.title(
    "ATP Calibration Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "ml/reports/calibration_curve_atp.png",
    dpi=300
)

print()
print(
    "Saved: ml/reports/calibration_curve_atp.png"
)

print(
    "Saved: ml/reports/probability_bins_atp.csv"
)
