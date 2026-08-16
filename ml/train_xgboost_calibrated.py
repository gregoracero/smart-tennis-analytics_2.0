
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

df = pd.read_parquet(DATASET)

df["match_date"] = pd.to_datetime(
    df["match_date"]
)

df = df[
    df["surface"] == "Hard"
]

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

base_model = XGBClassifier(

    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    eval_metric="logloss"
)

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
print(round(auc,4))

print()
print("BRIER SCORE")
print(round(brier,6))

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
    "ml/reports/probability_bins.csv",
    index=False
)

plt.figure(figsize=(8,6))

plt.plot(
    prob_pred,
    prob_true,
    marker="o"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title("Calibration Curve")

plt.savefig(
    "ml/reports/calibration_curve.png",
    dpi=300
)

print()
print("Saved calibration artifacts")
