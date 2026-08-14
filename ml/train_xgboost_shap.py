
import pandas as pd
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer

DATASET = (
    "data/parquet/training_matches_with_elo.parquet"
)

SURFACE = "Hard"

TARGET = "target"

SPLIT_DATE = "2023-01-01"

REPORT_DIR = "ml/reports"

print("Loading dataset...")

df = pd.read_parquet(
    DATASET
)

df["match_date"] = pd.to_datetime(
    df["match_date"]
)

df = df[
    df["surface"] == SURFACE
]

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

X = (
    df[
        feature_columns
    ]
    .select_dtypes(
        include=["number"]
    )
)

feature_columns = (
    X.columns.tolist()
)

y = df[
    TARGET
]

train_mask = (
    df["match_date"]
    < SPLIT_DATE
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

X_train = pd.DataFrame(
    imputer.fit_transform(
        X_train
    ),
    columns=feature_columns
)

X_test = pd.DataFrame(
    imputer.transform(
        X_test
    ),
    columns=feature_columns
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

print()
print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(
    model
)

sample = X_test.sample(
    min(
        5000,
        len(X_test)
    ),
    random_state=42
)

print()
print("Calculating SHAP values...")

shap_values = explainer.shap_values(
    sample
)

#
# SHAP SUMMARY
#

plt.figure()

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.tight_layout()

summary_path = (
    f"{REPORT_DIR}/shap_summary.png"
)

plt.savefig(
    summary_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print(
    f"Saved: {summary_path}"
)

#
# SHAP IMPORTANCE CSV
#

importance = pd.DataFrame({

    "feature": sample.columns,

    "mean_abs_shap": (
        abs(shap_values)
        .mean(axis=0)
    )
})

importance = (
    importance
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

importance_path = (
    f"{REPORT_DIR}/shap_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)

print(
    f"Saved: {importance_path}"
)

print()
print("TOP SHAP FEATURES")

print(
    importance.head(20)
)

#
# DEPENDENCE PLOTS
#

dependence_features = [

    "delta_elo",

    "delta_surface_elo",

    "delta_inactivity_days",

    "delta_surface_inactivity_days"
]

for feature in dependence_features:

    if feature not in sample.columns:

        continue

    plt.figure()

    shap.dependence_plot(
        feature,
        shap_values,
        sample,
        show=False
    )

    plt.tight_layout()

    output_path = (
        f"{REPORT_DIR}/shap_{feature}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )

print()
print("SHAP analysis complete.")
