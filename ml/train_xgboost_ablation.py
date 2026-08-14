import pandas as pd

from xgboost import XGBClassifier

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_auc_score
)

DATASET = (
    "data/parquet/training_matches_with_elo.parquet"
)

SURFACE = "Hard"

SPLIT_DATE = "2023-01-01"

TARGET = "target"

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

df["match_date"] = pd.to_datetime(
    df["match_date"]
)

df = df[
    df["surface"] == SURFACE
]

df = df[
    df["tourney_level"].isin(
        ATP_LEVELS
    )
]

ALL_FEATURES = [

    c

    for c in df.columns

    if c not in [
        "player_a",
        "player_b",
        "surface",
        "match_date",
        "tourney_level",
        TARGET
    ]
]

EXPERIMENTS = {

    "elo_only": [

        "elo_a",
        "elo_b",
        "surface_elo_a",
        "surface_elo_b",
        "delta_elo",
        "delta_surface_elo"
    ],

    "inactivity_only": [

        "days_inactive_a",
        "days_inactive_b",

        "surface_days_inactive_a",
        "surface_days_inactive_b",

        "delta_inactivity_days",
        "delta_surface_inactivity_days"
    ],

    "ranking_only": [

        "rank_a",
        "rank_b",
        "delta_rank",

        "rank_points_a",
        "rank_points_b",
        "delta_rank_points"
    ],

    "winrate_only": [

        "delta_win_pct_5",
        "delta_win_pct_10",
        "delta_win_pct_20",
        "delta_win_pct_50"
    ],

    "service_only": [

        c

        for c in ALL_FEATURES

        if (
            "serve" in c.lower()
            or
            "service_points" in c.lower()
            or
            "bp_saved" in c.lower()
        )
    ],

    "return_only": [

        c

        for c in ALL_FEATURES

        if (
            "return" in c.lower()
            or
            "break_conversion" in c.lower()
        )
    ],

    "all_without_elo": [

        c

        for c in ALL_FEATURES

        if "elo" not in c.lower()
    ],

    "all_without_inactivity": [

        c

        for c in ALL_FEATURES

        if "inactive" not in c.lower()
    ],

    "all": ALL_FEATURES
}

results = []

for experiment_name, features in (
    EXPERIMENTS.items()
):

    print()
    print("=" * 60)
    print(experiment_name)

    features = [

        c

        for c in features

        if c in df.columns
    ]

    X = df[
        features
    ]

    X = X.select_dtypes(
        include=["number"]
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

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = (
        imputer.fit_transform(
            X_train
        )
    )

    X_test = (
        imputer.transform(
            X_test
        )
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

    probabilities = (
        model.predict_proba(
            X_test
        )[:,1]
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        "AUC:",
        round(
            auc,
            4
        )
    )

    results.append({
        "experiment": experiment_name,
        "features": len(features),
        "auc": auc
    })

results = pd.DataFrame(
    results
)

results = results.sort_values(
    "auc",
    ascending=False
)

print()
print(results)

results.to_csv(
    "ml/reports/ablation_results.csv",
    index=False
)

print()
print(
    "Saved: ml/reports/ablation_results.csv"
)