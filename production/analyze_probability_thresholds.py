
import pandas as pd

from production.services.model_loader import (
    load_model
)

assets = load_model(
    tour="atp",
    surface="hard",
    engine="xgboost",
    version="v1"
)

df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

features = assets["features"]

for col in features:

    if col not in df.columns:

        df[col] = None

X = df[features]

X = assets["imputer"].transform(X)

probabilities = (
    assets["model"]
    .predict_proba(X)[:, 1]
)

results = pd.DataFrame({

    "actual":
        df["target"],

    "probability":
        probabilities
})

print()
print("=" * 80)
print("PROBABILITY THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = [

    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95

]

rows = []

for threshold in thresholds:

    subset = results[
        results["probability"]
        >= threshold
    ]

    if len(subset) == 0:

        continue

    win_rate = (
        subset["actual"]
        .mean()
    )

    rows.append({

        "threshold":
            threshold,

        "matches":
            len(subset),

        "win_rate":
            round(
                win_rate,
                4
            ),

        "avg_probability":
            round(
                subset[
                    "probability"
                ].mean(),
                4
            )
    })

summary = pd.DataFrame(
    rows
)

print()
print(summary)
print()

print("=" * 80)
print("PROBABILITY BUCKET ANALYSIS")
print("=" * 80)

results["bucket"] = pd.cut(

    results["probability"],

    bins=[
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00
    ]
)

bucket_summary = (

    results

    .groupby(
        "bucket",
        observed=False
    )

    .agg(

        matches=(
            "actual",
            "count"
        ),

        win_rate=(
            "actual",
            "mean"
        ),

        avg_probability=(
            "probability",
            "mean"
        )

    )

    .reset_index()
)

print()
print(bucket_summary)
