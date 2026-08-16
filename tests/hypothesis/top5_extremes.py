
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions_unique.parquet"
)

top = (
    df
    .sort_values(
        "top5_probability",
        ascending=False
    )
    .head(100)
)

print()
print("=" * 70)
print("TOP 100 MOST CONFIDENT PREDICTIONS")
print("=" * 70)

print()

print(
    "HIT RATE:",
    round(
        top["target"].mean() * 100,
        2
    ),
    "%"
)

print()

print(
    top[
        [
            "top5_probability",
            "delta_surface_inactivity_days",
            "delta_elo",
            "delta_inactivity_days"
        ]
    ]
    .describe()
)
