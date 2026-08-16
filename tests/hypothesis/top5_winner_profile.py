
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_predictions_unique.parquet"
)

bets = df[
    df["top5_probability"] >= 0.80
]

print()
print("=" * 80)
print("TOP5 HIGH-CONFIDENCE PROFILE")
print("=" * 80)

columns = [

    "delta_surface_inactivity_days",

    "delta_inactivity_days",

    "delta_elo"

]

print()

print(
    bets[
        columns
    ].describe()
)
