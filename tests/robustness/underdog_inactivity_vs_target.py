
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

dogs = df[
    df["market_probability_a"] < 0.50
].copy()

bins = [
    -1000,
    -60,
    -30,
    -14,
    -7,
    0,
    7,
    14,
    30,
    60,
    1000
]

labels = [
    "< -60",
    "-60 to -30",
    "-30 to -14",
    "-14 to -7",
    "-7 to 0",
    "0 to 7",
    "7 to 14",
    "14 to 30",
    "30 to 60",
    "> 60"
]

dogs["bucket"] = pd.cut(
    dogs["delta_inactivity_days"],
    bins=bins,
    labels=labels
)

summary = (
    dogs
    .groupby("bucket")
    .agg(
        matches=("target","size"),
        win_rate=("target","mean"),
        avg_odds=("odds_a","mean")
    )
)

print()
print("=" * 100)
print("UNDERDOGS: INACTIVITY VS TARGET")
print("=" * 100)

print(summary)
