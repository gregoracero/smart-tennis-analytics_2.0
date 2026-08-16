
import pandas as pd

df = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

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

df["bucket"] = pd.cut(
    df["delta_inactivity_days"],
    bins=bins,
    labels=labels
)

summary = (
    df
    .groupby("bucket")
    .agg(
        matches=("target","size"),
        win_rate=("target","mean"),
        avg_delta_inactivity=(
            "delta_inactivity_days",
            "mean"
        )
    )
)

print()
print("=" * 100)
print("DELTA INACTIVITY VS TARGET")
print("=" * 100)

print(summary)
