
import pandas as pd

df = pd.read_parquet(
    "tests/reports/top5_temporal_predictions.parquet"
)

bets = df[
    df["top5_edge"] > 0.05
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

elo_bins = [
    -9999,
    -200,
    -100,
    0,
    100,
    200,
    9999
]

elo_labels = [
    "< -200",
    "-200:-100",
    "-100:0",
    "0:100",
    "100:200",
    ">200"
]

activity_bins = [
    -9999,
    -30,
    -14,
    -7,
    0,
    7,
    14,
    30,
    9999
]

activity_labels = [
    "< -30",
    "-30:-14",
    "-14:-7",
    "-7:0",
    "0:7",
    "7:14",
    "14:30",
    ">30"
]

bets["elo_bucket"] = pd.cut(
    bets["delta_elo"],
    bins=elo_bins,
    labels=elo_labels
)

bets["activity_bucket"] = pd.cut(
    bets["delta_surface_inactivity_days"],
    bins=activity_bins,
    labels=activity_labels
)

summary = (
    bets
    .groupby(
        [
            "elo_bucket",
            "activity_bucket"
        ]
    )
    .agg(
        bets=("target","size"),
        hit_rate=("target","mean"),
        roi=("profit","mean")
    )
)

print()
print(summary)

summary.to_csv(
    "tests/reports/core_model_roi_matrix.csv"
)

print()
print(
    "Saved: tests/reports/core_model_roi_matrix.csv"
)
