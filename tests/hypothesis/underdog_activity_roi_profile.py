
import pandas as pd

df = pd.read_parquet(
    "data/parquet/temporal_predictions_2023_plus.parquet"
)

bets = df[
    (df["edge"] > 0.05)
    &
    (df["market_probability_a"] < 0.50)
].copy()

bets["profit"] = bets.apply(
    lambda r:
    r["odds_a"] - 1
    if r["target"] == 1
    else -1,
    axis=1
)

bins = [
    -1000,
    -30,
    -14,
    -7,
    0,
    7,
    14,
    30,
    1000
]

labels = [
    "< -30",
    "-30 to -14",
    "-14 to -7",
    "-7 to 0",
    "0 to 7",
    "7 to 14",
    "14 to 30",
    "> 30"
]

bets["bucket"] = pd.cut(
    bets["delta_inactivity_days"],
    bins=bins,
    labels=labels
)

summary = (
    bets
    .groupby("bucket")
    .agg(
        bets=("target","size"),
        hit_rate=("target","mean"),
        roi=("profit","mean"),
        avg_odds=("odds_a","mean"),
        avg_delta_elo=("delta_elo","mean")
    )
)

print()
print(summary)
