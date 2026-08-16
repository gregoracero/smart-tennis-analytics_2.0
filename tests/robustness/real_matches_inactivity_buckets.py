
import pandas as pd

elo = pd.read_parquet(
    "data/parquet/player_elo_history.parquet"
)

elo = elo[
    (elo["winner_days_inactive"] >= 0)
    &
    (elo["loser_days_inactive"] >= 0)
].copy()

elo["delta_inactivity"] = (
    elo["winner_days_inactive"]
    -
    elo["loser_days_inactive"]
)

bins = [
    -10000,
    -60,
    -30,
    -14,
    -7,
    0,
    7,
    14,
    30,
    60,
    10000
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

elo["bucket"] = pd.cut(
    elo["delta_inactivity"],
    bins=bins,
    labels=labels
)

summary = (
    elo
    .groupby("bucket")
    .size()
)

print()
print("=" * 80)
print("REAL MATCHES - WINNER INACTIVITY")
print("=" * 80)

print()
print(summary)
