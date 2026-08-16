
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

winner_more_active = (
    elo["delta_inactivity"] < 0
).mean()

winner_less_active = (
    elo["delta_inactivity"] > 0
).mean()

same_activity = (
    elo["delta_inactivity"] == 0
).mean()

print()
print("=" * 80)
print("REAL MATCHES - ACTIVITY ADVANTAGE")
print("=" * 80)

print()
print(
    "Winner more active:",
    round(
        winner_more_active * 100,
        2
    ),
    "%"
)

print(
    "Winner less active:",
    round(
        winner_less_active * 100,
        2
    ),
    "%"
)

print(
    "Same activity:",
    round(
        same_activity * 100,
        2
    ),
    "%"
)
