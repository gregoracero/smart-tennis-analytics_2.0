
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

elo["delta_elo"] = (
    elo["winner_elo_before"]
    -
    elo["loser_elo_before"]
)

subset = elo[
    elo["delta_elo"].abs() < 50
]

print()
print("=" * 80)
print("REAL MATCHES - ELO NEUTRAL")
print("=" * 80)

print()

print(
    "Matches:",
    len(subset)
)

print()

print(
    "Winner more active:",
    round(
        (
            subset["delta_inactivity"] < 0
        ).mean() * 100,
        2
    ),
    "%"
)

print(
    "Winner less active:",
    round(
        (
            subset["delta_inactivity"] > 0
        ).mean() * 100,
        2
    ),
    "%"
)
