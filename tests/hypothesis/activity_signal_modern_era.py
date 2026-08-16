
import pandas as pd

elo = pd.read_parquet(
    "data/parquet/player_elo_history.parquet"
)

elo = elo[
    elo["match_date"] >= "2020-01-01"
].copy()

elo = elo[
    (elo["winner_days_inactive"] >= 0)
    &
    (elo["loser_days_inactive"] >= 0)
]

different = elo[
    elo["winner_days_inactive"]
    !=
    elo["loser_days_inactive"]
]

winner_more_active = (
    different["winner_days_inactive"]
    <
    different["loser_days_inactive"]
).mean()

print()

print("MATCHES")
print(len(different))

print()

print("WINNER MORE ACTIVE")
print(
    round(
        winner_more_active * 100,
        2
    ),
    "%"
)
