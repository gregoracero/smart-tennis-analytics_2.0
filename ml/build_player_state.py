
import pandas as pd

elo = pd.read_parquet(
    "data/parquet/player_elo_history.parquet"
)

elo = elo.sort_values(
    "match_date"
)

print()
print("ROWS")
print(len(elo))

print()
print("LAST DATE")
print(
    elo["match_date"].max()
)

winner_rows = elo[
    [
        "winner_id",
        "winner_elo_before",
        "winner_surface_elo_before",
        "winner_days_inactive",
        "winner_surface_days_inactive",
        "winner_matches_played",
        "winner_surface_matches_played",
        "match_date",
        "surface"
    ]
].copy()

winner_rows.columns = [
    "player_id",
    "elo",
    "surface_elo",
    "days_inactive",
    "surface_days_inactive",
    "matches_played",
    "surface_matches_played",
    "match_date",
    "surface"
]

loser_rows = elo[
    [
        "loser_id",
        "loser_elo_before",
        "loser_surface_elo_before",
        "loser_days_inactive",
        "loser_surface_days_inactive",
        "loser_matches_played",
        "loser_surface_matches_played",
        "match_date",
        "surface"
    ]
].copy()

loser_rows.columns = winner_rows.columns

players = pd.concat(
    [
        winner_rows,
        loser_rows
    ],
    ignore_index=True
)

players = players.sort_values(
    "match_date"
)

state = (
    players
    .groupby("player_id")
    .tail(1)
    .copy()
)

state.to_parquet(
    "data/parquet/player_state.parquet",
    index=False
)

print()
print("PLAYERS")
print(len(state))

print()
print("SAVED")
print(
    "data/parquet/player_state.parquet"
)
