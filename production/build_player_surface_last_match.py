
import pandas as pd

elo = pd.read_parquet(
    "data/parquet/player_elo_history.parquet"
)

elo = elo.sort_values(
    "match_date"
)

winner_rows = elo[
    [
        "winner_id",
        "surface",
        "winner_surface_elo_before",
        "winner_surface_matches_played",
        "match_date"
    ]
].copy()

winner_rows.columns = [
    "player_id",
    "surface",
    "surface_elo",
    "surface_matches_played",
    "last_surface_match_date"
]

loser_rows = elo[
    [
        "loser_id",
        "surface",
        "loser_surface_elo_before",
        "loser_surface_matches_played",
        "match_date"
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

players = players[
    players["surface"].notna()
]

surface_state = (
    players
    .sort_values(
        "last_surface_match_date"
    )
    .groupby(
        [
            "player_id",
            "surface"
        ]
    )
    .tail(1)
    .copy()
)

surface_state.to_parquet(
    "data/parquet/player_surface_last_match.parquet",
    index=False
)

print()
print("ROWS")
print(len(surface_state))

print()

print("PLAYERS")
print(
    surface_state["player_id"]
    .nunique()
)

print()

print("SURFACES")
print(
    surface_state["surface"]
    .value_counts()
)

print()

print(
    "Saved: data/parquet/player_surface_last_match.parquet"
)
