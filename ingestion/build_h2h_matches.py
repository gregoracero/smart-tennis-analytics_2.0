from pathlib import Path

import pandas as pd

INPUT = Path(
    "data/parquet/analytics_matches.parquet"
)

OUTPUT = Path(
    "data/parquet/h2h_matches.parquet"
)

print("Loading analytics_matches...")

df = pd.read_parquet(INPUT)

df["player_a"] = df[
    ["winner_name", "loser_name"]
].min(axis=1)

df["player_b"] = df[
    ["winner_name", "loser_name"]
].max(axis=1)

out = df[
    [
        "player_a",
        "player_b",
        "tourney_date",
        "tourney_name",
        "surface",
        "round",
        "winner_name",
        "loser_name",
        "score",
        "minutes"
    ]
].copy()

out = out.rename(
    columns={
        "winner_name": "winner",
        "loser_name": "loser"
    }
)

out.to_parquet(
    OUTPUT,
    index=False
)

print()
print("Rows :", len(out))
print("Cols :", len(out.columns))
print()
print(f"Saved -> {OUTPUT}")

