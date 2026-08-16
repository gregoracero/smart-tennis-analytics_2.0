
import pandas as pd

MASTER_PATH = (
    "data/parquet/master_matches.parquet"
)

OUTPUT_PATH = (
    "data/parquet/player_lookup.parquet"
)

print("Loading master matches...")

master = pd.read_parquet(
    MASTER_PATH
)

print()
print("MASTER SHAPE")
print(master.shape)

print()
print("Building winners lookup...")

winners = master[
    [
        "winner_id",
        "winner_name"
    ]
].copy()

winners.columns = [
    "player_id",
    "player_name"
]

print()
print("Building losers lookup...")

losers = master[
    [
        "loser_id",
        "loser_name"
    ]
].copy()

losers.columns = [
    "player_id",
    "player_name"
]

print()
print("Concatenating...")

lookup = pd.concat(
    [
        winners,
        losers
    ],
    ignore_index=True
)

print()
print("Removing nulls...")

lookup = lookup.dropna(
    subset=[
        "player_id",
        "player_name"
    ]
)

lookup["player_id"] = (
    lookup["player_id"]
    .astype(str)
    .str.strip()
)

lookup["player_name"] = (
    lookup["player_name"]
    .astype(str)
    .str.strip()
)

lookup = lookup[
    lookup["player_id"] != ""
]

lookup = lookup[
    lookup["player_name"] != ""
]

print()
print("Removing duplicates...")

lookup = (
    lookup
    .drop_duplicates(
        subset=["player_id"]
    )
    .sort_values(
        "player_name"
    )
)

lookup.to_parquet(
    OUTPUT_PATH,
    index=False
)

print()
print("PLAYERS")
print(len(lookup))

print()

print("SAMPLE")
print(
    lookup.head(10)
)

print()

print("SAVED")
print(OUTPUT_PATH)
