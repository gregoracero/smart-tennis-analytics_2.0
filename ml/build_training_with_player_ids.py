
import pandas as pd

TRAINING_PATH = (
    "data/parquet/training_matches.parquet"
)

LOOKUP_PATH = (
    "data/parquet/player_lookup.parquet"
)

OUTPUT_PATH = (
    "data/parquet/training_matches_with_ids.parquet"
)

print("Loading training dataset...")

training = pd.read_parquet(
    TRAINING_PATH
)

print("Loading player lookup...")

lookup = pd.read_parquet(
    LOOKUP_PATH
)

print()
print("TRAINING")
print(training.shape)

print()
print("LOOKUP")
print(lookup.shape)

lookup_a = lookup.rename(
    columns={
        "player_name": "player_a",
        "player_id": "player_a_id"
    }
)

lookup_b = lookup.rename(
    columns={
        "player_name": "player_b",
        "player_id": "player_b_id"
    }
)

print()
print("Merging Player A IDs...")

df = training.merge(
    lookup_a,
    on="player_a",
    how="left"
)

print()
print("Merging Player B IDs...")

df = df.merge(
    lookup_b,
    on="player_b",
    how="left"
)

print()
print("Coverage")

print(
    "Player A missing:",
    df["player_a_id"].isna().sum()
)

print(
    "Player B missing:",
    df["player_b_id"].isna().sum()
)

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print()
print("FINAL SHAPE")
print(df.shape)

print()
print("SAVED")
print(OUTPUT_PATH)
