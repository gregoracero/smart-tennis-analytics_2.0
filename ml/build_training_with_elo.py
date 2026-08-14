"""
Build Training Dataset With Elo Features

Inputs
------
data/parquet/training_matches_with_ids.parquet
data/parquet/player_elo_history.parquet

Outputs
-------
data/parquet/training_matches_with_elo.parquet

Features
--------
elo_a
elo_b

surface_elo_a
surface_elo_b

days_inactive_a
days_inactive_b

surface_days_inactive_a
surface_days_inactive_b

matches_played_a
matches_played_b

surface_matches_played_a
surface_matches_played_b

delta_elo
delta_surface_elo

delta_matches_played
delta_surface_matches_played

delta_inactivity_days
delta_surface_inactivity_days
"""

import pandas as pd


TRAINING_PATH = (
    "data/parquet/training_matches_with_ids.parquet"
)

ELO_PATH = (
    "data/parquet/player_elo_history.parquet"
)

OUTPUT_PATH = (
    "data/parquet/training_matches_with_elo.parquet"
)


print("Loading datasets...")


training = pd.read_parquet(
    TRAINING_PATH
)

elo = pd.read_parquet(
    ELO_PATH
)

training["match_date"] = pd.to_datetime(
    training["match_date"]
)

elo["match_date"] = pd.to_datetime(
    elo["match_date"]
)


print()
print("Creating direct merge...")


direct = training.merge(
    elo,
    left_on=[
        "player_a_id",
        "player_b_id",
        "match_date"
    ],
    right_on=[
        "winner_id",
        "loser_id",
        "match_date"
    ],
    how="inner"
)


direct["elo_a"] = (
    direct["winner_elo_before"]
)

direct["elo_b"] = (
    direct["loser_elo_before"]
)

direct["surface_elo_a"] = (
    direct["winner_surface_elo_before"]
)

direct["surface_elo_b"] = (
    direct["loser_surface_elo_before"]
)

direct["days_inactive_a"] = (
    direct["winner_days_inactive"]
)

direct["days_inactive_b"] = (
    direct["loser_days_inactive"]
)

direct["surface_days_inactive_a"] = (
    direct["winner_surface_days_inactive"]
)

direct["surface_days_inactive_b"] = (
    direct["loser_surface_days_inactive"]
)

direct["matches_played_a"] = (
    direct["winner_matches_played"]
)

direct["matches_played_b"] = (
    direct["loser_matches_played"]
)

direct["surface_matches_played_a"] = (
    direct["winner_surface_matches_played"]
)

direct["surface_matches_played_b"] = (
    direct["loser_surface_matches_played"]
)


print()
print("Creating reverse merge...")


reverse = training.merge(
    elo,
    left_on=[
        "player_a_id",
        "player_b_id",
        "match_date"
    ],
    right_on=[
        "loser_id",
        "winner_id",
        "match_date"
    ],
    how="inner"
)


reverse["elo_a"] = (
    reverse["loser_elo_before"]
)

reverse["elo_b"] = (
    reverse["winner_elo_before"]
)

reverse["surface_elo_a"] = (
    reverse["loser_surface_elo_before"]
)

reverse["surface_elo_b"] = (
    reverse["winner_surface_elo_before"]
)

reverse["days_inactive_a"] = (
    reverse["loser_days_inactive"]
)

reverse["days_inactive_b"] = (
    reverse["winner_days_inactive"]
)

reverse["surface_days_inactive_a"] = (
    reverse["loser_surface_days_inactive"]
)

reverse["surface_days_inactive_b"] = (
    reverse["winner_surface_days_inactive"]
)

reverse["matches_played_a"] = (
    reverse["loser_matches_played"]
)

reverse["matches_played_b"] = (
    reverse["winner_matches_played"]
)

reverse["surface_matches_played_a"] = (
    reverse["loser_surface_matches_played"]
)

reverse["surface_matches_played_b"] = (
    reverse["winner_surface_matches_played"]
)


print()
print("Concatenating...")


df = pd.concat(
    [
        direct,
        reverse
    ],
    ignore_index=True
)


print()
print("Building delta features...")


df["delta_elo"] = (
    df["elo_a"]
    -
    df["elo_b"]
)

df["delta_surface_elo"] = (
    df["surface_elo_a"]
    -
    df["surface_elo_b"]
)

df["delta_matches_played"] = (
    df["matches_played_a"]
    -
    df["matches_played_b"]
)

df["delta_surface_matches_played"] = (
    df["surface_matches_played_a"]
    -
    df["surface_matches_played_b"]
)

df["delta_inactivity_days"] = (
    df["days_inactive_a"]
    -
    df["days_inactive_b"]
)

df["delta_surface_inactivity_days"] = (
    df["surface_days_inactive_a"]
    -
    df["surface_days_inactive_b"]
)


print()
print("Saving...")

columns_to_drop = [

    "winner_id",
    "loser_id",

    "winner_elo_before",
    "loser_elo_before",

    "winner_stored_elo",
    "loser_stored_elo",

    "winner_surface_elo_before",
    "loser_surface_elo_before",

    "winner_surface_stored_elo",
    "loser_surface_stored_elo",

    "winner_is_new_player",
    "loser_is_new_player",

    "winner_days_inactive",
    "loser_days_inactive",

    "winner_surface_days_inactive",
    "loser_surface_days_inactive",

    "winner_matches_played",
    "loser_matches_played",

    "winner_surface_matches_played",
    "loser_surface_matches_played"
]

df = df.drop(
    columns=[
        c
        for c in columns_to_drop
        if c in df.columns
    ]
)

if "surface_x" in df.columns:

    df = df.rename(
        columns={
            "surface_x": "surface"
        }
    )

if "surface_y" in df.columns:

    df = df.drop(
        columns=["surface_y"]
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

print()
print("NEW FEATURES")

for col in [
    "delta_elo",
    "delta_surface_elo",
    "delta_matches_played",
    "delta_surface_matches_played",
    "delta_inactivity_days",
    "delta_surface_inactivity_days"
]:
    print(col)
