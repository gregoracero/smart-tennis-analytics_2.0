
import pandas as pd

def normalize_training_name(name):

    parts = name.strip().split()

    first = parts[0]
    last = parts[-1]

    return (
        last.lower()
        + "_"
        + first[0].lower()
    )

def normalize_odds_name(name):

    parts = name.replace(".", "").split()

    surname = parts[0]
    initial = parts[1][0]

    return (
        surname.lower()
        + "_"
        + initial.lower()
    )

matches = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

odds = pd.read_parquet(
    "data/parquet/atp_odds_2020_2026.parquet"
)

matches["player_a_key"] = (
    matches["player_a"]
    .apply(normalize_training_name)
)

matches["player_b_key"] = (
    matches["player_b"]
    .apply(normalize_training_name)
)

odds["winner_key"] = (
    odds["Winner"]
    .apply(normalize_odds_name)
)

odds["loser_key"] = (
    odds["Loser"]
    .apply(normalize_odds_name)
)

train_players = set(
    matches["player_a_key"]
).union(
    set(matches["player_b_key"])
)

odds_players = set(
    odds["winner_key"]
).union(
    set(odds["loser_key"])
)

overlap = (
    train_players
    &
    odds_players
)

print()
print("TRAIN PLAYERS")
print(len(train_players))

print()
print("ODDS PLAYERS")
print(len(odds_players))

print()
print("COMMON PLAYERS")
print(len(overlap))

print()
print(
    "PLAYER COVERAGE:",
    round(
        100 * len(overlap)
        / len(odds_players),
        2
    ),
    "%"
)
