
import pandas as pd

def normalize_training_name(name):

    parts = str(name).strip().split()

    return (
        parts[-1].lower()
        + "_"
        + parts[0][0].lower()
    )

def normalize_odds_name(name):

    parts = (
        str(name)
        .replace(".", "")
        .split()
    )

    return (
        parts[0].lower()
        + "_"
        + parts[1][0].lower()
    )

matches = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

odds = pd.read_parquet(
    "data/parquet/atp_odds_2020_2026.parquet"
)

train_players = set(
    matches["player_a"]
    .apply(normalize_training_name)
).union(
    set(
        matches["player_b"]
        .apply(normalize_training_name)
    )
)

odds_players = set(
    odds["Winner"]
    .apply(normalize_odds_name)
).union(
    set(
        odds["Loser"]
        .apply(normalize_odds_name)
    )
)

unmatched = sorted(
    odds_players - train_players
)

print()
print("UNMATCHED PLAYERS")
print(len(unmatched))

print()

for p in unmatched[:200]:
    print(p)
