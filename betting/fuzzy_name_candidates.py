
import pandas as pd
from difflib import get_close_matches

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

train_names = sorted(
    set(
        matches["player_a"]
        .apply(normalize_training_name)
    )
)

odds_names = sorted(
    set(
        odds["Winner"]
        .apply(normalize_odds_name)
    )
)

missing = [

    n

    for n in odds_names

    if n not in train_names
]

print()

for name in missing[:100]:

    matches_found = get_close_matches(
        name,
        train_names,
        n=3,
        cutoff=0.75
    )

    if matches_found:

        print()

        print("ODDS :", name)

        print(
            "CANDIDATES :",
            matches_found
        )
