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

matches = matches[
    matches["match_date"] >= "2020-01-01"
].copy()

matches["match_date"] = (
    pd.to_datetime(
        matches["match_date"]
    )
    .dt.date
)

odds["Date"] = (
    pd.to_datetime(
        odds["Date"]
    )
    .dt.date
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

matches["match_key"] = (
    matches["match_date"].astype(str)
    + "|"
    + matches["player_a_key"]
    + "|"
    + matches["player_b_key"]
)

odds["match_key"] = (
    odds["Date"].astype(str)
    + "|"
    + odds["winner_key"]
    + "|"
    + odds["loser_key"]
)

matched = matches[
    matches["match_key"].isin(
        set(
            odds["match_key"]
        )
    )
]

print()
print("MATCHES FROM 2020")
print(len(matches))

print()
print("MATCHED")
print(len(matched))

print()
print(
    "MATCH RATE:",
    round(
        100
        * len(matched)
        / len(matches),
        2
    ),
    "%"
)