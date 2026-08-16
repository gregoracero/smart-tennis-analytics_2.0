
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

print("Loading...")

matches = pd.read_parquet(
    "data/parquet/training_matches_with_elo.parquet"
)

odds = pd.read_parquet(
    "data/parquet/atp_odds_2020_2026.parquet"
)

matches = matches[
    matches["match_date"] >= "2020-01-01"
].copy()

matches = matches[
    matches["tourney_level"] != "C"
].copy()

matches["match_date"] = pd.to_datetime(
    matches["match_date"]
).dt.date

odds["Date"] = pd.to_datetime(
    odds["Date"]
).dt.date

matches["a_key"] = (
    matches["player_a"]
    .apply(normalize_training_name)
)

matches["b_key"] = (
    matches["player_b"]
    .apply(normalize_training_name)
)

odds["w_key"] = (
    odds["Winner"]
    .apply(normalize_odds_name)
)

odds["l_key"] = (
    odds["Loser"]
    .apply(normalize_odds_name)
)

matches["pair_key"] = matches.apply(
    lambda r:
    "|".join(
        sorted([
            r["a_key"],
            r["b_key"]
        ])
    ),
    axis=1
)

odds["pair_key"] = odds.apply(
    lambda r:
    "|".join(
        sorted([
            r["w_key"],
            r["l_key"]
        ])
    ),
    axis=1
)

matches["lookup"] = (
    matches["match_date"].astype(str)
    + "|"
    + matches["pair_key"]
)

odds["lookup"] = (
    odds["Date"].astype(str)
    + "|"
    + odds["pair_key"]
)

merged = matches.merge(

    odds[
        [
            "lookup",
            "AvgW",
            "AvgL",
            "market_prob_winner",
            "market_prob_loser"
        ]
    ],

    on="lookup",

    how="inner"
)

print()
print("MERGED ROWS")
print(len(merged))

output = (
    "data/parquet/"
    "atp_matches_with_odds.parquet"
)

merged.to_parquet(
    output
)

print()
print(
    f"Saved: {output}"
)
