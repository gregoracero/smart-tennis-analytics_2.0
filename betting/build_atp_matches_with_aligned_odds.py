
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

matches = matches[
    matches["match_date"] >= "2020-01-01"
].copy()

matches = matches[
    matches["tourney_level"] != "C"
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

matches["a_key"] = (
    matches["player_a"]
    .apply(normalize_training_name)
)

matches["b_key"] = (
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
            r["winner_key"],
            r["loser_key"]
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
    odds,
    on="lookup",
    how="inner"
)

#
# Align odds to player_a
#

player_a_is_winner = (
    merged["a_key"]
    ==
    merged["winner_key"]
)

merged["odds_a"] = (
    merged["AvgW"]
    .where(
        player_a_is_winner,
        merged["AvgL"]
    )
)

merged["odds_b"] = (
    merged["AvgL"]
    .where(
        player_a_is_winner,
        merged["AvgW"]
    )
)

merged["market_probability_a"] = (
    1 / merged["odds_a"]
)

merged["market_probability_b"] = (
    1 / merged["odds_b"]
)

output = (
    "data/parquet/"
    "atp_matches_with_aligned_odds.parquet"
)

merged.to_parquet(
    output
)

print()
print("ROWS")
print(len(merged))

print()
print(
    merged[
        [
            "player_a",
            "player_b",
            "odds_a",
            "odds_b",
            "market_probability_a"
        ]
    ]
    .head()
)

print()
print("Saved:", output)
