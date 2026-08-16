
import pandas as pd

MAX_DAY_DIFF = 7

def normalize_training_name(name):
    parts = str(name).strip().split()
    return parts[-1].lower() + parts[0][0].lower()

def normalize_odds_name(name):
    parts = str(name).replace(".", "").split()
    return parts[0].lower() + parts[1][0].lower()

print("Loading datasets...")

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
)

odds["Date"] = pd.to_datetime(
    odds["Date"]
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
    lambda r: "|".join(
        sorted([r["a_key"], r["b_key"]])
    ),
    axis=1
)

odds["pair_key"] = odds.apply(
    lambda r: "|".join(
        sorted([r["winner_key"], r["loser_key"]])
    ),
    axis=1
)

matches = matches.reset_index(drop=True)
matches["match_id"] = matches.index

odds = odds.reset_index(drop=True)
odds["odds_id"] = odds.index

print("Building candidate matches...")

merged = matches.merge(
    odds,
    on="pair_key",
    how="inner"
)

merged["day_diff"] = (
    merged["match_date"]
    - merged["Date"]
).dt.days.abs()

merged = merged[
    merged["day_diff"] <= MAX_DAY_DIFF
].copy()

print("Candidates:", len(merged))

merged = merged.sort_values(
    ["match_id", "day_diff"]
)

best = (
    merged
    .groupby("match_id", as_index=False)
    .first()
)

print("Best matches:", len(best))

player_a_is_winner = (
    best["a_key"]
    == best["winner_key"]
)

best["odds_a"] = best["AvgW"].where(
    player_a_is_winner,
    best["AvgL"]
)

best["odds_b"] = best["AvgL"].where(
    player_a_is_winner,
    best["AvgW"]
)

raw_a = 1 / best["odds_a"]
raw_b = 1 / best["odds_b"]

best["market_probability_a"] = (
    raw_a / (raw_a + raw_b)
)

best["market_probability_b"] = (
    raw_b / (raw_a + raw_b)
)

output = (
    "data/parquet/"
    "atp_matches_with_nearest_odds.parquet"
)

best.to_parquet(output)

print()
print("ROWS:", len(best))
print()

print(
    best["day_diff"]
    .describe()
)

print()
print("Saved:", output)
