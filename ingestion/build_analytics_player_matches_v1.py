import pandas as pd
from pathlib import Path

INPUT = Path(
    "data/parquet/analytics_matches.parquet"
)

OUTPUT = Path(
    "data/parquet/analytics_player_matches.parquet"
)

print("Loading all_matches...")

df = pd.read_parquet(INPUT)

print(df.shape)

winner = pd.DataFrame({

    "tourney_date": df["tourney_date"],
    "surface": df["surface"],

    "player": df["winner_name"],
    "opponent": df["loser_name"],

    "player_id": df["winner_id"],
    "opponent_id": df["loser_id"],

    "won_match": 1,

    "minutes": df["minutes"],

    "ace": df["w_ace"],
    "df": df["w_df"],

    "svpt": df["w_svpt"],
    "first_in": df["w_1stIn"],
    "first_won": df["w_1stWon"],
    "second_won": df["w_2ndWon"],

    "bp_saved": df["w_bpSaved"],
    "bp_faced": df["w_bpFaced"]
})

loser = pd.DataFrame({

    "tourney_date": df["tourney_date"],
    "surface": df["surface"],

    "player": df["loser_name"],
    "opponent": df["winner_name"],

    "player_id": df["loser_id"],
    "opponent_id": df["winner_id"],

    "won_match": 0,

    "minutes": df["minutes"],

    "ace": df["l_ace"],
    "df": df["l_df"],

    "svpt": df["l_svpt"],
    "first_in": df["l_1stIn"],
    "first_won": df["l_1stWon"],
    "second_won": df["l_2ndWon"],

    "bp_saved": df["l_bpSaved"],
    "bp_faced": df["l_bpFaced"]
})

result = pd.concat(
    [winner, loser],
    ignore_index=True
)
print()
print("Normalizing numeric columns...")

numeric_cols = [
    "won_match",
    "minutes",
    "ace",
    "df",
    "svpt",
    "first_in",
    "first_won",
    "second_won",
    "bp_saved",
    "bp_faced"
]

for col in numeric_cols:

    result[col] = pd.to_numeric(
        result[col],
        errors="coerce"
    )

print("Numeric normalization completed")

result.to_parquet(
    OUTPUT,
    index=False
)

print()
print("DONE")
print("Rows :", len(result))
print("Cols :", len(result.columns))
print("Saved:", OUTPUT)
