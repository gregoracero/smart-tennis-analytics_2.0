import pandas as pd
from pathlib import Path

INPUT = Path(
    "data/parquet/analytics_matches_tiebreaks.parquet"
)

OUTPUT = Path(
    "data/parquet/analytics_player_matches.parquet"
)

print("Loading analytics_matches_tiebreaks...")

df = pd.read_parquet(INPUT)

bp_cols = [
    "w_bpSaved",
    "w_bpFaced",
    "l_bpSaved",
    "l_bpFaced"
]

for col in bp_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

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
    "service_games": df["w_SvGms"],

    "opp_svpt": df["l_svpt"],
    "opp_first_in": df["l_1stIn"],
    "opp_first_won": df["l_1stWon"],
    "opp_second_won": df["l_2ndWon"],

    "opp_bp_faced": df["l_bpFaced"],
    "opp_bp_saved": df["l_bpSaved"],

    "bp_saved": df["w_bpSaved"],
    "bp_faced": df["w_bpFaced"],

    "tb_played": df["winner_tb_played"],
    "tb_won": df["winner_tb_won"],

    "break_points_generated": df["l_bpFaced"],
    "breaks_converted":
        df["l_bpFaced"] - df["l_bpSaved"]

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
    "service_games": df["l_SvGms"],

    "opp_svpt": df["w_svpt"],
    "opp_first_in": df["w_1stIn"],
    "opp_first_won": df["w_1stWon"],
    "opp_second_won": df["w_2ndWon"],

    "opp_bp_faced": df["w_bpFaced"],
    "opp_bp_saved": df["w_bpSaved"],

    "bp_saved": df["l_bpSaved"],
    "bp_faced": df["l_bpFaced"],

    "tb_played": df["loser_tb_played"],
    "tb_won": df["loser_tb_won"],

    "break_points_generated": df["w_bpFaced"],
    "breaks_converted":
        df["w_bpFaced"] - df["w_bpSaved"]

})

result = pd.concat(
    [winner, loser],
    ignore_index=True
)

numeric_cols = [
    "won_match",
    "minutes",

    "ace",
    "df",

    "svpt",
    "first_in",
    "first_won",
    "second_won",

    "service_games",

    "bp_saved",
    "bp_faced",

    "tb_played",
    "tb_won",

    "break_points_generated",
    "breaks_converted",

    "opp_svpt",
    "opp_first_in",
    "opp_first_won",
    "opp_second_won",

    "opp_bp_faced",
    "opp_bp_saved"
]

for col in numeric_cols:

    result[col] = pd.to_numeric(
        result[col],
        errors="coerce"
    )

result.to_parquet(
    OUTPUT,
    index=False
)

print()
print("DONE")
print(result.shape)
print(f"Saved: {OUTPUT}")
