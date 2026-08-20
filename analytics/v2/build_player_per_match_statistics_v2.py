from pathlib import Path

import numpy as np
import pandas as pd

INPUT = (
    "data/parquet/v2/"
    "master_matches_v2.parquet"
)

OUTPUT = (
    "data/parquet/v2/"
    "player_per_match_statistics_v2.parquet"
)


print("Loading master matches...")

master = pd.read_parquet(INPUT)

# ==========================================================
# HELPERS
# ==========================================================

def safe_value(value):

    if pd.isna(value):
        return np.nan

    return value


# ==========================================================
# BUILD
# ==========================================================

print("Building player per match statistics...")

rows = []

for idx, (_, row) in enumerate(
    master.iterrows(),
    start=1
):

    if idx % 1000 == 0:

        print(
            f"Processed {idx:,}"
        )

    # ======================================================
    # WINNER RECORD
    # ======================================================

    winner = {

        # ----------------------------------
        # Match Identity
        # ----------------------------------

        "master_match_id":
            row["master_match_id"],

        "match_date":
            row["tml_tourney_date"],

        "season":
            row["td_season"],

        "surface":
            row["tml_surface"],

        "tournament_id":
            row["tournament_id"],

        "tournament_key":
            row["tournament_key"],

        "tournament_name":
            row["tml_tourney_name"],

        "tournament_level":
            row["tml_tourney_level"],

        "round":
            row["tml_round"],

        # ----------------------------------
        # Player
        # ----------------------------------

        "player_key":
            row["winner_player_key"],

        "player_name":
            row["tml_winner_name"],

        "player_rank":
            row["tml_winner_rank"],

        "player_rank_points":
            row["tml_winner_rank_points"],

        "player_age":
            row["tml_winner_age"],

        "player_height":
            row["tml_winner_ht"],

        "player_hand":
            row["tml_winner_hand"],

        # ----------------------------------
        # Opponent
        # ----------------------------------

        "opponent_player_key":
            row["loser_player_key"],

        "opponent_name":
            row["tml_loser_name"],

        "opponent_rank":
            row["tml_loser_rank"],

        "opponent_rank_points":
            row["tml_loser_rank_points"],

        "opponent_age":
            row["tml_loser_age"],

        # ----------------------------------
        # Result
        # ----------------------------------

        "result":
            "W",

        "match":
            1,

        "win":
            1,

        "loss":
            0,

        # ----------------------------------
        # Sets
        # ----------------------------------

        "sets_won":
            safe_value(
                row["td_Wsets"]
            ),

        "sets_lost":
            safe_value(
                row["td_Lsets"]
            ),

        # ----------------------------------
        # Service
        # ----------------------------------

        "aces":
            safe_value(
                row["tml_w_ace"]
            ),

        "double_faults":
            safe_value(
                row["tml_w_df"]
            ),

        "first_serve_in":
            safe_value(
                row["tml_w_1stIn"]
            ),

        "first_serve_points_won":
            safe_value(
                row["tml_w_1stWon"]
            ),

        "second_serve_points_won":
            safe_value(
                row["tml_w_2ndWon"]
            ),

        "service_points_played":
            safe_value(
                row["tml_w_svpt"]
            ),

        "service_games_played":
            safe_value(
                row["tml_w_SvGms"]
            ),

        # ----------------------------------
        # Break Points
        # ----------------------------------

        "break_points_faced":
            safe_value(
                row["tml_w_bpFaced"]
            ),

        "break_points_saved":
            safe_value(
                row["tml_w_bpSaved"]
            ),

        # ----------------------------------
        # Derived
        # ----------------------------------

        "breaks_suffered":
            max(
                0,
                safe_value(
                    row["tml_w_bpFaced"]
                )
                -
                safe_value(
                    row["tml_w_bpSaved"]
                )
            ),

        "break_points_chances":
            safe_value(
                row["tml_l_bpFaced"]
            ),

        "break_points_converted":
            max(
                0,
                safe_value(
                    row["tml_l_bpFaced"]
                )
                -
                safe_value(
                    row["tml_l_bpSaved"]
                )
            ),

        # ----------------------------------
        # Minutes
        # ----------------------------------

        "minutes":
            safe_value(
                row["tml_minutes"]
            ),

        # ----------------------------------
        # Betting
        # ----------------------------------

        "player_b365":
            row["td_B365W"],

        "opponent_b365":
            row["td_B365L"]

    }

    winner["service_games_won"] = (

        winner["service_games_played"]

        -

        winner["breaks_suffered"]

    )

    rows.append(
        winner
    )

    # ======================================================
    # LOSER RECORD
    # ======================================================

    loser = winner.copy()

    loser.update({

        "player_key":
            row["loser_player_key"],

        "player_name":
            row["tml_loser_name"],

        "player_rank":
            row["tml_loser_rank"],

        "player_rank_points":
            row["tml_loser_rank_points"],

        "player_age":
            row["tml_loser_age"],

        "player_height":
            row["tml_loser_ht"],

        "player_hand":
            row["tml_loser_hand"],

        "opponent_player_key":
            row["winner_player_key"],

        "opponent_name":
            row["tml_winner_name"],

        "opponent_rank":
            row["tml_winner_rank"],

        "opponent_rank_points":
            row["tml_winner_rank_points"],

        "opponent_age":
            row["tml_winner_age"],

        "result":
            "L",

        "win":
            0,

        "loss":
            1,

        "sets_won":
            safe_value(
                row["td_Lsets"]
            ),

        "sets_lost":
            safe_value(
                row["td_Wsets"]
            ),

        "aces":
            safe_value(
                row["tml_l_ace"]
            ),

        "double_faults":
            safe_value(
                row["tml_l_df"]
            ),

        "first_serve_in":
            safe_value(
                row["tml_l_1stIn"]
            ),

        "first_serve_points_won":
            safe_value(
                row["tml_l_1stWon"]
            ),

        "second_serve_points_won":
            safe_value(
                row["tml_l_2ndWon"]
            ),

        "service_points_played":
            safe_value(
                row["tml_l_svpt"]
            ),

        "service_games_played":
            safe_value(
                row["tml_l_SvGms"]
            ),

        "break_points_faced":
            safe_value(
                row["tml_l_bpFaced"]
            ),

        "break_points_saved":
            safe_value(
                row["tml_l_bpSaved"]
            ),

        "breaks_suffered":
            max(
                0,
                safe_value(
                    row["tml_l_bpFaced"]
                )
                -
                safe_value(
                    row["tml_l_bpSaved"]
                )
            ),

        "break_points_chances":
            safe_value(
                row["tml_w_bpFaced"]
            ),

        "break_points_converted":
            max(
                0,
                safe_value(
                    row["tml_w_bpFaced"]
                )
                -
                safe_value(
                    row["tml_w_bpSaved"]
                )
            ),

        "player_b365":
            row["td_B365L"],

        "opponent_b365":
            row["td_B365W"]

    })

    loser["service_games_won"] = (

        loser["service_games_played"]

        -

        loser["breaks_suffered"]

    )

    rows.append(
        loser
    )
    
    
   

# ==========================================================
# SAVE
# ==========================================================

player_match = pd.DataFrame(
    rows
)

# ==========================================================
# ODDS NORMALIZATION
# ==========================================================

for col in [
    "player_b365",
    "opponent_b365"
]:

    player_match[col] = pd.to_numeric(
        player_match[col],
        errors="coerce"
    )

Path(
    "data/parquet/v2"
).mkdir(
    parents=True,
    exist_ok=True
)

player_match.to_parquet(
    OUTPUT,
    index=False
)

print()
print("=" * 80)
print("PLAYER PER MATCH STATISTICS V2")
print("=" * 80)

print()
print("ROWS:", len(player_match))

print()
print("COLS:", len(player_match.columns))

print()
print("PLAYERS:",
      player_match["player_key"]
      .nunique())

print()
print("SAVED:", OUTPUT)