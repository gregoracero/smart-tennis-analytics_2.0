import pandas as pd

INPUT = "data/parquet/analytics_matches.parquet"

OUT_OVERALL = "data/parquet/h2h_overall.parquet"
OUT_SURFACE = "data/parquet/h2h_surface.parquet"

print("Loading analytics_matches...")

df = pd.read_parquet(INPUT)

df["tourney_date"] = pd.to_numeric(
    df["tourney_date"],
    errors="coerce"
)

# --------------------------------------------------
# Overall H2H
# --------------------------------------------------

overall = {}

# --------------------------------------------------
# Surface H2H
# --------------------------------------------------

surface_h2h = {}

for _, row in df.iterrows():

    winner = str(row["winner_name"])
    loser = str(row["loser_name"])
    surface = str(row["surface"])

    player_a, player_b = sorted(
        [winner, loser]
    )

    overall_key = (
        player_a,
        player_b
    )

    surface_key = (
        player_a,
        player_b,
        surface
    )

    # --------------------------
    # OVERALL
    # --------------------------

    if overall_key not in overall:

        overall[overall_key] = {

            "player_a": player_a,
            "player_b": player_b,

            "matches": 0,

            "wins_a": 0,
            "wins_b": 0,

            "last_match_date": None
        }

    overall[overall_key]["matches"] += 1

    if winner == player_a:
        overall[overall_key]["wins_a"] += 1
    else:
        overall[overall_key]["wins_b"] += 1

    current_date = row["tourney_date"]

    prev = overall[overall_key]["last_match_date"]

    if prev is None or current_date > prev:
        overall[overall_key]["last_match_date"] = current_date

    # --------------------------
    # SURFACE
    # --------------------------

    if surface_key not in surface_h2h:

        surface_h2h[surface_key] = {

            "player_a": player_a,
            "player_b": player_b,
            "surface": surface,

            "matches": 0,

            "wins_a": 0,
            "wins_b": 0,

            "last_match_date": None
        }

    surface_h2h[surface_key]["matches"] += 1

    if winner == player_a:
        surface_h2h[surface_key]["wins_a"] += 1
    else:
        surface_h2h[surface_key]["wins_b"] += 1

    prev = surface_h2h[surface_key]["last_match_date"]

    if prev is None or current_date > prev:
        surface_h2h[surface_key]["last_match_date"] = current_date

# --------------------------------------------------
# DataFrames
# --------------------------------------------------

overall_df = pd.DataFrame(
    overall.values()
)

overall_df["h2h_pct_a"] = round(
    overall_df["wins_a"] * 100 / overall_df["matches"],
    2
)

overall_df["h2h_pct_b"] = round(
    overall_df["wins_b"] * 100 / overall_df["matches"],
    2
)

surface_df = pd.DataFrame(
    surface_h2h.values()
)

surface_df["h2h_pct_a"] = round(
    surface_df["wins_a"] * 100 / surface_df["matches"],
    2
)

surface_df["h2h_pct_b"] = round(
    surface_df["wins_b"] * 100 / surface_df["matches"],
    2
)

overall_df.to_parquet(
    OUT_OVERALL,
    index=False
)

surface_df.to_parquet(
    OUT_SURFACE,
    index=False
)

print()
print("OVERALL")
print(overall_df.shape)

print()

print("SURFACE")
print(surface_df.shape)

print()

print("Saved:")
print(OUT_OVERALL)
print(OUT_SURFACE)
