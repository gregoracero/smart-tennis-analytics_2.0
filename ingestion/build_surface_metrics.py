import pandas as pd

INPUT = "data/parquet/player_matches.parquet"
OUTPUT = "data/parquet/player_surface_metrics.parquet"

print("Loading player_matches...")

df = pd.read_parquet(INPUT)

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

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

metrics = (

    df.groupby(
        ["player", "surface"]
    )

    .agg(

        matches=("player","count"),

        wins=("won_match","sum"),

        avg_minutes=("minutes","mean"),

        avg_aces=("ace","mean"),

        avg_double_faults=("df","mean"),

        first_serve_in_pct=(
            "first_in",
            "mean"
        ),

        first_serve_won_pct=(
            "first_won",
            "mean"
        ),

        second_serve_won_pct=(
            "second_won",
            "mean"
        ),

        bp_saved_avg=(
            "bp_saved",
            "mean"
        )

    )

    .reset_index()

)

metrics["losses"] = (
    metrics["matches"]
    - metrics["wins"]
)

metrics.to_parquet(
    OUTPUT,
    index=False
)

print()
print("DONE")
print(metrics.shape)
print(f"Saved: {OUTPUT}")
