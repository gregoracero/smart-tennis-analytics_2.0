import pandas as pd

INPUT = "data/parquet/analytics_player_matches.parquet"
OUTPUT = "data/parquet/rolling_metrics.parquet"

print("Loading analytics_player_matches...")

df = pd.read_parquet(INPUT)

df["tourney_date"] = pd.to_datetime(
    df["tourney_date"].astype(str),
    format="%Y%m%d",
    errors="coerce"
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
    "bp_saved",
    "bp_faced"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

rows = []

players = df["player"].dropna().unique()

for player in players:

    player_df = df[
        df["player"] == player
    ]

    for surface in player_df["surface"].dropna().unique():

        subset = player_df[
            player_df["surface"] == surface
        ].sort_values(
            "tourney_date",
            ascending=False
        )

        last10 = subset.head(10)

        if len(last10) == 0:
            continue

        rows.append({

            "player": player,
            "surface": surface,
            "window": "LAST_10_MATCHES",

            "matches": len(last10),

            "wins": last10["won_match"].sum(),

            "win_pct":
                round(
                    last10["won_match"].mean() * 100,
                    2
                ),

            "avg_minutes":
                round(
                    last10["minutes"].mean(),
                    2
                ),

            "avg_ace":
                round(
                    last10["ace"].mean(),
                    2
                ),

            "first_serve_won":
                round(
                    last10["first_won"].mean(),
                    2
                ),

            "second_serve_won":
                round(
                    last10["second_won"].mean(),
                    2
                ),

            "bp_saved":
                round(
                    last10["bp_saved"].mean(),
                    2
                )

        })

result = pd.DataFrame(rows)

result.to_parquet(
    OUTPUT,
    index=False
)

print()
print("DONE")
print(result.shape)
print(f"Saved: {OUTPUT}")
