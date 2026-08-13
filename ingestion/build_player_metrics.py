import pandas as pd

INPUT = "data/parquet/analytics_player_matches.parquet"
OUTPUT = "data/parquet/player_metrics.parquet"

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
    "service_games",

    "opp_svpt",
    "opp_first_in",
    "opp_first_won",
    "opp_second_won",

    "opp_bp_faced",
    "opp_bp_saved",
    "bp_saved",
    "bp_faced",

    "tb_played",
    "tb_won",

    "break_points_generated",
    "breaks_converted"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# --------------------------------------------------
# DATA QUALITY FILTERS
# --------------------------------------------------

# Estadísticas de servicio inválidas

df = df[
    (
        (df["svpt"] > 0)
        &
        (df["first_in"] >= 0)
        &
        (df["first_in"] <= df["svpt"])
    )
    |
    (
        df["svpt"].isna()
        |
        df["first_in"].isna()
    )
]

# Estadísticas de break inválidas

df = df[
    (
        (df["bp_faced"] >= 0)
        &
        (df["bp_saved"] >= 0)
        &
        (df["bp_saved"] <= df["bp_faced"])
    )
    |
    (
        df["bp_saved"].isna()
        |
        df["bp_faced"].isna()
    )
]

print()
print("After data quality filters:")
print(df.shape)

rows = []

latest_date = df["tourney_date"].max()

WINDOWS = {
    "LAST_10_MATCHES": None,
    "LAST_3_MONTHS": latest_date - pd.DateOffset(months=3),
    "LAST_6_MONTHS": latest_date - pd.DateOffset(months=6),
    "CAREER_SURFACE": pd.Timestamp("1900-01-01")
}

for player in df["player"].dropna().unique():

    player_df = df[
        df["player"] == player
    ]

    for surface in player_df["surface"].dropna().unique():

        surface_df = player_df[
            player_df["surface"] == surface
        ]

        for window_name, cutoff in WINDOWS.items():

            if window_name == "LAST_10_MATCHES":

                subset = (
                    surface_df
                    .sort_values(
                        "tourney_date",
                        ascending=False
                    )
                    .head(10)
                )

            else:

                subset = surface_df[
                    surface_df["tourney_date"] >= cutoff
                ]

            if len(subset) < 3:
                continue

            matches = len(subset)
            wins = subset["won_match"].sum()

            svpt = subset["svpt"].sum()
            first_in = subset["first_in"].sum()
            first_won = subset["first_won"].sum()
            second_won = subset["second_won"].sum()

            second_attempts = (
                svpt - first_in
            )

            aces = subset["ace"].sum()

            double_faults = subset["df"].sum()

            service_points_won = (
                first_won
                + second_won
            )
            service_games = subset["service_games"].sum()

            opp_svpt = subset["opp_svpt"].sum()

            opp_first_in = subset["opp_first_in"].sum()

            opp_first_won = subset["opp_first_won"].sum()

            opp_second_won = subset["opp_second_won"].sum()

            opp_bp_faced = subset["opp_bp_faced"].sum()

            opp_bp_saved = subset["opp_bp_saved"].sum()
            
            return_points_won = (
                opp_svpt
                - (
                    opp_first_won
                    + opp_second_won
                )
            )

            return_points_won_pct = (
                return_points_won
                / opp_svpt
                * 100
                if opp_svpt > 0
                else None
            )

            first_return_points_won_pct = (
                (
                    opp_first_in
                    - opp_first_won
                )
                / opp_first_in
                * 100
                if opp_first_in > 0
                else None
            )

            opp_second_attempts = (
                opp_svpt
                - opp_first_in
            )

            second_return_points_won_pct = (
                (
                    opp_second_attempts
                    - opp_second_won
                )
                / opp_second_attempts
                * 100
                if opp_second_attempts > 0
                else None
            )

            return_games_won_pct = (
                (
                    opp_bp_faced
                    - opp_bp_saved
                )
                / opp_bp_faced
                * 100
                if opp_bp_faced > 0
                else None
            )
            aces_per_match = (
                aces / matches
                if matches > 0
                else None
            )

            double_faults_per_match = (
                double_faults / matches
                if matches > 0
                else None
            )

            service_points_won_pct = (
                service_points_won / svpt * 100
                if svpt > 0
                else None
            )

            service_points_won_per_match = (
                service_points_won / matches
                if matches > 0
                else None
            )

            bp_saved = subset["bp_saved"].sum()
            bp_faced = subset["bp_faced"].sum()

            service_breaks_conceded = (
                bp_faced - bp_saved
            )

            service_games_won_pct = (
                (
                    service_games
                    - service_breaks_conceded
                )
                / service_games
                * 100
                if service_games > 0
                else None
            )

            break_points_generated = subset[
                "break_points_generated"
            ].sum()

            breaks_converted = subset[
                "breaks_converted"
            ].sum()

            break_points_generated_per_match = (
                break_points_generated / matches
                if matches > 0
                else None
            )

            breaks_converted_per_match = (
                breaks_converted / matches
                if matches > 0
                else None
            )

            break_conversion_pct = (
                breaks_converted
                / break_points_generated
                * 100
                if break_points_generated > 0
                else None
            )

            tb_played = subset[
                "tb_played"
            ].sum()

            tb_won = subset[
                "tb_won"
            ].sum()

            tiebreak_win_pct = (
                tb_won
                / tb_played
                * 100
                if tb_played > 0
                else None
            )

            tiebreaks_per_match = (
                tb_played
                / matches
                if matches > 0
                else None
            )

            rows.append({

                "player": player,
                "surface": surface,
                "window": window_name,

                "matches": matches,
                "wins": wins,
                "losses": matches - wins,

                "surface_recent_win_pct":
                    round(
                        wins / matches * 100,
                        2
                    ),

                "first_serve_in_pct":
                    round(
                        first_in / svpt * 100,
                        2
                    )
                    if svpt > 0
                    else None,

                "first_serve_won_pct":
                    round(
                        first_won / first_in * 100,
                        2
                    )
                    if first_in > 0
                    else None,

                "second_serve_won_pct":
                    round(
                        second_won / second_attempts * 100,
                        2
                    )
                    if second_attempts > 0
                    else None,
                "service_games_won_pct":
                    round(service_games_won_pct, 2)
                    if service_games_won_pct is not None
                    else None,

                "return_points_won_pct":
                    round(return_points_won_pct, 2)
                    if return_points_won_pct is not None
                    else None,

                "first_return_points_won_pct":
                    round(first_return_points_won_pct, 2)
                    if first_return_points_won_pct is not None
                    else None,

                "second_return_points_won_pct":
                    round(second_return_points_won_pct, 2)
                    if second_return_points_won_pct is not None
                    else None,

                "return_games_won_pct":
                    round(return_games_won_pct, 2)
                    if return_games_won_pct is not None
                    else None,

                "bp_saved_pct":
                    round(
                        bp_saved / bp_faced * 100,
                        2
                    )
                    if bp_faced > 0
                    else None,

                "aces_per_match":
                    round(
                        aces_per_match,
                        2
                    ) if aces_per_match is not None else None,

                "double_faults_per_match":
                    round(
                        double_faults_per_match,
                        2
                    ) if double_faults_per_match is not None else None,

                "service_points_won_pct":
                    round(
                        service_points_won_pct,
                        2
                    ) if service_points_won_pct is not None else None,

                "service_points_won_per_match":
                    round(
                        service_points_won_per_match,
                        2
                    ) if service_points_won_per_match is not None else None,

                "avg_minutes":
                    round(
                        subset["minutes"].mean(),
                        2
                    ),
                "break_points_generated":
                    int(break_points_generated),

                "breaks_converted":
                    int(breaks_converted),

                "break_points_generated_per_match":
                    round(
                        break_points_generated_per_match,
                        2
                    ) if break_points_generated_per_match is not None else None,

                "breaks_converted_per_match":
                    round(
                        breaks_converted_per_match,
                        2
                    ) if breaks_converted_per_match is not None else None,

                "break_conversion_pct":
                    round(
                        break_conversion_pct,
                        2
                    ) if break_conversion_pct is not None else None,

                "tiebreaks_played":
                    int(tb_played),

                "tiebreaks_won":
                    int(tb_won),

                "tiebreak_win_pct":
                    round(
                        tiebreak_win_pct,
                        2
                    ) if tiebreak_win_pct is not None else None,

                "tiebreaks_per_match":
                    round(
                        tiebreaks_per_match,
                        2
                    ),

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
