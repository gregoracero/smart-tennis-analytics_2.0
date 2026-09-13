from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================================
# PROJECT PATH
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(
        str(ROOT)
    )


from streamlit_app.config.data_paths import MATCH_HISTORY_PATH

# ==========================================================
# DATA ACCESS
# ==========================================================

from data_access.players_adapted import (
    get_players,
    get_available_seasons,
    get_available_surfaces,
    get_player_recent_form,
    get_latest_snapshot,
    get_player_statistics,
    get_player_elo_history,
    get_player_match_history,
    get_player_matches,
    get_player_surface_breakdown,
    get_player_tournament_breakdown,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Player Analysis",
    page_icon="🎾",
    layout="wide",
)

st.title(
    "🎾 Player Analysis"
)

st.caption(
    "Statistics derived from the canonical v5 feature history: "
    f"{MATCH_HISTORY_PATH.name}"
)


# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def safe_number(
    value,
    decimals=0,
    default="-",
):
    """
    Formats a numeric value safely.
    """

    if value is None or pd.isna(value):
        return default

    try:
        numeric_value = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default

    if decimals == 0:
        return f"{numeric_value:,.0f}"

    return f"{numeric_value:,.{decimals}f}"


def safe_integer(
    value,
    default="-",
):
    """
    Formats a value as an integer safely.
    """

    if value is None or pd.isna(value):
        return default

    try:
        return f"{int(round(float(value))):,}"
    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_percentage(
    value,
    decimals=1,
    default="-",
):
    """
    Formats a decimal value as a percentage.

    Example:
        0.654 -> 65.4%
    """

    if value is None or pd.isna(value):
        return default

    try:
        return (
            f"{float(value):.{decimals}%}"
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def get_value(
    row,
    column,
    default=np.nan,
):
    """
    Safely retrieves a value from a Series.
    """

    if row is None:
        return default

    if column not in row.index:
        return default

    return row[column]


def get_existing_columns(
    frame,
    columns,
):
    """
    Returns only columns that exist in a DataFrame.
    """

    return [
        column
        for column in columns
        if column in frame.columns
    ]


def display_metric(
    label,
    value,
    *,
    kind="number",
    decimals=1,
    help_text=None,
):
    """
    Displays a Streamlit metric with safe formatting.
    """

    if kind == "percentage":
        displayed_value = safe_percentage(
            value,
            decimals=decimals,
        )

    elif kind == "integer":
        displayed_value = safe_integer(
            value
        )

    else:
        displayed_value = safe_number(
            value,
            decimals=decimals,
        )

    st.metric(
        label=label,
        value=displayed_value,
        help=help_text,
    )


# ==========================================================
# LOAD PLAYER LIST
# ==========================================================

players = get_players()

if players is None or players.empty:
    st.error(
        "No players were found in the source Parquet."
    )

    st.stop()


# Avoid duplicate names producing ambiguous selections.
players = (
    players
    .dropna(
        subset=[
            "player_id",
            "player_name",
        ]
    )
    .drop_duplicates(
        subset=[
            "player_id",
        ]
    )
    .sort_values(
        "player_name",
        kind="mergesort",
    )
    .reset_index(
        drop=True,
    )
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "Filters"
)

player_options = {
    (
        f"{row.player_name} "
        f"[{row.player_id}]"
    ): row.player_id
    for row in players.itertuples(
        index=False
    )
}

selected_player_label = (
    st.sidebar.selectbox(
        "Player",
        options=list(
            player_options.keys()
        ),
    )
)

player_id = player_options[
    selected_player_label
]

player_row = players[
    players["player_id"].eq(
        player_id
    )
].iloc[0]

player_name = player_row[
    "player_name"
]


# ----------------------------------------------------------
# Surface filter
# ----------------------------------------------------------

try:
    raw_surfaces = (
        get_available_surfaces()
    )

except (
    AttributeError,
    ImportError,
):
    raw_surfaces = [
        "ALL",
        "Hard",
        "Clay",
        "Grass",
        "Carpet",
    ]


surface_mapping = {
    "all": "ALL",
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
    "indoor": "Indoor",
}


available_surfaces = []

for value in raw_surfaces:

    if value is None:
        continue

    normalized_value = (
        surface_mapping.get(
            str(value)
            .strip()
            .lower()
        )
    )

    if (
        normalized_value is not None
        and normalized_value
        not in available_surfaces
    ):
        available_surfaces.append(
            normalized_value
        )


preferred_surface_order = [
    "ALL",
    "Hard",
    "Clay",
    "Grass",
    "Carpet",
    "Indoor",
]


available_surfaces = [
    value
    for value in preferred_surface_order
    if value in available_surfaces
]


if "ALL" not in available_surfaces:
    available_surfaces.insert(
        0,
        "ALL",
    )


surface = st.sidebar.selectbox(
    "Surface",
    available_surfaces,
)



# ----------------------------------------------------------
# Season filter
# ----------------------------------------------------------

seasons = get_available_seasons()

season_options = [
    "CAREER",
    *[
        str(value)
        for value in seasons
        if str(value) != "CAREER"
    ],
]

season = st.sidebar.selectbox(
    "Season",
    season_options,
)


# ----------------------------------------------------------
# Last N matches
# ----------------------------------------------------------

last_n_enabled = (
    st.sidebar.checkbox(
        "Limit to last N matches",
        value=False,
    )
)

last_n = None

if last_n_enabled:
    last_n = st.sidebar.selectbox(
        "Number of matches",
        [
            5,
            10,
            20,
            50,
            100,
        ],
        index=1,
    )


# ==========================================================
# LOAD PLAYER DATA
# ==========================================================

snapshot = get_latest_snapshot(
    player_id
)

stats_raw = get_player_statistics(
    player_id=player_id,
    season=(
        None
        if season == "CAREER"
        else season
    ),
    surface=surface,
    scope=(
        "CAREER"
        if season == "CAREER"
        else "SEASON"
    ),
    last_n_matches=last_n,
)

matches = get_player_matches(
    player_id=player_id,
    season=season,
    surface=surface,
    last_n_matches=last_n,
)

history = get_player_match_history(
    player_id=player_id,
    season=season,
    surface=surface,
    last_n_matches=last_n,
)

elo = get_player_elo_history(
    player_id
)

surface_breakdown = (
    get_player_surface_breakdown(
        player_id
    )
)

tournament_breakdown = (
    get_player_tournament_breakdown(
        player_id
    )
)

recent_form = get_player_recent_form(
    player_id=player_id,
    last_n=10,
)


# ==========================================================
# SELECT AGGREGATED STATISTICS
# ==========================================================

player_stats = None

if (
    stats_raw is not None
    and not stats_raw.empty
):
    stats_filter = (
        stats_raw.copy()
    )

    if (
        "competition_type"
        in stats_filter.columns
    ):
        stats_filter = stats_filter[
            stats_filter[
                "competition_type"
            ]
            .astype(str)
            .eq("ALL")
        ]

    if (
        "tournament_level"
        in stats_filter.columns
    ):
        stats_filter = stats_filter[
            stats_filter[
                "tournament_level"
            ]
            .astype(str)
            .eq("ALL")
        ]

    if not stats_filter.empty:
        player_stats = (
            stats_filter
            .sort_values(
                "matches",
                ascending=False,
                kind="mergesort",
            )
            .iloc[0]
        )


# ==========================================================
# PLAYER HEADER
# ==========================================================

st.header(
    f"🎾 {player_name}"
)

if snapshot is None:
    st.warning(
        "No latest snapshot is available "
        "for this player."
    )

else:
    header_1, header_2, header_3, header_4 = (
        st.columns(4)
    )

    with header_1:
        display_metric(
            "Current Rank",
            get_value(
                snapshot,
                "player_rank",
            ),
            kind="integer",
        )

    with header_2:
        display_metric(
            "ATP Points",
            get_value(
                snapshot,
                "player_rank_points",
            ),
            kind="integer",
        )

    with header_3:
        display_metric(
            "Age",
            get_value(
                snapshot,
                "player_age",
            ),
            decimals=1,
        )

    with header_4:
        height = get_value(
            snapshot,
            "player_height",
        )

        st.metric(
            "Height",
            (
                f"{safe_integer(height)} cm"
                if pd.notna(height)
                else "-"
            ),
        )


# ==========================================================
# RECENT FORM CHIPS
# ==========================================================

st.markdown(
    "### Recent Form"
)

if recent_form:
    for (
        surface_name,
        results,
    ) in recent_form.items():

        chips = [
            "🟢"
            if result == "W"
            else "🔴"
            for result in results
        ]

        st.markdown(
            f"**{surface_name}**  "
            + " ".join(chips)
        )

else:
    st.info(
        "Recent form is not available."
    )


# ==========================================================
# ELO AND PERFORMANCE SUMMARY
# ==========================================================

current_elo = np.nan
peak_elo = np.nan
current_surface_elo = np.nan

if elo is not None and not elo.empty:

    elo_sorted = (
        elo
        .sort_values(
            "match_date",
            kind="mergesort",
        )
    )

    if "elo_before" in elo_sorted.columns:
        current_elo = (
            elo_sorted[
                "elo_before"
            ]
            .dropna()
            .iloc[-1]
            if elo_sorted[
                "elo_before"
            ]
            .notna()
            .any()
            else np.nan
        )

        peak_elo = (
            elo_sorted[
                "elo_before"
            ].max()
        )

    if (
        "surface_elo_before"
        in elo_sorted.columns
    ):
        current_surface_elo = (
            elo_sorted[
                "surface_elo_before"
            ]
            .dropna()
            .iloc[-1]
            if elo_sorted[
                "surface_elo_before"
            ]
            .notna()
            .any()
            else np.nan
        )


summary_1, summary_2, summary_3, summary_4 = (
    st.columns(4)
)

with summary_1:
    display_metric(
        "Current Elo",
        current_elo,
        kind="integer",
        help_text=(
            "Elo before the player's "
            "latest recorded match."
        ),
    )

with summary_2:
    display_metric(
        "Peak Elo",
        peak_elo,
        kind="integer",
    )

with summary_3:
    display_metric(
        "Matches",
        (
            get_value(
                player_stats,
                "matches",
            )
            if player_stats is not None
            else len(matches)
        ),
        kind="integer",
    )

with summary_4:
    display_metric(
        "Win Rate",
        (
            get_value(
                player_stats,
                "win_rate",
            )
            if player_stats is not None
            else (
                matches["won"].mean()
                if (
                    matches is not None
                    and not matches.empty
                    and "won"
                    in matches.columns
                )
                else np.nan
            )
        ),
        kind="percentage",
    )


# ==========================================================
# LATEST PREMATCH SNAPSHOT
# ==========================================================

if snapshot is not None:

    st.markdown(
        "### Latest Prematch Snapshot"
    )

    snap_1, snap_2, snap_3, snap_4 = (
        st.columns(4)
    )

    with snap_1:
        display_metric(
            "Career Win Rate",
            get_value(
                snapshot,
                "career_win_rate_before",
            ),
            kind="percentage",
        )

    with snap_2:
        display_metric(
            "Surface Win Rate",
            get_value(
                snapshot,
                "surface_win_rate_before",
            ),
            kind="percentage",
        )

    with snap_3:
        display_metric(
            "Last 10 Win Rate",
            get_value(
                snapshot,
                "form_last_10_before",
            ),
            kind="percentage",
        )

    with snap_4:
        display_metric(
            "Days Since Last Match",
            get_value(
                snapshot,
                "days_since_last_match",
            ),
            decimals=0,
        )

    snap_5, snap_6, snap_7, snap_8 = (
        st.columns(4)
    )

    with snap_5:
        display_metric(
            "Career Matches",
            get_value(
                snapshot,
                "career_matches_before",
            ),
            kind="integer",
        )

    with snap_6:
        display_metric(
            "Surface Matches",
            get_value(
                snapshot,
                "surface_matches_before",
            ),
            kind="integer",
        )

    with snap_7:
        display_metric(
            "Surface Elo",
            current_surface_elo,
            kind="integer",
        )

    with snap_8:
        display_metric(
            "Avg Minutes, Last 5",
            get_value(
                snapshot,
                "avg_minutes_last_5_before",
            ),
            decimals=1,
        )


# ==========================================================
# TABS
# ==========================================================

stats_tab, matches_tab = st.tabs(
    [
        "📊 Statistics",
        "🎾 Matches",
    ]
)


# ==========================================================
# STATISTICS TAB
# ==========================================================

with stats_tab:

    if player_stats is None:
        st.warning(
            "No aggregated statistics are available "
            "for the selected filters."
        )

    else:

        # --------------------------------------------------
        # Performance overview
        # --------------------------------------------------

        st.subheader(
            "Performance Overview"
        )

        perf_1, perf_2, perf_3, perf_4 = (
            st.columns(4)
        )

        with perf_1:
            display_metric(
                "Matches",
                get_value(
                    player_stats,
                    "matches",
                ),
                kind="integer",
            )

        with perf_2:
            display_metric(
                "Wins",
                get_value(
                    player_stats,
                    "wins",
                ),
                kind="integer",
            )

        with perf_3:
            display_metric(
                "Losses",
                get_value(
                    player_stats,
                    "losses",
                ),
                kind="integer",
            )

        with perf_4:
            display_metric(
                "Win Rate",
                get_value(
                    player_stats,
                    "win_rate",
                ),
                kind="percentage",
            )

        perf_5, perf_6, perf_7, perf_8 = (
            st.columns(4)
        )

        with perf_5:
            display_metric(
                "Latest Rank",
                get_value(
                    player_stats,
                    "latest_rank",
                ),
                kind="integer",
            )

        with perf_6:
            display_metric(
                "Latest ATP Points",
                get_value(
                    player_stats,
                    "latest_rank_points",
                ),
                kind="integer",
            )

        with perf_7:
            display_metric(
                "Latest Elo",
                get_value(
                    player_stats,
                    "latest_elo",
                ),
                kind="integer",
            )

        with perf_8:
            display_metric(
                "Average Match Minutes",
                get_value(
                    player_stats,
                    "average_match_minutes",
                ),
                decimals=1,
            )

        # --------------------------------------------------
        # Serve metrics
        # --------------------------------------------------

        st.subheader(
            "Serve Metrics"
        )

        serve_1, serve_2, serve_3, serve_4 = (
            st.columns(4)
        )

        with serve_1:
            display_metric(
                "Aces / Match",
                get_value(
                    player_stats,
                    "aces_per_match",
                ),
                decimals=2,
            )

        with serve_2:
            display_metric(
                "Double Faults / Match",
                get_value(
                    player_stats,
                    "double_faults_per_match",
                ),
                decimals=2,
            )

        with serve_3:
            display_metric(
                "Aces / Service Game",
                get_value(
                    player_stats,
                    "aces_per_service_game",
                ),
                decimals=3,
            )

        with serve_4:
            display_metric(
                "DF / Service Game",
                get_value(
                    player_stats,
                    "double_faults_per_service_game",
                ),
                decimals=3,
            )

        serve_5, serve_6, serve_7, serve_8 = (
            st.columns(4)
        )

        with serve_5:
            display_metric(
                "First Serve In",
                get_value(
                    player_stats,
                    "first_serve_in_pct",
                ),
                kind="percentage",
            )

        with serve_6:
            display_metric(
                "First Serve Won",
                get_value(
                    player_stats,
                    "first_serve_win_pct",
                ),
                kind="percentage",
            )

        with serve_7:
            display_metric(
                "Second Serve Won",
                get_value(
                    player_stats,
                    "second_serve_win_pct",
                ),
                kind="percentage",
            )

        with serve_8:
            display_metric(
                "Service Points Won",
                get_value(
                    player_stats,
                    "service_points_won_pct",
                ),
                kind="percentage",
            )

        # --------------------------------------------------
        # Return and pressure metrics
        # --------------------------------------------------

        st.subheader(
            "Return and Break Points"
        )

        return_1, return_2, return_3, return_4 = (
            st.columns(4)
        )

        with return_1:
            display_metric(
                "Return Points Won",
                get_value(
                    player_stats,
                    "return_points_won_pct",
                ),
                kind="percentage",
            )

        with return_2:
            display_metric(
                "Break Points Saved",
                get_value(
                    player_stats,
                    "break_points_saved_pct",
                ),
                kind="percentage",
            )

        with return_3:
            display_metric(
                "BP Faced",
                get_value(
                    player_stats,
                    "break_points_faced",
                ),
                kind="integer",
            )

        with return_4:
            display_metric(
                "BP Saved",
                get_value(
                    player_stats,
                    "break_points_saved",
                ),
                kind="integer",
            )

        # --------------------------------------------------
        # Volume totals
        # --------------------------------------------------

        st.subheader(
            "Statistical Volume"
        )

        volume_1, volume_2, volume_3, volume_4 = (
            st.columns(4)
        )

        with volume_1:
            display_metric(
                "Aces",
                get_value(
                    player_stats,
                    "aces",
                ),
                kind="integer",
            )

        with volume_2:
            display_metric(
                "Double Faults",
                get_value(
                    player_stats,
                    "double_faults",
                ),
                kind="integer",
            )

        with volume_3:
            display_metric(
                "Service Points",
                get_value(
                    player_stats,
                    "service_points",
                ),
                kind="integer",
            )

        with volume_4:
            display_metric(
                "Return Points Played",
                get_value(
                    player_stats,
                    "return_points_played",
                ),
                kind="integer",
            )


# ==========================================================
# MATCHES TAB
# ==========================================================

with matches_tab:

    if matches is None or matches.empty:
        st.info(
            "No matches are available "
            "for the selected filters."
        )

    else:
        st.caption(
            f"Showing {len(matches):,} matches"
        )

        for match in matches.itertuples(
            index=False
        ):

            match_data = match._asdict()

            result = match_data.get(
                "result",
                "L",
            )

            icon = (
                "🟢"
                if result == "W"
                else "🔴"
            )

            match_date_value = pd.to_datetime(
                match_data.get(
                    "match_date"
                ),
                errors="coerce",
            )

            displayed_date = (
                match_date_value.date()
                if pd.notna(
                    match_date_value
                )
                else "-"
            )

            tournament_name = (
                match_data.get(
                    "tourney_name"
                )
                or match_data.get(
                    "tournament_name"
                )
                or "-"
            )

            opponent_name = (
                match_data.get(
                    "opponent_name",
                    "-",
                )
            )

            expander_title = (
                f"{icon} "
                f"{displayed_date} | "
                f"{tournament_name} | "
                f"{opponent_name}"
            )

            with st.expander(
                expander_title
            ):

                match_1, match_2, match_3, match_4 = (
                    st.columns(4)
                )

                with match_1:
                    st.write(
                        "Surface:",
                        match_data.get(
                            "surface",
                            "-",
                        ),
                    )

                with match_2:
                    st.write(
                        "Round:",
                        match_data.get(
                            "round",
                            "-",
                        ),
                    )

                with match_3:
                    st.write(
                        "Player Rank:",
                        safe_integer(
                            match_data.get(
                                "player_rank"
                            )
                        ),
                    )

                with match_4:
                    st.write(
                        "Opponent Rank:",
                        safe_integer(
                            match_data.get(
                                "opponent_rank"
                            )
                        ),
                    )

                st.divider()

                match_5, match_6, match_7, match_8 = (
                    st.columns(4)
                )

                with match_5:
                    display_metric(
                        "Aces",
                        match_data.get(
                            "aces"
                        ),
                        kind="integer",
                    )

                with match_6:
                    display_metric(
                        "Double Faults",
                        match_data.get(
                            "double_faults"
                        ),
                        kind="integer",
                    )

                with match_7:
                    bp_saved = match_data.get(
                        "break_points_saved"
                    )

                    bp_faced = match_data.get(
                        "break_points_faced"
                    )

                    st.metric(
                        "Break Points Saved",
                        (
                            f"{safe_integer(bp_saved)}"
                            f" / "
                            f"{safe_integer(bp_faced)}"
                        ),
                    )

                with match_8:
                    display_metric(
                        "Minutes",
                        match_data.get(
                            "minutes"
                        ),
                        kind="integer",
                    )

                match_9, match_10, match_11, match_12 = (
                    st.columns(4)
                )

                with match_9:
                    display_metric(
                        "First Serve In",
                        match_data.get(
                            "first_serve_in_pct"
                        ),
                        kind="percentage",
                    )

                with match_10:
                    display_metric(
                        "First Serve Won",
                        match_data.get(
                            "first_serve_win_pct"
                        ),
                        kind="percentage",
                    )

                with match_11:
                    display_metric(
                        "Second Serve Won",
                        match_data.get(
                            "second_serve_win_pct"
                        ),
                        kind="percentage",
                    )

                with match_12:
                    display_metric(
                        "Return Points Won",
                        match_data.get(
                            "return_points_won_pct"
                        ),
                        kind="percentage",
                    )


# ==========================================================
# PERFORMANCE BY SURFACE
# ==========================================================

st.subheader(
    "Performance by Surface"
)

if (
    surface_breakdown is not None
    and not surface_breakdown.empty
):

    surface_columns = (
        get_existing_columns(
            surface_breakdown,
            [
                "surface",
                "matches",
                "wins",
                "losses",
                "win_rate",
                "aces_per_match",
                "first_serve_win_pct",
                "second_serve_win_pct",
                "return_points_won_pct",
            ],
        )
    )

    surface_display = (
        surface_breakdown[
            surface_columns
        ]
        .copy()
    )

    percentage_columns = [
        "win_rate",
        "first_serve_win_pct",
        "second_serve_win_pct",
        "return_points_won_pct",
    ]

    st.dataframe(
        surface_display.style.format(
            {
                column: "{:.1%}"
                for column
                in percentage_columns
                if column
                in surface_display.columns
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if (
        "win_rate"
        in surface_breakdown.columns
    ):
        figure = px.bar(
            surface_breakdown,
            x="surface",
            y="win_rate",
            color="surface",
            title="Win Rate by Surface",
            labels={
                "surface": "Surface",
                "win_rate": "Win Rate",
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

else:
    st.info(
        "No surface breakdown is available."
    )


# ==========================================================
# PERFORMANCE BY TOURNAMENT LEVEL
# ==========================================================

st.subheader(
    "Performance by Tournament Level"
)

if (
    tournament_breakdown is not None
    and not tournament_breakdown.empty
):

    tournament_columns = (
        get_existing_columns(
            tournament_breakdown,
            [
                "tournament_level",
                "matches",
                "wins",
                "losses",
                "win_rate",
                "aces_per_match",
                "first_serve_win_pct",
                "return_points_won_pct",
            ],
        )
    )

    tournament_display = (
        tournament_breakdown[
            tournament_columns
        ]
        .copy()
    )

    tournament_percentage_columns = [
        "win_rate",
        "first_serve_win_pct",
        "return_points_won_pct",
    ]

    st.dataframe(
        tournament_display.style.format(
            {
                column: "{:.1%}"
                for column
                in tournament_percentage_columns
                if column
                in tournament_display.columns
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    if (
        "win_rate"
        in tournament_breakdown.columns
    ):
        figure = px.bar(
            tournament_breakdown,
            x="tournament_level",
            y="win_rate",
            color="tournament_level",
            title=(
                "Win Rate by Tournament Level"
            ),
            labels={
                "tournament_level": (
                    "Tournament Level"
                ),
                "win_rate": "Win Rate",
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

else:
    st.info(
        "No tournament-level breakdown "
        "is available."
    )


# ==========================================================
# ELO CHARTS
# ==========================================================

st.subheader(
    "Elo Evolution"
)

if elo is not None and not elo.empty:

    elo_chart = (
        elo
        .sort_values(
            "match_date",
            kind="mergesort",
        )
        .copy()
    )

    elo_columns = get_existing_columns(
        elo_chart,
        [
            "elo_before",
            "surface_elo_before",
        ],
    )

    if elo_columns:
        elo_long = elo_chart.melt(
            id_vars=[
                "match_date",
            ],
            value_vars=elo_columns,
            var_name="rating_type",
            value_name="elo",
        )

        rating_labels = {
            "elo_before": "Global Elo",
            "surface_elo_before": (
                "Surface Elo"
            ),
        }

        elo_long["rating_type"] = (
            elo_long["rating_type"]
            .map(rating_labels)
            .fillna(
                elo_long["rating_type"]
            )
        )

        figure = px.line(
            elo_long,
            x="match_date",
            y="elo",
            color="rating_type",
            title="Prematch Elo Evolution",
            labels={
                "match_date": "Date",
                "elo": "Elo",
                "rating_type": "Rating",
            },
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

else:
    st.info(
        "No Elo history is available."
    )


# ==========================================================
# FORM CHART
# ==========================================================

st.subheader(
    "Rolling Form"
)

if history is not None and not history.empty:

    history_chart = (
        history
        .sort_values(
            "match_date",
            kind="mergesort",
        )
        .copy()
    )

    form_columns = get_existing_columns(
        history_chart,
        [
            "form_last_5_before",
            "form_last_10_before",
        ],
    )

    if form_columns:
        form_long = history_chart.melt(
            id_vars=[
                "match_date",
            ],
            value_vars=form_columns,
            var_name="window",
            value_name="win_rate",
        )

        window_labels = {
            "form_last_5_before": (
                "Last 5"
            ),
            "form_last_10_before": (
                "Last 10"
            ),
        }

        form_long["window"] = (
            form_long["window"]
            .map(window_labels)
            .fillna(form_long["window"])
        )

        figure = px.line(
            form_long,
            x="match_date",
            y="win_rate",
            color="window",
            title="Prematch Rolling Win Rate",
            labels={
                "match_date": "Date",
                "win_rate": "Win Rate",
                "window": "Window",
            },
        )

        figure.update_yaxes(
            tickformat=".0%",
            range=[
                0,
                1,
            ],
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    else:
        st.info(
            "Rolling form columns are not available."
        )

else:
    st.info(
        "No match history is available "
        "for the selected filters."
    )


# ==========================================================
# ACTIVITY CHART
# ==========================================================

st.subheader(
    "Activity and Rest"
)

if history is not None and not history.empty:

    activity_columns = (
        get_existing_columns(
            history,
            [
                "days_since_last_match",
                "avg_minutes_last_5_before",
            ],
        )
    )

    if activity_columns:
        activity_long = (
            history
            .sort_values(
                "match_date",
                kind="mergesort",
            )
            .melt(
                id_vars=[
                    "match_date",
                ],
                value_vars=activity_columns,
                var_name="metric",
                value_name="value",
            )
        )

        activity_labels = {
            "days_since_last_match": (
                "Days Since Previous Match"
            ),
            "avg_minutes_last_5_before": (
                "Average Minutes, Last 5"
            ),
        }

        activity_long["metric"] = (
            activity_long["metric"]
            .map(activity_labels)
            .fillna(
                activity_long["metric"]
            )
        )

        figure = px.line(
            activity_long,
            x="match_date",
            y="value",
            color="metric",
            title="Prematch Activity Indicators",
            labels={
                "match_date": "Date",
                "value": "Value",
                "metric": "Metric",
            },
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

    else:
        st.info(
            "Activity columns are not available."
        )


# ==========================================================
# DATA NOTES
# ==========================================================

with st.expander(
    "ℹ️ Data definitions"
):
    st.markdown(
        """
        - **Elo before** is the player's rating immediately
          before the corresponding match.
        - **Surface Elo before** is the rating maintained
          separately for the match surface.
        - **Rolling form** only uses matches completed before
          the displayed match.
        - **Current rank** is the ranking stored in the latest
          available player snapshot.
        - Serve and return statistics are calculated from
          completed matches with point statistics available.
        - Missing values are shown as `-` and are not
          automatically replaced with zero.
        """
    )