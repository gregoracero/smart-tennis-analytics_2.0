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

from data_access.tournaments_adapted import (
    get_tournaments,
    get_tournament_stats,
    get_tournament_matches,
    get_tournament_editions,
    get_tournament_players,
    get_tournament_round_breakdown,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Tournament Analysis",
    page_icon="🏆",
    layout="wide",
)

st.title(
    "🏆 Tournament Analysis"
)

st.caption(
    "Tournament statistics derived from the canonical v5 feature history: "
    f"{MATCH_HISTORY_PATH.name}"
)


# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def safe_number(
    value,
    decimals=1,
    default="-",
):
    """
    Formats a numeric value safely.
    """

    if value is None or pd.isna(value):
        return default

    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return default


def safe_integer(
    value,
    default="-",
):
    """
    Formats a numeric value as an integer safely.
    """

    if value is None or pd.isna(value):
        return default

    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return default


def safe_percentage(
    value,
    decimals=1,
    default="-",
):
    """
    Formats a decimal value as a percentage.
    """

    if value is None or pd.isna(value):
        return default

    try:
        return f"{float(value):.{decimals}%}"
    except (TypeError, ValueError):
        return default


def row_value(
    row,
    column,
    default=np.nan,
):
    """
    Safely reads a value from a pandas Series.
    """

    if row is None:
        return default

    if column not in row.index:
        return default

    return row[column]


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


def existing_columns(
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


def calculate_participant_strength(
    matches,
):
    """
    Calculates participant ranking metrics using both
    players from every match.

    To avoid counting the same player repeatedly, the latest
    available ranking for each player in the edition is used.
    """

    empty_result = {
        "players": 0,
        "ranked_players": 0,
        "average_rank": np.nan,
        "median_rank": np.nan,
        "top10_players": 0,
        "top20_players": 0,
        "top50_players": 0,
        "top100_players": 0,
    }

    if matches is None or matches.empty:
        return empty_result

    required_player_1 = {
        "player_1_id",
        "player_1_name",
        "player_1_rank",
    }

    required_player_2 = {
        "player_2_id",
        "player_2_name",
        "player_2_rank",
    }

    if not required_player_1.issubset(
        matches.columns
    ):
        return empty_result

    if not required_player_2.issubset(
        matches.columns
    ):
        return empty_result

    player_1 = matches[
        [
            "player_1_id",
            "player_1_name",
            "player_1_rank",
            "match_date",
        ]
    ].rename(
        columns={
            "player_1_id": "player_id",
            "player_1_name": "player_name",
            "player_1_rank": "rank",
        }
    )

    player_2 = matches[
        [
            "player_2_id",
            "player_2_name",
            "player_2_rank",
            "match_date",
        ]
    ].rename(
        columns={
            "player_2_id": "player_id",
            "player_2_name": "player_name",
            "player_2_rank": "rank",
        }
    )

    participants = pd.concat(
        [
            player_1,
            player_2,
        ],
        ignore_index=True,
    )

    participants["rank"] = pd.to_numeric(
        participants["rank"],
        errors="coerce",
    )

    participants["match_date"] = pd.to_datetime(
        participants["match_date"],
        errors="coerce",
    )

    participants = (
        participants
        .dropna(
            subset=[
                "player_id",
                "player_name",
            ]
        )
        .sort_values(
            "match_date",
            kind="mergesort",
        )
        .drop_duplicates(
            subset=[
                "player_id",
            ],
            keep="last",
        )
        .reset_index(
            drop=True,
        )
    )

    ranked = participants[
        participants["rank"].notna()
    ].copy()

    return {
        "players": int(
            len(participants)
        ),
        "ranked_players": int(
            len(ranked)
        ),
        "average_rank": (
            ranked["rank"].mean()
            if not ranked.empty
            else np.nan
        ),
        "median_rank": (
            ranked["rank"].median()
            if not ranked.empty
            else np.nan
        ),
        "top10_players": int(
            ranked["rank"].le(10).sum()
        ),
        "top20_players": int(
            ranked["rank"].le(20).sum()
        ),
        "top50_players": int(
            ranked["rank"].le(50).sum()
        ),
        "top100_players": int(
            ranked["rank"].le(100).sum()
        ),
    }


def prepare_history(
    history,
):
    """
    Normalizes historical tournament edition data before
    displaying charts.
    """

    if history is None or history.empty:
        return pd.DataFrame()

    result = history.copy()

    if "season" in result.columns:
        result["season"] = pd.to_numeric(
            result["season"],
            errors="coerce",
        )

    numeric_columns = [
        "matches",
        "unique_players",
        "average_match_minutes",
        "average_winner_rank",
        "average_loser_rank",
        "average_winner_elo",
        "average_loser_elo",
        "aces_per_match",
        "double_faults_per_match",
        "service_points_per_match",
        "break_points_faced_per_match",
        "break_points_saved_pct",
        "higher_ranked_winner_rate",
        "upset_rate",
        "odds_coverage",
        "favorite_win_rate",
        "average_market_overround",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    if "season" in result.columns:
        result = result.sort_values(
            "season",
            kind="mergesort",
        )

    return result.reset_index(
        drop=True
    )


# ==========================================================
# LOAD TOURNAMENT EDITIONS
# ==========================================================

tournaments = get_tournaments()

if (
    tournaments is None
    or tournaments.empty
):
    st.error(
        "No tournament editions were found."
    )

    st.stop()


required_tournament_columns = {
    "tournament_id",
    "tournament_name",
}

missing_tournament_columns = (
    required_tournament_columns
    - set(tournaments.columns)
)

if missing_tournament_columns:
    st.error(
        "The tournament data access layer is missing "
        f"these columns: {sorted(missing_tournament_columns)}"
    )

    st.stop()


# ==========================================================
# TOURNAMENT SELECTOR
# ==========================================================

st.sidebar.header(
    "Tournament Selection"
)


def create_tournament_label(
    row,
):
    """
    Creates a unique and readable label for each edition.
    """

    tournament_name = (
        row.get(
            "tournament_name",
            "Unknown Tournament",
        )
    )

    season = row.get(
        "season",
        np.nan,
    )

    surface = row.get(
        "surface",
        None,
    )

    level = row.get(
        "tournament_level",
        None,
    )

    label_parts = [
        str(tournament_name)
    ]

    if pd.notna(season):
        try:
            label_parts.append(
                str(int(float(season)))
            )
        except (TypeError, ValueError):
            label_parts.append(
                str(season)
            )

    if (
        surface is not None
        and pd.notna(surface)
    ):
        label_parts.append(
            str(surface)
        )

    if (
        level is not None
        and pd.notna(level)
    ):
        label_parts.append(
            str(level)
        )

    return " | ".join(
        label_parts
    )


tournaments = tournaments.copy()

tournaments["selector_label"] = (
    tournaments.apply(
        create_tournament_label,
        axis=1,
    )
)

tournaments = (
    tournaments
    .sort_values(
        [
            "tournament_name",
            "season",
        ],
        ascending=[
            True,
            False,
        ],
        kind="mergesort",
    )
    .reset_index(
        drop=True,
    )
)


selected_label = st.sidebar.selectbox(
    "Tournament Edition",
    options=tournaments[
        "selector_label"
    ].tolist(),
)

selected_tournament = tournaments[
    tournaments["selector_label"].eq(
        selected_label
    )
].iloc[0]

tournament_id = selected_tournament[
    "tournament_id"
]

tournament_name = selected_tournament[
    "tournament_name"
]


# ==========================================================
# LOAD DATA
# ==========================================================

stats = get_tournament_stats(
    tournament_id
)

matches = get_tournament_matches(
    tournament_id
)

players = get_tournament_players(
    tournament_id
)

round_breakdown = (
    get_tournament_round_breakdown(
        tournament_id
    )
)

history = get_tournament_editions(
    tournament_name
)


if stats is None or stats.empty:
    st.warning(
        "No tournament statistics were found "
        "for the selected edition."
    )

    st.stop()


latest = stats.iloc[0]

history = prepare_history(
    history
)

participant_strength = (
    calculate_participant_strength(
        matches
    )
)


# ==========================================================
# HEADER
# ==========================================================

season_value = row_value(
    latest,
    "season",
)

displayed_season = safe_integer(
    season_value
)

st.header(
    f"🏆 {tournament_name} "
    f"({displayed_season})"
)

header_1, header_2, header_3, header_4 = (
    st.columns(4)
)

with header_1:
    st.metric(
        "Surface",
        (
            str(
                row_value(
                    latest,
                    "surface",
                    "-",
                )
            )
        ),
    )

with header_2:
    st.metric(
        "Level",
        (
            str(
                row_value(
                    latest,
                    "tournament_level",
                    "-",
                )
            )
        ),
    )

with header_3:
    display_metric(
        "Season",
        season_value,
        kind="integer",
    )

with header_4:
    st.metric(
        "Tournament ID",
        str(tournament_id),
    )


# ==========================================================
# TABS
# ==========================================================

(
    overview_tab,
    strength_tab,
    style_tab,
    rounds_tab,
    players_tab,
    history_tab,
    matches_tab,
) = st.tabs(
    [
        "📊 Overview",
        "💪 Strength",
        "🎾 Playing Style",
        "🧭 Rounds",
        "👥 Players",
        "📈 History",
        "🗂️ Matches",
    ]
)


# ==========================================================
# OVERVIEW TAB
# ==========================================================

with overview_tab:

    st.subheader(
        "Tournament Overview"
    )

    overview_1, overview_2, overview_3, overview_4 = (
        st.columns(4)
    )

    with overview_1:
        display_metric(
            "Matches",
            row_value(
                latest,
                "matches",
            ),
            kind="integer",
        )

    with overview_2:
        display_metric(
            "Players",
            row_value(
                latest,
                "unique_players",
                participant_strength[
                    "players"
                ],
            ),
            kind="integer",
        )

    with overview_3:
        display_metric(
            "Average Match Minutes",
            row_value(
                latest,
                "average_match_minutes",
            ),
            decimals=1,
        )

    with overview_4:
        display_metric(
            "Total Match Minutes",
            row_value(
                latest,
                "total_match_minutes",
            ),
            kind="integer",
        )

    overview_5, overview_6, overview_7, overview_8 = (
        st.columns(4)
    )

    with overview_5:
        first_match_date = pd.to_datetime(
            row_value(
                latest,
                "first_match_date",
            ),
            errors="coerce",
        )

        st.metric(
            "First Match",
            (
                str(first_match_date.date())
                if pd.notna(first_match_date)
                else "-"
            ),
        )

    with overview_6:
        last_match_date = pd.to_datetime(
            row_value(
                latest,
                "last_match_date",
            ),
            errors="coerce",
        )

        st.metric(
            "Last Match",
            (
                str(last_match_date.date())
                if pd.notna(last_match_date)
                else "-"
            ),
        )

    with overview_7:
        display_metric(
            "Odds Coverage",
            row_value(
                latest,
                "odds_coverage",
            ),
            kind="percentage",
        )

    with overview_8:
        display_metric(
            "Matches with Odds",
            row_value(
                latest,
                "matches_with_odds",
            ),
            kind="integer",
        )

    st.subheader(
        "Results and Market"
    )

    market_1, market_2, market_3, market_4 = (
        st.columns(4)
    )

    with market_1:
        display_metric(
            "Higher-Ranked Winner Rate",
            row_value(
                latest,
                "higher_ranked_winner_rate",
            ),
            kind="percentage",
        )

    with market_2:
        display_metric(
            "Upset Rate",
            row_value(
                latest,
                "upset_rate",
            ),
            kind="percentage",
        )

    with market_3:
        display_metric(
            "Favourite Win Rate",
            row_value(
                latest,
                "favorite_win_rate",
            ),
            kind="percentage",
        )

    with market_4:
        display_metric(
            "Market Overround",
            row_value(
                latest,
                "average_market_overround",
            ),
            kind="percentage",
        )


# ==========================================================
# STRENGTH TAB
# ==========================================================

with strength_tab:

    st.subheader(
        "Participant Ranking Strength"
    )

    strength_1, strength_2, strength_3, strength_4 = (
        st.columns(4)
    )

    with strength_1:
        display_metric(
            "Average Rank",
            participant_strength[
                "average_rank"
            ],
            decimals=1,
        )

    with strength_2:
        display_metric(
            "Median Rank",
            participant_strength[
                "median_rank"
            ],
            decimals=1,
        )

    with strength_3:
        display_metric(
            "Ranked Players",
            participant_strength[
                "ranked_players"
            ],
            kind="integer",
        )

    with strength_4:
        display_metric(
            "Total Players",
            participant_strength[
                "players"
            ],
            kind="integer",
        )

    st.subheader(
        "Player Depth"
    )

    depth_1, depth_2, depth_3, depth_4 = (
        st.columns(4)
    )

    with depth_1:
        display_metric(
            "Top 10 Players",
            participant_strength[
                "top10_players"
            ],
            kind="integer",
        )

    with depth_2:
        display_metric(
            "Top 20 Players",
            participant_strength[
                "top20_players"
            ],
            kind="integer",
        )

    with depth_3:
        display_metric(
            "Top 50 Players",
            participant_strength[
                "top50_players"
            ],
            kind="integer",
        )

    with depth_4:
        display_metric(
            "Top 100 Players",
            participant_strength[
                "top100_players"
            ],
            kind="integer",
        )

    st.subheader(
        "Winner and Loser Quality"
    )

    quality_1, quality_2, quality_3, quality_4 = (
        st.columns(4)
    )

    with quality_1:
        display_metric(
            "Average Winner Rank",
            row_value(
                latest,
                "average_winner_rank",
            ),
            decimals=1,
        )

    with quality_2:
        display_metric(
            "Average Loser Rank",
            row_value(
                latest,
                "average_loser_rank",
            ),
            decimals=1,
        )

    with quality_3:
        display_metric(
            "Average Winner Elo",
            row_value(
                latest,
                "average_winner_elo",
            ),
            decimals=0,
        )

    with quality_4:
        display_metric(
            "Average Loser Elo",
            row_value(
                latest,
                "average_loser_elo",
            ),
            decimals=0,
        )

    if (
        players is not None
        and not players.empty
    ):
        st.subheader(
            "Best Tournament Results"
        )

        player_columns = existing_columns(
            players,
            [
                "player_name",
                "matches",
                "wins",
                "losses",
                "win_rate",
            ],
        )

        st.dataframe(
            players[
                player_columns
            ]
            .head(20)
            .style.format(
                {
                    "win_rate": "{:.1%}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# PLAYING STYLE TAB
# ==========================================================

with style_tab:

    st.subheader(
        "Playing Style"
    )

    style_1, style_2, style_3, style_4 = (
        st.columns(4)
    )

    with style_1:
        display_metric(
            "Aces / Match",
            row_value(
                latest,
                "aces_per_match",
            ),
            decimals=2,
        )

    with style_2:
        display_metric(
            "Double Faults / Match",
            row_value(
                latest,
                "double_faults_per_match",
            ),
            decimals=2,
        )

    with style_3:
        display_metric(
            "Service Points / Match",
            row_value(
                latest,
                "service_points_per_match",
            ),
            decimals=1,
        )

    with style_4:
        display_metric(
            "Average Match Minutes",
            row_value(
                latest,
                "average_match_minutes",
            ),
            decimals=1,
        )

    style_5, style_6, style_7, style_8 = (
        st.columns(4)
    )

    with style_5:
        display_metric(
            "Break Points Faced / Match",
            row_value(
                latest,
                "break_points_faced_per_match",
            ),
            decimals=2,
        )

    with style_6:
        display_metric(
            "Break Points Saved",
            row_value(
                latest,
                "break_points_saved_pct",
            ),
            kind="percentage",
        )

    with style_7:
        display_metric(
            "Total Aces",
            row_value(
                latest,
                "total_aces",
            ),
            kind="integer",
        )

    with style_8:
        display_metric(
            "Total Double Faults",
            row_value(
                latest,
                "total_double_faults",
            ),
            kind="integer",
        )

    if (
        history is not None
        and not history.empty
    ):
        style_history_columns = existing_columns(
            history,
            [
                "season",
                "aces_per_match",
                "double_faults_per_match",
                "average_match_minutes",
            ],
        )

        style_value_columns = [
            column
            for column in [
                "aces_per_match",
                "double_faults_per_match",
                "average_match_minutes",
            ]
            if column in style_history_columns
        ]

        if style_value_columns:
            style_history = history.melt(
                id_vars=[
                    "season",
                ],
                value_vars=style_value_columns,
                var_name="metric",
                value_name="value",
            )

            style_labels = {
                "aces_per_match": (
                    "Aces / Match"
                ),
                "double_faults_per_match": (
                    "Double Faults / Match"
                ),
                "average_match_minutes": (
                    "Average Match Minutes"
                ),
            }

            style_history["metric"] = (
                style_history["metric"]
                .map(style_labels)
                .fillna(
                    style_history["metric"]
                )
            )

            figure = px.line(
                style_history,
                x="season",
                y="value",
                color="metric",
                markers=True,
                title=(
                    "Tournament Style Evolution"
                ),
                labels={
                    "season": "Season",
                    "value": "Value",
                    "metric": "Metric",
                },
            )

            figure.update_xaxes(
                dtick=1
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )


# ==========================================================
# ROUND BREAKDOWN TAB
# ==========================================================

with rounds_tab:

    st.subheader(
        "Round Breakdown"
    )

    if (
        round_breakdown is None
        or round_breakdown.empty
    ):
        st.info(
            "No round statistics are available."
        )

    else:
        round_order = [
            "Q1",
            "Q2",
            "Q3",
            "R128",
            "R64",
            "R32",
            "R16",
            "QF",
            "SF",
            "F",
            "RR",
        ]

        round_breakdown = (
            round_breakdown.copy()
        )

        round_breakdown["round_order"] = (
            round_breakdown["round"]
            .astype(str)
            .map(
                {
                    value: index
                    for index, value
                    in enumerate(
                        round_order
                    )
                }
            )
            .fillna(999)
        )

        round_breakdown = (
            round_breakdown
            .sort_values(
                "round_order"
            )
            .drop(
                columns=[
                    "round_order",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        round_columns = existing_columns(
            round_breakdown,
            [
                "round",
                "matches",
                "average_match_minutes",
                "average_winner_rank",
                "average_loser_rank",
                "favorite_win_rate",
                "total_aces",
                "aces_per_match",
                "total_double_faults",
            ],
        )

        round_display = round_breakdown[
            round_columns
        ].copy()

        st.dataframe(
            round_display.style.format(
                {
                    "average_match_minutes": (
                        "{:.1f}"
                    ),
                    "average_winner_rank": (
                        "{:.1f}"
                    ),
                    "average_loser_rank": (
                        "{:.1f}"
                    ),
                    "favorite_win_rate": (
                        "{:.1%}"
                    ),
                    "aces_per_match": (
                        "{:.2f}"
                    ),
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        if (
            "average_match_minutes"
            in round_breakdown.columns
        ):
            figure = px.bar(
                round_breakdown,
                x="round",
                y="average_match_minutes",
                color="round",
                title=(
                    "Average Match Duration by Round"
                ),
                labels={
                    "round": "Round",
                    "average_match_minutes": (
                        "Average Minutes"
                    ),
                },
                category_orders={
                    "round": round_order,
                },
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        if (
            "favorite_win_rate"
            in round_breakdown.columns
        ):
            figure = px.bar(
                round_breakdown,
                x="round",
                y="favorite_win_rate",
                color="round",
                title=(
                    "Favourite Win Rate by Round"
                ),
                labels={
                    "round": "Round",
                    "favorite_win_rate": (
                        "Favourite Win Rate"
                    ),
                },
                category_orders={
                    "round": round_order,
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


# ==========================================================
# PLAYERS TAB
# ==========================================================

with players_tab:

    st.subheader(
        "Tournament Participants"
    )

    if players is None or players.empty:
        st.info(
            "No player results are available."
        )

    else:
        player_display = players.copy()

        player_columns = existing_columns(
            player_display,
            [
                "player_name",
                "matches",
                "wins",
                "losses",
                "win_rate",
            ],
        )

        st.dataframe(
            player_display[
                player_columns
            ]
            .style.format(
                {
                    "win_rate": "{:.1%}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        top_players = (
            player_display
            .head(20)
            .copy()
        )

        figure = px.bar(
            top_players,
            x="player_name",
            y="wins",
            color="wins",
            title=(
                "Tournament Wins by Player"
            ),
            labels={
                "player_name": "Player",
                "wins": "Wins",
            },
        )

        figure.update_xaxes(
            tickangle=-45
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# ==========================================================
# HISTORY TAB
# ==========================================================

with history_tab:

    st.subheader(
        "Historical Tournament Strength"
    )

    if history is None or history.empty:
        st.info(
            "No historical editions are available."
        )

    else:
        rank_columns = [
            column
            for column in [
                "average_winner_rank",
                "average_loser_rank",
            ]
            if column in history.columns
        ]

        if rank_columns:
            rank_history = history.melt(
                id_vars=[
                    "season",
                ],
                value_vars=rank_columns,
                var_name="rank_type",
                value_name="average_rank",
            )

            rank_labels = {
                "average_winner_rank": (
                    "Average Winner Rank"
                ),
                "average_loser_rank": (
                    "Average Loser Rank"
                ),
            }

            rank_history["rank_type"] = (
                rank_history["rank_type"]
                .map(rank_labels)
                .fillna(
                    rank_history["rank_type"]
                )
            )

            figure = px.line(
                rank_history,
                x="season",
                y="average_rank",
                color="rank_type",
                markers=True,
                title=(
                    "Average Ranking Over Time"
                ),
                labels={
                    "season": "Season",
                    "average_rank": (
                        "Average Rank"
                    ),
                    "rank_type": (
                        "Ranking Type"
                    ),
                },
            )

            figure.update_yaxes(
                autorange="reversed"
            )

            figure.update_xaxes(
                dtick=1
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        strength_columns = [
            column
            for column in [
                "higher_ranked_winner_rate",
                "upset_rate",
                "favorite_win_rate",
            ]
            if column in history.columns
        ]

        if strength_columns:
            outcome_history = history.melt(
                id_vars=[
                    "season",
                ],
                value_vars=strength_columns,
                var_name="metric",
                value_name="rate",
            )

            outcome_labels = {
                "higher_ranked_winner_rate": (
                    "Higher-Ranked Winner"
                ),
                "upset_rate": "Upset Rate",
                "favorite_win_rate": (
                    "Favourite Win Rate"
                ),
            }

            outcome_history["metric"] = (
                outcome_history["metric"]
                .map(outcome_labels)
                .fillna(
                    outcome_history["metric"]
                )
            )

            figure = px.line(
                outcome_history,
                x="season",
                y="rate",
                color="metric",
                markers=True,
                title=(
                    "Tournament Outcome Evolution"
                ),
                labels={
                    "season": "Season",
                    "rate": "Rate",
                    "metric": "Metric",
                },
            )

            figure.update_yaxes(
                tickformat=".0%",
                range=[
                    0,
                    1,
                ],
            )

            figure.update_xaxes(
                dtick=1
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        if (
            "unique_players"
            in history.columns
        ):
            figure = px.line(
                history,
                x="season",
                y="unique_players",
                markers=True,
                title=(
                    "Player Field Size Over Time"
                ),
                labels={
                    "season": "Season",
                    "unique_players": (
                        "Unique Players"
                    ),
                },
            )

            figure.update_xaxes(
                dtick=1
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        st.subheader(
            "Tournament Edition History"
        )

        history_columns = existing_columns(
            history,
            [
                "season",
                "surface",
                "tournament_level",
                "matches",
                "unique_players",
                "average_match_minutes",
                "average_winner_rank",
                "average_loser_rank",
                "average_winner_elo",
                "average_loser_elo",
                "aces_per_match",
                "double_faults_per_match",
                "higher_ranked_winner_rate",
                "upset_rate",
                "odds_coverage",
                "favorite_win_rate",
            ],
        )

        history_display = history[
            history_columns
        ].sort_values(
            "season",
            ascending=False,
            kind="mergesort",
        )

        st.dataframe(
            history_display.style.format(
                {
                    "season": "{:.0f}",
                    "average_match_minutes": (
                        "{:.1f}"
                    ),
                    "average_winner_rank": (
                        "{:.1f}"
                    ),
                    "average_loser_rank": (
                        "{:.1f}"
                    ),
                    "average_winner_elo": (
                        "{:.0f}"
                    ),
                    "average_loser_elo": (
                        "{:.0f}"
                    ),
                    "aces_per_match": (
                        "{:.2f}"
                    ),
                    "double_faults_per_match": (
                        "{:.2f}"
                    ),
                    "higher_ranked_winner_rate": (
                        "{:.1%}"
                    ),
                    "upset_rate": "{:.1%}",
                    "odds_coverage": "{:.1%}",
                    "favorite_win_rate": (
                        "{:.1%}"
                    ),
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# MATCHES TAB
# ==========================================================

with matches_tab:

    st.subheader(
        "Tournament Matches"
    )

    if matches is None or matches.empty:
        st.info(
            "No matches are available "
            "for this tournament edition."
        )

    else:
        match_columns = existing_columns(
            matches,
            [
                "match_date",
                "round",
                "player_1_name",
                "player_2_name",
                "winner_name",
                "loser_name",
                "winner_rank",
                "loser_rank",
                "winner_elo_before",
                "loser_elo_before",
                "minutes",
                "total_aces",
                "total_double_faults",
                "favorite_won",
                "player_1_odds",
                "player_2_odds",
                "odds_source",
            ],
        )

        match_display = (
            matches[
                match_columns
            ]
            .sort_values(
                [
                    column
                    for column in [
                        "match_date",
                        "round",
                    ]
                    if column
                    in match_columns
                ],
                ascending=False,
                kind="mergesort",
            )
            .copy()
        )

        match_rename = {
            "match_date": "Date",
            "round": "Round",
            "player_1_name": "Player 1",
            "player_2_name": "Player 2",
            "winner_name": "Winner",
            "loser_name": "Loser",
            "winner_rank": "Winner Rank",
            "loser_rank": "Loser Rank",
            "winner_elo_before": (
                "Winner Elo"
            ),
            "loser_elo_before": (
                "Loser Elo"
            ),
            "minutes": "Minutes",
            "total_aces": "Total Aces",
            "total_double_faults": (
                "Total Double Faults"
            ),
            "favorite_won": (
                "Favourite Won"
            ),
            "player_1_odds": (
                "Player 1 Odds"
            ),
            "player_2_odds": (
                "Player 2 Odds"
            ),
            "odds_source": (
                "Odds Source"
            ),
        }

        match_display = (
            match_display.rename(
                columns=match_rename
            )
        )

        st.dataframe(
            match_display.style.format(
                {
                    "Winner Rank": "{:.0f}",
                    "Loser Rank": "{:.0f}",
                    "Winner Elo": "{:.0f}",
                    "Loser Elo": "{:.0f}",
                    "Minutes": "{:.0f}",
                    "Total Aces": "{:.0f}",
                    "Total Double Faults": (
                        "{:.0f}"
                    ),
                    "Player 1 Odds": "{:.2f}",
                    "Player 2 Odds": "{:.2f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# DATA DEFINITIONS
# ==========================================================

with st.expander(
    "ℹ️ Data definitions"
):
    st.markdown(
        """
        - A **tournament ID** identifies a specific tournament
          edition rather than the tournament across all years.
        - **Average winner rank** and **average loser rank**
          use ATP ranking values available before each match.
        - A **higher-ranked winner** has a numerically lower ATP
          rank than the loser.
        - An **upset** occurs when the winner has a numerically
          higher ATP rank than the loser.
        - **Global Elo** and **surface Elo** are measured before
          each match.
        - **Odds coverage** is the percentage of matches with
          a valid pair of historical pre-match odds.
        - Favourite statistics exclude matches with equal odds
          or missing prices.
        - Missing values are shown as `-` and are not replaced
          automatically with zero.
        """
    )