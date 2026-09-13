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


# ==========================================================
# DATA ACCESS
# ==========================================================

from data_access.tournament_betting_adapted import (
    get_tournaments,
    get_tournament_detail,
    get_tournament_matches,
    get_tournament_summary,
    get_available_tournament_seasons,
    get_available_tournament_surfaces,
    get_available_tournament_levels,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Tournament Betting",
    page_icon="💰",
    layout="wide",
)

st.title(
    "🏆 Tournament Betting Analysis"
)

st.info(
    """
    Historical betting metrics use matches with a valid pair
    of pre-match odds.

    Flat-stake simulations assume one unit on the favourite or
    one unit on the underdog in every eligible match. These
    statistics are descriptive and do not guarantee future
    results.
    """
)


# ==========================================================
# DISPLAY HELPERS
# ==========================================================

def safe_number(
    value,
    decimals=2,
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
    Formats a numeric value as an integer.
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


def display_metric(
    label,
    value,
    *,
    kind="number",
    decimals=2,
    help_text=None,
):
    """
    Displays a safely formatted Streamlit metric.
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


def numeric_series(
    frame,
    column,
    default=0.0,
):
    """
    Returns a numeric Series.

    If the column does not exist, a Series containing the
    default value is returned.
    """

    if column not in frame.columns:
        return pd.Series(
            default,
            index=frame.index,
            dtype="float64",
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def weighted_average(
    frame,
    value_column,
    weight_column,
):
    """
    Calculates a weighted average while ignoring null values.
    """

    if (
        frame is None
        or frame.empty
        or value_column not in frame.columns
        or weight_column not in frame.columns
    ):
        return np.nan

    values = pd.to_numeric(
        frame[value_column],
        errors="coerce",
    )

    weights = pd.to_numeric(
        frame[weight_column],
        errors="coerce",
    )

    valid = (
        values.notna()
        & weights.notna()
        & weights.gt(0)
    )

    if not valid.any():
        return np.nan

    return np.average(
        values[valid],
        weights=weights[valid],
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


def calculate_combined_metrics(
    frame,
):
    """
    Combines several tournament editions correctly.

    ROI is calculated from total profit divided by the number
    of eligible matches. It is not calculated as a simple
    average of edition-level ROI values.
    """

    if frame is None or frame.empty:
        return {
            "editions": 0,
            "matches": 0,
            "matches_with_odds": 0,
            "matches_without_odds": 0,
            "odds_coverage": np.nan,
            "favorite_matches": 0,
            "favorite_wins": 0,
            "favorite_losses": 0,
            "favorite_win_rate": np.nan,
            "favorite_profit": 0.0,
            "favorite_roi": np.nan,
            "underdog_wins": 0,
            "underdog_win_rate": np.nan,
            "underdog_profit": 0.0,
            "underdog_roi": np.nan,
            "average_favorite_odds": np.nan,
            "average_underdog_odds": np.nan,
            "average_winner_odds": np.nan,
            "average_market_overround": np.nan,
        }

    matches = numeric_series(
        frame,
        "matches",
    ).fillna(0).sum()

    matches_with_odds = numeric_series(
        frame,
        "matches_with_odds",
    ).fillna(0).sum()

    favorite_matches = numeric_series(
        frame,
        "valid_favorite_matches",
    ).fillna(0).sum()

    favorite_wins = numeric_series(
        frame,
        "favorite_wins",
    ).fillna(0).sum()

    favorite_profit = numeric_series(
        frame,
        "favorite_flat_stake_profit",
    ).fillna(0).sum()

    underdog_profit = numeric_series(
        frame,
        "underdog_flat_stake_profit",
    ).fillna(0).sum()

    favorite_losses = (
        favorite_matches
        - favorite_wins
    )

    underdog_wins = favorite_losses

    return {
        "editions": int(
            len(frame)
        ),
        "matches": int(
            matches
        ),
        "matches_with_odds": int(
            matches_with_odds
        ),
        "matches_without_odds": int(
            max(
                matches - matches_with_odds,
                0,
            )
        ),
        "odds_coverage": (
            matches_with_odds / matches
            if matches
            else np.nan
        ),
        "favorite_matches": int(
            favorite_matches
        ),
        "favorite_wins": int(
            favorite_wins
        ),
        "favorite_losses": int(
            favorite_losses
        ),
        "favorite_win_rate": (
            favorite_wins
            / favorite_matches
            if favorite_matches
            else np.nan
        ),
        "favorite_profit": float(
            favorite_profit
        ),
        "favorite_roi": (
            favorite_profit
            / favorite_matches
            if favorite_matches
            else np.nan
        ),
        "underdog_wins": int(
            underdog_wins
        ),
        "underdog_win_rate": (
            underdog_wins
            / favorite_matches
            if favorite_matches
            else np.nan
        ),
        "underdog_profit": float(
            underdog_profit
        ),
        "underdog_roi": (
            underdog_profit
            / favorite_matches
            if favorite_matches
            else np.nan
        ),
        "average_favorite_odds": (
            weighted_average(
                frame,
                "average_favorite_odds",
                "valid_favorite_matches",
            )
        ),
        "average_underdog_odds": (
            weighted_average(
                frame,
                "average_underdog_odds",
                "valid_favorite_matches",
            )
        ),
        "average_winner_odds": (
            weighted_average(
                frame,
                "average_winner_odds",
                "matches_with_odds",
            )
        ),
        "average_market_overround": (
            weighted_average(
                frame,
                "average_market_overround",
                "matches_with_odds",
            )
        ),
    }


def aggregate_tournament_dimension(
    frame,
    dimension,
):
    """
    Aggregates edition statistics by surface, season or level.
    """

    if (
        frame is None
        or frame.empty
        or dimension not in frame.columns
    ):
        return pd.DataFrame()

    records = []

    for value, subset in frame.groupby(
        dimension,
        dropna=False,
        observed=True,
    ):
        metrics = calculate_combined_metrics(
            subset
        )

        records.append(
            {
                dimension: value,
                **metrics,
            }
        )

    return pd.DataFrame(
        records
    )


def prepare_match_detail(
    matches,
):
    """
    Normalizes individual tournament matches for display.
    """

    if matches is None:
        return pd.DataFrame()

    result = matches.copy()

    if "match_date" in result.columns:
        result["match_date"] = pd.to_datetime(
            result["match_date"],
            errors="coerce",
        )

    numeric_columns = [
        "player_1_odds",
        "player_2_odds",
        "winner_odds",
        "favorite_odds",
        "underdog_odds",
        "market_overround",
        "favorite_flat_stake_profit",
        "underdog_flat_stake_profit",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


# ==========================================================
# LOAD TOURNAMENTS
# ==========================================================

tournament_names = get_tournaments()

if (
    tournament_names is None
    or len(tournament_names) == 0
):
    st.error(
        "No tournament betting data was found."
    )

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "Filters"
)

tournament_name = st.sidebar.selectbox(
    "Tournament",
    tournament_names,
)


# ----------------------------------------------------------
# Available filters for selected tournament
# ----------------------------------------------------------

season_options = (
    get_available_tournament_seasons(
        tournament_name
    )
)

surface_options = (
    get_available_tournament_surfaces(
        tournament_name
    )
)

level_options = (
    get_available_tournament_levels()
)

if "ALL" not in season_options:
    season_options = [
        "ALL",
        *season_options,
    ]

if "ALL" not in surface_options:
    surface_options = [
        "ALL",
        *surface_options,
    ]

if "ALL" not in level_options:
    level_options = [
        "ALL",
        *level_options,
    ]


selected_season = st.sidebar.selectbox(
    "Season",
    season_options,
)

selected_surface = st.sidebar.selectbox(
    "Surface",
    surface_options,
)

selected_level = st.sidebar.selectbox(
    "Tournament Level",
    level_options,
)


# ==========================================================
# LOAD DATA
# ==========================================================

detail = get_tournament_detail(
    tournament_name=tournament_name,
    season=selected_season,
    surface=selected_surface,
    tourney_level=selected_level,
)

all_tournament_detail = (
    get_tournament_detail(
        tournament_name=tournament_name,
    )
)

if detail is None or detail.empty:
    st.warning(
        "No tournament betting data was found "
        "for the selected filters."
    )

    st.stop()


# ==========================================================
# NORMALIZE DATA TYPES
# ==========================================================

detail = detail.copy()

if "season" in detail.columns:
    detail["season"] = pd.to_numeric(
        detail["season"],
        errors="coerce",
    )

numeric_columns = [
    "matches",
    "matches_with_odds",
    "matches_without_odds",
    "odds_coverage",
    "average_winner_odds",
    "average_favorite_odds",
    "average_underdog_odds",
    "favorite_wins",
    "favorite_losses",
    "valid_favorite_matches",
    "favorite_win_rate",
    "underdog_wins",
    "underdog_win_rate",
    "favorite_flat_stake_profit",
    "favorite_flat_stake_roi",
    "underdog_flat_stake_profit",
    "underdog_flat_stake_roi",
    "average_market_overround",
]

for column in numeric_columns:
    if column in detail.columns:
        detail[column] = pd.to_numeric(
            detail[column],
            errors="coerce",
        )


# ==========================================================
# SUMMARY
# ==========================================================

summary = calculate_combined_metrics(
    detail
)


# ==========================================================
# HEADER
# ==========================================================

st.header(
    f"🏆 {tournament_name}"
)

filter_description = []

if selected_season != "ALL":
    filter_description.append(
        f"Season: {selected_season}"
    )

if selected_surface != "ALL":
    filter_description.append(
        f"Surface: {selected_surface}"
    )

if selected_level != "ALL":
    filter_description.append(
        f"Level: {selected_level}"
    )

if filter_description:
    st.caption(
        " | ".join(
            filter_description
        )
    )

else:
    st.caption(
        "All available editions"
    )


# ==========================================================
# TABS
# ==========================================================

(
    overview_tab,
    favorite_tab,
    underdog_tab,
    surface_tab,
    season_tab,
    matches_tab,
) = st.tabs(
    [
        "📈 Overview",
        "⭐ Favourite",
        "🎯 Underdog",
        "🎾 Surface",
        "📅 Season",
        "🗂️ Matches",
    ]
)


# ==========================================================
# OVERVIEW TAB
# ==========================================================

with overview_tab:

    st.subheader(
        "Tournament Betting Overview"
    )

    overview_1, overview_2, overview_3, overview_4 = (
        st.columns(4)
    )

    with overview_1:
        display_metric(
            "Editions",
            summary["editions"],
            kind="integer",
        )

    with overview_2:
        display_metric(
            "Matches",
            summary["matches"],
            kind="integer",
        )

    with overview_3:
        display_metric(
            "Matches with Odds",
            summary[
                "matches_with_odds"
            ],
            kind="integer",
        )

    with overview_4:
        display_metric(
            "Odds Coverage",
            summary["odds_coverage"],
            kind="percentage",
        )

    overview_5, overview_6, overview_7, overview_8 = (
        st.columns(4)
    )

    with overview_5:
        display_metric(
            "Favourite Win Rate",
            summary[
                "favorite_win_rate"
            ],
            kind="percentage",
        )

    with overview_6:
        display_metric(
            "Underdog Win Rate",
            summary[
                "underdog_win_rate"
            ],
            kind="percentage",
        )

    with overview_7:
        display_metric(
            "Average Winner Odds",
            summary[
                "average_winner_odds"
            ],
            decimals=2,
        )

    with overview_8:
        display_metric(
            "Average Market Overround",
            summary[
                "average_market_overround"
            ],
            kind="percentage",
        )

    st.subheader(
        "Flat-Stake Results"
    )

    profit_1, profit_2, profit_3, profit_4 = (
        st.columns(4)
    )

    with profit_1:
        display_metric(
            "Favourite Profit",
            summary[
                "favorite_profit"
            ],
            decimals=2,
        )

    with profit_2:
        display_metric(
            "Favourite ROI",
            summary[
                "favorite_roi"
            ],
            kind="percentage",
        )

    with profit_3:
        display_metric(
            "Underdog Profit",
            summary[
                "underdog_profit"
            ],
            decimals=2,
        )

    with profit_4:
        display_metric(
            "Underdog ROI",
            summary[
                "underdog_roi"
            ],
            kind="percentage",
        )

    if (
        all_tournament_detail is not None
        and not all_tournament_detail.empty
    ):
        career_summary = (
            calculate_combined_metrics(
                all_tournament_detail
            )
        )

        st.divider()

        st.subheader(
            "All-Edition Summary"
        )

        career_1, career_2, career_3, career_4 = (
            st.columns(4)
        )

        with career_1:
            display_metric(
                "Historical Editions",
                career_summary[
                    "editions"
                ],
                kind="integer",
            )

        with career_2:
            display_metric(
                "Historical Matches",
                career_summary[
                    "matches"
                ],
                kind="integer",
            )

        with career_3:
            display_metric(
                "Historical Favourite ROI",
                career_summary[
                    "favorite_roi"
                ],
                kind="percentage",
            )

        with career_4:
            display_metric(
                "Historical Underdog ROI",
                career_summary[
                    "underdog_roi"
                ],
                kind="percentage",
            )


# ==========================================================
# FAVOURITE TAB
# ==========================================================

with favorite_tab:

    st.subheader(
        "⭐ Favourite Performance"
    )

    favorite_1, favorite_2, favorite_3 = (
        st.columns(3)
    )

    with favorite_1:
        display_metric(
            "Eligible Matches",
            summary[
                "favorite_matches"
            ],
            kind="integer",
        )

    with favorite_2:
        display_metric(
            "Wins",
            summary[
                "favorite_wins"
            ],
            kind="integer",
        )

    with favorite_3:
        display_metric(
            "Losses",
            summary[
                "favorite_losses"
            ],
            kind="integer",
        )

    favorite_4, favorite_5, favorite_6 = (
        st.columns(3)
    )

    with favorite_4:
        display_metric(
            "Win Rate",
            summary[
                "favorite_win_rate"
            ],
            kind="percentage",
        )

    with favorite_5:
        display_metric(
            "Profit Units",
            summary[
                "favorite_profit"
            ],
            decimals=2,
        )

    with favorite_6:
        display_metric(
            "Flat-Stake ROI",
            summary[
                "favorite_roi"
            ],
            kind="percentage",
        )

    favorite_7, favorite_8 = (
        st.columns(2)
    )

    with favorite_7:
        display_metric(
            "Average Favourite Odds",
            summary[
                "average_favorite_odds"
            ],
            decimals=2,
        )

    with favorite_8:
        display_metric(
            "Average Market Overround",
            summary[
                "average_market_overround"
            ],
            kind="percentage",
        )

    favorite_detail_columns = (
        existing_columns(
            detail,
            [
                "season",
                "surface",
                "tourney_level",
                "valid_favorite_matches",
                "favorite_wins",
                "favorite_losses",
                "favorite_win_rate",
                "average_favorite_odds",
                "favorite_flat_stake_profit",
                "favorite_flat_stake_roi",
            ],
        )
    )

    favorite_detail = detail[
        favorite_detail_columns
    ].copy()

    st.dataframe(
        favorite_detail.style.format(
            {
                "favorite_win_rate": (
                    "{:.1%}"
                ),
                "average_favorite_odds": (
                    "{:.2f}"
                ),
                "favorite_flat_stake_profit": (
                    "{:.2f}"
                ),
                "favorite_flat_stake_roi": (
                    "{:.1%}"
                ),
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    if "season" in detail.columns:
        favorite_season = (
            aggregate_tournament_dimension(
                detail,
                "season",
            )
        )

        favorite_season = (
            favorite_season
            .sort_values(
                "season"
            )
        )

        figure = px.line(
            favorite_season,
            x="season",
            y="favorite_roi",
            markers=True,
            title=(
                "Favourite ROI by Season"
            ),
            labels={
                "season": "Season",
                "favorite_roi": (
                    "Favourite ROI"
                ),
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        figure.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# ==========================================================
# UNDERDOG TAB
# ==========================================================

with underdog_tab:

    st.subheader(
        "🎯 Underdog Performance"
    )

    underdog_1, underdog_2, underdog_3 = (
        st.columns(3)
    )

    with underdog_1:
        display_metric(
            "Eligible Matches",
            summary[
                "favorite_matches"
            ],
            kind="integer",
        )

    with underdog_2:
        display_metric(
            "Wins",
            summary[
                "underdog_wins"
            ],
            kind="integer",
        )

    with underdog_3:
        display_metric(
            "Losses",
            summary[
                "favorite_wins"
            ],
            kind="integer",
        )

    underdog_4, underdog_5, underdog_6 = (
        st.columns(3)
    )

    with underdog_4:
        display_metric(
            "Win Rate",
            summary[
                "underdog_win_rate"
            ],
            kind="percentage",
        )

    with underdog_5:
        display_metric(
            "Profit Units",
            summary[
                "underdog_profit"
            ],
            decimals=2,
        )

    with underdog_6:
        display_metric(
            "Flat-Stake ROI",
            summary[
                "underdog_roi"
            ],
            kind="percentage",
        )

    underdog_7, underdog_8 = (
        st.columns(2)
    )

    with underdog_7:
        display_metric(
            "Average Underdog Odds",
            summary[
                "average_underdog_odds"
            ],
            decimals=2,
        )

    with underdog_8:
        display_metric(
            "Average Market Overround",
            summary[
                "average_market_overround"
            ],
            kind="percentage",
        )

    underdog_detail_columns = (
        existing_columns(
            detail,
            [
                "season",
                "surface",
                "tourney_level",
                "valid_favorite_matches",
                "underdog_wins",
                "underdog_win_rate",
                "average_underdog_odds",
                "underdog_flat_stake_profit",
                "underdog_flat_stake_roi",
            ],
        )
    )

    underdog_detail = detail[
        underdog_detail_columns
    ].copy()

    st.dataframe(
        underdog_detail.style.format(
            {
                "underdog_win_rate": (
                    "{:.1%}"
                ),
                "average_underdog_odds": (
                    "{:.2f}"
                ),
                "underdog_flat_stake_profit": (
                    "{:.2f}"
                ),
                "underdog_flat_stake_roi": (
                    "{:.1%}"
                ),
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    if "season" in detail.columns:
        underdog_season = (
            aggregate_tournament_dimension(
                detail,
                "season",
            )
        )

        underdog_season = (
            underdog_season
            .sort_values(
                "season"
            )
        )

        figure = px.line(
            underdog_season,
            x="season",
            y="underdog_roi",
            markers=True,
            title=(
                "Underdog ROI by Season"
            ),
            labels={
                "season": "Season",
                "underdog_roi": (
                    "Underdog ROI"
                ),
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        figure.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# ==========================================================
# SURFACE TAB
# ==========================================================

with surface_tab:

    st.subheader(
        "Performance by Surface"
    )

    surface_source = (
        all_tournament_detail.copy()
    )

    if selected_season != "ALL":
        surface_source = surface_source[
            surface_source["season"]
            .astype(str)
            .eq(str(selected_season))
        ]

    if selected_level != "ALL":
        surface_source = surface_source[
            surface_source["tourney_level"]
            .astype(str)
            .eq(str(selected_level))
        ]

    surface_summary = (
        aggregate_tournament_dimension(
            surface_source,
            "surface",
        )
    )

    if surface_summary.empty:
        st.info(
            "No surface statistics are available."
        )

    else:
        surface_columns = existing_columns(
            surface_summary,
            [
                "surface",
                "editions",
                "matches",
                "matches_with_odds",
                "odds_coverage",
                "favorite_win_rate",
                "favorite_profit",
                "favorite_roi",
                "underdog_win_rate",
                "underdog_profit",
                "underdog_roi",
            ],
        )

        st.dataframe(
            surface_summary[
                surface_columns
            ]
            .style.format(
                {
                    "odds_coverage": "{:.1%}",
                    "favorite_win_rate": (
                        "{:.1%}"
                    ),
                    "favorite_profit": (
                        "{:.2f}"
                    ),
                    "favorite_roi": "{:.1%}",
                    "underdog_win_rate": (
                        "{:.1%}"
                    ),
                    "underdog_profit": (
                        "{:.2f}"
                    ),
                    "underdog_roi": "{:.1%}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        roi_columns = [
            column
            for column in [
                "favorite_roi",
                "underdog_roi",
            ]
            if column in surface_summary.columns
        ]

        surface_long = surface_summary.melt(
            id_vars=[
                "surface",
            ],
            value_vars=roi_columns,
            var_name="strategy",
            value_name="roi",
        )

        surface_long["strategy"] = (
            surface_long["strategy"]
            .map(
                {
                    "favorite_roi": (
                        "Favourite"
                    ),
                    "underdog_roi": (
                        "Underdog"
                    ),
                }
            )
        )

        figure = px.bar(
            surface_long,
            x="surface",
            y="roi",
            color="strategy",
            barmode="group",
            title=(
                "Flat-Stake ROI by Surface"
            ),
            labels={
                "surface": "Surface",
                "roi": "Flat-Stake ROI",
                "strategy": "Strategy",
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )


# ==========================================================
# SEASON TAB
# ==========================================================

with season_tab:

    st.subheader(
        "Historical Evolution"
    )

    season_source = (
        all_tournament_detail.copy()
    )

    if selected_surface != "ALL":
        season_source = season_source[
            season_source["surface"]
            .astype(str)
            .eq(str(selected_surface))
        ]

    if selected_level != "ALL":
        season_source = season_source[
            season_source["tourney_level"]
            .astype(str)
            .eq(str(selected_level))
        ]

    season_summary = (
        aggregate_tournament_dimension(
            season_source,
            "season",
        )
    )

    if season_summary.empty:
        st.info(
            "No season statistics are available."
        )

    else:
        season_summary["season"] = (
            pd.to_numeric(
                season_summary["season"],
                errors="coerce",
            )
        )

        season_summary = (
            season_summary
            .dropna(
                subset=[
                    "season",
                ]
            )
            .sort_values(
                "season"
            )
            .reset_index(
                drop=True
            )
        )

        season_columns = existing_columns(
            season_summary,
            [
                "season",
                "editions",
                "matches",
                "matches_with_odds",
                "odds_coverage",
                "favorite_win_rate",
                "favorite_profit",
                "favorite_roi",
                "underdog_win_rate",
                "underdog_profit",
                "underdog_roi",
            ],
        )

        st.dataframe(
            season_summary[
                season_columns
            ]
            .style.format(
                {
                    "season": "{:.0f}",
                    "odds_coverage": "{:.1%}",
                    "favorite_win_rate": (
                        "{:.1%}"
                    ),
                    "favorite_profit": (
                        "{:.2f}"
                    ),
                    "favorite_roi": "{:.1%}",
                    "underdog_win_rate": (
                        "{:.1%}"
                    ),
                    "underdog_profit": (
                        "{:.2f}"
                    ),
                    "underdog_roi": "{:.1%}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        season_long = season_summary.melt(
            id_vars=[
                "season",
            ],
            value_vars=[
                "favorite_roi",
                "underdog_roi",
            ],
            var_name="strategy",
            value_name="roi",
        )

        season_long["strategy"] = (
            season_long["strategy"]
            .map(
                {
                    "favorite_roi": (
                        "Favourite"
                    ),
                    "underdog_roi": (
                        "Underdog"
                    ),
                }
            )
        )

        figure = px.line(
            season_long,
            x="season",
            y="roi",
            color="strategy",
            markers=True,
            title="ROI Evolution",
            labels={
                "season": "Season",
                "roi": "Flat-Stake ROI",
                "strategy": "Strategy",
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        figure.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        profit_long = season_summary.melt(
            id_vars=[
                "season",
            ],
            value_vars=[
                "favorite_profit",
                "underdog_profit",
            ],
            var_name="strategy",
            value_name="profit",
        )

        profit_long["strategy"] = (
            profit_long["strategy"]
            .map(
                {
                    "favorite_profit": (
                        "Favourite"
                    ),
                    "underdog_profit": (
                        "Underdog"
                    ),
                }
            )
        )

        profit_figure = px.bar(
            profit_long,
            x="season",
            y="profit",
            color="strategy",
            barmode="group",
            title="Profit Units by Season",
            labels={
                "season": "Season",
                "profit": "Profit Units",
                "strategy": "Strategy",
            },
        )

        profit_figure.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            profit_figure,
            use_container_width=True,
        )


# ==========================================================
# MATCH DETAIL TAB
# ==========================================================

with matches_tab:

    st.subheader(
        "Individual Match Detail"
    )

    match_frames = []

    for edition in detail.itertuples(
        index=False
    ):
        edition_id = getattr(
            edition,
            "tourney_id",
            None,
        )

        if edition_id is None:
            continue

        edition_matches = (
            get_tournament_matches(
                tournament_name=tournament_name,
                season=str(
                    getattr(
                        edition,
                        "season",
                        "ALL",
                    )
                ),
                surface=(
                    getattr(
                        edition,
                        "surface",
                        "ALL",
                    )
                ),
                with_odds_only=False,
            )
        )

        if (
            edition_matches is not None
            and not edition_matches.empty
        ):
            match_frames.append(
                edition_matches
            )

    if match_frames:
        tournament_matches = (
            pd.concat(
                match_frames,
                ignore_index=True,
            )
            .drop_duplicates(
                subset=[
                    column
                    for column in [
                        "tourney_id",
                        "match_num",
                    ]
                    if column
                    in match_frames[0].columns
                ]
            )
        )

    else:
        tournament_matches = (
            get_tournament_matches(
                tournament_name=tournament_name,
                season=selected_season,
                surface=selected_surface,
                with_odds_only=False,
            )
        )

    tournament_matches = (
        prepare_match_detail(
            tournament_matches
        )
    )

    if (
        tournament_matches is None
        or tournament_matches.empty
    ):
        st.info(
            "No individual matches are available."
        )

    else:
        match_columns = existing_columns(
            tournament_matches,
            [
                "match_date",
                "season",
                "surface",
                "round",
                "player_1_name",
                "player_2_name",
                "winner_name",
                "player_1_odds",
                "player_2_odds",
                "odds_source",
                "favorite_side",
                "favorite_odds",
                "favorite_won",
                "favorite_flat_stake_profit",
                "underdog_flat_stake_profit",
                "market_overround",
            ],
        )

        match_display = (
            tournament_matches[
                match_columns
            ]
            .sort_values(
                "match_date",
                ascending=False,
                kind="mergesort",
            )
            .copy()
        )

        match_renames = {
            "match_date": "Date",
            "season": "Season",
            "surface": "Surface",
            "round": "Round",
            "player_1_name": "Player 1",
            "player_2_name": "Player 2",
            "winner_name": "Winner",
            "player_1_odds": "Player 1 Odds",
            "player_2_odds": "Player 2 Odds",
            "odds_source": "Odds Source",
            "favorite_side": "Favourite Side",
            "favorite_odds": "Favourite Odds",
            "favorite_won": "Favourite Won",
            "favorite_flat_stake_profit": (
                "Favourite Profit"
            ),
            "underdog_flat_stake_profit": (
                "Underdog Profit"
            ),
            "market_overround": (
                "Market Overround"
            ),
        }

        match_display = (
            match_display.rename(
                columns=match_renames
            )
        )

        st.dataframe(
            match_display.style.format(
                {
                    "Player 1 Odds": "{:.2f}",
                    "Player 2 Odds": "{:.2f}",
                    "Favourite Odds": "{:.2f}",
                    "Favourite Profit": "{:.2f}",
                    "Underdog Profit": "{:.2f}",
                    "Market Overround": "{:.1%}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# COMPLETE AGGREGATED DETAIL
# ==========================================================

with st.expander(
    "Tournament Betting Detail"
):

    detail_columns = existing_columns(
        detail,
        [
            "tournament_name",
            "season",
            "surface",
            "tourney_level",
            "matches",
            "matches_with_odds",
            "matches_without_odds",
            "odds_coverage",
            "valid_favorite_matches",
            "favorite_wins",
            "favorite_losses",
            "favorite_win_rate",
            "average_favorite_odds",
            "favorite_flat_stake_profit",
            "favorite_flat_stake_roi",
            "underdog_wins",
            "underdog_win_rate",
            "average_underdog_odds",
            "underdog_flat_stake_profit",
            "underdog_flat_stake_roi",
            "average_winner_odds",
            "average_market_overround",
        ],
    )

    detail_display = (
        detail[
            detail_columns
        ]
        .sort_values(
            "season",
            ascending=False,
            kind="mergesort",
        )
        .copy()
    )

    st.dataframe(
        detail_display.style.format(
            {
                "odds_coverage": "{:.1%}",
                "favorite_win_rate": "{:.1%}",
                "average_favorite_odds": "{:.2f}",
                "favorite_flat_stake_profit": (
                    "{:.2f}"
                ),
                "favorite_flat_stake_roi": (
                    "{:.1%}"
                ),
                "underdog_win_rate": "{:.1%}",
                "average_underdog_odds": "{:.2f}",
                "underdog_flat_stake_profit": (
                    "{:.2f}"
                ),
                "underdog_flat_stake_roi": (
                    "{:.1%}"
                ),
                "average_winner_odds": "{:.2f}",
                "average_market_overround": (
                    "{:.1%}"
                ),
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )


# ==========================================================
# METHODOLOGY
# ==========================================================

with st.expander(
    "ℹ️ Methodology and definitions"
):
    st.markdown(
        """
        ### Favourite

        The favourite is the player with the lower pre-match
        decimal price.

        ### Underdog

        The underdog is the player with the higher pre-match
        decimal price.

        Matches with identical prices are not included in
        favourite-versus-underdog metrics.

        ### Flat-stake simulation

        The dashboard assumes a stake of one unit per eligible
        match.

        - Winning selection: `odds - 1`
        - Losing selection: `-1`
        - ROI: total profit divided by eligible matches

        ### Odds source

        The representative pair of prices is selected from the
        same source in this order:

        1. Average odds
        2. Pinnacle / PS
        3. Bet365
        4. Betfair Exchange

        ### Important limitation

        Historical returns do not include commission, liquidity,
        rejected stakes, betting limits or differences between
        quoted and executable prices. Historical profitability
        does not guarantee future profitability.
        """
    )