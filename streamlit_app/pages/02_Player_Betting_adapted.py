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
    sys.path.append(str(ROOT))


# ==========================================================
# DATA ACCESS
# ==========================================================

from data_access.betting_adapted import (
    get_betting_players,
    get_player_betting_summary,
    get_player_betting_detail,
    get_surfaces,
    get_years,
    calculate_betting_metrics,
)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Player Betting",
    page_icon="💰",
    layout="wide",
)

st.title(
    "💰 Player Betting Analysis"
)

st.info(
    """
    Betting statistics are calculated from matches with a valid
    pair of historical pre-match odds.

    Flat-stake profit assumes a stake of one unit on the selected
    player in every match with available odds. These figures are
    descriptive and do not represent guaranteed future returns.
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


def metric_value(
    metrics,
    key,
    default=np.nan,
):
    """
    Reads a value from a dictionary or pandas Series.
    """

    if metrics is None:
        return default

    if isinstance(metrics, dict):
        return metrics.get(
            key,
            default,
        )

    if isinstance(metrics, pd.Series):
        return metrics.get(
            key,
            default,
        )

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


def empty_metrics():
    """
    Default metric structure for empty subsets.
    """

    return {
        "matches": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": np.nan,
        "matches_with_odds": 0,
        "odds_coverage": np.nan,
        "average_odds": np.nan,
        "flat_stake_profit": 0.0,
        "flat_stake_roi": np.nan,
        "average_market_probability": np.nan,
    }


def calculate_subset_metrics(
    frame,
):
    """
    Calculates descriptive betting metrics for a subset.

    Uses calculate_betting_metrics from data_access when
    available. Falls back to a local calculation if needed.
    """

    if frame is None or frame.empty:
        return empty_metrics()

    try:
        return calculate_betting_metrics(
            frame
        )

    except (NameError, KeyError):
        matches = len(frame)

        wins = int(
            pd.to_numeric(
                frame.get(
                    "won",
                    pd.Series(
                        0,
                        index=frame.index,
                    ),
                ),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        if "has_odds" in frame.columns:
            valid_odds = (
                frame["has_odds"]
                .fillna(False)
                .astype(bool)
            )
        else:
            valid_odds = (
                pd.to_numeric(
                    frame.get(
                        "odds",
                        pd.Series(
                            np.nan,
                            index=frame.index,
                        ),
                    ),
                    errors="coerce",
                )
                .gt(1.0)
            )

        matches_with_odds = int(
            valid_odds.sum()
        )

        profit = pd.to_numeric(
            frame.get(
                "flat_stake_profit",
                pd.Series(
                    np.nan,
                    index=frame.index,
                ),
            ),
            errors="coerce",
        ).sum(
            min_count=1
        )

        if pd.isna(profit):
            profit = 0.0

        average_odds = pd.to_numeric(
            frame.loc[
                valid_odds,
                "odds",
            ]
            if "odds" in frame.columns
            else pd.Series(dtype=float),
            errors="coerce",
        ).mean()

        return {
            "matches": matches,
            "wins": wins,
            "losses": matches - wins,
            "win_rate": (
                wins / matches
                if matches
                else np.nan
            ),
            "matches_with_odds": (
                matches_with_odds
            ),
            "odds_coverage": (
                matches_with_odds / matches
                if matches
                else np.nan
            ),
            "average_odds": average_odds,
            "flat_stake_profit": float(
                profit
            ),
            "flat_stake_roi": (
                profit / matches_with_odds
                if matches_with_odds
                else np.nan
            ),
            "average_market_probability": (
                pd.to_numeric(
                    frame.get(
                        "market_probability_no_vig",
                        pd.Series(
                            np.nan,
                            index=frame.index,
                        ),
                    ),
                    errors="coerce",
                )
                .mean()
            ),
        }


def add_betting_dimensions(
    frame,
):
    """
    Adds dimensions used by the dashboard:

    - favourite / underdog status
    - odds bucket
    - profit in cumulative order
    """

    if frame is None:
        return pd.DataFrame()

    result = frame.copy()

    # ------------------------------------------------------
    # Numeric normalization
    # ------------------------------------------------------

    numeric_columns = [
        "odds",
        "flat_stake_profit",
        "market_probability_raw",
        "market_probability_no_vig",
        "market_overround",
        "won",
        "rank",
        "opponent_rank",
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    # ------------------------------------------------------
    # Valid odds
    # ------------------------------------------------------

    if "has_odds" not in result.columns:
        result["has_odds"] = (
            result["odds"].gt(1.0)
            if "odds" in result.columns
            else False
        )

    result["has_odds"] = (
        result["has_odds"]
        .fillna(False)
        .astype(bool)
    )

    # ------------------------------------------------------
    # Favourite or underdog
    #
    # market_probability_no_vig > 0.50 means the selected
    # player had the lower market price.
    # ------------------------------------------------------

    market_probability = pd.to_numeric(
        result.get(
            "market_probability_no_vig",
            pd.Series(
                np.nan,
                index=result.index,
            ),
        ),
        errors="coerce",
    )

    result["market_side"] = np.select(
        [
            result["has_odds"]
            & market_probability.gt(0.5),

            result["has_odds"]
            & market_probability.lt(0.5),

            result["has_odds"]
            & market_probability.eq(0.5),
        ],
        [
            "Favourite",
            "Underdog",
            "Even",
        ],
        default="No Odds",
    )

    # ------------------------------------------------------
    # Odds bucket
    # ------------------------------------------------------

    odds = pd.to_numeric(
        result.get(
            "odds",
            pd.Series(
                np.nan,
                index=result.index,
            ),
        ),
        errors="coerce",
    )

    odds_bucket_labels = [
        "1.01–1.25",
        "1.26–1.50",
        "1.51–1.75",
        "1.76–2.00",
        "2.01–2.50",
        "2.51–3.00",
        "3.01–4.00",
        "4.01+",
    ]

    result["odds_bucket"] = pd.cut(
        odds,
        bins=[
            1.0,
            1.25,
            1.50,
            1.75,
            2.00,
            2.50,
            3.00,
            4.00,
            np.inf,
        ],
        labels=odds_bucket_labels,
        include_lowest=False,
        right=True,
    )

    result["odds_bucket"] = (
        result["odds_bucket"]
        .astype("string")
        .fillna("No Odds")
    )

    # ------------------------------------------------------
    # Result label
    # ------------------------------------------------------

    if "result" not in result.columns:
        result["result"] = np.where(
            pd.to_numeric(
                result.get(
                    "won",
                    pd.Series(
                        0,
                        index=result.index,
                    ),
                ),
                errors="coerce",
            ).eq(1),
            "W",
            "L",
        )

    # ------------------------------------------------------
    # Cumulative profit
    # ------------------------------------------------------

    if "match_date" in result.columns:
        result["match_date"] = pd.to_datetime(
            result["match_date"],
            errors="coerce",
        )

        result = result.sort_values(
            [
                "match_date",
                "tourney_id",
                "match_num",
            ],
            kind="mergesort",
            na_position="last",
        )

    result["cumulative_profit"] = (
        pd.to_numeric(
            result.get(
                "flat_stake_profit",
                pd.Series(
                    np.nan,
                    index=result.index,
                ),
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .cumsum()
    )

    return result.reset_index(
        drop=True
    )


def aggregate_betting_dimension(
    frame,
    dimension,
):
    """
    Aggregates matches by season, surface, market side
    or odds bucket.
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
        metrics = calculate_subset_metrics(
            subset
        )

        records.append(
            {
                dimension: value,
                "matches": metric_value(
                    metrics,
                    "matches",
                    0,
                ),
                "wins": metric_value(
                    metrics,
                    "wins",
                    0,
                ),
                "losses": metric_value(
                    metrics,
                    "losses",
                    0,
                ),
                "win_rate": metric_value(
                    metrics,
                    "win_rate",
                ),
                "matches_with_odds": (
                    metric_value(
                        metrics,
                        "matches_with_odds",
                        0,
                    )
                ),
                "odds_coverage": (
                    metric_value(
                        metrics,
                        "odds_coverage",
                    )
                ),
                "average_odds": (
                    metric_value(
                        metrics,
                        "average_odds",
                    )
                ),
                "profit_units": (
                    metric_value(
                        metrics,
                        "flat_stake_profit",
                        0.0,
                    )
                ),
                "roi": metric_value(
                    metrics,
                    "flat_stake_roi",
                ),
                "average_market_probability": (
                    metric_value(
                        metrics,
                        "average_market_probability",
                    )
                ),
            }
        )

    return pd.DataFrame(
        records
    )


# ==========================================================
# LOAD PLAYERS
# ==========================================================

players = get_betting_players()

if players is None or players.empty:
    st.error(
        "No players with betting information were found."
    )
    st.stop()


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

selected_player = players[
    players["player_id"].eq(
        player_id
    )
].iloc[0]

player_name = selected_player[
    "player_name"
]


surface_options = get_surfaces()

if "ALL" not in surface_options:
    surface_options = [
        "ALL",
        *surface_options,
    ]

surface = st.sidebar.selectbox(
    "Surface",
    surface_options,
)


year_options = get_years()

if "ALL" not in year_options:
    year_options = [
        "ALL",
        *year_options,
    ]

year = st.sidebar.selectbox(
    "Year",
    year_options,
)


market_side_filter = (
    st.sidebar.selectbox(
        "Market Position",
        [
            "ALL",
            "Favourite",
            "Underdog",
            "Even",
        ],
    )
)


with_odds_only = (
    st.sidebar.checkbox(
        "Matches with odds only",
        value=True,
    )
)


# ==========================================================
# LOAD DATA
# ==========================================================

career_summary = (
    get_player_betting_summary(
        player_id
    )
)

detail = get_player_betting_detail(
    player_id
)

if detail is None:
    detail = pd.DataFrame()

detail = add_betting_dimensions(
    detail
)


# ==========================================================
# APPLY FILTERS
# ==========================================================

filtered_detail = detail.copy()

if surface != "ALL":
    filtered_detail = filtered_detail[
        filtered_detail["surface"]
        .astype(str)
        .eq(str(surface))
    ]

if year != "ALL":
    filtered_detail = filtered_detail[
        filtered_detail["season"]
        .astype(str)
        .eq(str(year))
    ]

if market_side_filter != "ALL":
    filtered_detail = filtered_detail[
        filtered_detail["market_side"]
        .astype(str)
        .eq(str(market_side_filter))
    ]

if with_odds_only:
    filtered_detail = filtered_detail[
        filtered_detail["has_odds"]
        .eq(True)
    ]

filtered_detail = (
    filtered_detail
    .reset_index(
        drop=True
    )
)


# ==========================================================
# FILTERED SUMMARY
# ==========================================================

filtered_metrics = (
    calculate_subset_metrics(
        filtered_detail
    )
)


# ==========================================================
# HEADER
# ==========================================================

st.header(
    f"💰 {player_name}"
)

filter_description = []

if surface != "ALL":
    filter_description.append(
        f"Surface: {surface}"
    )

if year != "ALL":
    filter_description.append(
        f"Season: {year}"
    )

if market_side_filter != "ALL":
    filter_description.append(
        f"Position: {market_side_filter}"
    )

if with_odds_only:
    filter_description.append(
        "Only matches with odds"
    )

if filter_description:
    st.caption(
        " | ".join(
            filter_description
        )
    )
else:
    st.caption(
        "All available matches"
    )


# ==========================================================
# TABS
# ==========================================================

(
    overview_tab,
    side_tab,
    surface_tab,
    odds_tab,
    season_tab,
    evolution_tab,
) = st.tabs(
    [
        "📈 Overview",
        "⭐ Favourite vs Underdog",
        "🎾 Surface",
        "💰 Odds Bucket",
        "📅 Season",
        "📉 Profit Evolution",
    ]
)


# ==========================================================
# OVERVIEW TAB
# ==========================================================

with overview_tab:

    st.subheader(
        "Filtered Performance"
    )

    overview_1, overview_2, overview_3 = (
        st.columns(3)
    )

    with overview_1:
        display_metric(
            "Matches",
            metric_value(
                filtered_metrics,
                "matches",
                0,
            ),
            kind="integer",
        )

    with overview_2:
        display_metric(
            "Wins",
            metric_value(
                filtered_metrics,
                "wins",
                0,
            ),
            kind="integer",
        )

    with overview_3:
        display_metric(
            "Losses",
            metric_value(
                filtered_metrics,
                "losses",
                0,
            ),
            kind="integer",
        )

    overview_4, overview_5, overview_6 = (
        st.columns(3)
    )

    with overview_4:
        display_metric(
            "Win Rate",
            metric_value(
                filtered_metrics,
                "win_rate",
            ),
            kind="percentage",
        )

    with overview_5:
        display_metric(
            "Matches with Odds",
            metric_value(
                filtered_metrics,
                "matches_with_odds",
                0,
            ),
            kind="integer",
        )

    with overview_6:
        display_metric(
            "Odds Coverage",
            metric_value(
                filtered_metrics,
                "odds_coverage",
            ),
            kind="percentage",
        )

    overview_7, overview_8, overview_9 = (
        st.columns(3)
    )

    with overview_7:
        display_metric(
            "Flat-Stake ROI",
            metric_value(
                filtered_metrics,
                "flat_stake_roi",
            ),
            kind="percentage",
            help_text=(
                "Profit divided by the number of "
                "matches with valid odds."
            ),
        )

    with overview_8:
        display_metric(
            "Profit Units",
            metric_value(
                filtered_metrics,
                "flat_stake_profit",
                0.0,
            ),
            decimals=2,
        )

    with overview_9:
        display_metric(
            "Average Odds",
            metric_value(
                filtered_metrics,
                "average_odds",
            ),
            decimals=2,
        )

    overview_10, overview_11 = (
        st.columns(2)
    )

    with overview_10:
        display_metric(
            "Average Market Probability",
            metric_value(
                filtered_metrics,
                "average_market_probability",
            ),
            kind="percentage",
        )

    with overview_11:
        if (
            career_summary is not None
            and "last_match_date"
            in career_summary.index
        ):
            last_match = pd.to_datetime(
                career_summary[
                    "last_match_date"
                ],
                errors="coerce",
            )

            st.metric(
                "Last Match",
                (
                    str(last_match.date())
                    if pd.notna(last_match)
                    else "-"
                ),
            )
        else:
            st.metric(
                "Last Match",
                "-",
            )

    st.divider()

    st.subheader(
        "Career Summary"
    )

    career_1, career_2, career_3, career_4 = (
        st.columns(4)
    )

    with career_1:
        display_metric(
            "Career Matches",
            metric_value(
                career_summary,
                "matches",
                0,
            ),
            kind="integer",
        )

    with career_2:
        display_metric(
            "Career Win Rate",
            metric_value(
                career_summary,
                "win_rate",
            ),
            kind="percentage",
        )

    with career_3:
        display_metric(
            "Career Matches with Odds",
            metric_value(
                career_summary,
                "matches_with_odds",
                0,
            ),
            kind="integer",
        )

    with career_4:
        display_metric(
            "Career Flat-Stake ROI",
            metric_value(
                career_summary,
                "flat_stake_roi",
            ),
            kind="percentage",
        )


# ==========================================================
# FAVOURITE VS UNDERDOG TAB
# ==========================================================

with side_tab:

    side_data = detail[
        detail["market_side"]
        .isin(
            [
                "Favourite",
                "Underdog",
                "Even",
            ]
        )
    ].copy()

    if surface != "ALL":
        side_data = side_data[
            side_data["surface"]
            .astype(str)
            .eq(str(surface))
        ]

    if year != "ALL":
        side_data = side_data[
            side_data["season"]
            .astype(str)
            .eq(str(year))
        ]

    side_summary = (
        aggregate_betting_dimension(
            side_data,
            "market_side",
        )
    )

    favourite_data = side_data[
        side_data["market_side"]
        .eq("Favourite")
    ]

    underdog_data = side_data[
        side_data["market_side"]
        .eq("Underdog")
    ]

    favourite_metrics = (
        calculate_subset_metrics(
            favourite_data
        )
    )

    underdog_metrics = (
        calculate_subset_metrics(
            underdog_data
        )
    )

    favourite_column, underdog_column = (
        st.columns(2)
    )

    with favourite_column:
        st.subheader(
            "⭐ Favourite"
        )

        favourite_1, favourite_2 = (
            st.columns(2)
        )

        with favourite_1:
            display_metric(
                "Matches",
                metric_value(
                    favourite_metrics,
                    "matches",
                    0,
                ),
                kind="integer",
            )

        with favourite_2:
            display_metric(
                "Wins",
                metric_value(
                    favourite_metrics,
                    "wins",
                    0,
                ),
                kind="integer",
            )

        favourite_3, favourite_4 = (
            st.columns(2)
        )

        with favourite_3:
            display_metric(
                "Win Rate",
                metric_value(
                    favourite_metrics,
                    "win_rate",
                ),
                kind="percentage",
            )

        with favourite_4:
            display_metric(
                "ROI",
                metric_value(
                    favourite_metrics,
                    "flat_stake_roi",
                ),
                kind="percentage",
            )

        favourite_5, favourite_6 = (
            st.columns(2)
        )

        with favourite_5:
            display_metric(
                "Profit Units",
                metric_value(
                    favourite_metrics,
                    "flat_stake_profit",
                    0.0,
                ),
                decimals=2,
            )

        with favourite_6:
            display_metric(
                "Average Odds",
                metric_value(
                    favourite_metrics,
                    "average_odds",
                ),
                decimals=2,
            )

    with underdog_column:
        st.subheader(
            "🎯 Underdog"
        )

        underdog_1, underdog_2 = (
            st.columns(2)
        )

        with underdog_1:
            display_metric(
                "Matches",
                metric_value(
                    underdog_metrics,
                    "matches",
                    0,
                ),
                kind="integer",
            )

        with underdog_2:
            display_metric(
                "Wins",
                metric_value(
                    underdog_metrics,
                    "wins",
                    0,
                ),
                kind="integer",
            )

        underdog_3, underdog_4 = (
            st.columns(2)
        )

        with underdog_3:
            display_metric(
                "Win Rate",
                metric_value(
                    underdog_metrics,
                    "win_rate",
                ),
                kind="percentage",
            )

        with underdog_4:
            display_metric(
                "ROI",
                metric_value(
                    underdog_metrics,
                    "flat_stake_roi",
                ),
                kind="percentage",
            )

        underdog_5, underdog_6 = (
            st.columns(2)
        )

        with underdog_5:
            display_metric(
                "Profit Units",
                metric_value(
                    underdog_metrics,
                    "flat_stake_profit",
                    0.0,
                ),
                decimals=2,
            )

        with underdog_6:
            display_metric(
                "Average Odds",
                metric_value(
                    underdog_metrics,
                    "average_odds",
                ),
                decimals=2,
            )

    if not side_summary.empty:
        st.divider()

        side_display = side_summary.copy()

        st.dataframe(
            side_display.style.format(
                {
                    "win_rate": "{:.1%}",
                    "odds_coverage": "{:.1%}",
                    "average_odds": "{:.2f}",
                    "profit_units": "{:.2f}",
                    "roi": "{:.1%}",
                    "average_market_probability": (
                        "{:.1%}"
                    ),
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        figure = px.bar(
            side_summary,
            x="market_side",
            y="roi",
            color="market_side",
            title=(
                "Flat-Stake ROI: "
                "Favourite vs Underdog"
            ),
            labels={
                "market_side": (
                    "Market Position"
                ),
                "roi": "Flat-Stake ROI",
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
# SURFACE TAB
# ==========================================================

with surface_tab:

    surface_source = detail.copy()

    if year != "ALL":
        surface_source = surface_source[
            surface_source["season"]
            .astype(str)
            .eq(str(year))
        ]

    if market_side_filter != "ALL":
        surface_source = surface_source[
            surface_source["market_side"]
            .astype(str)
            .eq(str(market_side_filter))
        ]

    if with_odds_only:
        surface_source = surface_source[
            surface_source["has_odds"]
            .eq(True)
        ]

    surface_summary = (
        aggregate_betting_dimension(
            surface_source,
            "surface",
        )
    )

    if surface_summary.empty:
        st.info(
            "No surface statistics are available "
            "for the selected filters."
        )

    else:
        st.dataframe(
            surface_summary.style.format(
                {
                    "win_rate": "{:.1%}",
                    "odds_coverage": "{:.1%}",
                    "average_odds": "{:.2f}",
                    "profit_units": "{:.2f}",
                    "roi": "{:.1%}",
                    "average_market_probability": (
                        "{:.1%}"
                    ),
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        figure = px.bar(
            surface_summary,
            x="surface",
            y="roi",
            color="surface",
            title="Flat-Stake ROI by Surface",
            labels={
                "surface": "Surface",
                "roi": "Flat-Stake ROI",
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        profit_figure = px.bar(
            surface_summary,
            x="surface",
            y="profit_units",
            color="surface",
            title="Profit Units by Surface",
            labels={
                "surface": "Surface",
                "profit_units": (
                    "Profit Units"
                ),
            },
        )

        st.plotly_chart(
            profit_figure,
            use_container_width=True,
        )


# ==========================================================
# ODDS BUCKET TAB
# ==========================================================

with odds_tab:

    odds_source = detail[
        detail["has_odds"]
        .eq(True)
    ].copy()

    if surface != "ALL":
        odds_source = odds_source[
            odds_source["surface"]
            .astype(str)
            .eq(str(surface))
        ]

    if year != "ALL":
        odds_source = odds_source[
            odds_source["season"]
            .astype(str)
            .eq(str(year))
        ]

    if market_side_filter != "ALL":
        odds_source = odds_source[
            odds_source["market_side"]
            .astype(str)
            .eq(str(market_side_filter))
        ]

    odds_summary = (
        aggregate_betting_dimension(
            odds_source,
            "odds_bucket",
        )
    )

    bucket_order = [
        "1.01–1.25",
        "1.26–1.50",
        "1.51–1.75",
        "1.76–2.00",
        "2.01–2.50",
        "2.51–3.00",
        "3.01–4.00",
        "4.01+",
        "No Odds",
    ]

    if not odds_summary.empty:
        odds_summary[
            "odds_bucket"
        ] = pd.Categorical(
            odds_summary[
                "odds_bucket"
            ],
            categories=bucket_order,
            ordered=True,
        )

        odds_summary = (
            odds_summary
            .sort_values(
                "odds_bucket"
            )
            .reset_index(
                drop=True
            )
        )

        st.dataframe(
            odds_summary.style.format(
                {
                    "win_rate": "{:.1%}",
                    "odds_coverage": "{:.1%}",
                    "average_odds": "{:.2f}",
                    "profit_units": "{:.2f}",
                    "roi": "{:.1%}",
                    "average_market_probability": (
                        "{:.1%}"
                    ),
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        figure = px.bar(
            odds_summary,
            x="odds_bucket",
            y="roi",
            color="odds_bucket",
            title="Flat-Stake ROI by Odds Bucket",
            labels={
                "odds_bucket": (
                    "Odds Bucket"
                ),
                "roi": "Flat-Stake ROI",
            },
            category_orders={
                "odds_bucket": (
                    bucket_order
                ),
            },
        )

        figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        win_rate_figure = px.bar(
            odds_summary,
            x="odds_bucket",
            y="win_rate",
            color="odds_bucket",
            title="Win Rate by Odds Bucket",
            labels={
                "odds_bucket": (
                    "Odds Bucket"
                ),
                "win_rate": "Win Rate",
            },
            category_orders={
                "odds_bucket": (
                    bucket_order
                ),
            },
        )

        win_rate_figure.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            win_rate_figure,
            use_container_width=True,
        )

    else:
        st.info(
            "No matches with odds are available "
            "for the selected filters."
        )


# ==========================================================
# SEASON TAB
# ==========================================================

with season_tab:

    season_source = detail.copy()

    if surface != "ALL":
        season_source = season_source[
            season_source["surface"]
            .astype(str)
            .eq(str(surface))
        ]

    if market_side_filter != "ALL":
        season_source = season_source[
            season_source["market_side"]
            .astype(str)
            .eq(str(market_side_filter))
        ]

    if with_odds_only:
        season_source = season_source[
            season_source["has_odds"]
            .eq(True)
        ]

    season_summary = (
        aggregate_betting_dimension(
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

        st.dataframe(
            season_summary.style.format(
                {
                    "season": "{:.0f}",
                    "win_rate": "{:.1%}",
                    "odds_coverage": "{:.1%}",
                    "average_odds": "{:.2f}",
                    "profit_units": "{:.2f}",
                    "roi": "{:.1%}",
                    "average_market_probability": (
                        "{:.1%}"
                    ),
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

        figure = px.line(
            season_summary,
            x="season",
            y="roi",
            markers=True,
            title="Flat-Stake ROI by Season",
            labels={
                "season": "Season",
                "roi": "Flat-Stake ROI",
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

        season_profit_figure = px.bar(
            season_summary,
            x="season",
            y="profit_units",
            color="profit_units",
            title="Profit Units by Season",
            labels={
                "season": "Season",
                "profit_units": (
                    "Profit Units"
                ),
            },
            color_continuous_scale=[
                "#d62728",
                "#f2f2f2",
                "#2ca02c",
            ],
        )

        season_profit_figure.update_xaxes(
            dtick=1
        )

        st.plotly_chart(
            season_profit_figure,
            use_container_width=True,
        )


# ==========================================================
# PROFIT EVOLUTION TAB
# ==========================================================

with evolution_tab:

    evolution = filtered_detail[
        filtered_detail["has_odds"]
        .eq(True)
    ].copy()

    if evolution.empty:
        st.info(
            "No matches with odds are available "
            "for the selected filters."
        )

    else:
        evolution = (
            evolution
            .sort_values(
                "match_date",
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        evolution[
            "cumulative_profit"
        ] = (
            pd.to_numeric(
                evolution[
                    "flat_stake_profit"
                ],
                errors="coerce",
            )
            .fillna(0.0)
            .cumsum()
        )

        figure = px.line(
            evolution,
            x="match_date",
            y="cumulative_profit",
            color="surface",
            markers=False,
            title=(
                "Cumulative Flat-Stake Profit"
            ),
            labels={
                "match_date": "Date",
                "cumulative_profit": (
                    "Cumulative Profit Units"
                ),
                "surface": "Surface",
            },
        )

        figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        # Overall cumulative line without splitting
        # the cumulative sum by surface.
        overall_figure = px.line(
            evolution,
            x="match_date",
            y="cumulative_profit",
            title=(
                "Overall Cumulative Profit"
            ),
            labels={
                "match_date": "Date",
                "cumulative_profit": (
                    "Cumulative Profit Units"
                ),
            },
        )

        overall_figure.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
        )

        st.plotly_chart(
            overall_figure,
            use_container_width=True,
        )

        drawdown_data = evolution[
            [
                "match_date",
                "cumulative_profit",
            ]
        ].copy()

        drawdown_data[
            "running_peak"
        ] = (
            drawdown_data[
                "cumulative_profit"
            ]
            .cummax()
        )

        drawdown_data[
            "drawdown"
        ] = (
            drawdown_data[
                "cumulative_profit"
            ]
            - drawdown_data[
                "running_peak"
            ]
        )

        drawdown_figure = px.area(
            drawdown_data,
            x="match_date",
            y="drawdown",
            title="Historical Drawdown",
            labels={
                "match_date": "Date",
                "drawdown": (
                    "Drawdown Units"
                ),
            },
        )

        st.plotly_chart(
            drawdown_figure,
            use_container_width=True,
        )


# ==========================================================
# DETAIL TABLE
# ==========================================================

with st.expander(
    "Detailed Betting Breakdown"
):

    if filtered_detail.empty:
        st.info(
            "No matches satisfy the selected filters."
        )

    else:
        detail_columns = [
            column
            for column in [
                "match_date",
                "season",
                "tourney_name",
                "surface",
                "round",
                "result",
                "opponent_name",
                "rank",
                "opponent_rank",
                "odds",
                "market_side",
                "market_probability_no_vig",
                "market_overround",
                "flat_stake_profit",
                "cumulative_profit",
                "odds_bucket",
            ]
            if column
            in filtered_detail.columns
        ]

        detailed_display = (
            filtered_detail[
                detail_columns
            ]
            .sort_values(
                "match_date",
                ascending=False,
                kind="mergesort",
            )
            .copy()
        )

        rename_columns = {
            "match_date": "Date",
            "season": "Season",
            "tourney_name": "Tournament",
            "surface": "Surface",
            "round": "Round",
            "result": "Result",
            "opponent_name": "Opponent",
            "rank": "Player Rank",
            "opponent_rank": (
                "Opponent Rank"
            ),
            "odds": "Odds",
            "market_side": (
                "Market Position"
            ),
            "market_probability_no_vig": (
                "Market Probability"
            ),
            "market_overround": (
                "Market Overround"
            ),
            "flat_stake_profit": (
                "Profit Units"
            ),
            "cumulative_profit": (
                "Cumulative Profit"
            ),
            "odds_bucket": (
                "Odds Bucket"
            ),
        }

        detailed_display = (
            detailed_display.rename(
                columns=rename_columns
            )
        )

        table_formats = {}

        if (
            "Market Probability"
            in detailed_display.columns
        ):
            table_formats[
                "Market Probability"
            ] = "{:.1%}"

        if (
            "Market Overround"
            in detailed_display.columns
        ):
            table_formats[
                "Market Overround"
            ] = "{:.1%}"

        if "Odds" in detailed_display.columns:
            table_formats[
                "Odds"
            ] = "{:.2f}"

        if (
            "Profit Units"
            in detailed_display.columns
        ):
            table_formats[
                "Profit Units"
            ] = "{:.2f}"

        if (
            "Cumulative Profit"
            in detailed_display.columns
        ):
            table_formats[
                "Cumulative Profit"
            ] = "{:.2f}"

        st.dataframe(
            detailed_display.style.format(
                table_formats,
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )


# ==========================================================
# DATA NOTES
# ==========================================================

with st.expander(
    "ℹ️ Methodology and definitions"
):
    st.markdown(
        """
        ### Flat-stake simulation

        The dashboard assumes a stake of one unit on the
        selected player in every match with valid historical
        odds.

        - If the player wins, profit is `odds - 1`.
        - If the player loses, profit is `-1`.
        - ROI is total profit divided by the number of matches
          with valid odds.

        ### Odds selection

        The representative odds are selected in this order:

        1. Average market odds
        2. Pinnacle / PS
        3. Bet365
        4. Betfair Exchange

        Both players' odds must come from the same provider.

        ### Favourite and underdog

        - A player is considered the favourite when their
          no-vig market probability is above 50%.
        - A player is considered the underdog when their
          no-vig market probability is below 50%.
        - Equal prices are classified as `Even`.

        ### Important limitation

        Historical ROI is descriptive. It does not account for
        commission, liquidity, rejected bets, market timing,
        limits or differences between available and closing
        prices. Historical profitability does not guarantee
        future profitability.
        """
    )