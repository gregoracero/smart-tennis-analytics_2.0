import pandas as pd
import streamlit as st
from utils.player_metrics_runtime import (
    build_player_metrics
)

st.set_page_config(
    page_title="Smart Tennis Analytics",
    layout="wide"
)

LEVEL_MAPPING = {
    "Grand Slam": ["G"],
    "Masters 1000": ["M", "1000"],
    "ATP 500": ["500"],
    "ATP 250": ["250"],
    "Challenger": ["C"],
    "Futures / ITF": ["F"],
    "Davis Cup": ["D"],
    "Olympic Games": ["O"]
}

PLAYER_METRICS = "data/parquet/player_metrics.parquet"
H2H_OVERALL = "data/parquet/h2h_overall.parquet"
H2H_SURFACE = "data/parquet/h2h_surface.parquet"
H2H_MATCHES = "data/parquet/h2h_matches.parquet"
PLAYER_MATCHES = (
    "data/parquet/analytics_player_matches.parquet"
)


@st.cache_data
def load_data():

    metrics = pd.read_parquet(
        PLAYER_METRICS
    )
    
    player_matches = pd.read_parquet(
        PLAYER_MATCHES
    )

    h2h_overall = pd.read_parquet(
        H2H_OVERALL
    )

    h2h_surface = pd.read_parquet(
        H2H_SURFACE
    )

    h2h_matches = pd.read_parquet(
        H2H_MATCHES
    )
            

    return (
        metrics,
        player_matches,
        h2h_overall,
        h2h_surface,
        h2h_matches
    )

df,player_matches, h2h_overall, h2h_surface, h2h_matches = load_data()

st.title("🎾 Smart Tennis Analytics")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

players = sorted(
    df["player"]
    .dropna()
    .unique()
)

player_a = st.sidebar.selectbox(
    "Player A",
    players,
    index=0
)

player_b = st.sidebar.selectbox(
    "Player B",
    players,
    index=1
)

surface = st.sidebar.selectbox(
    "Surface",
    sorted(
        df["surface"]
        .dropna()
        .unique()
    )
)

window = st.sidebar.selectbox(
    "Window",
    sorted(
        df["window"]
        .dropna()
        .unique()
    )
)
st.sidebar.subheader(
    "Tournament Categories"
)

include_gs = st.sidebar.checkbox(
    "Grand Slam",
    value=True
)

include_masters = st.sidebar.checkbox(
    "Masters 1000",
    value=True
)

include_500 = st.sidebar.checkbox(
    "ATP 500",
    value=True
)

include_250 = st.sidebar.checkbox(
    "ATP 250",
    value=True
)

include_challenger = st.sidebar.checkbox(
    "Challenger",
    value=False
)

include_futures = st.sidebar.checkbox(
    "Futures / ITF",
    value=False
)

include_davis = st.sidebar.checkbox(
    "Davis Cup",
    value=False
)

include_olympics = st.sidebar.checkbox(
    "Olympic Games",
    value=False
)

selected_levels = []

if include_gs:
    selected_levels.extend(["G"])

if include_masters:
    selected_levels.extend(["M", "1000"])

if include_500:
    selected_levels.extend(["500"])

if include_250:
    selected_levels.extend(["250"])

if include_challenger:
    selected_levels.extend(["C"])

if include_futures:
    selected_levels.extend(["F"])

if include_davis:
    selected_levels.extend(["D"])

if include_olympics:
    selected_levels.extend(["O"])
    
if not selected_levels:

    st.warning(
        "Select at least one tournament category."
    )

    st.stop()

# --------------------------------------------------
# Filter
# --------------------------------------------------

filtered = df[
    (df["surface"] == surface)
    &
    (df["window"] == window)
]

a = build_player_metrics(
    player_matches,
    player_a,
    surface,
    selected_levels,
    window
)

b = build_player_metrics(
    player_matches,
    player_b,
    surface,
    selected_levels,
    window
)

if a is None or b is None:

    st.warning(
        "No metrics available for this selection."
    )

    st.stop()




# --------------------------------------------------
# Header
# --------------------------------------------------

st.header(
    f"{player_a} vs {player_b}"
)

st.caption(
    f"Surface: {surface} | Window: {window}"
)


# --------------------------------------------------
# Overview
# --------------------------------------------------

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        f"{player_a} Win %",
        round(
            a["surface_recent_win_pct"],
            1
        )
    )

with col2:

    st.metric(
        f"{player_a} Avg Min",
        round(
            a["avg_minutes"],
            1
        )
    )

with col3:

    st.metric(
        f"{player_b} Win %",
        round(
            b["surface_recent_win_pct"],
            1
        )
    )

with col4:

    st.metric(
        f"{player_b} Avg Min",
        round(
            b["avg_minutes"],
            1
        )
    )

# --------------------------------------------------
# H2H OVERALL
# --------------------------------------------------

st.subheader("Head To Head")

overall = h2h_overall[
    (
        (
            (h2h_overall["player_a"] == player_a)
            &
            (h2h_overall["player_b"] == player_b)
        )
        |
        (
            (h2h_overall["player_a"] == player_b)
            &
            (h2h_overall["player_b"] == player_a)
        )
    )
]

if not overall.empty:

    row = overall.iloc[0]

    if row["player_a"] == player_a:

        wins_player_a = int(row["wins_a"])
        wins_player_b = int(row["wins_b"])

    else:

        wins_player_a = int(row["wins_b"])
        wins_player_b = int(row["wins_a"])

    h1, h2, h3 = st.columns([3, 2, 3])

    with h1:
        st.metric(
            player_a,
            wins_player_a
        )

    with h2:
        st.metric(
            "Matches",
            int(row["matches"])
        )

    with h3:
        st.metric(
            player_b,
            wins_player_b
        )

# --------------------------------------------------
# H2H SURFACE
# --------------------------------------------------

st.subheader("Event Breakdown")

event_df = h2h_matches[
    (
        (
            (h2h_matches["player_a"] == player_a)
            &
            (h2h_matches["player_b"] == player_b)
        )
        |
        (
            (h2h_matches["player_a"] == player_b)
            &
            (h2h_matches["player_b"] == player_a)
        )
    )
].copy()

if not event_df.empty:

    event_df = event_df.sort_values(
        "tourney_date",
        ascending=False
    )

    # --------------------------------------------------
    # Date formatting
    # --------------------------------------------------

    event_df["tourney_date"] = (
        event_df["tourney_date"]
        .fillna(0)
        .astype(int)
        .astype(str)
    )

    event_df["tourney_date"] = (
        pd.to_datetime(
            event_df["tourney_date"],
            format="%Y%m%d",
            errors="coerce"
        )
        .dt.strftime("%d %b %Y")
    )

    event_df = event_df[
        [
            "tourney_date",
            "tourney_name",
            "surface",
            "round",
            "winner",
            "score"
        ]
    ]

    event_df.columns = [
        "Date",
        "Tournament",
        "Surface",
        "Round",
        "Winner",
        "Score"
    ]

    st.dataframe(
        event_df,
        width="stretch",
        hide_index=True
    )

#---------------------------------------------------
# Player Form
#---------------------------------------------------

st.subheader("Players Form")
form_df = pd.DataFrame({

    "Metric": [

        "Matches",
        "Wins",
        "Win %",

        "Aces / Match",
        "Double Faults / Match",

        "1st Serve In %",
        "1st Serve Won %",
        "2nd Serve Won %",

        "Service Points Won %",
        "Service Games Won %",

        "Return Points Won %",
        "1st Return Won %",
        "2nd Return Won %",

        "Return Games Won %",

        "BP Chances / Match",
        "BP Converted / Match",
        "BP Conversion %",

        "Avg Minutes"
    ],

    player_a: [

        a["matches"],
        a["wins"],
        round(a["surface_recent_win_pct"], 1),

        round(a["aces_per_match"], 1),
        round(a["double_faults_per_match"], 1),

        round(a["first_serve_in_pct"], 1),
        round(a["first_serve_won_pct"], 1),
        round(a["second_serve_won_pct"], 1),

        round(a["service_points_won_pct"], 1),
        round(a["service_games_won_pct"], 1),

        round(a["return_points_won_pct"], 1),
        round(a["first_return_points_won_pct"], 1),
        round(a["second_return_points_won_pct"], 1),

        round(a["return_games_won_pct"], 1),

        round(a["break_points_generated_per_match"], 1),
        round(a["breaks_converted_per_match"], 1),
        round(a["break_conversion_pct"], 1),

        round(a["avg_minutes"], 1)
    ],

    player_b: [

        b["matches"],
        b["wins"],
        round(b["surface_recent_win_pct"], 1),

        round(b["aces_per_match"], 1),
        round(b["double_faults_per_match"], 1),

        round(b["first_serve_in_pct"], 1),
        round(b["first_serve_won_pct"], 1),
        round(b["second_serve_won_pct"], 1),

        round(b["service_points_won_pct"], 1),
        round(b["service_games_won_pct"], 1),

        round(b["return_points_won_pct"], 1),
        round(b["first_return_points_won_pct"], 1),
        round(b["second_return_points_won_pct"], 1),

        round(b["return_games_won_pct"], 1),

        round(b["break_points_generated_per_match"], 1),
        round(b["breaks_converted_per_match"], 1),
        round(b["break_conversion_pct"], 1),

        round(b["avg_minutes"], 1)
    ]
})

st.dataframe(
    form_df,
    width="stretch",
    hide_index=True
)

# --------------------------------------------------
# Comparison Table
# --------------------------------------------------

comparison = pd.DataFrame({

    "Metric": [

        "Matches",
        "Wins",
        "Losses",

        "Surface Win %",
        "1st Serve In %",
        "1st Serve Won %",
        "2nd Serve Won %",

        "BP Saved %",
        "Break Conversion %",

        "TB Win %",
        "TB Per Match",

        "Avg Minutes"
    ],

    player_a: [

        a["matches"],
        a["wins"],
        a["losses"],

        a["surface_recent_win_pct"],
        a["first_serve_in_pct"],
        a["first_serve_won_pct"],
        a["second_serve_won_pct"],

        a["bp_saved_pct"],
        a["break_conversion_pct"],

        a["tiebreak_win_pct"],
        a["tiebreaks_per_match"],

        a["avg_minutes"]
    ],

    player_b: [

        b["matches"],
        b["wins"],
        b["losses"],

        b["surface_recent_win_pct"],
        b["first_serve_in_pct"],
        b["first_serve_won_pct"],
        b["second_serve_won_pct"],

        b["bp_saved_pct"],
        b["break_conversion_pct"],

        b["tiebreak_win_pct"],
        b["tiebreaks_per_match"],

        b["avg_minutes"]
    ]
})

st.subheader("Comparison")

st.dataframe(
    comparison,
    width="stretch",
    hide_index=True
)

