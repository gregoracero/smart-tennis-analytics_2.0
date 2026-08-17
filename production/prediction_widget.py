
import streamlit as st

from production.services.player_lookup_service import (
    get_player_id,
    get_all_players
)

from production.services.prediction_service import (
    predict_match
)


def render_prediction():

    st.header("?? Match Prediction")

    players = get_all_players()

    col1, col2 = st.columns(2)

    with col1:

        tour = st.selectbox(
            "Tournament Type",
            [
                "atp",
                "challenger"
            ]
        )

        surface = st.selectbox(
            "Surface",
            [
                "hard",
                "clay",
                "grass"
            ]
        )

        match_date = st.date_input(
            "Match Date"
        )

        odds_a = st.number_input(
            "Player A Odds",
            min_value=1.01,
            value=2.00,
            step=0.01
        )

        odds_b = st.number_input(
            "Player B Odds",
            min_value=1.01,
            value=2.00,
            step=0.01
        )

    with col2:

        player_a = st.selectbox(
            "Player A",
            players,
            key="prediction_a"
        )

        player_b = st.selectbox(
            "Player B",
            players,
            key="prediction_b"
        )

    if player_a == player_b:

        st.warning(
            "Select two different players."
        )

        return

    if not st.button(
        "Predict",
        type="primary"
    ):
        return

    player_a_id = get_player_id(
        player_a
    )

    player_b_id = get_player_id(
        player_b
    )

    result = predict_match(

        player_a_id=player_a_id,

        player_b_id=player_b_id,

        tour=tour,

        surface=surface,

        match_date=str(match_date)

    )

    market_probability_a = (
        1 / odds_a
    )

    market_probability_b = (
        1 / odds_b
    )

    edge_a = (
        result["probability"]
        -
        market_probability_a
    )

    edge_b = (
        (1 - result["probability"])
        -
        market_probability_b
    )

    favorite = (
        player_a
        if result["probability"] >= 0.5
        else player_b
    )

    st.subheader("Prediction")

    st.caption(
        f"{player_a} ({player_a_id}) vs "
        f"{player_b} ({player_b_id})"
    )

    p1, p2 = st.columns(2)

    with p1:

        st.metric(
            player_a,
            f"{result['probability']:.2%}"
        )

    with p2:

        st.metric(
            player_b,
            f"{1-result['probability']:.2%}"
        )

    st.info(
        f"?? Model Favorite: {favorite}"
    )

    st.divider()

    st.subheader(
        "Player Readiness"
    )

    r1, r2 = st.columns(2)

    with r1:

        st.markdown(
            f"### {player_a}"
        )

        st.metric(
            "Elo",
            round(
                result["elo_a"],
                0
            )
        )

        st.metric(
            "Surface Elo",
            round(
                result["surface_elo_a"],
                0
            )
        )

        st.metric(
            "Days Inactive",
            round(
                result["days_inactive_a"],
                0
            )
        )

        st.metric(
            "Surface Days Inactive",
            round(
                result[
                    "surface_days_inactive_a"
                ],
                0
            )
        )

    with r2:

        st.markdown(
            f"### {player_b}"
        )

        st.metric(
            "Elo",
            round(
                result["elo_b"],
                0
            )
        )

        st.metric(
            "Surface Elo",
            round(
                result["surface_elo_b"],
                0
            )
        )

        st.metric(
            "Days Inactive",
            round(
                result["days_inactive_b"],
                0
            )
        )

        st.metric(
            "Surface Days Inactive",
            round(
                result[
                    "surface_days_inactive_b"
                ],
                0
            )
        )

    st.divider()

    st.subheader(
        "Model Drivers"
    )

    d1, d2, d3 = st.columns(3)

    with d1:

        st.metric(
            "Delta Elo",
            round(
                result["delta_elo"],
                2
            )
        )

    with d2:

        st.metric(
            "Delta Inactivity",
            round(
                result[
                    "delta_inactivity_days"
                ],
                2
            )
        )

    with d3:

        st.metric(
            "Delta Surface Inactivity",
            round(
                result[
                    "delta_surface_inactivity_days"
                ],
                2
            )
        )

    st.divider()

    st.subheader(
        "Market Edge"
    )

    e1, e2 = st.columns(2)

    with e1:

        st.metric(
            f"{player_a} Edge",
            f"{edge_a:.2%}"
        )

    with e2:

        st.metric(
            f"{player_b} Edge",
            f"{edge_b:.2%}"
        )

    st.subheader(
        "Recommendation"
    )

    if edge_a > 0.10:

        st.success(
            f"? BET PLAYER A\n\n"
            f"{player_a}\n\n"
            f"Edge: {edge_a:.2%}"
        )

    elif edge_b > 0.10:

        st.success(
            f"? BET PLAYER B\n\n"
            f"{player_b}\n\n"
            f"Edge: {edge_b:.2%}"
        )

    else:

        st.info(
            "No value bet detected."
        )

    st.divider()

    if result["strategy_v3"]:

        st.success(
            "? Strategy V3 MATCH"
        )

    else:

        st.warning(
            "? Strategy V3 NO MATCH"
        )

    st.divider()

    st.subheader(
        "Model Information"
    )

    m1, m2, m3, m4 = st.columns(4)

    with m1:

        st.metric(
            "Tour",
            result["metadata"]["tour"]
        )

    with m2:

        st.metric(
            "Surface",
            result["metadata"]["surface"]
        )

    with m3:

        st.metric(
            "Engine",
            result["metadata"]["engine"]
        )

    with m4:

        st.metric(
            "Version",
            result["metadata"]["version"]
        )

    st.subheader(
        "Model Metrics"
    )

    k1, k2 = st.columns(2)

    with k1:

        st.metric(
            "ROC AUC",
            round(
                result["metrics"]["roc_auc"],
                4
            )
        )

    with k2:

        st.metric(
            "Accuracy",
            round(
                result["metrics"]["accuracy"],
                4
            )
        )

    with st.expander(
        "Advanced Metrics"
    ):

        st.json(
            result["metrics"]
        )
