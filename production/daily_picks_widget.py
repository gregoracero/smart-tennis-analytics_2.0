
import streamlit as st

from production.services.daily_prediction_service import (
    predict_daily_fixtures
)


def render_daily_picks():

    st.header(
        "?? Daily Picks"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "Using local Matchstat cache."
        )

    st.info(
        "Using Matchstat today's fixtures."
    )

    if not st.button(
        "Load Matches",
        type="primary"
    ):
        return

    try:

        with st.spinner(
            "Loading fixtures and generating predictions..."
        ):

            df = predict_daily_fixtures()

        if len(df) == 0:

            st.warning(
                "No matches found."
            )

            return

        st.success(
            f"{len(df)} matches loaded."
        )

        # --------------------------------------------------
        # ALL MATCHES
        # --------------------------------------------------

        st.subheader(
            "📋 All Upcoming Matches"
        )

        all_matches = df.sort_values(
            "date"
        )

        st.dataframe(
            all_matches,
            width="stretch",
            hide_index=True
        )

        # --------------------------------------------------
        # TOP OPPORTUNITIES
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "🏆 Top Opportunities"
        )

        top_opportunities = (
            df
            .sort_values(
                "max_edge",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            top_opportunities,
            width="stretch",
            hide_index=True
        )

        # --------------------------------------------------
        # VALUE BETS
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "✅ Value Bets (>10% Edge)"
        )

        value_bets = df[
            df["max_edge"] >= 0.10
        ]

        if len(value_bets):

            st.dataframe(
                value_bets
                .sort_values(
                    "max_edge",
                    ascending=False
                ),
                width="stretch",
                hide_index=True
            )

        else:

            st.info(
                "No value bets found."
            )

        # --------------------------------------------------
        # STRATEGY V3
        # --------------------------------------------------

        if "strategy_v3" in df.columns:

            strategy_df = df[
                df["strategy_v3"] == True
            ]

            st.divider()

            st.subheader(
                "🎯 Strategy V3 Signals"
            )

            if len(strategy_df):

                st.dataframe(
                    strategy_df
                    .sort_values(
                        "max_edge",
                        ascending=False
                    ),
                    width="stretch",
                    hide_index=True
                )

            else:

                st.info(
                    "No Strategy V3 signals found."
                )

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Summary"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Matches",
                len(df)
            )

        with c2:

            st.metric(
                "Value Bets",
                len(
                    df[
                        df["max_edge"] >= 0.10
                    ]
                )
            )

        with c3:

            st.metric(
                "High Confidence",
                len(
                    df[
                        df["confidence"]
                        == "HIGH"
                    ]
                )
                if "confidence" in df.columns
                else 0
            )

        with c4:

            st.metric(
                "Strategy V3",
                len(
                    df[
                        df["strategy_v3"]
                        == True
                    ]
                )
                if "strategy_v3" in df.columns
                else 0
            )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )
