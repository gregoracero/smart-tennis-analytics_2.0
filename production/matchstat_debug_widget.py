
import streamlit as st

from production.services.matchstat_api import (
    get_today_odds
)


def render_matchstat_debug():

    st.header(
        "?? Matchstat Debug"
    )

    tour = st.selectbox(

        "Tour",

        [
            "atp",
            "challenger"
        ],

        key="debug_tour"
    )

    if not st.button(
        "Load Matchstat Data"
    ):
        return

    try:

        data = get_today_odds(
            tour=tour
        )

        st.success(
            f"Records: {len(data)}"
        )

        if len(data):

            st.subheader(
                "First Record"
            )

            st.json(
                data[0]
            )

            st.subheader(
                "Raw Records"
            )

            st.dataframe(
                data,
                width="stretch"
            )

    except Exception as e:

        st.error(
            str(e)
        )
