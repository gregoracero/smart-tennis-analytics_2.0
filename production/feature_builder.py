
from pathlib import Path

import pandas as pd


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

PLAYER_STATE_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "player_state.parquet"
)

SURFACE_STATE_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "player_surface_last_match.parquet"
)


player_state = pd.read_parquet(
    PLAYER_STATE_PATH
)

surface_state = pd.read_parquet(
    SURFACE_STATE_PATH
)

player_state["match_date"] = pd.to_datetime(
    player_state["match_date"]
)

surface_state["last_surface_match_date"] = pd.to_datetime(
    surface_state["last_surface_match_date"]
)


def build_features(
    player_a_id,
    player_b_id,
    surface,
    match_date
):

    match_date = pd.to_datetime(
        match_date
    )

    a_state = player_state[
        player_state["player_id"] == player_a_id
    ]

    if len(a_state) == 0:

        raise ValueError(
            f"Player A not found: {player_a_id}"
        )

    b_state = player_state[
        player_state["player_id"] == player_b_id
    ]

    if len(b_state) == 0:

        raise ValueError(
            f"Player B not found: {player_b_id}"
        )

    a_state = a_state.iloc[0]
    b_state = b_state.iloc[0]

    a_surface = surface_state[
        (surface_state["player_id"] == player_a_id)
        &
        (surface_state["surface"] == surface)
    ]

    b_surface = surface_state[
        (surface_state["player_id"] == player_b_id)
        &
        (surface_state["surface"] == surface)
    ]

    surface_elo_a = None
    surface_elo_b = None

    surface_days_a = None
    surface_days_b = None

    if len(a_surface):

        a_surface = a_surface.iloc[0]

        surface_elo_a = (
            a_surface["surface_elo"]
        )

        surface_days_a = (
            match_date
            -
            a_surface[
                "last_surface_match_date"
            ]
        ).days

    if len(b_surface):

        b_surface = b_surface.iloc[0]

        surface_elo_b = (
            b_surface["surface_elo"]
        )

        surface_days_b = (
            match_date
            -
            b_surface[
                "last_surface_match_date"
            ]
        ).days

    days_inactive_a = (
        match_date
        -
        a_state["match_date"]
    ).days

    days_inactive_b = (
        match_date
        -
        b_state["match_date"]
    ).days

    return pd.DataFrame([
        {

            "player_a_id":
                player_a_id,

            "player_b_id":
                player_b_id,

            "elo_a":
                a_state["elo"],

            "elo_b":
                b_state["elo"],

            "surface_elo_a":
                surface_elo_a,

            "surface_elo_b":
                surface_elo_b,

            "days_inactive_a":
                days_inactive_a,

            "days_inactive_b":
                days_inactive_b,

            "surface_days_inactive_a":
                surface_days_a,

            "surface_days_inactive_b":
                surface_days_b,

            "delta_elo":
                a_state["elo"]
                -
                b_state["elo"],

            "delta_inactivity_days":
                days_inactive_a
                -
                days_inactive_b,

            "delta_surface_inactivity_days":
                surface_days_a
                -
                surface_days_b
        }
    ])


if __name__ == "__main__":

    df = build_features(

        player_a_id="S0S1",

        player_b_id="N0AE",

        surface="Hard",

        match_date="2026-08-16"
    )

    print()

    print(df)
