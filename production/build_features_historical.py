#La diferencia respecto a build_features() es que:

#Busca el último estado anterior o igual al match_date.
#Busca el último estado de superficie anterior o igual al match_date.
#Evita inactividades negativas.
#Está pensada únicamente para backtesting/calibración histórica.



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

    print()

    print("=" * 80)
    print("FEATURE DEBUG")
    print("=" * 80)

    print(
        "PLAYER A:",
        player_a_id
    )

    print(
        "MATCH DATE:",
        match_date
    )

    print(
        "A STATE DATE:",
        a_state["match_date"]
    )

    print()

    print(
        "PLAYER B:",
        player_b_id
    )

    print(
        "B STATE DATE:",
        b_state["match_date"]
    )

    if len(a_surface):

        print(
            "A SURFACE DATE:",
            a_surface.iloc[0][
                "last_surface_match_date"
            ]
        )

    if len(b_surface):

        print(
            "B SURFACE DATE:",
            b_surface.iloc[0][
                "last_surface_match_date"
            ]
        )

    print("=" * 80)

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
    

def build_features_historical(
    player_a_id,
    player_b_id,
    surface,
    match_date
):

    match_date = pd.to_datetime(
        match_date
    )

    a_state = (

        player_state[

            (
                player_state["player_id"]
                == player_a_id
            )

            &

            (
                player_state["match_date"]
                <= match_date
            )

        ]

        .sort_values(
            "match_date"
        )

    )

    if len(a_state) == 0:

        raise ValueError(
            f"No historical state found for {player_a_id}"
        )

    a_state = a_state.iloc[-1]

    b_state = (

        player_state[

            (
                player_state["player_id"]
                == player_b_id
            )

            &

            (
                player_state["match_date"]
                <= match_date
            )

        ]

        .sort_values(
            "match_date"
        )

    )

    if len(b_state) == 0:

        raise ValueError(
            f"No historical state found for {player_b_id}"
        )

    b_state = b_state.iloc[-1]

    a_surface = (

        surface_state[

            (
                surface_state["player_id"]
                == player_a_id
            )

            &

            (
                surface_state["surface"]
                == surface
            )

            &

            (
                surface_state[
                    "last_surface_match_date"
                ]
                <= match_date
            )

        ]

        .sort_values(
            "last_surface_match_date"
        )

    )

    b_surface = (

        surface_state[

            (
                surface_state["player_id"]
                == player_b_id
            )

            &

            (
                surface_state["surface"]
                == surface
            )

            &

            (
                surface_state[
                    "last_surface_match_date"
                ]
                <= match_date
            )

        ]

        .sort_values(
            "last_surface_match_date"
        )

    )

    surface_elo_a = None
    surface_elo_b = None

    surface_days_a = None
    surface_days_b = None

    if len(a_surface) > 0:

        a_surface = a_surface.iloc[-1]

        surface_elo_a = (
            a_surface["surface_elo"]
        )

        surface_days_a = max(
            0,
            (
                match_date
                -
                a_surface[
                    "last_surface_match_date"
                ]
            ).days
        )

    if len(b_surface) > 0:

        b_surface = b_surface.iloc[-1]

        surface_elo_b = (
            b_surface["surface_elo"]
        )

        surface_days_b = max(
            0,
            (
                match_date
                -
                b_surface[
                    "last_surface_match_date"
                ]
            ).days
        )

    days_inactive_a = max(
        0,
        (
            match_date
            -
            a_state["match_date"]
        ).days
    )

    days_inactive_b = max(
        0,
        (
            match_date
            -
            b_state["match_date"]
        ).days
    )

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
                (
                    surface_days_a or 0
                )
                -
                (
                    surface_days_b or 0
                )
        }
    ])


if __name__ == "__main__":

    df = build_features_historical(
        player_a_id="V0DZ",
        player_b_id="D0DW",
        surface="Hard",
        match_date="2026-08-04"
    )

    print()

    print(df)
