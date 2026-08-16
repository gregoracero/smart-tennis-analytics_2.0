
import pandas as pd

future = pd.read_csv(
    "data/predictions/future_matches.csv"
)

future["match_date"] = pd.to_datetime(
    future["match_date"]
)

state = pd.read_parquet(
    "data/parquet/player_state.parquet"
)

surface_state = pd.read_parquet(
    "data/parquet/player_surface_last_match.parquet"
)

state["match_date"] = pd.to_datetime(
    state["match_date"]
)

surface_state[
    "last_surface_match_date"
] = pd.to_datetime(
    surface_state[
        "last_surface_match_date"
    ]
)

rows = []

for _,match in future.iterrows():

    player_a = match["player_a_id"]
    player_b = match["player_b_id"]

    surface = match["surface"]

    prediction_date = match["match_date"]

    a_state = state[
        state["player_id"] == player_a
    ].iloc[0]

    b_state = state[
        state["player_id"] == player_b
    ].iloc[0]

    a_surface = surface_state[
        (surface_state["player_id"] == player_a)
        &
        (surface_state["surface"] == surface)
    ]

    b_surface = surface_state[
        (surface_state["player_id"] == player_b)
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
            prediction_date
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
            prediction_date
            -
            b_surface[
                "last_surface_match_date"
            ]
        ).days

    days_inactive_a = (
        prediction_date
        -
        a_state["match_date"]
    ).days

    days_inactive_b = (
        prediction_date
        -
        b_state["match_date"]
    ).days

    rows.append({

        "player_a_id": player_a,
        "player_b_id": player_b,
        "surface": surface,

        "elo_a": a_state["elo"],
        "elo_b": b_state["elo"],

        "surface_elo_a": surface_elo_a,
        "surface_elo_b": surface_elo_b,

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

    })

features = pd.DataFrame(
    rows
)

features.to_parquet(
    "data/parquet/prediction_features.parquet",
    index=False
)

print()
print(features)

print()

print(
    "Saved: data/parquet/prediction_features.parquet"
)
