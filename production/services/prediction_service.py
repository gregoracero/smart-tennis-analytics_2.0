
from production.feature_builder import (
    build_features,
    build_features_historical
)

from production.services.model_loader import (
    load_model
)



def strategy_v3(
    probability,
    delta_elo,
    delta_surface_inactivity_days
):

    return (

        probability >= 0.5

        and

        delta_elo < 0

        and

        delta_surface_inactivity_days < -14
    )


def predict_match(

    player_a_id,
    player_b_id,

    tour,

    surface,

    match_date,

    engine="xgboost",

    version="v1"
):

    assets = load_model(

        tour=tour,

        surface=surface,

        engine=engine,

        version=version
    )

    df = build_features(

        player_a_id,

        player_b_id,

        surface.capitalize(),

        match_date
    )

    model_features = assets["features"]

    for col in model_features:

        if col not in df.columns:

            df[col] = None

    X = df[
        model_features
    ]

    X = (
        assets["imputer"]
        .transform(X)
    )

    probability = float(
        assets["model"]
        .predict_proba(X)[0][1]
    )

    delta_elo = float(
        df["delta_elo"].iloc[0]
    )

    delta_inactivity_days = float(
        df["delta_inactivity_days"].iloc[0]
    )

    delta_surface_inactivity_days = float(
        df[
            "delta_surface_inactivity_days"
        ].iloc[0]
    )

    return {

        "probability":
            probability,

        "elo_a":
            float(
                df["elo_a"].iloc[0]
            ),

        "elo_b":
            float(
                df["elo_b"].iloc[0]
            ),

        "surface_elo_a":
            float(
                df["surface_elo_a"].iloc[0]
            ),

        "surface_elo_b":
            float(
                df["surface_elo_b"].iloc[0]
            ),

        "days_inactive_a":
            float(
                df["days_inactive_a"].iloc[0]
            ),

        "days_inactive_b":
            float(
                df["days_inactive_b"].iloc[0]
            ),

        "surface_days_inactive_a":
            float(
                df[
                    "surface_days_inactive_a"
                ].iloc[0]
            ),

        "surface_days_inactive_b":
            float(
                df[
                    "surface_days_inactive_b"
                ].iloc[0]
            ),

        "delta_elo":
            delta_elo,

        "delta_inactivity_days":
            delta_inactivity_days,

        "delta_surface_inactivity_days":
            delta_surface_inactivity_days,

        "strategy_v3":
            strategy_v3(

                probability,

                delta_elo,

                delta_surface_inactivity_days

            ),

        "metadata":
            assets["metadata"],

        "metrics":
            assets["metrics"]
    }
    
def predict_match_historical(

    player_a_id,
    player_b_id,

    tour,

    surface,

    match_date,
    engine="xgboost",
    
    version="v1"  
):

    assets = load_model(

        tour=tour,

        surface=surface,

        engine=engine,

        version=version
    )

    df = build_features_historical(

        player_a_id,

        player_b_id,

        surface.capitalize(),

        match_date
    )

    model_features = assets["features"]

    for col in model_features:

        if col not in df.columns:

            df[col] = None

    X = df[
        model_features
    ]

    X = (
        assets["imputer"]
        .transform(X)
    )

    probability = float(
        assets["model"]
        .predict_proba(X)[0][1]
    )

    delta_elo = float(
        df["delta_elo"].iloc[0]
    )

    delta_inactivity_days = float(
        df["delta_inactivity_days"].iloc[0]
    )

    delta_surface_inactivity_days = float(
        df[
            "delta_surface_inactivity_days"
        ].iloc[0]
    )

    return {

        "probability":
            probability,

        "elo_a":
            float(
                df["elo_a"].iloc[0]
            ),

        "elo_b":
            float(
                df["elo_b"].iloc[0]
            ),

        "surface_elo_a":
            float(
                df["surface_elo_a"].iloc[0]
            ),

        "surface_elo_b":
            float(
                df["surface_elo_b"].iloc[0]
            ),

        "days_inactive_a":
            float(
                df["days_inactive_a"].iloc[0]
            ),

        "days_inactive_b":
            float(
                df["days_inactive_b"].iloc[0]
            ),

        "surface_days_inactive_a":
            float(
                df[
                    "surface_days_inactive_a"
                ].iloc[0]
            ),

        "surface_days_inactive_b":
            float(
                df[
                    "surface_days_inactive_b"
                ].iloc[0]
            ),

        "delta_elo":
            delta_elo,

        "delta_inactivity_days":
            delta_inactivity_days,

        "delta_surface_inactivity_days":
            delta_surface_inactivity_days,

        "strategy_v3":
            strategy_v3(

                probability,

                delta_elo,

                delta_surface_inactivity_days

            ),

        "metadata":
            assets["metadata"],

        "metrics":
            assets["metrics"]
    }
