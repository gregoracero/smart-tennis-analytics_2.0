
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)

from production.services.prediction_service import (
    predict_match_historical
)


ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MATCHES_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "analytics_player_matches.parquet"
)

TOURNAMENT = "Canada Masters"

START_DATE = 20260802
END_DATE = 20260814


def main():

    print("START")

    matches = pd.read_parquet(
        MATCHES_PATH
    )

    matches = matches[
        matches["tourney_name"]
        == TOURNAMENT
    ]

    matches = matches[
        matches["tourney_date"]
        .between(
            START_DATE,
            END_DATE
        )
    ]

    print()
    print(
        f"Tournament: {TOURNAMENT}"
    )

    print(
        f"Rows found: {len(matches)}"
    )

    rows = []

    for _, row in matches.iterrows():

        try:

            prediction = predict_match_historical(

                player_a_id=
                    row["player_id"],

                player_b_id=
                    row["opponent_id"],

                tour="atp",

                surface=
                    str(
                        row["surface"]
                    ).lower(),

                match_date=
                    str(
                        row["tourney_date"]
                    )
            )

            probability = prediction[
                "probability"
            ]

            rows.append({

                "player":
                    row["player"],

                "opponent":
                    row["opponent"],

                "player_id":
                    row["player_id"],

                "opponent_id":
                    row["opponent_id"],

                "surface":
                    row["surface"],

                "tourney_date":
                    row["tourney_date"],

                "actual":
                    int(
                        row["won_match"]
                    ),

                "probability":
                    probability,

                "prediction":
                    int(
                        probability >= 0.5
                    )
            })

        except Exception as e:

            print(
                "SKIPPED:",
                repr(e)
            )

    results = pd.DataFrame(
        rows
    )

    print()
    print(
        f"Predictions: {len(results)}"
    )

    if results.empty:

        print(
            "No predictions generated."
        )

        return

    accuracy = accuracy_score(

        results["actual"],

        results["prediction"]
    )

    brier = brier_score_loss(

        results["actual"],

        results["probability"]
    )

    logloss = log_loss(

        results["actual"],

        results["probability"]
    )

    print()
    print("=" * 60)

    print(
        "ACCURACY:",
        round(
            accuracy,
            4
        )
    )

    print(
        "BRIER SCORE:",
        round(
            brier,
            4
        )
    )

    print(
        "LOG LOSS:",
        round(
            logloss,
            4
        )
    )

    print("=" * 60)

    print()
    print(
        "PROBABILITY DISTRIBUTION"
    )

    print()

    print(
        results[
            "probability"
        ].describe()
    )

    favorite_accuracy = (

        (
            results[
                "probability"
            ] >= 0.5
        )

        ==

        (
            results[
                "actual"
            ] == 1
        )

    ).mean()

    print()

    print(
        "FAVORITE ACCURACY:",
        round(
            favorite_accuracy,
            4
        )
    )

    high_conf = results[
        results["probability"] >= 0.80
    ]

    print()

    print(
        "HIGH CONFIDENCE CASES:",
        len(high_conf)
    )

    print(
        "HIGH CONFIDENCE WIN RATE:",
        round(
            high_conf[
                "actual"
            ].mean(),
            4
        )
    )
    
    print()

    print(
        high_conf[
            [
                "player",
                "opponent",
                "actual",
                "probability"
            ]
        ]
        .head(100)
    )

    print()

    print(
        high_conf["actual"]
        .value_counts()
    )

    print()

    print(
        "CORRELATION"
    )

    print(
        results[
            [
                "actual",
                "probability"
            ]
        ].corr()
    )

    results["bucket"] = pd.cut(

        results["probability"],

        bins=[
            0.0,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            1.0
        ]
    )

    calibration = (

        results

        .groupby(
            "bucket",
            observed=False
        )

        .agg(

            avg_prediction=(
                "probability",
                "mean"
            ),

            actual_win_pct=(
                "actual",
                "mean"
            ),

            matches=(
                "actual",
                "count"
            )
        )

        .reset_index()
    )

    print()

    print(
        "CALIBRATION TABLE"
    )

    print()

    print(
        calibration
    )

    print()

    print(
        "TOP 50 HIGHEST PROBABILITIES"
    )

    print()

    top_probs = (

        results

        .sort_values(
            "probability",
            ascending=False
        )

        .head(50)
    )

    print(

        top_probs[
            [
                "player",
                "opponent",
                "actual",
                "probability"
            ]
        ]

    )

    print()

    print("=" * 80)
    print("FEATURE AUDIT")
    print("=" * 80)

    extreme_cases = (
        top_probs
        .head(20)
    )

    for _, row in extreme_cases.iterrows():

        try:

            prediction = predict_match(

                player_a_id=
                    row["player_id"],

                player_b_id=
                    row["opponent_id"],

                tour="atp",

                surface=
                    str(
                        row["surface"]
                    ).lower(),

                match_date=
                    str(
                        row["tourney_date"]
                    )
            )

            print()

            print(
                f"{row['player']} vs {row['opponent']}"
            )

            print(
                f"Actual={row['actual']}"
            )

            print(
                f"Probability={row['probability']:.4f}"
            )
            
            print(
                f"Inactivity A={prediction['days_inactive_a']}"
            )

            print(
                f"Inactivity B={prediction['days_inactive_b']}"
            )

            print(
                f"Surface Inactivity A={prediction['surface_days_inactive_a']}"
            )

            print(
                f"Surface Inactivity B={prediction['surface_days_inactive_b']}"
            )

            print(
                f"Elo A={prediction['elo_a']:.2f}"
            )

            print(
                f"Elo B={prediction['elo_b']:.2f}"
            )

            print(
                f"Delta Elo={prediction['delta_elo']:.2f}"
            )

        except Exception as e:

            print(
                "AUDIT FAILED:",
                repr(e)
            )
            
        print()

    print("=" * 80)
    print("FULL PREDICTION DUMP")
    print("=" * 80)

    audit_player = "Daniel Altmaier"
    audit_opponent = "Aleksandar Vukic"

    audit_case = results[

        (results["player"] == audit_player)

        &

        (results["opponent"] == audit_opponent)

    ]

    if len(audit_case):

        row = audit_case.iloc[0]

        prediction = predict_match(

            player_a_id=
                row["player_id"],

            player_b_id=
                row["opponent_id"],

            tour="atp",

            surface=
                str(
                    row["surface"]
                ).lower(),

            match_date=
                str(
                    row["tourney_date"]
                )
        )

        from pprint import pprint

        pprint(
            prediction
        )

    else:

        print(
            "Audit case not found."
        )


if __name__ == "__main__":

    main()
