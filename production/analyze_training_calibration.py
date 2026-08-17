
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)

from production.services.model_loader import (
    load_model
)

DATASET_PATH = (
    "data/parquet/training_matches_with_elo.parquet"
)


def main():

    assets = load_model(

        tour="atp",

        surface="hard",

        engine="xgboost",

        version="v1"
    )

    df = pd.read_parquet(
        DATASET_PATH
    )

    model_features = (
        assets["features"]
    )

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

    probabilities = (

        assets["model"]

        .predict_proba(X)

        [:, 1]

    )

    results = pd.DataFrame({

        "actual":
            df["target"],

        "probability":
            probabilities,

        "prediction":
            (
                probabilities >= 0.5
            ).astype(int)

    })

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
        "ROWS:",
        len(results)
    )

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
        calibration
    )


if __name__ == "__main__":

    main()
