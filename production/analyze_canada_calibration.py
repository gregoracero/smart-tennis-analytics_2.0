import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss
)

from production.services.model_loader import (
    load_model
)


def main():

    assets = load_model(

        tour="atp",

        surface="hard",

        engine="xgboost",

        version="v1"
    )

    train = pd.read_parquet(
        "data/parquet/training_matches_with_elo.parquet"
    )

    canada = pd.read_parquet(
        "data/parquet/analytics_player_matches.parquet"
    )

    canada = canada[
        (
            canada["tourney_name"]
            == "Canada Masters"
        )
        &
        (
            canada["tourney_date"]
            .between(
                20260802,
                20260814
            )
        )
    ]

    train["match_date_int"] = (
        pd.to_datetime(
            train["match_date"]
        )
        .dt.strftime("%Y%m%d")
        .astype(int)
    )

    merged = canada.merge(

        train,

        left_on=[
            "player_id",
            "opponent_id",
            "tourney_date"
        ],

        right_on=[
            "player_a_id",
            "player_b_id",
            "match_date_int"
        ],

        how="inner"
    )

    print()
    print(
        "CANADA ROWS:",
        len(canada)
    )

    print(
        "MATCHED ROWS:",
        len(merged)
    )

    model_features = (
        assets["features"]
    )

    X = merged[
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
            merged["target"],

        "probability":
            probabilities,

        "prediction":
            (
                probabilities >= 0.5
            ).astype(int),

        "delta_elo":
            merged["delta_elo"]

    })

    results["abs_delta_elo"] = (
        results["delta_elo"]
        .abs()
    )

    results["elo_bucket"] = pd.cut(

        results["abs_delta_elo"],

        bins=[
            0,
            50,
            100,
            150,
            200,
            9999
        ],

        labels=[
            "0-50",
            "50-100",
            "100-150",
            "150-200",
            "200+"
        ]
    )

    elo_analysis = (

        results

        .groupby(
            "elo_bucket",
            observed=False
        )

        .agg(

            matches=(
                "actual",
                "count"
            ),

            accuracy=(
                "actual",
                lambda x:
                (
                    x
                    ==
                    results.loc[
                        x.index,
                        "prediction"
                    ]
                ).mean()
            ),

            avg_probability=(
                "probability",
                "mean"
            )

        )

        .reset_index()
    )

    print()

    print("=" * 60)
    print("ELO ANALYSIS")
    print("=" * 60)

    print()

    print(
        elo_analysis
    )

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

    high_conf = results[
        results["probability"] >= 0.80
    ]

    print()

    print(
        "HIGH CONFIDENCE CASES:",
        len(high_conf)
    )

    if len(high_conf):

        print(
            "HIGH CONFIDENCE WIN RATE:",
            round(
                high_conf[
                    "actual"
                ].mean(),
                4
            )
        )

    results["player"] = (
        merged["player"]
    )

    results["opponent"] = (
        merged["opponent"]
    )

    print()

    print("=" * 60)
    print("BIGGEST WRONG PREDICTIONS")
    print("=" * 60)

    wrong = results[

        results["actual"]
        !=
        results["prediction"]

    ]

    wrong = wrong.sort_values(
        "probability",
        ascending=False
    )

    print()

    print(
        wrong[
            [
                "player",
                "opponent",
                "actual",
                "probability",
                "delta_elo"
            ]
        ]
        .head(25)
    )


    player_analysis = (

        results

        .groupby(
            "player",
            observed=False
        )

        .agg(

            matches=(
                "actual",
                "count"
            ),

            accuracy=(
                "actual",
                lambda x:
                (
                    x
                    ==
                    results.loc[
                        x.index,
                        "prediction"
                    ]
                ).mean()
            ),

            avg_probability=(
                "probability",
                "mean"
            )

        )

        .reset_index()
    )

    print()

    print("=" * 60)
    print("WORST PLAYER PERFORMANCE")
    print("=" * 60)

    print()

    print(
        player_analysis[
            player_analysis["matches"]
            >= 2
        ]
        .sort_values(
            [
                "accuracy",
                "matches"
            ],
            ascending=[
                True,
                False
            ]
        )
        .head(30)
    )

    print()

    print("=" * 60)
    print("BEST PLAYER PERFORMANCE")
    print("=" * 60)

    print()

    print(
        player_analysis[
            player_analysis["matches"]
            >= 2
        ]
        .sort_values(
            [
                "accuracy",
                "matches"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(30)
    )

    results["player"] = (
        merged["player"]
    )

    results["opponent"] = (
        merged["opponent"]
    )

    print()

    print("=" * 60)
    print("BIGGEST WRONG PREDICTIONS")
    print("=" * 60)

    wrong = results[

        results["actual"]
        !=
        results["prediction"]

    ]

    wrong = wrong.sort_values(
        "probability",
        ascending=False
    )

    print()

    print(
        wrong[
            [
                "player",
                "opponent",
                "actual",
                "probability",
                "delta_elo"
            ]
        ]
        .head(25)
    )
    
    correct = results[
        results["actual"]
        ==
        results["prediction"]
    ]

    wrong = results[
        results["actual"]
        !=
        results["prediction"]
    ]

    print()
    print("=" * 60)
    print("CORRECT VS WRONG")
    print("=" * 60)
    print()

    print(
        "CORRECT:",
        len(correct)
    )

    print(
        "WRONG:",
        len(wrong)
    )

    print()

    print(
        "CORRECT DELTA ELO"
    )

    print(
        correct[
            "delta_elo"
        ]
        .abs()
        .describe()
    )

    print()

    print(
        "WRONG DELTA ELO"
    )

    print(
        wrong[
            "delta_elo"
        ]
        .abs()
        .describe()
    )

    print()

    print(
        "CORRECT PROBABILITY"
    )

    print(
        correct[
            "probability"
        ]
        .describe()
    )

    print()

    print(
        "WRONG PROBABILITY"
    )

    print(
        wrong[
            "probability"
        ]
        .describe()
    )
        
    results["player"] = merged["player"]
    results["opponent"] = merged["opponent"]

    print()
    print("=" * 60)
    print("BIGGEST WRONG PREDICTIONS")
    print("=" * 60)

    wrong = results[
        results["actual"]
        != results["prediction"]
    ]

    wrong = wrong.sort_values(
        "probability",
        ascending=False
    )

    print(
        wrong[
            [
                "player",
                "opponent",
                "actual",
                "probability",
                "delta_elo"
            ]
        ]
        .head(25)
    )


if __name__ == "__main__":

    main()

    
