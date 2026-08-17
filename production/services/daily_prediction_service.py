
import json
from pathlib import Path

import pandas as pd

from production.services.player_lookup_service import (
    get_player_id
)

from production.services.prediction_service import (
    predict_match
)

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

MATCHSTAT_CACHE = (
    ROOT
    / "cache"
    / "matchstat"
    / "upcoming_matches.json"
)


def predict_daily_fixtures():

    with open(
        MATCHSTAT_CACHE,
        "r",
        encoding="utf-8"
    ) as f:

        payload = json.load(f)

    matches = payload["matches"]

    rows = []

    for match in matches:

        try:

            player_a = (
                match["player1"]["name"]
            )

            player_b = (
                match["player2"]["name"]
            )

            if (
                "/" in player_a
                or
                "/" in player_b
            ):
                continue

            odds = (
                match.get("odds")
                or {}
            )

            odds_a = odds.get("k1")
            odds_b = odds.get("k2")

            if (
                odds_a is None
                or
                odds_b is None
            ):
                continue

            surface = (
                match["court"]
                .lower()
            )

            tour = (
                match["type"]
                .lower()
            )

            player_a_id = get_player_id(
                player_a
            )

            player_b_id = get_player_id(
                player_b
            )
  
            
            if (
                player_a_id is None
                or
                player_b_id is None
            ):

                print()

                print(
                    "LOOKUP FAILED:"
                )

                print(
                    player_a,
                    "->",
                    player_a_id
                )

                print(
                    player_b,
                    "->",
                    player_b_id
                )

                continue

            prediction = predict_match(

                player_a_id=
                    player_a_id,

                player_b_id=
                    player_b_id,

                tour=tour,

                surface=surface,

                match_date=
                    match["date"][:10]
            )

            probability_a = (
                prediction[
                    "probability"
                ]
            )

            probability_b = (
                1
                -
                probability_a
            )

            market_prob_a = (
                1 / odds_a
            )

            market_prob_b = (
                1 / odds_b
            )

            edge_a = (
                probability_a
                -
                market_prob_a
            )

            edge_b = (
                probability_b
                -
                market_prob_b
            )

            recommendation = None

            max_edge = max(
                edge_a,
                edge_b
            )

            if max_edge >= 0.05:

                recommendation = (
                    player_a
                    if edge_a > edge_b
                    else player_b
                )

            if max_edge >= 0.15:

                confidence = "HIGH"

            elif max_edge >= 0.10:

                confidence = "MEDIUM"

            elif max_edge >= 0.05:

                confidence = "LOW"

            else:

                confidence = "NONE"

            rows.append({

                "tournament":
                    match[
                        "tournament"
                    ][
                        "name"
                    ],

                "surface":
                    surface,

                "date":
                    match[
                        "date"
                    ],

                "player_a":
                    player_a,

                "player_b":
                    player_b,

                "odds_a":
                    odds_a,

                "odds_b":
                    odds_b,

                "market_prob_a":
                    round(
                        market_prob_a,
                        4
                    ),

                "market_prob_b":
                    round(
                        market_prob_b,
                        4
                    ),

                "market_gap_a":
                    round(
                        probability_a
                        -
                        market_prob_a,
                        4
                    ),

                "market_gap_b":
                    round(
                        probability_b
                        -
                        market_prob_b,
                        4
                    ),

                "market_prob_a":
                    round(
                        market_prob_a,
                        4
                    ),

                "market_prob_b":
                    round(
                        market_prob_b,
                        4
                    ),

                "market_gap_a":
                    round(
                        probability_a
                        -
                        market_prob_a,
                        4
                    ),

                "market_gap_b":
                    round(
                        probability_b
                        -
                        market_prob_b,
                        4
                    ),

                "probability_a":
                    round(
                        probability_a,
                        4
                    ),

                "probability_b":
                    round(
                        probability_b,
                        4
                    ),

                "edge_a":
                    round(
                        edge_a,
                        4
                    ),

                "edge_b":
                    round(
                        edge_b,
                        4
                    ),

                "max_edge":
                    round(
                        max_edge,
                        4
                    ),

                "confidence":
                    confidence,

                "h2h":
                    match.get(
                        "h2h"
                    ),

                "strategy_v3":
                    prediction[
                        "strategy_v3"
                    ],

                "recommendation":
                    recommendation
            })

        except Exception as e:

            print(
                f"Skipped: {e}"
            )

    df = pd.DataFrame(
        rows
    )

    if len(df):

        df = df.sort_values(
            "max_edge",
            ascending=False
        )

    return df
