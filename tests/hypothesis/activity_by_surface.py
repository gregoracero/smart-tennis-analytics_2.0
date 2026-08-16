
import pandas as pd

elo = pd.read_parquet(
    "data/parquet/player_elo_history.parquet"
)

elo = elo[
    (elo["winner_surface_days_inactive"] >= 0)
    &
    (elo["loser_surface_days_inactive"] >= 0)
]

for surface, tmp in elo.groupby(
    "surface"
):

    different = tmp[
        tmp["winner_surface_days_inactive"]
        !=
        tmp["loser_surface_days_inactive"]
    ]

    rate = (
        different[
            "winner_surface_days_inactive"
        ]
        <
        different[
            "loser_surface_days_inactive"
        ]
    ).mean()

    print()
    print(surface)
    print(
        round(
            rate * 100,
            2
        ),
        "%"
    )
