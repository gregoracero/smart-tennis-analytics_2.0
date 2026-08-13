import pandas as pd

ROLLING = "data/parquet/rolling_metrics.parquet"
SIGNALS = "data/parquet/signal_engine.parquet"
H2H = "data/parquet/h2h_surface.parquet"

rolling_df = pd.read_parquet(ROLLING)
signal_df = pd.read_parquet(SIGNALS)
h2h_df = pd.read_parquet(H2H)


def analyze(player_a, player_b, surface):

    a = signal_df[
        (signal_df["player"] == player_a)
        &
        (signal_df["surface"] == surface)
    ]

    b = signal_df[
        (signal_df["player"] == player_b)
        &
        (signal_df["surface"] == surface)
    ]

    h2h = h2h_df[
        (
            (
                (h2h_df["player_a"] == player_a)
                &
                (h2h_df["player_b"] == player_b)
            )
            |
            (
                (h2h_df["player_a"] == player_b)
                &
                (h2h_df["player_b"] == player_a)
            )
        )
        &
        (h2h_df["surface"] == surface)
    ]

    return {
        "player_a": a.to_dict("records"),
        "player_b": b.to_dict("records"),
        "h2h": h2h.to_dict("records")
    }


if __name__ == "__main__":

    result = analyze(
        "Carlos Alcaraz",
        "Jannik Sinner",
        "Clay"
    )

    print(result)
