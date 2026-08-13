import pandas as pd

METRICS = "data/parquet/player_metrics.parquet"

df = pd.read_parquet(METRICS)

METRIC_COLUMNS = [

    "surface_recent_win_pct",

    "first_serve_in_pct",
    "first_serve_won_pct",
    "second_serve_won_pct",

    "bp_saved_pct",
    "break_conversion_pct",

    "tiebreak_win_pct"
]


def compare(player_a, player_b, surface, window):

    a = df[
        (df["player"] == player_a)
        &
        (df["surface"] == surface)
        &
        (df["window"] == window)
    ]

    b = df[
        (df["player"] == player_b)
        &
        (df["surface"] == surface)
        &
        (df["window"] == window)
    ]

    if a.empty:
        print(f"No metrics for {player_a}")
        return

    if b.empty:
        print(f"No metrics for {player_b}")
        return

    a = a.iloc[0]
    b = b.iloc[0]

    rows = []

    for metric in METRIC_COLUMNS:

        if pd.isna(a[metric]) or pd.isna(b[metric]):
            continue

        diff = round(
            a[metric] - b[metric],
            2
        )

        if diff > 0:
            leader = player_a
        elif diff < 0:
            leader = player_b
        else:
            leader = "TIE"

        rows.append({
            "metric": metric,
            "player_a": round(a[metric], 2),
            "player_b": round(b[metric], 2),
            "edge": abs(diff),
            "leader": leader
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "edge",
        ascending=False
    )

    print()
    print("=" * 90)
    print("METRIC EDGES")
    print("=" * 90)

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    compare(
        "Carlos Alcaraz",
        "Jannik Sinner",
        "Clay",
        "LAST_10_MATCHES"
    )
