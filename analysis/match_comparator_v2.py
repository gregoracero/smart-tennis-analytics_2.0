import pandas as pd

METRICS = "data/parquet/player_metrics.parquet"
H2H_OVERALL = "data/parquet/h2h_overall.parquet"
H2H_SURFACE = "data/parquet/h2h_surface.parquet"

metrics = pd.read_parquet(METRICS)
h2h_overall = pd.read_parquet(H2H_OVERALL)
h2h_surface = pd.read_parquet(H2H_SURFACE)


def get_player_metrics(player, surface, window):

    result = metrics[
        (metrics["player"] == player)
        &
        (metrics["surface"] == surface)
        &
        (metrics["window"] == window)
    ]

    if result.empty:
        return None

    return result.iloc[0]


def get_h2h(player_a, player_b, surface):

    overall = h2h_overall[
        (
            (
                (h2h_overall["player_a"] == player_a)
                &
                (h2h_overall["player_b"] == player_b)
            )
            |
            (
                (h2h_overall["player_a"] == player_b)
                &
                (h2h_overall["player_b"] == player_a)
            )
        )
    ]

    surface_h2h = h2h_surface[
        (
            (
                (h2h_surface["player_a"] == player_a)
                &
                (h2h_surface["player_b"] == player_b)
            )
            |
            (
                (h2h_surface["player_a"] == player_b)
                &
                (h2h_surface["player_b"] == player_a)
            )
        )
        &
        (h2h_surface["surface"] == surface)
    ]

    return overall, surface_h2h


def print_player(name, data):

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    fields = [
        "matches",
        "wins",
        "losses",
        "surface_recent_win_pct",
        "first_serve_in_pct",
        "first_serve_won_pct",
        "second_serve_won_pct",
        "bp_saved_pct",
        "break_conversion_pct",
        "tiebreaks_played",
        "tiebreaks_won",
        "tiebreak_win_pct",
        "tiebreaks_per_match",
        "avg_minutes"
    ]

    for field in fields:
        print(f"{field:<30} {data[field]}")


def compare(player_a, player_b, surface, window):

    a = get_player_metrics(
        player_a,
        surface,
        window
    )

    b = get_player_metrics(
        player_b,
        surface,
        window
    )

    if a is None:
        print(f"No metrics found for {player_a}")
        return

    if b is None:
        print(f"No metrics found for {player_b}")
        return

    overall, surface_h2h = get_h2h(
        player_a,
        player_b,
        surface
    )

    print()
    print("=" * 70)
    print("MATCH COMPARATOR V2")
    print("=" * 70)

    print()
    print(f"Surface: {surface}")
    print(f"Window : {window}")

    print_player(player_a, a)
    print_player(player_b, b)

    print()
    print("=" * 70)
    print("H2H OVERALL")
    print("=" * 70)

    if len(overall):
        print(overall.to_string(index=False))
    else:
        print("No data")

    print()
    print("=" * 70)
    print("H2H SURFACE")
    print("=" * 70)

    if len(surface_h2h):
        print(surface_h2h.to_string(index=False))
    else:
        print("No data")


if __name__ == "__main__":

    compare(
        "Carlos Alcaraz",
        "Jannik Sinner",
        "Clay",
        "LAST_10_MATCHES"
    )
