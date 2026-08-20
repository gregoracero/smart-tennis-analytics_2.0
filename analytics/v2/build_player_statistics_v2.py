from pathlib import Path

import numpy as np
import pandas as pd

INPUT = (
    "data/parquet/v2/"
    "player_per_match_statistics_v2.parquet"
)

OUTPUT = (
    "data/parquet/v2/"
    "player_statistics_v2.parquet"
)

def print_progress(
    current,
    total,
    label="Progress"
):

    pct = (
        current
        /
        total
        *
        100
    )

    print(
        f"\r{label}: "
        f"{current:,}/{total:,} "
        f"({pct:5.1f}%)",
        end=""
    )



print("Loading player per match statistics...")

df = pd.read_parquet(INPUT)

# ==========================================================
# NOS ASEGURAMOS QUE TODAS LAS CUOTAS SON NUMERICAS
# ==========================================================

df["player_b365"] = pd.to_numeric(
    df["player_b365"],
    errors="coerce"
)


# ==========================================================
# HELPERS
# ==========================================================

def safe_ratio(numerator, denominator):

    denominator = denominator.sum()

    if denominator == 0:
        return np.nan

    return numerator.sum() / denominator


def per_match(series):

    valid_matches = series.count()

    if valid_matches == 0:
        return np.nan

    return series.sum() / valid_matches


def aggregate_group(g):

    matches = len(g)

    wins = g["win"].sum()
    losses = g["loss"].sum()

    sets_won = g["sets_won"].sum()
    sets_lost = g["sets_lost"].sum()

    first_serve_in = g["first_serve_in"].sum()

    first_serve_points_won = (
        g["first_serve_points_won"].sum()
    )

    second_serve_points_won = (
        g["second_serve_points_won"].sum()
    )

    service_points_played = (
        g["service_points_played"].sum()
    )

    service_games_played = (
        g["service_games_played"].sum()
    )

    service_games_won = (
        g["service_games_won"].sum()
    )

    break_points_faced = (
        g["break_points_faced"].sum()
    )

    break_points_saved = (
        g["break_points_saved"].sum()
    )

    break_points_chances = (
        g["break_points_chances"].sum()
    )

    break_points_converted = (
        g["break_points_converted"].sum()
    )

    minutes_played = (
        g["minutes"].sum()
    )

    avg_opponent_rank = (
        g["opponent_rank"]
        .mean()
    )

    median_opponent_rank = (
        g["opponent_rank"]
        .median()
    )

    top10_matches = (
        g["opponent_rank"]
        .le(10)
        .sum()
    )

    top10_wins = g.loc[
        (
            g["opponent_rank"] <= 10
        )
        &
        (
            g["win"] == 1
        )
    ].shape[0]

    top20_matches = (
        g["opponent_rank"]
        .le(20)
        .sum()
    )

    top20_wins = g.loc[
        (
            g["opponent_rank"] <= 20
        )
        &
        (
            g["win"] == 1
        )
    ].shape[0]

    top50_matches = (
        g["opponent_rank"]
        .le(50)
        .sum()
    )

    top50_wins = g.loc[
        (
            g["opponent_rank"] <= 50
        )
        &
        (
            g["win"] == 1
        )
    ].shape[0]

    top100_matches = (
        g["opponent_rank"]
        .le(100)
        .sum()
    )

    top100_wins = g.loc[
        (
            g["opponent_rank"] <= 100
        )
        &
        (
            g["win"] == 1
        )
    ].shape[0]

    betting_matches = (
        g["player_b365"]
        .notna()
        .sum()
    )

    yield_total = np.nan

    if betting_matches > 0:

        valid_bets = g[
            g["player_b365"].notna()
        ]

        profit = np.where(

            valid_bets["win"] == 1,

            valid_bets["player_b365"] - 1,

            -1

        ).sum()

        yield_total = (

            profit

            /

            betting_matches

        )

    pressure_index = np.nan

    bp_saved_pct = safe_ratio(

        g["break_points_saved"],

        g["break_points_faced"]

    )

    bp_conv_pct = safe_ratio(

        g["break_points_converted"],

        g["break_points_chances"]

    )

    if (
        pd.notna(bp_saved_pct)
        and
        pd.notna(bp_conv_pct)
    ):

        pressure_index = (

            bp_saved_pct

            +

            bp_conv_pct

        ) / 2

    return pd.Series({

        "matches":
            matches,

        "wins":
            wins,

        "losses":
            losses,

        "win_pct":
            wins / matches,

        "loss_pct":
            losses / matches,

        "sets_won":
            sets_won,

        "sets_lost":
            sets_lost,

        "set_win_pct":
            (
                sets_won
                /
                (sets_won + sets_lost)
            )
            if (
                sets_won + sets_lost
            ) > 0
            else np.nan,

        "aces":
            g["aces"].sum(),

        "aces_per_match":
            per_match(g["aces"]),

        "aces_sample_matches":
            g["aces"].count(),

        "double_faults":
            g["double_faults"].sum(),

        "double_faults_per_match":
            per_match(
                g["double_faults"]
            ),

        "double_faults_sample_matches":
            g["double_faults"].count(),

        "first_serve_in_pct":
            safe_ratio(
                g["first_serve_in"],
                g["service_points_played"]
            ),

        "first_serve_won_pct":
            safe_ratio(
                g["first_serve_points_won"],
                g["first_serve_in"]
            ),

        "second_serve_won_pct":
            (
                second_serve_points_won
                /
                (
                    service_points_played
                    -
                    first_serve_in
                )
            )
            if (
                service_points_played
                -
                first_serve_in
            ) > 0
            else np.nan,

        "service_points_won_pct":
            (
                first_serve_points_won
                +
                second_serve_points_won
            )
            /
            service_points_played
            if service_points_played > 0
            else np.nan,

        "service_games_won_pct":
            (
                service_games_won
                /
                service_games_played
            )
            if service_games_played > 0
            else np.nan,

        "break_points_faced":
            break_points_faced,

        "break_points_saved":
            break_points_saved,

        "break_points_saved_pct":
            bp_saved_pct,

        "break_points_chances":
            break_points_chances,

        "break_points_converted":
            break_points_converted,

        "break_points_converted_pct":
            bp_conv_pct,

        "minutes_played":
            minutes_played,

        "minutes_per_match":
            per_match(g["minutes"]),

        "minutes_sample_matches":
            g["minutes"].count(),

        "avg_opponent_rank":
            avg_opponent_rank,

        "median_opponent_rank":
            median_opponent_rank,

        "top10_matches":
            top10_matches,

        "top10_wins":
            top10_wins,

        "top20_matches":
            top20_matches,

        "top20_wins":
            top20_wins,

        "top50_matches":
            top50_matches,

        "top50_wins":
            top50_wins,

        "top100_matches":
            top100_matches,

        "top100_wins":
            top100_wins,

        "pressure_index":
            pressure_index,

        "yield_total":
            yield_total

    })


# ==========================================================
# BUILD CONTEXTS
# ==========================================================

print("Creating context combinations...")

rows = []

contexts = [

    ("CAREER", False),

    ("SEASON", True)

]

total_contexts = len(contexts)

for idx_context,(scope, by_season) in enumerate(contexts,start=1):
    
    print()
    print(
        f"[{idx_context}/{total_contexts}] "
        f"Building {scope}..."
    )

    tmp = df.copy()

    # ALL SURFACES

    all_surface = tmp.copy()

    all_surface["surface"] = "ALL"

    tmp = pd.concat(
        [tmp, all_surface],
        ignore_index=True
    )

    # ALL LEVELS

    all_level = tmp.copy()

    all_level[
        "tournament_level"
    ] = "ALL"

    tmp = pd.concat(
        [tmp, all_level],
        ignore_index=True
    )

    group_cols = [

        "player_key",

        "surface",

        "tournament_level"

    ]

    if by_season:

        group_cols.append(
            "season"
        )

    grouped = (
        tmp.groupby(
            group_cols
        )
    )

    total_groups = len(grouped)

    records = []

    for idx_group, (
        keys,
        group
    ) in enumerate(
        grouped,
        start=1
    ):

        if (
            idx_group % 100 == 0
            or
            idx_group == total_groups
        ):

            print_progress(
                idx_group,
                total_groups,
                scope
            )

        row = aggregate_group(
            group
        )

        if not isinstance(
            keys,
            tuple
        ):
            keys = (keys,)

        record = dict(
            zip(
                group_cols,
                keys
            )
        )

        record.update(
            row.to_dict()
        )

        records.append(
            record
        )

    print()

    result = pd.DataFrame(
        records
    )

    result["scope"] = scope

    if not by_season:

        result["season"] = "ALL"

    rows.append(result)

# ==========================================================
# OUTPUT
# ==========================================================

player_statistics = pd.concat(
    rows,
    ignore_index=True
)

Path(
    "data/parquet/v2"
).mkdir(
    parents=True,
    exist_ok=True
)

for col in [
    "surface",
    "tournament_level",
    "scope",
    "season"
]:

    player_statistics[col] = (
        player_statistics[col]
        .astype(str)
    )

player_statistics.to_parquet(
    OUTPUT,
    index=False
)

player_statistics.to_parquet(
    OUTPUT,
    index=False
)

print()
print("=" * 80)
print("PLAYER STATISTICS V2")
print("=" * 80)

print()
print("ROWS:", len(player_statistics))

print()
print("COLS:", len(player_statistics.columns))

print()
print(
    "PLAYERS:",
    player_statistics[
        "player_key"
    ].nunique()
)

print()
print("SAVED:", OUTPUT)