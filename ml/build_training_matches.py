import pandas as pd

MASTER_MATCHES = "data/parquet/master_matches.parquet"
PLAYER_MATCHES = "data/parquet/analytics_player_matches.parquet"

START_YEAR = 2010

VALID_LEVELS = [
    "G",
    "M",
    "1000",
    "500",
    "250",
    "C"
]

VALID_SURFACES = [
    "Hard",
    "Clay",
    "Grass"
]

WINDOWS = [5, 10, 20, 50]

print("Loading datasets...")

master = pd.read_parquet(
    MASTER_MATCHES
)

player_matches = pd.read_parquet(
    PLAYER_MATCHES
)

master["match_date"] = pd.to_datetime(
    master["tourney_date"]
        .astype(str)
        .str.extract(r"(\d{8})")[0],
    format="%Y%m%d",
    errors="coerce"
)

master = master[
    master["match_date"].dt.year >= START_YEAR
]

master = master[
    master["surface"].isin(
        VALID_SURFACES
    )
]

master = master[
    master["tourney_level"].isin(
        VALID_LEVELS
    )
]

player_matches["tourney_date"] = (
    player_matches["tourney_date"]
    .astype(str)
    .str.extract(r"(\d{8})")[0]
)

player_matches["match_date"] = pd.to_datetime(
    player_matches["tourney_date"]
        .astype(str)
        .str.extract(r"(\d{8})")[0],
    format="%Y%m%d",
    errors="coerce"
)

master = master[
    master["match_date"].notna()
]

player_matches = player_matches[
    player_matches["match_date"].notna()
]

def get_history(
    player,
    surface,
    match_date
):

    history = player_matches[
        (player_matches["player"] == player)
        &
        (player_matches["surface"] == surface)
        &
        (player_matches["match_date"] < match_date)
    ]

    history = history.sort_values(
        "match_date",
        ascending=False
    )

    return history


def get_win_pct(
    history,
    matches
):

    subset = history.head(matches)

    if len(subset) == 0:
        return None

    return round(
        subset["won_match"].mean() * 100,
        2
    )


print()
print("MASTER")
print(master.shape)

print()
print("PLAYER_MATCHES")
print(player_matches.shape)


def get_player_features(
    player,
    surface,
    match_date
):

    history = get_history(
        player,
        surface,
        match_date
    )

    features = {}

    features["history_size"] = len(
        history
    )

    features["matches_available"] = len(
        history
    )

    for window in WINDOWS:

        subset = history.head(window)

        features[
            f"matches_used_{window}"
        ] = len(subset)

        if len(subset) == 0:
            continue

        svpt = subset["svpt"].sum()

        first_in = subset["first_in"].sum()

        first_won = subset["first_won"].sum()

        second_won = subset["second_won"].sum()

        second_attempts = (
            svpt - first_in
        )

        service_points_won = (
            first_won + second_won
        )

        opp_svpt = subset["opp_svpt"].sum()

        opp_first_won = subset[
            "opp_first_won"
        ].sum()

        opp_second_won = subset[
            "opp_second_won"
        ].sum()

        return_points_won = (
            opp_svpt
            - (
                opp_first_won
                + opp_second_won
            )
        )

        break_points_generated = subset[
            "break_points_generated"
        ].sum()

        breaks_converted = subset[
            "breaks_converted"
        ].sum()

        bp_saved = subset[
            "bp_saved"
        ].sum()

        bp_faced = subset[
            "bp_faced"
        ].sum()

        tb_played = subset[
            "tb_played"
        ].sum()

        tb_won = subset[
            "tb_won"
        ].sum()

        features[
            f"win_pct_{window}"
        ] = round(
            subset["won_match"].mean() * 100,
            2
        )

        features[
            f"first_serve_in_pct_{window}"
        ] = (
            round(first_in / svpt * 100, 2)
            if svpt > 0
            else None
        )

        features[
            f"first_serve_won_pct_{window}"
        ] = (
            round(first_won / first_in * 100, 2)
            if first_in > 0
            else None
        )

        features[
            f"second_serve_won_pct_{window}"
        ] = (
            round(
                second_won
                / second_attempts
                * 100,
                2
            )
            if second_attempts > 0
            else None
        )

        features[
            f"service_points_won_pct_{window}"
        ] = (
            round(
                service_points_won
                / svpt
                * 100,
                2
            )
            if svpt > 0
            else None
        )

        features[
            f"return_points_won_pct_{window}"
        ] = (
            round(
                return_points_won
                / opp_svpt
                * 100,
                2
            )
            if opp_svpt > 0
            else None
        )

        features[
            f"break_conversion_pct_{window}"
        ] = (
            round(
                breaks_converted
                / break_points_generated
                * 100,
                2
            )
            if break_points_generated > 0
            else None
        )

        features[
            f"bp_saved_pct_{window}"
        ] = (
            round(
                bp_saved
                / bp_faced
                * 100,
                2
            )
            if bp_faced > 0
            else None
        )

        features[
            f"tiebreak_win_pct_{window}"
        ] = (
            round(
                tb_won
                / tb_played
                * 100,
                2
            )
            if tb_played > 0
            else None
        )

    return features


def delta(
    a,
    b
):

    if a is None or b is None:
        return None

    return round(
        a - b,
        2
    )


def build_match_row(
    player_a,
    player_b,

    rank_a,
    rank_b,

    rank_points_a,
    rank_points_b,

    age_a,
    age_b,

    height_a,
    height_b,

    hand_a,
    hand_b,

    tourney_level,
    surface,
    match_date,
    target
):

    features_a = get_player_features(
        player_a,
        surface,
        match_date
    )

    features_b = get_player_features(
        player_b,
        surface,
        match_date
    )
    if (
        features_a["history_size"] < 5
        or
        features_b["history_size"] < 5
    ):
        return None

    row = {

        "player_a": player_a,

        "player_b": player_b,

        "tourney_level": tourney_level,

        "surface": surface,

        "match_date": match_date,

        "match_year": match_date.year,

        "target": target
    }

    row["history_size_a"] = (
        features_a.get(
            "history_size"
        )
    )

    row["history_size_b"] = (
        features_b.get(
            "history_size"
        )
    )

    row["matches_available_a"] = (
        features_a.get(
            "matches_available"
        )
    )

    row["matches_available_b"] = (
        features_b.get(
            "matches_available"
        )
    )

    row["delta_history_size"] = delta(
        features_a.get(
            "history_size"
        ),
        features_b.get(
            "history_size"
        )
    )
    row["rank_a"] = rank_a

    row["rank_b"] = rank_b

    row["delta_rank"] = (
        rank_b - rank_a
    )
    row["rank_points_a"] = (
        rank_points_a
    )

    row["rank_points_b"] = (
        rank_points_b
    )

    row["delta_rank_points"] = (
        rank_points_a
        - rank_points_b
    )
    
    row["age_a"] = age_a

    row["age_b"] = age_b

    row["delta_age"] = (
        age_a - age_b
        if pd.notna(age_a)
        and pd.notna(age_b)
        else None
    )

    row["height_a"] = height_a

    row["height_b"] = height_b

    row["delta_height"] = (
        height_a - height_b
        if pd.notna(height_a)
        and pd.notna(height_b)
        else None
    )

    row["is_left_handed_a"] = (
        1 if hand_a == "L"
        else 0
    )

    row["is_left_handed_b"] = (
        1 if hand_b == "L"
        else 0
    )

    row["tourney_level"] = tourney_level

    common_features = [

        "win_pct",

        "first_serve_in_pct",

        "first_serve_won_pct",

        "second_serve_won_pct",

        "service_points_won_pct",

        "return_points_won_pct",

        "break_conversion_pct",

        "bp_saved_pct",

        "tiebreak_win_pct"
    ]

    for window in WINDOWS:

        row[
            f"matches_used_{window}_a"
        ] = features_a.get(
            f"matches_used_{window}"
        )

        row[
            f"matches_used_{window}_b"
        ] = features_b.get(
            f"matches_used_{window}"
        )

        for feature in common_features:

            key = (
                f"{feature}_{window}"
            )

            row[
                f"delta_{key}"
            ] = delta(
                features_a.get(key),
                features_b.get(key)
            )

    return row


def build_training_dataset(
    matches
):

    rows = []
    skipped_matches = 0

    for idx, (_, match) in enumerate(
        matches.iterrows(),
        start=1
    ):
        if idx % 250 == 0:

            print(
                f"Processed {idx:,} matches..."
            )

        row_win = build_match_row(
            match["winner_name"],
            match["loser_name"],

            match["winner_rank"],
            match["loser_rank"],

            match["winner_rank_points"],
            match["loser_rank_points"],

            match["winner_age"],
            match["loser_age"],

            match["winner_ht"],
            match["loser_ht"],

            match["winner_hand"],
            match["loser_hand"],

            match["tourney_level"],
            match["surface"],
            match["match_date"],
            1
        )

        if row_win is not None:
            rows.append(row_win)
        else:
            skipped_matches += 1

        row_loss = build_match_row(
            match["loser_name"],
            match["winner_name"],

            match["loser_rank"],
            match["winner_rank"],

            match["loser_rank_points"],
            match["winner_rank_points"],

            match["loser_age"],
            match["winner_age"],

            match["loser_ht"],
            match["winner_ht"],

            match["loser_hand"],
            match["winner_hand"],

            match["tourney_level"],
            match["surface"],
            match["match_date"],
            0
        )

        if row_loss is not None:
            rows.append(row_loss)
        else:
            skipped_matches += 1
            

    print()
    print(
        f"Skipped rows: {skipped_matches:,}"
    )
    #borrar inicio
    print()
    print(f"ROWS GENERATED: {len(rows):,}")
    #borrar fin

    return pd.DataFrame(rows)


training_df = build_training_dataset(
   master
   # master.head(100)
    #master.sample(
        #n=5000,
        #random_state=42
    #)
)
#borrar
print()
print("TRAINING DF SHAPE")
print(training_df.shape)
#borrar fin

training_df.to_parquet(
    "data/parquet/training_matches.parquet",
    index=False
)

print()
print(
    "Saved: data/parquet/training_matches.parquet"
)

print()

print("FINAL DATASET")

print(
    training_df.shape
)

print()

print(
    training_df["target"]
    .value_counts()
)

print()

print(
    training_df[
        [
            "history_size_a",
            "history_size_b"
        ]
    ]
    .describe()
)
