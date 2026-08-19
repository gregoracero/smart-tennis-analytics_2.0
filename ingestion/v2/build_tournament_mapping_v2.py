import pandas as pd

INPUT_TD = (
    "data/parquet/v2/"
    "tennis_data_co_uk_v2.parquet"
)

INPUT_TML = (
    "data/parquet/v2/"
    "tml_v2.parquet"
)

INPUT_PLAYERS = (
    "data/parquet/v2/"
    "player_mapping_v2.parquet"
)

OUTPUT = (
    "data/parquet/v2/"
    "tournament_mapping_v2.parquet"
)

MIN_OVERLAP = 10

# ==========================================================
# MANUAL TOURNAMENT ALIASES
# ==========================================================

MANUAL_TOURNAMENT_ALIASES = {

    "Grand Prix Hassan II": [
        "Casablanca",
        "Marrakech"
    ],

    "Rio Open": [
        "Rio De Janeiro",
        "Rio de Janeiro"
    ],

    "Masters Cup": [
        "Masters Cup",
        "Tour Finals",
        "ATP Finals",
        "ATP Tour Finals"
    ],

    "Shanghai Masters": [
        "Shanghai Masters",
        "Shanghai"
    ],

    "Brasil Open": [
        "Costa do Sauipe",
        "Sao Paulo",
        "Salvador"
    ],

    "Movistar Open": [
        "Vina del Mar",
        "Santiago"
    ],
    "Astana Open": [
        "Astana",
        "Nur-Sultan"
    ],
    "Monte Carlo Masters": [
        "Monte Carlo Masters",
        "Monte Carlo"
    ],
    "Dutch Open": [
        "Amersfoort",
        "Amsterdam"
    ],
    "European Open": [
        "Antwerp",
        "Brussels"
    ],
    "Portugal Open": [
        "Estoril",
        "Oeiras"
    ],
    "New York Open": [
        "New York",
        "New York Open"
    ],
    "Channel Open": [
        "Las Vegas",
        "Scottsdale"
    ],
    "Abierto Mexicano Mifel": [
        "Los Cabos",
        "Cabo San Lucas"
    ],
    "Serbia Open": [
        "Belgrade",
        "Serbia"
    ],
    "BMW Open": ["Munich"],
    "BNP Paribas": ["Paris Masters"],
    "Generali Open": ["Kitzbuhel"],
    "Hall of Fame Championships": ["Newport"],
}

print("Loading datasets...")

td = pd.read_parquet(INPUT_TD)

tml = pd.read_parquet(INPUT_TML)

players = pd.read_parquet(
    INPUT_PLAYERS
)



# ==========================================================
# EXCLUDE DAVIS CUP
# ==========================================================

td = td[
    ~td["Tournament"]
    .astype(str)
    .str.contains(
        "Davis",
        case=False,
        na=False
    )
]

tml = tml[
    ~tml["tourney_name"]
    .astype(str)
    .str.contains(
        "Davis",
        case=False,
        na=False
    )
]

# ==========================================================
# PLAYER LOOKUPS
# ==========================================================

td_player_map = dict(
    zip(
        players["tennis_data_name"],
        players["player_key"]
    )
)

tml_player_map = dict(
    zip(
        players["tml_name"],
        players["player_key"]
    )
)

# ==========================================================
# APPLY PLAYER KEYS
# ==========================================================

td["winner_key"] = (
    td["Winner"]
    .map(td_player_map)
)

td["loser_key"] = (
    td["Loser"]
    .map(td_player_map)
)

tml["winner_key"] = (
    tml["winner_name"]
    .map(tml_player_map)
)

tml["loser_key"] = (
    tml["loser_name"]
    .map(tml_player_map)
)

# ==========================================================
# REMOVE UNMAPPED PLAYERS
# ==========================================================

td = td.dropna(
    subset=[
        "winner_key",
        "loser_key"
    ]
)

tml = tml.dropna(
    subset=[
        "winner_key",
        "loser_key"
    ]
)


# ==========================================================
# PRECOMPUTE TML PAIRS
# ==========================================================

print("Building TML pair cache...")

tml_pairs_by_tournament = {}

for tournament in sorted(
    tml["tourney_name"]
    .dropna()
    .unique()
):

    subset = tml[
        tml["tourney_name"]
        == tournament
    ]

    tml_pairs_by_tournament[
        tournament
    ] = set(
        zip(
            subset["winner_key"],
            subset["loser_key"]
        )
    )

# ==========================================================
# BUILD MAPPING
# ==========================================================

rows = []

td_tournaments = sorted(
    td["Tournament"]
    .dropna()
    .unique()
)

for idx, td_tournament in enumerate(
    td_tournaments,
    start=1
):

    if idx % 25 == 0:

        print(
            f"Processed {idx}/{len(td_tournaments)}"
        )

    td_subset = td[
        td["Tournament"]
        == td_tournament
    ]

    td_pairs = set(
        zip(
            td_subset["winner_key"],
            td_subset["loser_key"]
        )
    )

    td_matches = len(
        td_subset
    )

    # ======================================================
    # MANUAL ALIAS
    # ======================================================

    if td_tournament in MANUAL_TOURNAMENT_ALIASES:

        for tml_tournament in sorted(
            MANUAL_TOURNAMENT_ALIASES[
                td_tournament
            ]
        ):

            overlap = 0

            if (
                tml_tournament
                in tml_pairs_by_tournament
            ):

                overlap = len(

                    td_pairs
                    &
                    tml_pairs_by_tournament[
                        tml_tournament
                    ]

                )

            rows.append({

                "tennis_data_tournament":
                    td_tournament,

                "tml_tournament":
                    tml_tournament,

                "overlap":
                    overlap,

                "td_matches":
                    td_matches,

                "mapping_method":
                    "MANUAL_ALIAS",

                "confidence":
                    round(
                        overlap
                        /
                        max(
                            td_matches,
                            1
                        ),
                        4
                    )

            })

        continue

    # ======================================================
    # AUTOMATIC OVERLAP MATCHING
    # ======================================================

    best_match = None

    best_overlap = 0

    for (
        tml_tournament,
        tml_pairs
    ) in tml_pairs_by_tournament.items():

        overlap = len(
            td_pairs & tml_pairs
        )

        if overlap > best_overlap:

            best_overlap = overlap

            best_match = (
                tml_tournament
            )

    rows.append({

        "tennis_data_tournament":
            td_tournament,

        "tml_tournament":
            best_match,

        "overlap":
            best_overlap,

        "td_matches":
            td_matches,

        "mapping_method":
            "PLAYER_MAPPING_OVERLAP",

        "confidence":
            round(
                best_overlap
                /
                max(
                    td_matches,
                    1
                ),
                4
            )

    })

# ==========================================================
# DATAFRAME
# ==========================================================

result = pd.DataFrame(rows)

# ==========================================================
# TOURNAMENT IDs
# ==========================================================

unique_tournaments = sorted(

    result[
        "tennis_data_tournament"
    ]
    .dropna()
    .unique()

)

tournament_ids = {

    tournament: idx

    for idx, tournament in enumerate(
        unique_tournaments,
        start=1
    )

}

result["tournament_id"] = (
    result[
        "tennis_data_tournament"
    ]
    .map(
        tournament_ids
    )
)

result["tournament_key"] = (
    result[
        "tournament_id"
    ]
    .apply(
        lambda x:
        f"T{x:06d}"
    )
)

# ==========================================================
# COLUMN ORDER
# ==========================================================

result = result[

    [

        "tournament_id",

        "tournament_key",

        "tennis_data_tournament",

        "tml_tournament",

        "overlap",

        "td_matches",

        "mapping_method",

        "confidence"

    ]

]

# ==========================================================
# SORT
# ==========================================================

result = result.sort_values(

    [

        "tournament_id",

        "confidence",

        "overlap"

    ],

    ascending=[

        True,

        False,

        False

    ]

)

# ==========================================================
# SAVE
# ==========================================================

result.to_parquet(
    OUTPUT,
    index=False
)

print()
print("=" * 80)
print("TOURNAMENT MAPPING V2")
print("=" * 80)

print()

print(
    "TOURNAMENTS:",
    result[
        "tournament_key"
    ].nunique()
)

print()

print(
    "ROWS:",
    len(result)
)

print()

print(
    result.head(100)
)

print()

print(
    "Saved:",
    OUTPUT
)
