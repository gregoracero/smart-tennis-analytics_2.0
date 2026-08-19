from pathlib import Path

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

INPUT_TOURNAMENTS = (
    "data/parquet/v2/"
    "tournament_mapping_v2.parquet"
)

OUTPUT = (
    "data/parquet/v2/"
    "match_link_audit_v2.parquet"
)

print("Loading datasets...")

td = pd.read_parquet(INPUT_TD)

tml = pd.read_parquet(INPUT_TML)

players = pd.read_parquet(
    INPUT_PLAYERS
)

tournaments = pd.read_parquet(
    INPUT_TOURNAMENTS
)
#Players que no están en TML por lo que no tenemos datos completos
NO_TML_PLAYERS = {
    "Bahrouzyan O.": "Dubai local wildcard",
    "Awadhy O.": "Dubai local wildcard",
    "Tyurnev E.": "",
    "Kutac R.":"",
    "Ruevski P.":"",
    "Chekov P.":"",
    "Haji A.":"",
    "Marin L.":""
}

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
# TOURNAMENT LOOKUPS (1 -> N)
# ==========================================================

td_to_tml_tournaments = {}

for _, row in tournaments.iterrows():

    td_name = row[
        "tennis_data_tournament"
    ]

    tml_name = row[
        "tml_tournament"
    ]

    if pd.isna(tml_name):
        continue

    td_to_tml_tournaments.setdefault(
        td_name,
        set()
    ).add(
        tml_name
    )

# ==========================================================
# DATES
# ==========================================================

td["match_year"] = td["season"]

tml["match_year"] = tml["season"]

# ==========================================================
# BUILD TML INDEX
# ==========================================================

print("Building TML index...")

tml_index = {}

for _, row in tml.iterrows():

    winner_key = tml_player_map.get(
        row["winner_name"]
    )

    loser_key = tml_player_map.get(
        row["loser_name"]
    )

    key = (

        row["match_year"],

        str(row["surface"]),

        row["tourney_name"],

        winner_key,

        loser_key

    )

    tml_index.setdefault(
        key,
        []
    ).append(row)

print(
    f"TML INDEX KEYS: {len(tml_index):,}"
)

# ==========================================================
# AUDIT
# ==========================================================

print("Building audit...")

rows = []

for idx, (_, row) in enumerate(
    td.iterrows(),
    start=1
):

    if idx % 500 == 0:

        print(
            f"Processed {idx:,}"
        )

    winner_key = td_player_map.get(
        row["Winner"]
    )

    loser_key = td_player_map.get(
        row["Loser"]
    )

    mapped_tournaments = list(

        td_to_tml_tournaments.get(
            row["Tournament"],
            []
        )

    )

    status = "MATCHED"

    if len(mapped_tournaments) == 0:

        status = "NO_TOURNAMENT_MATCH"

    elif winner_key is None:

        if row["Winner"] in NO_TML_PLAYERS:
            status = "NO_TML_COVERAGE"
        else:
            status = "WINNER_UNMATCHED"

    elif loser_key is None:

        if row["Loser"] in NO_TML_PLAYERS:
            status = "NO_TML_COVERAGE"
        else:
            status = "LOSER_UNMATCHED"

    candidates = []

    matched_tournament = None

    if status == "MATCHED":

        for tournament_name in mapped_tournaments:

            lookup_key = (

                row["match_year"],

                str(row["Surface"]),

                tournament_name,

                winner_key,

                loser_key

            )

            found = tml_index.get(
                lookup_key,
                []
            )

            if found:

                candidates.extend(
                    found
                )

                matched_tournament = (
                    tournament_name
                )

        if len(candidates) == 0:

            status = "NO_MATCH"

        elif len(candidates) == 1:

            status = "MATCHED"

        else:

            candidate_df = pd.DataFrame(candidates)

            # ¿son realmente duplicados?
            duplicate_cols = [
                "tourney_id",
                "winner_name",
                "loser_name",
                "score",
                "round",
                "tourney_date"
            ]

            if (
                candidate_df[duplicate_cols]
                .drop_duplicates()
                .shape[0]
                == 1
            ):
                status = "DUPLICATE_CANDIDATES"

            else:
                status = "VALID_MULTIPLE_MATCH"

    rows.append({

        "year":
            row["match_year"],

        "surface":
            row["Surface"],

        "td_tournament":
            row["Tournament"],

        "tml_tournament":
            matched_tournament,

        "winner_td":
            row["Winner"],

        "loser_td":
            row["Loser"],

        "winner_player_key":
            winner_key,

        "loser_player_key":
            loser_key,

        "candidate_matches":
            len(candidates),

        "status":
            status

    })

audit = pd.DataFrame(rows)

Path(
    "data/parquet/v2"
).mkdir(
    parents=True,
    exist_ok=True
)

audit.to_parquet(
    OUTPUT,
    index=False
)

print()
print("=" * 80)
print("MATCH LINK AUDIT")
print("=" * 80)

print()
print(
    audit["status"]
    .value_counts()
)

print()

print(
    (
        audit["status"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)

print()

print(
    "SAVED:",
    OUTPUT
)
