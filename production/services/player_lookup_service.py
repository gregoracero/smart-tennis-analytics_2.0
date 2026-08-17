
from pathlib import Path
import re

import pandas as pd

ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

LOOKUP_PATH = (
    ROOT
    / "data"
    / "parquet"
    / "player_lookup.parquet"
)

lookup = pd.read_parquet(
    LOOKUP_PATH
)


def _canonical_name(
    name
):

    if not name:
        return ""

    name = str(name).lower()

    name = (
        name
        .replace("-", " ")
        .replace(".", " ")
        .replace("'", " ")
    )

    name = re.sub(
        r"\s+",
        " ",
        name
    )

    return name.strip()


lookup[
    "canonical_name"
] = lookup[
    "player_name"
].apply(
    _canonical_name
)


def get_player_id(
    player_name
):

    canonical = _canonical_name(
        player_name
    )

    row = lookup[
        lookup[
            "canonical_name"
        ]
        ==
        canonical
    ]

    if row.empty:
        return None

    return row.iloc[0]["player_id"]


def get_player_name(
    player_id
):

    row = lookup[
        lookup["player_id"]
        ==
        player_id
    ]

    if row.empty:
        return None

    return row.iloc[0]["player_name"]


def get_all_players():

    return sorted(
        lookup[
            "player_name"
        ]
        .dropna()
        .unique()
        .tolist()
    )
