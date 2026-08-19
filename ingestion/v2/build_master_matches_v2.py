from pathlib import Path
import hashlib
import pandas as pd

INPUT_TD = (
    'data/parquet/v2/'
    'tennis_data_co_uk_v2.parquet'
)

INPUT_TML = (
    'data/parquet/v2/'
    'tml_v2.parquet'
)

INPUT_PLAYERS = (
    'data/parquet/v2/'
    'player_mapping_v2.parquet'
)

INPUT_TOURNAMENTS = (
    'data/parquet/v2/'
    'tournament_mapping_v2.parquet'
)

INPUT_AUDIT = (
    'data/parquet/v2/'
    'match_link_audit_v2.parquet'
)

OUTPUT = (
    'data/parquet/v2/'
    'master_matches_v2.parquet'
)

VALID_STATUSES = {
    'MATCHED',
    'VALID_MULTIPLE_MATCH'
}

print('Loading datasets...')

td = pd.read_parquet(INPUT_TD)
tml = pd.read_parquet(INPUT_TML)

players = pd.read_parquet(INPUT_PLAYERS)
tournaments = pd.read_parquet(INPUT_TOURNAMENTS)

audit = pd.read_parquet(INPUT_AUDIT)

# ==========================================================
# LOOKUPS
# ==========================================================

td_player_map = dict(
    zip(
        players['tennis_data_name'],
        players['player_key']
    )
)

tml_player_map = dict(
    zip(
        players['tml_name'],
        players['player_key']
    )
)

tournament_lookup = (
    tournaments[
        [
            'tennis_data_tournament',
            'tournament_id',
            'tournament_key'
        ]
    ]
    .drop_duplicates()
)

# ==========================================================
# PREPARE KEYS
# ==========================================================

td['winner_player_key'] = (
    td['Winner']
    .map(td_player_map)
)

td['loser_player_key'] = (
    td['Loser']
    .map(td_player_map)
)

tml['winner_player_key'] = (
    tml['winner_name']
    .map(tml_player_map)
)

tml['loser_player_key'] = (
    tml['loser_name']
    .map(tml_player_map)
)

# ==========================================================
# TD INDEX
# ==========================================================

print('Building TD index...')

td_index = {}

for _, row in td.iterrows():

    key = (
        row['season'],
        str(row['Surface']),
        row['Tournament'],
        row['Winner'],
        row['Loser']
    )

    td_index[key] = row

# ==========================================================
# TML INDEX
# ==========================================================

print('Building TML index...')

tml_index = {}

for _, row in tml.iterrows():

    key = (
        row['season'],
        str(row['surface']),
        row['tourney_name'],
        row['winner_player_key'],
        row['loser_player_key']
    )

    tml_index.setdefault(
        key,
        []
    ).append(row)

# ==========================================================
# MASTER BUILD
# ==========================================================

print('Building master matches...')

rows = []

valid_audit = audit[
    audit['status'].isin(
        VALID_STATUSES
    )
]

for idx, (_, a) in enumerate(
    valid_audit.iterrows(),
    start=1
):

    if idx % 1000 == 0:

        print(
            f'Processed {idx:,}'
        )

    td_key = (
        a['year'],
        str(a['surface']),
        a['td_tournament'],
        a['winner_td'],
        a['loser_td']
    )

    td_match = td_index.get(td_key)

    if td_match is None:
        continue

    tournament_row = (
        tournament_lookup[
            tournament_lookup[
                'tennis_data_tournament'
            ]
            ==
            a['td_tournament']
        ]
    )

    if len(tournament_row) == 0:
        continue

    tournament_row = (
        tournament_row
        .iloc[0]
    )

    tml_key = (
        a['year'],
        str(a['surface']),
        a['tml_tournament'],
        a['winner_player_key'],
        a['loser_player_key']
    )

    tml_candidates = (
    tml_index.get(
        tml_key,
        []
    )
)

    if len(tml_candidates) == 0:
        continue

    tml_match = None

    for candidate in tml_candidates:

        if (
            candidate["round"] == a["tml_round"]
            and
            candidate["score"] == a["tml_score"]
        ):

            tml_match = candidate
            break

    if tml_match is None:
        continue

    master_match_id = hashlib.sha1(
        (
            f"{a['year']}|"
            f"{tournament_row['tournament_key']}|"
            f"{a['winner_player_key']}|"
            f"{a['loser_player_key']}|"
            f"{a['tml_round']}|"
            f"{a['tml_score']}"
        ).encode()
    ).hexdigest()

    record = {

        'master_match_id':
            master_match_id,

        'match_link_status':
            a['status'],

        'candidate_matches':
            a['candidate_matches'],

        'tournament_id':
            tournament_row[
                'tournament_id'
            ],

        'tournament_key':
            tournament_row[
                'tournament_key'
            ],

        'winner_player_key':
            a['winner_player_key'],

        'loser_player_key':
            a['loser_player_key']

    }

    # --------------------------------------
    # TD COLUMNS
    # --------------------------------------

    for col, value in td_match.items():

        record[
            f'td_{col}'
        ] = value

    # --------------------------------------
    # TML COLUMNS
    # --------------------------------------

    for col, value in tml_match.items():

        record[
            f'tml_{col}'
        ] = value

    rows.append(record)

master = pd.DataFrame(rows)

master = pd.DataFrame(rows)

print()
print(
    "DUPLICATE IDS BEFORE DEDUP:",
    master["master_match_id"]
    .duplicated()
    .sum()
)

master = master.drop_duplicates(
    subset=["master_match_id"]
)

print(
    "ROWS AFTER DEDUP:",
    len(master)
)

# ==========================================================
# SAVE
# ==========================================================

Path(
    'data/parquet/v2'
).mkdir(
    parents=True,
    exist_ok=True
)

master.to_parquet(
    OUTPUT,
    index=False
)

print()
print('=' * 80)
print('MASTER MATCHES V2')
print('=' * 80)

print()
print('ROWS:', len(master))

print()
print('COLS:', len(master.columns))

print()
print('SAVED:', OUTPUT)
