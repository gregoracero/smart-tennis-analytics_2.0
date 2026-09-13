#!/usr/bin/env python3
"""Build a professional Sackmann-compatible Parquet with odds and TML updates.

Sources
-------
1. Jeff Sackmann ATP CSV files, including main tour, Futures and qual/chall.
2. TennisMyLife 50-column CSV files for main tour, ATP qualifying and Challenger.
3. tennis-data.co.uk XLS/XLSX files for historical odds.
4. ATP_Database.csv and atp_players.csv for player identity reconciliation.

The final Parquet keeps the 49-column Sackmann schema, traceability columns,
TML's indoor flag and tennis-data columns prefixed with ``td_``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import unicodedata
import warnings
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Unknown extension is not supported and will be removed",
    category=UserWarning,
    module="openpyxl",
)

SACKMANN_COLUMNS = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level",
    "tourney_date", "match_num", "winner_id", "winner_seed", "winner_entry",
    "winner_name", "winner_hand", "winner_ht", "winner_ioc", "winner_age",
    "loser_id", "loser_seed", "loser_entry", "loser_name", "loser_hand",
    "loser_ht", "loser_ioc", "loser_age", "score", "best_of", "round",
    "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
    "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df",
    "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
]

TML_COLUMNS = SACKMANN_COLUMNS[:5] + ["indoor"] + SACKMANN_COLUMNS[5:]

ROUND_MAP = {
    "1st round": "R128/R64/R32", "2nd round": "R64/R32/R16",
    "3rd round": "R32", "4th round": "R16", "round robin": "RR",
    "quarterfinals": "QF", "semifinals": "SF", "the final": "F",
    "final": "F", "robin": "RR",
}

ROUND_ORDER = {
    "Q1": 1, "Q2": 2, "Q3": 3, "R128": 10, "R64": 20,
    "R32": 30, "R16": 40, "QF": 50, "SF": 60, "F": 70, "RR": 80,
}

PARTICLES = {"de", "del", "den", "der", "van", "von", "da", "dos", "di", "la", "le"}

TD_NUMERIC_COLUMNS = [
    "td_atp", "td_best_of", "td_wrank", "td_lrank", "td_wpts", "td_lpts",
    "td_w1", "td_l1", "td_w2", "td_l2", "td_w3", "td_l3",
    "td_w4", "td_l4", "td_w5", "td_l5", "td_wsets", "td_lsets",
    "td_b365w", "td_b365l", "td_psw", "td_psl", "td_maxw", "td_maxl",
    "td_avgw", "td_avgl", "td_bfew", "td_bfel",
]

INTEGER_COLUMNS = [
    "draw_size", "tourney_date", "match_num", "winner_id", "winner_seed",
    "winner_ht", "loser_id", "loser_seed", "loser_ht", "best_of", "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt",
    "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved",
    "l_bpFaced", "winner_rank", "winner_rank_points", "loser_rank",
    "loser_rank_points",
]

FLOAT_COLUMNS = ["winner_age", "loser_age"]

SURFACE_MAP = {
    "hard": "Hard", "clay": "Clay", "grass": "Grass",
    "carpet": "Carpet", "unknown": "Unknown",
}

MAIN_LEVEL_MAP = {
    "250": "A", "500": "A", "A": "A", "M": "M", "G": "G",
    "D": "D", "F": "F", "O": "O",
}

TML_DB_ALIASES = {
    "tml_id": ["tml_id", "player_id", "playerid", "id", "atp_id", "code"],
    "sackmann_id": [
        "sackmann_id", "jeff_sackmann_id", "jeff_id", "numeric_id",
        "atp_player_id", "player_id_sackmann",
    ],
    "full_name": ["player_name", "full_name", "name", "player", "atp_name"],
    "first_name": ["first_name", "firstname", "first", "name_first"],
    "last_name": ["last_name", "lastname", "last", "surname", "name_last"],
    "dob": ["dob", "birth_date", "date_of_birth", "birthdate"],
    "ioc": ["ioc", "country", "country_code", "nationality"],
}


def ratio(a: str, b: str) -> float:
    return 100.0 * SequenceMatcher(None, a, b).ratio()


def ascii_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalized_name(value) -> str:
    return " ".join(p for p in ascii_text(value).split() if p not in PARTICLES)


def sack_name_key(value) -> str:
    parts = normalized_name(value).split()
    if not parts:
        return ""
    return parts[-1] + "_" + "".join(p[0] for p in parts[:-1])


def td_name_key(value) -> str:
    parts = normalized_name(value).split()
    if not parts:
        return ""
    initials = "".join(p for p in parts if len(p) <= 2)
    surnames = [p for p in parts if len(p) > 2]
    return ("_".join(surnames) + "_" + initials) if surnames else parts[0]


def surname(key: str) -> str:
    return key.split("_")[0] if key else ""


def initial(key: str) -> str:
    parts = key.split("_")
    return parts[-1][:1] if len(parts) > 1 else ""


def clean_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ascii_text(value)).strip("_")


def first_existing(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lookup = {clean_column_name(column): column for column in columns}
    for alias in aliases:
        if clean_column_name(alias) in lookup:
            return lookup[clean_column_name(alias)]
    return None


def stable_synthetic_id(tml_id: str, used_ids: set[int]) -> int:
    """Create a stable positive ID outside the usual Sackmann range."""
    digest = hashlib.blake2b(str(tml_id).encode("utf-8"), digest_size=8).digest()
    candidate = 90_000_000 + int.from_bytes(digest, "big") % 9_000_000
    while candidate in used_ids:
        candidate += 1
        if candidate > 99_999_999:
            candidate = 90_000_000
    used_ids.add(candidate)
    return candidate


def read_csv_flexible(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise RuntimeError(f"No se pudo leer {path}: {last_error}")


def enforce_sackmann_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in SACKMANN_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[SACKMANN_COLUMNS + [c for c in result.columns if c not in SACKMANN_COLUMNS]]


def normalize_common(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for column in ["tourney_id", "tourney_name", "winner_name", "loser_name", "score"]:
        if column in result.columns:
            result[column] = result[column].astype("string").str.strip()

    result["surface"] = (
        result["surface"].astype("string").str.strip().str.casefold().map(SURFACE_MAP).fillna("Unknown")
    )
    result["round"] = result["round"].astype("string").str.strip().str.upper()
    result["winner_hand"] = result["winner_hand"].astype("string").str.strip().str.upper().replace("", pd.NA)
    result["loser_hand"] = result["loser_hand"].astype("string").str.strip().str.upper().replace("", pd.NA)
    result["winner_ioc"] = result["winner_ioc"].astype("string").str.strip().str.upper().replace("", pd.NA)
    result["loser_ioc"] = result["loser_ioc"].astype("string").str.strip().str.upper().replace("", pd.NA)

    for column in INTEGER_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").round().astype("Int64")

    for column in FLOAT_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Float64")

    return result


def read_sackmann(root: Path) -> pd.DataFrame:
    patterns = [
        "atp_matches_[0-9][0-9][0-9][0-9].csv",
        "atp_matches_futures_[0-9][0-9][0-9][0-9].csv",
        "atp_matches_qual_[0-9][0-9][0-9][0-9].csv",
        "atp_matches_chall_[0-9][0-9][0-9][0-9].csv",
        "atp_matches_qual_chall_[0-9][0-9][0-9][0-9].csv",
    ]
    files = sorted({path for pattern in patterns for path in root.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No hay CSV de Sackmann en: {root.resolve()}")

    frames = []
    for index, file_path in enumerate(files, start=1):
        print(f"Leyendo Sackmann {index}/{len(files)}: {file_path.name}")
        frame = read_csv_flexible(file_path)
        frame = enforce_sackmann_schema(frame)
        frame["source_sackmann_file"] = file_path.name
        frame["source_origin"] = "SACKMANN"
        frame["source_priority"] = 10
        if "competition_type" not in frame.columns:
            name = file_path.name.casefold()
            if "futures" in name:
                frame["competition_type"] = "FUTURES"
            elif "qual_chall" in name:
                frame["competition_type"] = np.where(
                    frame["tourney_level"].astype("string").eq("C"), "CHALLENGER", "ATP_QUALIFYING"
                )
            elif "chall" in name:
                frame["competition_type"] = "CHALLENGER"
            elif "qual" in name:
                frame["competition_type"] = "ATP_QUALIFYING"
            else:
                frame["competition_type"] = "ATP"
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    gc.collect()
    result = normalize_common(result)
    print(f"Sackmann cargado: {len(result):,} partidos")
    return result


def build_existing_player_index(sackmann: pd.DataFrame, atp_players_path: Path | None) -> tuple[dict[str, int], dict[str, list[int]], set[int]]:
    records: list[pd.DataFrame] = []

    if atp_players_path and atp_players_path.exists():
        players = read_csv_flexible(atp_players_path)
        id_col = first_existing(players.columns, ["player_id", "id"])
        name_col = first_existing(players.columns, ["name", "player_name", "full_name"])
        first_col = first_existing(players.columns, ["name_first", "first_name", "firstname"])
        last_col = first_existing(players.columns, ["name_last", "last_name", "lastname", "surname"])
        if id_col:
            if name_col:
                full_name = players[name_col].astype("string")
            elif first_col and last_col:
                full_name = players[first_col].astype("string").fillna("") + " " + players[last_col].astype("string").fillna("")
            else:
                full_name = pd.Series(pd.NA, index=players.index, dtype="string")
            records.append(pd.DataFrame({"player_id": players[id_col], "player_name": full_name}))

    for side in ("winner", "loser"):
        records.append(pd.DataFrame({
            "player_id": sackmann[f"{side}_id"],
            "player_name": sackmann[f"{side}_name"],
        }))

    identities = pd.concat(records, ignore_index=True).dropna(subset=["player_id", "player_name"])
    identities["player_id"] = pd.to_numeric(identities["player_id"], errors="coerce")
    identities = identities.dropna(subset=["player_id"])
    identities["player_id"] = identities["player_id"].astype("int64")
    identities["name_key"] = identities["player_name"].map(normalized_name)
    identities = identities.loc[identities["name_key"].ne("")]

    ids_by_name: dict[str, list[int]] = defaultdict(list)
    for key, player_id in identities[["name_key", "player_id"]].drop_duplicates().itertuples(index=False):
        ids_by_name[key].append(int(player_id))

    unique_name_map = {key: values[0] for key, values in ids_by_name.items() if len(set(values)) == 1}
    all_ids = set(identities["player_id"].astype(int).tolist())
    return unique_name_map, ids_by_name, all_ids


def load_tml_database(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    database = read_csv_flexible(path)
    database.columns = [str(column).strip() for column in database.columns]
    return database


def make_tml_identity_maps(
    database: pd.DataFrame,
    unique_name_map: dict[str, int],
) -> tuple[dict[str, int], dict[str, str]]:
    if database.empty:
        return {}, {}

    tml_id_col = first_existing(database.columns, TML_DB_ALIASES["tml_id"])
    sackmann_id_col = first_existing(database.columns, TML_DB_ALIASES["sackmann_id"])
    full_name_col = first_existing(database.columns, TML_DB_ALIASES["full_name"])
    first_name_col = first_existing(database.columns, TML_DB_ALIASES["first_name"])
    last_name_col = first_existing(database.columns, TML_DB_ALIASES["last_name"])

    if not tml_id_col:
        print("AVISO: ATP_Database.csv no tiene una columna de ID TML reconocible.")
        return {}, {}

    if full_name_col:
        names = database[full_name_col].astype("string")
    elif first_name_col and last_name_col:
        names = database[first_name_col].astype("string").fillna("") + " " + database[last_name_col].astype("string").fillna("")
    else:
        names = pd.Series(pd.NA, index=database.index, dtype="string")

    tml_to_name: dict[str, str] = {}
    tml_to_sackmann: dict[str, int] = {}

    for index, raw_tml_id in database[tml_id_col].items():
        if pd.isna(raw_tml_id):
            continue
        tml_id = str(raw_tml_id).strip()
        player_name = "" if pd.isna(names.loc[index]) else str(names.loc[index]).strip()
        if player_name:
            tml_to_name[tml_id] = player_name

        mapped_id = None
        if sackmann_id_col:
            numeric = pd.to_numeric(pd.Series([database.at[index, sackmann_id_col]]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                mapped_id = int(numeric)
        if mapped_id is None and player_name:
            mapped_id = unique_name_map.get(normalized_name(player_name))
        if mapped_id is not None:
            tml_to_sackmann[tml_id] = mapped_id

    return tml_to_sackmann, tml_to_name


def resolve_tml_players(
    frame: pd.DataFrame,
    tml_to_sackmann: dict[str, int],
    unique_name_map: dict[str, int],
    used_ids: set[int],
    synthetic_id_map: dict[str, int],
    unmatched_policy: str,
    source_file: str,
) -> tuple[pd.DataFrame, list[dict]]:
    result = frame.copy()
    audit_rows: list[dict] = []

    for side in ("winner", "loser"):
        original_ids = result[f"{side}_id"].astype("string")
        names = result[f"{side}_name"].astype("string")
        resolved: list[int | pd.NA] = []

        for raw_id, name in zip(original_ids.tolist(), names.tolist()):
            tml_id = "" if pd.isna(raw_id) else str(raw_id).strip()
            player_name = "" if pd.isna(name) else str(name).strip()
            mapped_id = tml_to_sackmann.get(tml_id)
            method = "tml_database"

            if mapped_id is None and player_name:
                mapped_id = unique_name_map.get(normalized_name(player_name))
                method = "exact_normalized_name"

            if mapped_id is None:
                if unmatched_policy == "error":
                    raise ValueError(f"Jugador sin mapear: {player_name!r}, TML ID={tml_id!r}")
                if unmatched_policy == "synthetic":
                    identity_key = tml_id or normalized_name(player_name)
                    if identity_key not in synthetic_id_map:
                        synthetic_id_map[identity_key] = stable_synthetic_id(identity_key, used_ids)
                    mapped_id = synthetic_id_map[identity_key]
                    method = "synthetic"
                else:
                    mapped_id = pd.NA
                    method = "unmatched_empty"

            resolved.append(mapped_id)
            audit_rows.append({
                "source_file": source_file,
                "side": side,
                "tml_id": tml_id,
                "player_name": player_name,
                "resolved_player_id": mapped_id,
                "mapping_method": method,
            })

        result[f"{side}_id_tml"] = original_ids
        result[f"{side}_id"] = pd.Series(resolved, index=result.index, dtype="Int64")

    return result, audit_rows


def validate_tml_columns(frame: pd.DataFrame, source_file: Path) -> None:
    missing = sorted(set(TML_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(TML_COLUMNS))
    if missing:
        raise ValueError(f"{source_file.name}: faltan columnas TML: {missing}")
    if extra:
        print(f"AVISO {source_file.name}: columnas adicionales conservadas: {extra}")


def normalize_tml_level(series: pd.Series, competition_type: str) -> tuple[pd.Series, pd.Series]:
    original = series.astype("string").str.strip().str.upper()
    if competition_type == "CHALLENGER":
        normalized = pd.Series("C", index=series.index, dtype="string")
    else:
        normalized = original.map(MAIN_LEVEL_MAP).fillna(original)
    return normalized, original


def read_tml_file(
    path: Path,
    competition_type: str,
    tml_to_sackmann: dict[str, int],
    unique_name_map: dict[str, int],
    used_ids: set[int],
    synthetic_id_map: dict[str, int],
    unmatched_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"Leyendo TennisMyLife: {path.name} [{competition_type}]")
    frame = read_csv_flexible(path)
    validate_tml_columns(frame, path)

    frame["source_tml_file"] = path.name
    frame["source_origin"] = "TML"
    frame["source_priority"] = 30
    frame["competition_type"] = competition_type
    frame["indoor"] = frame["indoor"].astype("string").str.strip().str.upper().map({"I": True, "O": False}).astype("boolean")
    frame["tourney_level"], frame["tourney_level_original"] = normalize_tml_level(frame["tourney_level"], competition_type)

    frame, mapping_audit = resolve_tml_players(
        frame, tml_to_sackmann, unique_name_map, used_ids, synthetic_id_map,
        unmatched_policy=unmatched_policy, source_file=path.name,
    )

    frame = enforce_sackmann_schema(frame)
    frame = normalize_common(frame)

    invalid_rows = []
    checks = [
        ("w_1stIn", "w_svpt", "winner_first_in_gt_service_points"),
        ("l_1stIn", "l_svpt", "loser_first_in_gt_service_points"),
        ("w_bpSaved", "w_bpFaced", "winner_bp_saved_gt_faced"),
        ("l_bpSaved", "l_bpFaced", "loser_bp_saved_gt_faced"),
    ]
    for left_column, right_column, issue in checks:
        invalid = frame[left_column].notna() & frame[right_column].notna() & frame[left_column].gt(frame[right_column])
        if invalid.any():
            for row_index in frame.index[invalid]:
                invalid_rows.append({
                    "source_file": path.name,
                    "row_index": int(row_index),
                    "tourney_id": frame.at[row_index, "tourney_id"],
                    "match_num": frame.at[row_index, "match_num"],
                    "issue": issue,
                    "left_value": frame.at[row_index, left_column],
                    "right_value": frame.at[row_index, right_column],
                })
            frame.loc[invalid, [left_column, right_column]] = pd.NA

    if competition_type == "ATP_QUALIFYING":
        bad_round = ~frame["round"].isin(["Q1", "Q2", "Q3"])
        for row_index in frame.index[bad_round]:
            invalid_rows.append({
                "source_file": path.name,
                "row_index": int(row_index),
                "tourney_id": frame.at[row_index, "tourney_id"],
                "match_num": frame.at[row_index, "match_num"],
                "issue": "non_qualifying_round_in_qualifying_file",
                "left_value": frame.at[row_index, "round"],
                "right_value": pd.NA,
            })

    return frame, pd.DataFrame(mapping_audit), pd.DataFrame(invalid_rows)


def logical_match_key(frame: pd.DataFrame) -> pd.Series:
    date = pd.to_numeric(frame["tourney_date"], errors="coerce").astype("Int64").astype("string")
    tournament = frame["tourney_name"].map(ascii_text).astype("string")
    winner = frame["winner_name"].map(normalized_name).astype("string")
    loser = frame["loser_name"].map(normalized_name).astype("string")
    round_key = frame["round"].astype("string").fillna("")
    return date + "|" + tournament + "|" + winner + "|" + loser + "|" + round_key


def repair_match_num_collisions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    result["source_match_num"] = result["match_num"]
    key_columns = ["tourney_id", "competition_type", "match_num"]
    duplicate_mask = result.duplicated(key_columns, keep=False) & result["match_num"].notna()
    collisions = result.loc[duplicate_mask, key_columns + ["winner_name", "loser_name", "round", "source_origin"]].copy()

    if collisions.empty:
        return result, collisions

    for (tourney_id, competition_type), indices in result.groupby(["tourney_id", "competition_type"], dropna=False).groups.items():
        group_indices = list(indices)
        group = result.loc[group_indices]
        duplicated_numbers = group["match_num"].duplicated(keep=False) & group["match_num"].notna()
        if not duplicated_numbers.any():
            continue
        next_number = int(pd.to_numeric(group["match_num"], errors="coerce").max() or 0) + 1
        duplicate_rows = group.loc[duplicated_numbers].sort_values(
            ["match_num", "tourney_date", "round"], kind="mergesort"
        )
        for _, same_number_rows in duplicate_rows.groupby("match_num", dropna=False):
            keep_index = same_number_rows.index[0]
            for row_index in same_number_rows.index[1:]:
                result.at[row_index, "match_num"] = next_number
                next_number += 1

    result["match_num"] = pd.to_numeric(result["match_num"], errors="coerce").astype("Int64")
    return result, collisions


def deduplicate_matches(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    result["_logical_match_key"] = logical_match_key(result)
    result["_row_order"] = np.arange(len(result), dtype=np.int64)
    result["_stat_completeness"] = result[[
        "minutes", "w_svpt", "l_svpt", "w_ace", "l_ace", "winner_rank", "loser_rank"
    ]].notna().sum(axis=1).astype("int8")

    result = result.sort_values(
        ["_logical_match_key", "source_priority", "_stat_completeness", "_row_order"],
        ascending=[True, False, False, False], kind="mergesort",
    )
    duplicate_mask = result.duplicated("_logical_match_key", keep="first")
    duplicates = result.loc[duplicate_mask].copy()
    result = result.loc[~duplicate_mask].copy()
    result = result.drop(columns=["_logical_match_key", "_row_order", "_stat_completeness"])
    duplicates = duplicates.drop(columns=["_row_order", "_stat_completeness"])
    result, collisions = repair_match_num_collisions(result)
    if not collisions.empty:
        collisions["duplicate_reason"] = "match_num_collision_repaired"
        duplicates = pd.concat([duplicates, collisions], ignore_index=True, sort=False)
    return result.reset_index(drop=True), duplicates.reset_index(drop=True)


def read_tennis_data(root: Path) -> pd.DataFrame:
    chosen: dict[str, Path] = {}
    for file_path in sorted(root.glob("*.xls")):
        chosen[file_path.stem] = file_path
    for file_path in sorted(root.glob("*.xlsx")):
        chosen[file_path.stem] = file_path
    if not chosen:
        raise FileNotFoundError(f"No hay archivos XLS/XLSX de tennis-data en: {root.resolve()}")

    frames = []
    selected_files = sorted(chosen.items())
    for index, (_, file_path) in enumerate(selected_files, start=1):
        print(f"Leyendo cuotas {index}/{len(selected_files)}: {file_path.name}")
        engine = "openpyxl" if file_path.suffix.lower() == ".xlsx" else "xlrd"
        frame = pd.read_excel(file_path, engine=engine)
        frame.columns = [str(column).strip() for column in frame.columns]
        frame["source_tennis_data_file"] = file_path.name
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    del frames
    gc.collect()
    print(f"Tennis-data cargado: {len(result):,} partidos")
    return result


def validate_columns(sackmann: pd.DataFrame, tennis_data: pd.DataFrame) -> None:
    required_sackmann = {
        "tourney_date", "tourney_name", "surface", "best_of", "round",
        "winner_name", "loser_name", "winner_rank", "loser_rank",
        "winner_rank_points", "loser_rank_points",
    }
    required_tennis = {
        "Date", "Tournament", "Location", "Surface", "Best of", "Round",
        "Winner", "Loser", "WRank", "LRank", "WPts", "LPts",
    }
    missing_sackmann = sorted(required_sackmann - set(sackmann.columns))
    missing_tennis = sorted(required_tennis - set(tennis_data.columns))
    if missing_sackmann:
        raise ValueError(f"Faltan columnas en Sackmann/TML combinado: {missing_sackmann}")
    if missing_tennis:
        raise ValueError(f"Faltan columnas en tennis-data: {missing_tennis}")


def prepare(sackmann: pd.DataFrame, tennis_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(sackmann, tennis_data)
    sackmann = sackmann.copy()
    tennis_data = tennis_data.copy()
    sackmann["_srow"] = np.arange(len(sackmann), dtype=np.int32)
    tennis_data["_trow"] = np.arange(len(tennis_data), dtype=np.int32)

    sackmann_date_number = pd.to_numeric(sackmann["tourney_date"], errors="coerce")
    sackmann["_year"] = sackmann_date_number.floordiv(10000).astype("Int16")
    sackmann["_event_date"] = pd.to_datetime(
        sackmann_date_number.astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    tennis_data["_match_date"] = pd.to_datetime(tennis_data["Date"], errors="coerce")
    tennis_data["_year"] = tennis_data["_match_date"].dt.year.astype("Int16")
    sackmann["_wk"] = sackmann["winner_name"].map(sack_name_key)
    sackmann["_lk"] = sackmann["loser_name"].map(sack_name_key)
    tennis_data["_wk"] = tennis_data["Winner"].map(td_name_key)
    tennis_data["_lk"] = tennis_data["Loser"].map(td_name_key)
    sackmann["_tour"] = sackmann["tourney_name"].map(ascii_text)
    tennis_data["_tour"] = tennis_data["Tournament"].map(ascii_text)
    tennis_data["_loc"] = tennis_data["Location"].map(ascii_text)
    return sackmann, tennis_data


def match_rows(sackmann: pd.DataFrame, tennis_data: pd.DataFrame) -> pd.DataFrame:
    pair_frames = []
    tennis_years = sorted(tennis_data["_year"].dropna().astype(int).unique().tolist())
    sackmann_columns = [
        "_srow", "_year", "_event_date", "_wk", "_lk", "_tour", "surface",
        "best_of", "round", "winner_rank", "loser_rank", "winner_rank_points",
        "loser_rank_points",
    ]
    tennis_columns = [
        "_trow", "_year", "_match_date", "_wk", "_lk", "_tour", "_loc",
        "Surface", "Best of", "Round", "WRank", "LRank", "WPts", "LPts",
    ]

    for year in tennis_years:
        sy = sackmann.loc[sackmann["_year"].eq(year), sackmann_columns].copy()
        ty = tennis_data.loc[tennis_data["_year"].eq(year), tennis_columns].copy()
        if sy.empty or ty.empty:
            continue
        print(f"Matching {year}: {len(sy):,} partidos / {len(ty):,} cuotas")
        for candidate_frame in (sy, ty):
            candidate_frame["_wsurname"] = candidate_frame["_wk"].map(surname)
            candidate_frame["_lsurname"] = candidate_frame["_lk"].map(surname)
            candidate_frame["_winitial"] = candidate_frame["_wk"].map(initial)
            candidate_frame["_linitial"] = candidate_frame["_lk"].map(initial)

        merge_keys = ["_year", "_wsurname", "_lsurname", "_winitial", "_linitial"]
        candidates = sy.merge(ty, how="inner", on=merge_keys, suffixes=("_s", "_t"))
        del sy, ty
        if candidates.empty:
            continue

        candidates["_days"] = (candidates["_match_date"] - candidates["_event_date"]).dt.days
        candidates = candidates.loc[candidates["_days"].between(-3, 24)].copy()
        if candidates.empty:
            continue

        candidates["match_score"] = (100.0 - 1.5 * candidates["_days"].abs()).astype("float32")
        surface_match = (
            candidates["surface"].astype("string").str.lower()
            == candidates["Surface"].astype("string").str.lower()
        ).fillna(False)
        candidates.loc[surface_match, "match_score"] += 5.0

        sack_best_of = pd.to_numeric(candidates["best_of"], errors="coerce")
        tennis_best_of = pd.to_numeric(candidates["Best of"], errors="coerce")
        candidates.loc[sack_best_of.eq(tennis_best_of).fillna(False), "match_score"] += 4.0

        for sack_column, tennis_column, weight in [
            ("winner_rank", "WRank", 5.0), ("loser_rank", "LRank", 5.0),
            ("winner_rank_points", "WPts", 3.0), ("loser_rank_points", "LPts", 3.0),
        ]:
            left = pd.to_numeric(candidates[sack_column], errors="coerce")
            right = pd.to_numeric(candidates[tennis_column], errors="coerce")
            exact = left.notna() & right.notna() & left.eq(right)
            candidates.loc[exact, "match_score"] += weight

        mapped_rounds = candidates["Round"].map(ascii_text).map(ROUND_MAP).fillna("")
        round_match = np.fromiter(
            (str(sr) in str(tr).split("/") for sr, tr in zip(candidates["round"], mapped_rounds)),
            dtype=bool, count=len(candidates),
        )
        candidates.loc[round_match, "match_score"] += 5.0

        tournament_score = np.fromiter(
            (
                max(ratio(str(st), str(tt)), ratio(str(st), str(loc)))
                for st, tt, loc in zip(candidates["_tour_s"], candidates["_tour_t"], candidates["_loc"])
            ), dtype=np.float32, count=len(candidates),
        )
        candidates["match_score"] += tournament_score * np.float32(0.35)
        candidates = candidates.loc[
            candidates["match_score"].ge(112.0), ["_srow", "_trow", "match_score"]
        ].sort_values("match_score", ascending=False)
        if candidates.empty:
            continue
        candidates = candidates.drop_duplicates("_srow", keep="first").drop_duplicates("_trow", keep="first")
        candidates["_srow"] = candidates["_srow"].astype("int32")
        candidates["_trow"] = candidates["_trow"].astype("int32")
        candidates["match_score"] = candidates["match_score"].astype("float32")
        print(f"  Coincidencias aceptadas: {len(candidates):,}")
        pair_frames.append(candidates)
        gc.collect()

    if not pair_frames:
        return pd.DataFrame({
            "_srow": pd.Series(dtype="int32"),
            "_trow": pd.Series(dtype="int32"),
            "match_score": pd.Series(dtype="float32"),
        })
    return pd.concat(pair_frames, ignore_index=True)


def clean_numeric_odds(output: pd.DataFrame) -> pd.DataFrame:
    invalid_values = {"-": np.nan, "\\t": np.nan, "\t": np.nan, "": np.nan, " ": np.nan}
    for column in TD_NUMERIC_COLUMNS:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column].replace(invalid_values), errors="coerce")
    return output


def make_parquet_safe(output: pd.DataFrame) -> pd.DataFrame:
    for column in output.columns:
        series = output[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            output[column] = series.astype("string")
            continue
        if pd.api.types.is_object_dtype(series.dtype):
            non_null = series.dropna()
            if not non_null.empty and len(non_null.map(type).drop_duplicates()) > 1:
                output[column] = series.astype("string")
    return output


def merge_odds(matches: pd.DataFrame, tennis_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches, tennis_data = prepare(matches, tennis_data)
    pairs = match_rows(matches, tennis_data)
    original_columns = [column for column in tennis_data.columns if not column.startswith("_")]
    tennis_for_merge = tennis_data[["_trow"] + original_columns].copy().rename(
        columns={column: "td_" + column.replace(" ", "_").lower() for column in original_columns}
    )
    helper_columns = [column for column in matches.columns if column.startswith("_") and column != "_srow"]
    matches = matches.drop(columns=helper_columns)
    output = matches.merge(pairs, on="_srow", how="left", validate="one_to_one")
    output = output.merge(tennis_for_merge, on="_trow", how="left", validate="many_to_one")
    output["odds_matched"] = output["_trow"].notna()
    output = output.drop(columns=["_srow", "_trow"])
    return make_parquet_safe(clean_numeric_odds(output)), pairs


def build_tournament_dimension(matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "tourney_id", "tourney_name", "surface", "tourney_level",
        "tourney_level_original", "competition_type", "indoor", "tourney_date",
    ]
    selected = [column for column in columns if column in matches.columns]
    dimension = matches[selected].copy()
    dimension["_priority"] = matches["source_priority"].to_numpy()
    dimension = dimension.sort_values("_priority", ascending=False, kind="mergesort")
    dimension = dimension.drop_duplicates(["tourney_id", "competition_type"], keep="first")
    return dimension.drop(columns="_priority").sort_values(["tourney_date", "tourney_id"]).reset_index(drop=True)


def build_player_dimension(matches: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for side in ("winner", "loser"):
        mapping = {
            f"{side}_id": "player_id", f"{side}_name": "player_name",
            f"{side}_hand": "player_hand", f"{side}_ht": "player_ht",
            f"{side}_ioc": "player_ioc",
        }
        available = {source: target for source, target in mapping.items() if source in matches.columns}
        frame = matches[list(available)].rename(columns=available)
        frame["tourney_date"] = matches["tourney_date"].to_numpy()
        frame["source_priority"] = matches["source_priority"].to_numpy()
        frames.append(frame)
    players = pd.concat(frames, ignore_index=True).dropna(subset=["player_id", "player_name"])
    players = players.sort_values(["source_priority", "tourney_date"], ascending=[False, False], kind="mergesort")
    players = players.drop_duplicates("player_id", keep="first")
    return players.drop(columns=["tourney_date", "source_priority"]).sort_values("player_name").reset_index(drop=True)


def find_project_root(start: Path) -> Path:
    """Locate the repository root without depending on the script depth."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (
            (candidate / "modeling").exists()
            or (candidate / "streamlit_app").exists()
            or (candidate / ".git").exists()
        ):
            return candidate
    return start.parent


def parse_arguments() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    parser = argparse.ArgumentParser(description="Integra Sackmann, TennisMyLife y tennis-data.co.uk")
    parser.add_argument("--sackmann-dir", default=str(project_root / "data" / "raw" / "tennis_atp-master"))
    parser.add_argument("--odds-dir", default=str(project_root / "data" / "raw" / "tennis_data_co_uk_v2"))
    tml_current_season_dir = (
        project_root
        / "data"
        / "raw"
        / "tml_v2"
        / "current_season"
    )

    parser.add_argument(
        "--tml-main",
        default=str(
            tml_current_season_dir
            / "2026.csv"
        ),
    )

    parser.add_argument(
        "--tml-qualifying",
        default=str(
            tml_current_season_dir
            / "2026_atp_quali.csv"
        ),
    )

    parser.add_argument(
        "--tml-challenger",
        default=str(
            tml_current_season_dir
            / "2026_challenger.csv"
        ),
    )

    parser.add_argument(
        "--tml-player-database",
        default=str(
            tml_current_season_dir
            / "ATP_Database.csv"
        ),
    )
    parser.add_argument("--atp-players", default=str(project_root / "data" / "raw" / "tennis_atp-master" / "atp_players.csv"))
    parser.add_argument("--unmatched-player-policy", choices=["synthetic", "empty", "error"], default="synthetic")
    parser.add_argument("--output", default=str(project_root / "data" / "processed" / "jeff_sackmann_with_odds.parquet"))
    parser.add_argument("--report-dir", default=str(project_root / "data" / "processed" / "tml_integration_reports"))
    parser.add_argument("--skip-odds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    sackmann = read_sackmann(Path(args.sackmann_dir))
    unique_name_map, _, used_ids = build_existing_player_index(sackmann, Path(args.atp_players))
    tml_database = load_tml_database(Path(args.tml_player_database))
    tml_to_sackmann, _ = make_tml_identity_maps(tml_database, unique_name_map)

    tml_specs = [
        (Path(args.tml_main), "ATP"),
        (Path(args.tml_qualifying), "ATP_QUALIFYING"),
        (Path(args.tml_challenger), "CHALLENGER"),
    ]
    tml_frames = []
    synthetic_id_map: dict[str, int] = {}
    mapping_reports = []
    invalid_reports = []
    for path, competition_type in tml_specs:
        if not path.exists():
            print(f"AVISO: no existe {path}; se omite.")
            continue
        frame, mapping_report, invalid_report = read_tml_file(
            path, competition_type, tml_to_sackmann, unique_name_map,
            used_ids, synthetic_id_map, args.unmatched_player_policy,
        )
        tml_frames.append(frame)
        mapping_reports.append(mapping_report)
        invalid_reports.append(invalid_report)

    combined = pd.concat([sackmann] + tml_frames, ignore_index=True, sort=False)
    del sackmann, tml_frames
    gc.collect()
    combined, duplicate_report = deduplicate_matches(combined)
    print(f"Partidos tras deduplicar: {len(combined):,}")

    if args.skip_odds:
        output = make_parquet_safe(combined)
        pairs = pd.DataFrame()
    else:
        tennis_data = read_tennis_data(Path(args.odds_dir))
        output, pairs = merge_odds(combined, tennis_data)

    output_path = Path(args.output)
    report_dir = Path(args.report_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    output = enforce_sackmann_schema(output)
    output = make_parquet_safe(output)
    output.to_parquet(output_path, index=False, engine="pyarrow", compression="zstd")

    mapping_report = pd.concat(mapping_reports, ignore_index=True, sort=False) if mapping_reports else pd.DataFrame()
    invalid_report = pd.concat(invalid_reports, ignore_index=True, sort=False) if invalid_reports else pd.DataFrame()
    tournament_dimension = build_tournament_dimension(output)
    player_dimension = build_player_dimension(output)

    mapping_report.to_csv(report_dir / "tml_player_mapping_report.csv", index=False)
    mapping_report.loc[mapping_report.get("mapping_method", pd.Series(dtype="string")).isin(["synthetic", "unmatched_empty"])].to_csv(
        report_dir / "tml_unmatched_players.csv", index=False
    )
    invalid_report.to_csv(report_dir / "tml_invalid_statistics.csv", index=False)
    duplicate_report.to_csv(report_dir / "tml_duplicate_matches.csv", index=False)
    tournament_dimension.to_csv(report_dir / "tournament_dimension.csv", index=False)
    player_dimension.to_csv(report_dir / "player_dimension.csv", index=False)

    summary = {
        "output_rows": int(len(output)),
        "output_columns": int(len(output.columns)),
        "unique_logical_matches": int(logical_match_key(output).nunique()),
        "unique_players": int(player_dimension["player_id"].nunique()),
        "unique_tournaments": int(tournament_dimension[["tourney_id", "competition_type"]].drop_duplicates().shape[0]),
        "tml_mapping_rows": int(len(mapping_report)),
        "tml_synthetic_mappings": int(mapping_report.get("mapping_method", pd.Series(dtype="string")).eq("synthetic").sum()),
        "duplicates_removed_or_repaired": int(len(duplicate_report)),
        "invalid_statistics": int(len(invalid_report)),
        "odds_matches": int(len(pairs)),
    }
    (report_dir / "integration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nCreado: {output_path}")
    print(f"Informes: {report_dir}")


if __name__ == "__main__":
    main()
