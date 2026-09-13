#!/usr/bin/env python3
"""Postprocesa el Parquet integrado Sackmann + TML de forma segura.

Correcciones incluidas
---------------------
1. Repara colisiones de identidad confirmadas por nombre canónico.
2. Conserva los IDs TML originales como trazabilidad.
3. Detecta y fusiona duplicados cross-source Sackmann/TML con el mismo
   torneo, competición, fecha, ronda, ganador y perdedor.
4. Conserva conflictos históricos con ganador invertido.
5. Conserva partidos distintos aunque una fuente haya reutilizado un ID TML.
6. Crea tournament_key y canonical_tournament_name.
7. Conserva tourney_name_original.
8. Repara colisiones de match_num después de la fusión.
9. Escribe el Parquet de forma atómica y genera informes de auditoría.

Uso
---
python ingestion/v2/postprocess_sackmann_tml_integrated_parquet.py

También puede ejecutarse desde build_jeff_sackmann_with_odds_tml.py llamando a:

    from postprocess_sackmann_tml_integrated_parquet import postprocess_parquet
    postprocess_parquet(output_path, report_dir)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TOURNAMENT_KEY_SEPARATOR = "|"
SYNTHETIC_ID_MIN = 90_000_000

# Reglas manuales verificadas contra cientos de observaciones Sackmann.
CANONICAL_PLAYER_BY_NAME = {
    "alexander bublik": {
        "player_id": 122330,
        "canonical_name": "Alexander Bublik",
        "ioc": "KAZ",
    },
    "alexander shevchenko": {
        "player_id": 207686,
        "canonical_name": "Alexander Shevchenko",
        "ioc": None,
    },
    "aleksander shevchenko": {
        "player_id": 207686,
        "canonical_name": "Alexander Shevchenko",
        "ioc": None,
    },
    "aleksandr shevchenko": {
        "player_id": 207686,
        "canonical_name": "Alexander Shevchenko",
        "ioc": None,
    },
    "jerome kym": {
        "player_id": 208843,
        "canonical_name": "Jerome Kym",
        "ioc": "SUI",
    },
    "moez echargui": {
        "player_id": 121411,
        "canonical_name": "Moez Echargui",
        "ioc": "TUN",
    },
    # Identidad confirmada: el ID sintético TML 91985805 corresponde al
    # jugador ATP/Sackmann 210017. La auditoría confirmó continuidad de IOC,
    # edad, ranking, calendario y rivales.
    "daniel merida": {
        "player_id": 210017,
        "canonical_name": "Daniel Merida Aguilar",
        "ioc": "ESP",
    },
    "daniel merida aguilar": {
        "player_id": 210017,
        "canonical_name": "Daniel Merida Aguilar",
        "ioc": "ESP",
    },
}

# IDs sintéticos retirados tras una reconciliación manual confirmada. La
# validación final falla si cualquiera de estos IDs sobrevive en el resultado.
RETIRED_CONFIRMED_PLAYER_IDS = {
    91985805: {
        "canonical_player_id": 210017,
        "canonical_name": "Daniel Merida Aguilar",
        "reason": "confirmed_tml_synthetic_alias",
    },
}

STAT_COLUMNS = [
    "minutes",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]

ODDS_COLUMNS = [
    "odds_matched",
    "td_b365w", "td_b365l", "td_psw", "td_psl",
    "td_maxw", "td_maxl", "td_avgw", "td_avgl",
    "td_bfew", "td_bfel",
]

CONFIRMED_DUPLICATE_KEY = [
    "tourney_id",
    "competition_type",
    "tourney_date",
    "round",
    "winner_id",
    "loser_id",
]

SUSPECT_PAIR_KEY = [
    "tourney_id",
    "competition_type",
    "tourney_date",
    "round",
    "player_low",
    "player_high",
]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "processed").exists():
            return candidate
    return start.parent


def ascii_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalized_person_name(value: Any) -> str:
    return ascii_text(value)


def normalize_competition_type(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip().upper()
    return text or "UNKNOWN"


def normalize_tournament_name_for_comparison(value: Any) -> str:
    text = ascii_text(value)
    text = re.sub(r"\s+challenger$", "", text).strip()
    text = re.sub(r"\s+ch$", "", text).strip()
    return re.sub(r"\s+", " ", text).strip()


def make_tournament_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["tourney_id"].astype("string").str.strip()
        + TOURNAMENT_KEY_SEPARATOR
        + frame["competition_type"].astype("string").str.strip().str.upper()
    )


def available_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def safe_bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).astype(bool)


def first_non_null(values: list[Any]) -> Any:
    for value in values:
        if value is not None and not pd.isna(value):
            return value
    return pd.NA


def values_equal(left: Any, right: Any) -> bool:
    if (left is None or pd.isna(left)) and (right is None or pd.isna(right)):
        return True
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return False
    return str(left) == str(right)


# ---------------------------------------------------------------------------
# Identity repair
# ---------------------------------------------------------------------------


def build_player_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """Crea una dimensión dominante a partir de filas no TML y del histórico."""
    rows = []
    for side in ("winner", "loser"):
        columns = [
            f"{side}_id", f"{side}_name", f"{side}_ioc",
            f"{side}_hand", f"{side}_ht", f"{side}_age",
            f"{side}_rank", f"{side}_rank_points",
            "source_origin", "tourney_date",
        ]
        selected = [column for column in columns if column in frame.columns]
        local = frame[selected].copy()
        local = local.rename(columns={
            f"{side}_id": "player_id",
            f"{side}_name": "player_name",
            f"{side}_ioc": "player_ioc",
            f"{side}_hand": "player_hand",
            f"{side}_ht": "player_ht",
            f"{side}_age": "player_age",
            f"{side}_rank": "player_rank",
            f"{side}_rank_points": "player_rank_points",
        })
        rows.append(local)

    players = pd.concat(rows, ignore_index=True, sort=False)
    players = players.dropna(subset=["player_id", "player_name"])
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce")
    players = players.dropna(subset=["player_id"])
    players["player_id"] = players["player_id"].astype("int64")
    players["normalized_name"] = players["player_name"].map(normalized_person_name)
    players["source_order"] = (
        players.get("source_origin", pd.Series("", index=players.index))
        .astype("string")
        .str.upper()
        .map({"SACKMANN": 0, "SACKMANN_TML": 1, "TML": 2})
        .fillna(9)
    )
    players["date_order"] = pd.to_numeric(
        players.get("tourney_date", pd.Series(pd.NA, index=players.index)),
        errors="coerce",
    ).fillna(-1)

    # Una fila representativa por ID. Se prioriza Sackmann, y dentro de la
    # fuente la observación más reciente para atributos mutables.
    players = players.sort_values(
        ["player_id", "source_order", "date_order"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    reference = players.drop_duplicates("player_id", keep="first").copy()

    requested = [
        "player_id", "player_name", "player_ioc", "player_hand", "player_ht",
        "player_age", "player_rank", "player_rank_points",
    ]
    return reference[[column for column in requested if column in reference.columns]]


def repair_confirmed_identity_collisions(
    frame: pd.DataFrame,
    player_reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    reports: list[dict[str, Any]] = []

    reference_by_id = (
        player_reference.set_index("player_id").to_dict("index")
        if not player_reference.empty
        else {}
    )

    if "identity_correction_applied" not in result.columns:
        result["identity_correction_applied"] = False
    if "identity_correction_reason" not in result.columns:
        result["identity_correction_reason"] = pd.Series(
            pd.NA, index=result.index, dtype="string"
        )

    for side in ("winner", "loser"):
        id_column = f"{side}_id"
        name_column = f"{side}_name"
        ioc_column = f"{side}_ioc"
        normalized = result[name_column].map(normalized_person_name)

        for alias, rule in CANONICAL_PLAYER_BY_NAME.items():
            mask = normalized.eq(alias)
            if not mask.any():
                continue

            expected_id = int(rule["player_id"])
            current_ids = pd.to_numeric(result.loc[mask, id_column], errors="coerce")
            correction_mask = mask.copy()
            correction_mask.loc[mask] = current_ids.ne(expected_id).to_numpy()

            for row_index in result.index[correction_mask]:
                old_id = result.at[row_index, id_column]
                old_name = result.at[row_index, name_column]
                reports.append({
                    "row_index": int(row_index),
                    "tourney_id": result.at[row_index, "tourney_id"],
                    "competition_type": result.at[row_index, "competition_type"],
                    "match_num": result.at[row_index, "match_num"],
                    "source_origin": result.at[row_index, "source_origin"],
                    "side": side,
                    "raw_tml_id": result.at[row_index, f"{side}_id_tml"]
                    if f"{side}_id_tml" in result.columns else pd.NA,
                    "old_player_id": old_id,
                    "new_player_id": expected_id,
                    "old_player_name": old_name,
                    "canonical_player_name": rule["canonical_name"],
                    "reason": "confirmed_name_id_collision",
                })

            if not correction_mask.any():
                continue

            result.loc[correction_mask, id_column] = expected_id
            result.loc[correction_mask, name_column] = rule["canonical_name"]

            # Restaura atributos canónicos conocidos desde la dimensión.
            reference = reference_by_id.get(expected_id, {})
            attribute_mapping = {
                "player_ioc": ioc_column,
                "player_hand": f"{side}_hand",
                "player_ht": f"{side}_ht",
                "player_age": f"{side}_age",
                "player_rank": f"{side}_rank",
                "player_rank_points": f"{side}_rank_points",
            }
            for reference_column, target_column in attribute_mapping.items():
                if target_column not in result.columns:
                    continue
                value = reference.get(reference_column, pd.NA)
                # Ranking/edad varían temporalmente. Solo usamos dimensión para
                # IOC, mano y altura. Rank/age se limpian si la identidad cambió.
                if reference_column in {"player_ioc", "player_hand", "player_ht"}:
                    if value is not None and not pd.isna(value):
                        result.loc[correction_mask, target_column] = value
                else:
                    result.loc[correction_mask, target_column] = pd.NA

            if rule.get("ioc") and ioc_column in result.columns:
                result.loc[correction_mask, ioc_column] = rule["ioc"]

            result.loc[correction_mask, "identity_correction_applied"] = True
            result.loc[
                correction_mask, "identity_correction_reason"
            ] = "confirmed_name_id_collision"

    for column in ("winner_id", "loser_id"):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")

    return result, pd.DataFrame(reports)


# ---------------------------------------------------------------------------
# Tournament canonicalization
# ---------------------------------------------------------------------------


def build_canonical_tournament_names(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = frame[
        [
            "tournament_key", "tourney_id", "competition_type",
            "tourney_name", "source_origin",
        ]
    ].dropna(subset=["tournament_key", "tourney_name"]).copy()

    candidates["source_order"] = (
        candidates["source_origin"]
        .astype("string")
        .str.upper()
        .map({"TML": 0, "SACKMANN_TML": 1, "SACKMANN": 2})
        .fillna(9)
    )
    candidates["clean_name"] = candidates["tourney_name"].map(
        normalize_tournament_name_for_comparison
    )
    candidates["technical_suffix_penalty"] = (
        candidates["tourney_name"]
        .astype("string")
        .str.upper()
        .str.contains(r"\sCH$|CHALLENGER", regex=True, na=False)
        .astype("int8")
    )
    candidates["name_length"] = candidates["tourney_name"].astype("string").str.len()

    frequency = (
        candidates.groupby(
            ["tournament_key", "tourney_name"],
            dropna=False,
            as_index=False,
        )
        .size()
        .rename(columns={"size": "name_frequency"})
    )
    candidates = candidates.merge(
        frequency,
        on=["tournament_key", "tourney_name"],
        how="left",
        validate="many_to_one",
    )

    # Para la temporada TML se prioriza el nombre TML, que en los casos
    # auditados elimina sufijos técnicos como "CH". Luego frecuencia y limpieza.
    candidates = candidates.sort_values(
        [
            "tournament_key", "source_order", "technical_suffix_penalty",
            "name_frequency", "name_length", "tourney_name",
        ],
        ascending=[True, True, True, False, True, True],
        kind="mergesort",
    )

    canonical = candidates.drop_duplicates("tournament_key", keep="first").copy()
    canonical = canonical.rename(
        columns={"tourney_name": "canonical_tournament_name"}
    )
    return canonical[
        [
            "tournament_key", "tourney_id", "competition_type",
            "canonical_tournament_name",
        ]
    ].reset_index(drop=True)


def apply_tournament_canonicalization(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    result["competition_type"] = (
        result["competition_type"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )
    result["tournament_key"] = make_tournament_key(result)
    result["tourney_name_original"] = result["tourney_name"].astype("string")

    canonical = build_canonical_tournament_names(result)
    result = result.merge(
        canonical[["tournament_key", "canonical_tournament_name"]],
        on="tournament_key",
        how="left",
        validate="many_to_one",
    )
    result["tourney_name"] = (
        result["canonical_tournament_name"].fillna(result["tourney_name"])
    )
    return result, canonical


# ---------------------------------------------------------------------------
# Cross-source duplicate fusion
# ---------------------------------------------------------------------------


def duplicate_quality_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    stat_columns = [column for column in STAT_COLUMNS if column in result.columns]
    result["_stats_coverage"] = (
        result[stat_columns].notna().sum(axis=1).astype("int16")
        if stat_columns
        else 0
    )
    result["_has_odds"] = safe_bool_series(result, "odds_matched").astype("int8")
    result["_identity_ok"] = (~safe_bool_series(
        result, "identity_correction_applied"
    )).astype("int8")
    result["_source_order"] = (
        result["source_origin"]
        .astype("string")
        .str.upper()
        .map({"TML": 2, "SACKMANN": 1, "SACKMANN_TML": 3})
        .fillna(0)
        .astype("int8")
    )
    result["_row_order"] = np.arange(len(result), dtype=np.int64)
    return result


def merge_two_rows(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    merged = primary.copy()

    # Completa nulos desde la segunda fuente. No sustituye valores válidos del
    # registro de mayor calidad salvo en trazabilidad/nombre canónico.
    for column in merged.index:
        if column.startswith("_"):
            continue
        if column in {"source_origin", "source_priority"}:
            continue
        if (merged[column] is None or pd.isna(merged[column])) and (
            secondary[column] is not None and not pd.isna(secondary[column])
        ):
            merged[column] = secondary[column]

    merged["source_origin"] = "SACKMANN_TML"
    if "source_priority" in merged.index:
        priorities = pd.to_numeric(
            pd.Series([primary.get("source_priority"), secondary.get("source_priority")]),
            errors="coerce",
        )
        merged["source_priority"] = priorities.max()

    merged["source_merged"] = True
    merged["source_duplicate_count"] = 2

    # Preserva archivos de ambas fuentes independientemente de la fila base.
    for column in ("source_sackmann_file", "source_tml_file"):
        if column in merged.index:
            merged[column] = first_non_null([
                primary.get(column, pd.NA),
                secondary.get(column, pd.NA),
            ])

    # Conserva match_num de Sackmann cuando existe, porque suele ser estable en
    # el histórico. source_match_num conserva el número original de la fila base.
    sackmann_rows = [row for row in (primary, secondary) if str(row.get("source_origin", "")).upper() == "SACKMANN"]
    if sackmann_rows and "match_num" in merged.index:
        sackmann_match_num = sackmann_rows[0].get("match_num", pd.NA)
        if sackmann_match_num is not None and not pd.isna(sackmann_match_num):
            merged["match_num"] = sackmann_match_num

    return merged


def fuse_cross_source_duplicates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = duplicate_quality_columns(frame)

    group_stats = (
        result.groupby(CONFIRMED_DUPLICATE_KEY, dropna=False)
        .agg(
            rows=("source_origin", "size"),
            source_count=("source_origin", "nunique"),
            sources=(
                "source_origin",
                lambda values: "+".join(sorted(set(values.dropna().astype(str)))),
            ),
        )
        .reset_index()
    )

    candidates = group_stats.loc[
        group_stats["rows"].eq(2)
        & group_stats["source_count"].eq(2)
        & group_stats["sources"].eq("SACKMANN+TML")
    ].copy()

    candidate_key_index = pd.MultiIndex.from_frame(candidates[CONFIRMED_DUPLICATE_KEY])
    row_key_index = pd.MultiIndex.from_frame(result[CONFIRMED_DUPLICATE_KEY])
    candidate_mask = row_key_index.isin(candidate_key_index)

    candidate_rows = result.loc[candidate_mask].copy()
    untouched = result.loc[~candidate_mask].copy()

    merged_rows: list[pd.Series] = []
    reports: list[dict[str, Any]] = []

    for key_values, group in candidate_rows.groupby(
        CONFIRMED_DUPLICATE_KEY,
        dropna=False,
        sort=False,
    ):
        ordered = group.sort_values(
            ["_stats_coverage", "_has_odds", "_identity_ok", "_source_order", "_row_order"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        )
        primary = ordered.iloc[0]
        secondary = ordered.iloc[1]
        merged = merge_two_rows(primary, secondary)
        merged_rows.append(merged)

        reports.append({
            **dict(zip(CONFIRMED_DUPLICATE_KEY, key_values if isinstance(key_values, tuple) else (key_values,))),
            "primary_source": primary["source_origin"],
            "secondary_source": secondary["source_origin"],
            "primary_match_num": primary.get("match_num", pd.NA),
            "secondary_match_num": secondary.get("match_num", pd.NA),
            "primary_stats_coverage": int(primary["_stats_coverage"]),
            "secondary_stats_coverage": int(secondary["_stats_coverage"]),
            "primary_has_odds": bool(primary["_has_odds"]),
            "secondary_has_odds": bool(secondary["_has_odds"]),
            "fusion_status": "merged_cross_source_same_result",
        })

    merged_frame = pd.DataFrame(merged_rows)
    output = pd.concat([untouched, merged_frame], ignore_index=True, sort=False)
    helper_columns = [column for column in output.columns if column.startswith("_")]
    output = output.drop(columns=helper_columns, errors="ignore")

    # Grupos sospechosos por pareja no orientada que se conservaron.
    output["player_low"] = output[["winner_id", "loser_id"]].min(axis=1)
    output["player_high"] = output[["winner_id", "loser_id"]].max(axis=1)
    suspect_mask = output.duplicated(SUSPECT_PAIR_KEY, keep=False)
    conflicts = output.loc[suspect_mask].copy()
    if not conflicts.empty:
        conflicts["winner_count"] = conflicts.groupby(
            SUSPECT_PAIR_KEY,
            dropna=False,
        )["winner_id"].transform("nunique")
        conflicts["source_count"] = conflicts.groupby(
            SUSPECT_PAIR_KEY,
            dropna=False,
        )["source_origin"].transform("nunique")
        conflicts["conflict_status"] = np.where(
            conflicts["winner_count"].gt(1),
            "kept_opposite_winners",
            "kept_same_pair_requires_review",
        )

    output = output.drop(columns=["player_low", "player_high"], errors="ignore")
    return output, pd.DataFrame(reports), conflicts


# ---------------------------------------------------------------------------
# match_num repair and validation
# ---------------------------------------------------------------------------


def repair_match_num_collisions(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    if "source_match_num" not in result.columns:
        result["source_match_num"] = result["match_num"]

    key = ["tourney_id", "competition_type", "match_num"]
    duplicate_mask = result.duplicated(key, keep=False) & result["match_num"].notna()
    report = result.loc[
        duplicate_mask,
        [column for column in key + [
            "winner_name", "loser_name", "round", "source_origin",
        ] if column in result.columns],
    ].copy()

    if report.empty:
        return result, report

    for _, indices in result.groupby(
        ["tourney_id", "competition_type"],
        dropna=False,
    ).groups.items():
        group = result.loc[list(indices)]
        duplicated = group["match_num"].duplicated(keep=False) & group["match_num"].notna()
        if not duplicated.any():
            continue

        numeric_match_num = pd.to_numeric(group["match_num"], errors="coerce")
        maximum = numeric_match_num.max()
        next_number = int(maximum) + 1 if pd.notna(maximum) else 1

        for _, same_number in group.loc[duplicated].groupby("match_num", dropna=False):
            ordered = same_number.sort_values(
                ["tourney_date", "round", "winner_id", "loser_id"],
                kind="mergesort",
            )
            for row_index in ordered.index[1:]:
                result.at[row_index, "match_num"] = next_number
                next_number += 1

    result["match_num"] = pd.to_numeric(result["match_num"], errors="coerce").astype("Int64")
    return result, report


def validate_final_frame(
    frame: pd.DataFrame,
    expected_cross_source_merges: int,
) -> dict[str, Any]:
    operational_key = ["tourney_id", "competition_type", "match_num"]
    operational_duplicates = int(frame.duplicated(operational_key).sum())

    confirmed_duplicates = int(
        frame.duplicated(CONFIRMED_DUPLICATE_KEY).sum()
    )

    # Conflictos históricos por pareja orientada diferente se conservan.
    local = frame.copy()
    local["player_low"] = local[["winner_id", "loser_id"]].min(axis=1)
    local["player_high"] = local[["winner_id", "loser_id"]].max(axis=1)
    suspect = local.loc[local.duplicated(SUSPECT_PAIR_KEY, keep=False)]
    conflict_groups = 0
    if not suspect.empty:
        summary = suspect.groupby(SUSPECT_PAIR_KEY, dropna=False).agg(
            winner_count=("winner_id", "nunique"),
            source_count=("source_origin", "nunique"),
        )
        conflict_groups = int(summary["winner_count"].gt(1).sum())

    cross_source_duplicates = 0
    same_source_duplicate_groups = 0
    if confirmed_duplicates:
        duplicated = frame.loc[
            frame.duplicated(CONFIRMED_DUPLICATE_KEY, keep=False)
        ]
        duplicate_sources = duplicated.groupby(
            CONFIRMED_DUPLICATE_KEY,
            dropna=False,
        )["source_origin"].nunique()
        cross_source_duplicates = int(duplicate_sources.gt(1).sum())
        same_source_duplicate_groups = int(duplicate_sources.eq(1).sum())

    wrong_identity_rows = 0
    retired_confirmed_identity_rows = 0
    retired_ids = set(RETIRED_CONFIRMED_PLAYER_IDS)
    for side in ("winner", "loser"):
        normalized = frame[f"{side}_name"].map(normalized_person_name)
        ids = pd.to_numeric(frame[f"{side}_id"], errors="coerce")
        for alias, rule in CANONICAL_PLAYER_BY_NAME.items():
            wrong_identity_rows += int(
                (normalized.eq(alias) & ids.ne(rule["player_id"])).sum()
            )
        retired_confirmed_identity_rows += int(ids.isin(retired_ids).sum())

    canonical_name_counts = (
        frame.groupby("tournament_key", dropna=False)["canonical_tournament_name"]
        .nunique(dropna=True)
    )

    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "operational_duplicates": operational_duplicates,
        "confirmed_same_result_duplicate_rows": confirmed_duplicates,
        "same_source_duplicate_groups_kept": same_source_duplicate_groups,
        "cross_source_duplicates_remaining": cross_source_duplicates,
        "historical_opposite_winner_groups_kept": conflict_groups,
        "wrong_confirmed_identity_rows": wrong_identity_rows,
        "retired_confirmed_identity_rows": retired_confirmed_identity_rows,
        "tournament_keys_with_multiple_canonical_names": int(
            canonical_name_counts.gt(1).sum()
        ),
        "winner_id_nulls": int(frame["winner_id"].isna().sum()),
        "loser_id_nulls": int(frame["loser_id"].isna().sum()),
        "expected_cross_source_merges": int(expected_cross_source_merges),
        "max_tourney_date": int(
            pd.to_numeric(frame["tourney_date"], errors="coerce").max()
        ),
    }

    fatal = {
        "operational_duplicates": summary["operational_duplicates"],
        # Las repeticiones dentro de una sola fuente se conservan y auditan.
        # Solo los duplicados que siguen cruzando fuentes son un fallo fatal.
        "cross_source_duplicates_remaining": summary["cross_source_duplicates_remaining"],
        "wrong_confirmed_identity_rows": summary["wrong_confirmed_identity_rows"],
        "retired_confirmed_identity_rows": summary[
            "retired_confirmed_identity_rows"
        ],
        "tournament_keys_with_multiple_canonical_names": summary[
            "tournament_keys_with_multiple_canonical_names"
        ],
        "winner_id_nulls": summary["winner_id_nulls"],
        "loser_id_nulls": summary["loser_id_nulls"],
    }
    failing = {key: value for key, value in fatal.items() if value != 0}
    if failing:
        raise RuntimeError(f"Validación final fallida: {failing}")

    return summary


# ---------------------------------------------------------------------------
# Atomic output and public workflow
# ---------------------------------------------------------------------------


def write_parquet_atomically(
    frame: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    temporary = output_path.with_suffix(".postprocess.new.parquet")
    backup = output_path.with_suffix(".before_cross_source_dedup.parquet")
    temporary.unlink(missing_ok=True)

    frame.to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
        row_group_size=100_000,
    )

    verification = pd.read_parquet(
        temporary,
        columns=[
            "tourney_id", "competition_type", "match_num",
            "winner_id", "loser_id", "tournament_key",
        ],
        engine="pyarrow",
    )
    if len(verification) != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("La escritura temporal perdió filas")

    if output_path.exists() and not backup.exists():
        shutil.copy2(output_path, backup)

    temporary.replace(output_path)
    return backup if backup.exists() else None


def postprocess_parquet(
    parquet_path: Path,
    report_dir: Path,
    expected_cross_source_merges: int | None = None,
) -> dict[str, Any]:
    parquet_path = parquet_path.resolve()
    report_dir = report_dir.resolve()

    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"Leyendo Parquet integrado: {parquet_path}")
    frame = pd.read_parquet(parquet_path, engine="pyarrow")
    input_rows = len(frame)

    required = {
        "tourney_id", "tourney_name", "competition_type", "tourney_date",
        "round", "match_num", "winner_id", "winner_name", "loser_id",
        "loser_name", "source_origin",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    frame["competition_type"] = (
        frame["competition_type"].astype("string").str.strip().str.upper()
    )
    frame["source_origin"] = (
        frame["source_origin"].astype("string").str.strip().str.upper()
    )

    player_reference = build_player_reference(frame)
    frame, identity_report = repair_confirmed_identity_collisions(
        frame,
        player_reference,
    )
    print(f"Correcciones de identidad: {len(identity_report):,}")

    frame, canonical_tournaments = apply_tournament_canonicalization(frame)
    print(f"Ediciones canónicas: {len(canonical_tournaments):,}")

    frame, fusion_report, conflict_report = fuse_cross_source_duplicates(frame)
    merge_count = len(fusion_report)
    print(f"Pares Sackmann/TML fusionados: {merge_count:,}")

    if expected_cross_source_merges is not None and merge_count != expected_cross_source_merges:
        raise RuntimeError(
            "Número de fusiones inesperado: "
            f"esperado={expected_cross_source_merges:,}, obtenido={merge_count:,}"
        )

    frame, match_num_report = repair_match_num_collisions(frame)
    print(f"Colisiones match_num reparadas: {len(match_num_report):,}")

    # Orden determinista del Parquet final.
    frame = frame.sort_values(
        [
            "tourney_date", "tourney_id", "competition_type",
            "round", "match_num", "winner_id", "loser_id",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    validation = validate_final_frame(frame, merge_count)
    validation["input_rows"] = int(input_rows)
    validation["rows_removed_by_fusion"] = int(input_rows - len(frame))
    if identity_report.empty:
        validation["retired_identity_corrections"] = 0
    else:
        old_ids = pd.to_numeric(
            identity_report["old_player_id"], errors="coerce"
        )
        validation["retired_identity_corrections"] = int(
            old_ids.isin(set(RETIRED_CONFIRMED_PLAYER_IDS)).sum()
        )

    identity_report.to_csv(
        report_dir / "identity_corrections_upstream.csv",
        index=False,
    )
    fusion_report.to_csv(
        report_dir / "cross_source_match_fusions.csv",
        index=False,
    )
    conflict_report.to_csv(
        report_dir / "logical_match_conflicts_kept.csv",
        index=False,
    )
    match_num_report.to_csv(
        report_dir / "match_num_collisions_after_fusion.csv",
        index=False,
    )

    remaining_same_result = frame.loc[
        frame.duplicated(CONFIRMED_DUPLICATE_KEY, keep=False)
    ].copy()
    if not remaining_same_result.empty:
        remaining_same_result["duplicate_source_count"] = (
            remaining_same_result.groupby(
                CONFIRMED_DUPLICATE_KEY,
                dropna=False,
            )["source_origin"].transform("nunique")
        )
        remaining_same_result["review_status"] = np.where(
            remaining_same_result["duplicate_source_count"].eq(1),
            "kept_same_source_requires_review",
            "unexpected_cross_source_duplicate",
        )
    remaining_same_result.to_csv(
        report_dir / "same_result_duplicates_kept.csv",
        index=False,
    )
    canonical_tournaments.to_csv(
        report_dir / "canonical_tournament_names.csv",
        index=False,
    )
    player_reference.to_csv(
        report_dir / "player_reference_before_upstream_repair.csv",
        index=False,
    )

    backup = write_parquet_atomically(frame, parquet_path)
    validation["backup"] = str(backup) if backup else None
    validation["output"] = str(parquet_path)

    (report_dir / "upstream_postprocess_summary.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print("\n" + json.dumps(validation, indent=2, ensure_ascii=False, default=str))
    print(f"\nParquet reparado: {parquet_path}")
    if backup:
        print(f"Backup: {backup}")
    print(f"Informes: {report_dir}")
    return validation


def parse_args() -> argparse.Namespace:
    root = find_project_root(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(
        description="Repara identidades, duplicados cross-source y nombres de torneo"
    )
    parser.add_argument(
        "--parquet",
        default=str(root / "data" / "processed" / "jeff_sackmann_with_odds.parquet"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(root / "data" / "processed" / "tml_integration_reports"),
    )
    parser.add_argument(
        "--expected-cross-source-merges",
        type=int,
        default=2742,
        help="Protección basada en la auditoría actual. Use -1 para desactivar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = (
        None
        if args.expected_cross_source_merges < 0
        else args.expected_cross_source_merges
    )
    postprocess_parquet(
        Path(args.parquet),
        Path(args.report_dir),
        expected_cross_source_merges=expected,
    )


if __name__ == "__main__":
    main()
