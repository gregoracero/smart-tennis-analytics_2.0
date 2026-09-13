#!/usr/bin/env python3
"""Postprocesa la integración Sackmann + TML de forma conservadora.

Acciones:
- Repara mojibake en nombres.
- Aplica alias explícitos y auditables.
- Reconcilia IDs sintéticos contra identidades Sackmann usando nombre, IOC y edad.
- Reclasifica rondas no qualifying cuando corresponde.
- Normaliza superficies, niveles y tipos.
- Convierte ceros imposibles en estadísticas de servicio a valores ausentes.
- Elimina duplicados lógicos de forma determinista.
- Repara colisiones de match_num.
- Regenera dimensiones e informes de calidad.
- Escribe el Parquet de forma atómica y conserva un backup.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

SYNTHETIC_ID_MIN = 90_000_000

PARTICLES = {
    "de", "del", "den", "der", "van", "von", "da", "dos", "di", "la", "le"
}

PLAYER_NAME_ALIASES = {
    "aleksandr shevchenko": "alexander shevchenko",
    "faris zakaria": "fares zakaria",
    "faris zakaryia": "fares zakaria",
    "abedallah shelbayh": "abdullah shelbayh",
    "gabi adrian boitan": "adrian boitan",
    "olukayode alafia damina ayeni": "alafia ayeni",
    "alan sau franco": "alan raul sau franco",
    "christopher oconnell": "christopher o connell",
    "diego dedura palomero": "diego dedura",
    "felipe virgili berini": "felipe virgili",
    "ignacio parisca": "ignacio parisca romera",
    "igor ribeiro marcondes": "igor marcondes",
    "jay dylan hara friend": "jay friend",
    "joaquin aguilar cardozo": "joaquin aguilar cardozo",
    "herbert jody maginley": "jody maginley",
    "johan alexander rodriguez rodriguez": "johan alexander rodriguez",
    "kaan isak kosaner": "kaan isik kosaner",
    "khumoun sultanov": "khumoyun sultanov",
    "murkel alejandro dellien velasco": "murkel dellien",
    "mykyta mashtakov": "nikita mashtakov",
    "roberto cid": "roberto cid subervi",
    "san hui sin": "sanhui shin",
    "santiago fa rodriguez taverna": "santiago rodriguez taverna",
    "seong chan hong": "seongchan hong",
    "soon woo kwon": "soonwoo kwon",
}

IOC_EQUIVALENTS = {
    "URY": "URU",
    "URU": "URU",
    "PRY": "PAR",
    "PAR": "PAR",
    "ATG": "ANT",
    "ANT": "ANT",
    "CAR": "ANT",
}

# Mappings confirmed from the reconciliation table supplied on 2026-09-06.
# Ambiguous rows are intentionally excluded and remain synthetic for review.
AUDITED_SYNTHETIC_ID_MAP = {
    95206237: (209406, "Abedallah Shelbayh"),
    91926606: (206589, "Gabi Adrian Boitan"),
    95954860: (200366, "Olukayode Alafia Damina Ayeni"),
    95024766: (206827, "Alan Sau Franco"),
    95655904: (211346, "Andres Martin"),
    92150983: (207736, "Arklon Huertas Del Pino"),
    92364176: (210044, "Bruno Kuzuhara"),
    92347820: (106331, "Christopher Oconnell"),
    94931935: (202297, "Dan Martin"),
    95593554: (202369, "Daniel Salazar Martinez"),
    95570815: (212309, "Diego Dedura Palomero"),
    95569438: (209403, "Faris Zakaryia"),
    93111985: (212854, "Felipe Virgili Berini"),
    97993706: (105714, "Hugo Nys"),
    98745961: (212306, "Ignacio Parisca"),
    96664031: (127108, "Igor Ribeiro Marcondes"),
    93231725: (213605, "Jack Kennedy"),
    94381856: (210221, "Jay Dylan Hara Friend"),
    92831549: (212051, "Joaquin Aguilar Cardozo"),
    95391857: (200692, "Herbert Jody Maginley"),
    90970329: (208909, "Johan Alexander Rodriguez Rodriguez"),
    93541759: (213933, "Kaan Isak Kosaner"),
    94239780: (126939, "Khumoun Sultanov"),
    93049811: (210510, "Lautaro Midon"),
    90735975: (149117, "Lorenzo Angelini"),
    98344440: (210747, "Lui Maxted"),
    97194414: (212021, "Martin Landaluce"),
    90773735: (123961, "Murkel Alejandro Dellien Velasco"),
    94064806: (200125, "Mykyta Mashtakov"),
    93899342: (106232, "Roberto Cid"),
    92989429: (212837, "Roger Pascual Ferra"),
    94097102: (200321, "San Hui Sin"),
    97862782: (144973, "Santiago Fa Rodriguez Taverna"),
    92640798: (111805, "Seong Chan Hong"),
    98573986: (126952, "Soon Woo Kwon"),
}

SURFACE_MAP = {
    "hard": "Hard",
    "clay": "Clay",
    "grass": "Grass",
    "carpet": "Carpet",
    "unknown": "Unknown",
}

MAIN_LEVEL_MAP = {
    "250": "A",
    "500": "A",
    "A": "A",
    "M": "M",
    "G": "G",
    "D": "D",
    "F": "F",
    "O": "O",
}

PROFILE_COLUMNS = [
    "player_id", "player_name", "player_ioc", "player_age", "player_rank",
    "tourney_date", "source_origin",
]


def repair_mojibake(value):
    if value is None or pd.isna(value):
        return value
    text = str(value)
    if not any(marker in text for marker in ("Ã", "Â", "â€")):
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def ascii_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalized_name(value) -> str:
    key = " ".join(
        part for part in ascii_text(repair_mojibake(value)).split()
        if part not in PARTICLES
    )
    return PLAYER_NAME_ALIASES.get(key, key)


def normalized_ioc(value) -> str:
    if value is None or pd.isna(value):
        return ""
    key = str(value).strip().upper()
    return IOC_EQUIVALENTS.get(key, key)


def token_name(value) -> str:
    """Order-insensitive key, useful for surname-first provider variants."""
    return " ".join(sorted(normalized_name(value).split()))


def estimate_birth_year(tourney_date, age) -> float:
    if pd.isna(tourney_date) or pd.isna(age):
        return np.nan
    text = str(int(tourney_date))
    if len(text) < 4:
        return np.nan
    return float(int(text[:4]) - float(age))


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_matches(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"Leyendo Parquet: {path}")
    frame = pd.read_parquet(path, engine="pyarrow")
    print(f"Filas leídas: {len(frame):,}; columnas: {len(frame.columns):,}")
    return frame


def normalize_core(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    for column in ("winner_name", "loser_name"):
        result[column] = result[column].map(repair_mojibake).astype("string").str.strip()

    result["surface"] = (
        result["surface"].astype("string").str.strip().str.casefold()
        .map(SURFACE_MAP).fillna("Unknown")
    )

    result["round"] = result["round"].astype("string").str.strip().str.upper()
    result["competition_type"] = result["competition_type"].astype("string").str.strip().str.upper()
    result["tourney_level"] = result["tourney_level"].astype("string").str.strip().str.upper()

    tml_main = result["source_origin"].astype("string").eq("TML") & result["competition_type"].isin(
        ["ATP", "ATP_QUALIFYING"]
    )
    result.loc[tml_main, "tourney_level"] = (
        result.loc[tml_main, "tourney_level"].map(MAIN_LEVEL_MAP)
        .fillna(result.loc[tml_main, "tourney_level"])
    )
    result.loc[result["competition_type"].eq("CHALLENGER"), "tourney_level"] = "C"

    for column in ("winner_ioc", "loser_ioc", "winner_hand", "loser_hand"):
        if column in result.columns:
            result[column] = result[column].astype("string").str.strip().str.upper()

    return result


def player_observations(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side in ("winner", "loser"):
        part = pd.DataFrame({
            "player_id": frame[f"{side}_id"],
            "player_name": frame[f"{side}_name"],
            "player_ioc": frame.get(f"{side}_ioc"),
            "player_age": frame.get(f"{side}_age"),
            "player_rank": frame.get(f"{side}_rank"),
            "tourney_date": frame["tourney_date"],
            "source_origin": frame["source_origin"],
        })
        parts.append(part)

    players = pd.concat(parts, ignore_index=True)
    players["player_id"] = safe_numeric(players["player_id"]).astype("Int64")
    players["player_age"] = safe_numeric(players["player_age"])
    players["player_rank"] = safe_numeric(players["player_rank"])
    players["tourney_date"] = safe_numeric(players["tourney_date"]).astype("Int64")
    players["player_name"] = players["player_name"].map(repair_mojibake).astype("string").str.strip()
    players["name_key"] = players["player_name"].map(normalized_name)
    players["token_name_key"] = players["player_name"].map(token_name)
    players["player_ioc"] = players["player_ioc"].map(normalized_ioc).astype("string")
    players["estimated_birth_year"] = [
        estimate_birth_year(date, age)
        for date, age in zip(players["tourney_date"], players["player_age"])
    ]
    return players.dropna(subset=["player_id", "player_name"])


def most_common_nonempty(series: pd.Series):
    values = series.dropna().astype("string").str.strip()
    values = values.loc[values.ne("") & values.ne("<NA>")]
    return values.mode().iloc[0] if not values.empty else pd.NA


def build_profiles(players: pd.DataFrame) -> pd.DataFrame:
    """Create one identity profile per ID, retaining all observed spellings."""
    ordered = players.sort_values("tourney_date", ascending=False, kind="mergesort")
    profiles = (
        ordered.groupby("player_id", as_index=False, dropna=False)
        .agg(
            player_name=("player_name", "first"),
            player_ioc=("player_ioc", most_common_nonempty),
            latest_age=("player_age", "first"),
            median_age=("player_age", "median"),
            estimated_birth_year=("estimated_birth_year", "median"),
            best_rank=("player_rank", "min"),
            appearances=("player_id", "size"),
            latest_date=("tourney_date", "max"),
            first_date=("tourney_date", "min"),
        )
    )
    variants = ordered.groupby("player_id")["player_name"].agg(
        lambda values: "|".join(dict.fromkeys(str(v) for v in values.dropna()))
    )
    profiles["name_variants"] = profiles["player_id"].map(variants)
    profiles["name_key"] = profiles["player_name"].map(normalized_name)
    profiles["token_name_key"] = profiles["player_name"].map(token_name)
    profiles["is_synthetic"] = profiles["player_id"].ge(SYNTHETIC_ID_MIN)
    return profiles


def score_candidate(source: pd.Series, candidate: pd.Series) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    source_key = str(source["name_key"])
    candidate_key = str(candidate["name_key"])
    source_tokens = str(source["token_name_key"])
    candidate_tokens = str(candidate["token_name_key"])
    ordered_similarity = 100.0 * SequenceMatcher(None, source_key, candidate_key).ratio()
    token_similarity = 100.0 * SequenceMatcher(None, source_tokens, candidate_tokens).ratio()
    similarity = max(ordered_similarity, token_similarity)

    if source_key == candidate_key:
        score += 72.0
        reasons.append("exact_normalized_name")
    elif source_tokens == candidate_tokens:
        score += 68.0
        reasons.append("exact_token_name")
    elif source_key in candidate_key or candidate_key in source_key:
        shorter = min(len(source_key), len(candidate_key))
        longer = max(len(source_key), len(candidate_key))
        containment = shorter / longer if longer else 0.0
        if containment >= 0.55:
            score += 52.0
            reasons.append(f"name_containment_{containment:.2f}")
        else:
            return -999.0, [f"name_similarity_{similarity:.1f}"]
    elif similarity >= 88.0:
        score += 46.0
        reasons.append(f"fuzzy_name_{similarity:.1f}")
    else:
        return -999.0, [f"name_similarity_{similarity:.1f}"]

    source_ioc = normalized_ioc(source.get("player_ioc"))
    candidate_ioc = normalized_ioc(candidate.get("player_ioc"))
    if source_ioc and candidate_ioc:
        if source_ioc == candidate_ioc:
            score += 20.0
            reasons.append("same_ioc")
        else:
            score -= 55.0
            reasons.append("different_ioc")

    source_birth = source.get("estimated_birth_year")
    candidate_birth = candidate.get("estimated_birth_year")
    if pd.notna(source_birth) and pd.notna(candidate_birth):
        difference = abs(float(source_birth) - float(candidate_birth))
        if difference <= 0.75:
            score += 18.0
            reasons.append(f"birth_year_diff_{difference:.2f}")
        elif difference <= 1.5:
            score += 7.0
            reasons.append(f"birth_year_diff_{difference:.2f}")
        else:
            score -= 45.0
            reasons.append(f"birth_year_diff_{difference:.2f}")

    source_rank = source.get("best_rank")
    candidate_rank = candidate.get("best_rank")
    if pd.notna(source_rank) and pd.notna(candidate_rank):
        ratio = max(float(source_rank), float(candidate_rank)) / max(
            min(float(source_rank), float(candidate_rank)), 1.0
        )
        if ratio <= 1.35:
            score += 8.0
            reasons.append(f"compatible_rank_{ratio:.2f}")
        elif ratio >= 4.0:
            score -= 12.0
            reasons.append(f"incompatible_rank_{ratio:.2f}")

    appearances = int(candidate.get("appearances", 0) or 0)
    if appearances >= 20:
        score += 8.0
        reasons.append("established_history")
    elif appearances >= 5:
        score += 4.0
        reasons.append("sufficient_history")
    return score, reasons


def audited_map(profiles: pd.DataFrame) -> pd.DataFrame:
    real_ids = set(
        profiles.loc[~profiles["is_synthetic"], "player_id"].dropna().astype(int)
    )
    rows = []
    for alias_id, (canonical_id, canonical_name) in AUDITED_SYNTHETIC_ID_MAP.items():
        if canonical_id not in real_ids:
            continue
        rows.append({
            "alias_player_id": alias_id,
            "canonical_player_id": canonical_id,
            "canonical_name": canonical_name,
            "mapping_reason": "audited_from_reconciliation_table_2026_09_06",
            "confidence": "audited",
            "score": 999.0,
            "margin_to_second": 999.0,
        })
    return pd.DataFrame(rows)


def propose_canonical_map(profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    synthetic_profiles = profiles.loc[profiles["is_synthetic"]].copy()
    real_profiles = profiles.loc[~profiles["is_synthetic"]].copy()
    proposals: list[dict] = []
    accepted: list[dict] = []

    for _, source in synthetic_profiles.iterrows():
        if int(source["player_id"]) in AUDITED_SYNTHETIC_ID_MAP:
            continue
        source_ioc = normalized_ioc(source.get("player_ioc"))
        pool = real_profiles.copy()
        if source_ioc:
            same_ioc = pool["player_ioc"].map(normalized_ioc).eq(source_ioc)
            if same_ioc.any():
                pool = pool.loc[same_ioc]

        similarities = pool.apply(
            lambda row: max(
                SequenceMatcher(None, str(source["name_key"]), str(row["name_key"])).ratio(),
                SequenceMatcher(None, str(source["token_name_key"]), str(row["token_name_key"])).ratio(),
            ),
            axis=1,
        )
        containment = pool["name_key"].map(
            lambda value: str(source["name_key"]) in str(value)
            or str(value) in str(source["name_key"])
        )
        candidates = pool.loc[similarities.ge(0.70) | containment].copy()

        scored = []
        for _, candidate in candidates.iterrows():
            score, reasons = score_candidate(source, candidate)
            if score > -900:
                scored.append((score, reasons, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)

        for position, (score, reasons, candidate) in enumerate(scored[:5], start=1):
            proposals.append({
                "synthetic_player_id": int(source["player_id"]),
                "synthetic_name": source["player_name"],
                "synthetic_ioc": source["player_ioc"],
                "synthetic_age": source["latest_age"],
                "synthetic_rank": source["best_rank"],
                "candidate_position": position,
                "candidate_player_id": int(candidate["player_id"]),
                "candidate_name": candidate["player_name"],
                "candidate_ioc": candidate["player_ioc"],
                "candidate_age": candidate["latest_age"],
                "candidate_rank": candidate["best_rank"],
                "score": round(score, 2),
                "reasons": "|".join(reasons),
            })

        if not scored:
            continue
        best_score, best_reasons, best_candidate = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -999.0
        margin = best_score - second_score
        strong_identity = any(
            reason.startswith(("exact_normalized", "exact_token", "name_containment"))
            for reason in best_reasons
        )
        safe = (
            best_score >= 88.0
            and margin >= 12.0
            and strong_identity
            and "different_ioc" not in best_reasons
            and int(best_candidate["player_id"]) < SYNTHETIC_ID_MIN
        )
        if safe:
            accepted.append({
                "alias_player_id": int(source["player_id"]),
                "canonical_player_id": int(best_candidate["player_id"]),
                "canonical_name": best_candidate["player_name"],
                "mapping_reason": "|".join(best_reasons),
                "confidence": "high",
                "score": round(best_score, 2),
                "margin_to_second": round(margin, 2),
            })

    auto = pd.DataFrame(accepted)
    audited = audited_map(profiles)
    if audited.empty:
        return auto, pd.DataFrame(proposals)
    combined = pd.concat([audited, auto], ignore_index=True)
    combined = combined.drop_duplicates("alias_player_id", keep="first")
    return combined, pd.DataFrame(proposals)

def apply_canonical_map(frame: pd.DataFrame, canonical_map: pd.DataFrame) -> pd.DataFrame:
    if canonical_map.empty:
        return frame

    result = frame.copy()
    id_map = dict(zip(canonical_map["alias_player_id"], canonical_map["canonical_player_id"]))
    name_map = dict(zip(canonical_map["alias_player_id"], canonical_map["canonical_name"]))

    for side in ("winner", "loser"):
        ids = safe_numeric(result[f"{side}_id"]).astype("Int64")
        alias_mask = ids.isin(id_map)
        old_ids = ids.copy()
        result.loc[alias_mask, f"{side}_id"] = ids.loc[alias_mask].map(id_map).astype("Int64")
        result.loc[alias_mask, f"{side}_name"] = old_ids.loc[alias_mask].map(name_map).astype("string")

    return result


def reclassify_qualifying_rounds(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    unexpected = (
        result["source_origin"].astype("string").eq("TML")
        & result["competition_type"].eq("ATP_QUALIFYING")
        & ~result["round"].isin(["Q1", "Q2", "Q3"])
    )

    audit = result.loc[unexpected, [
        "tourney_id", "tourney_name", "tourney_date", "match_num",
        "winner_name", "loser_name", "round", "competition_type",
    ]].copy()

    if not audit.empty:
        result.loc[unexpected, "competition_type"] = "ATP"
        audit["new_competition_type"] = "ATP"
        audit["reason"] = "main_draw_round_in_tml_qualifying_file"

    return result, audit


def clean_impossible_statistics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()

    # Las correcciones de calidad de esta fase se limitan a TennisMyLife.
    # El histórico Sackmann conserva sus convenciones originales.
    tml_mask = (
        result["source_origin"]
        .astype("string")
        .eq("TML")
    )
    if "stats_quality_flag" not in result.columns:
        result["stats_quality_flag"] = pd.Series(pd.NA, index=result.index, dtype="string")

    audit_parts = []

    for column in ("w_svpt", "l_svpt", "w_SvGms", "l_SvGms"):
        values = safe_numeric(result[column])
        mask = (
            tml_mask
            & values.le(0)
            & result["score"].astype("string").notna()
        )
        if mask.any():
            audit_parts.append(pd.DataFrame({
                "row_index": result.index[mask],
                "tourney_id": result.loc[mask, "tourney_id"].to_numpy(),
                "match_num": result.loc[mask, "match_num"].to_numpy(),
                "issue": f"non_positive_{column}",
                "value": result.loc[mask, column].to_numpy(),
            }))
            result.loc[mask, column] = pd.NA

    checks = [
        ("w_1stIn", "w_svpt", "winner_first_in_gt_service_points"),
        ("l_1stIn", "l_svpt", "loser_first_in_gt_service_points"),
        ("w_bpSaved", "w_bpFaced", "winner_bp_saved_gt_faced"),
        ("l_bpSaved", "l_bpFaced", "loser_bp_saved_gt_faced"),
    ]

    for left, right, issue in checks:
        left_values = safe_numeric(result[left])
        right_values = safe_numeric(result[right])
        mask = (
            tml_mask
            & left_values.notna()
            & right_values.notna()
            & left_values.gt(right_values)
        )
        if not mask.any():
            continue

        audit_parts.append(pd.DataFrame({
            "row_index": result.index[mask],
            "tourney_id": result.loc[mask, "tourney_id"].to_numpy(),
            "match_num": result.loc[mask, "match_num"].to_numpy(),
            "issue": issue,
            "left_value": result.loc[mask, left].to_numpy(),
            "right_value": result.loc[mask, right].to_numpy(),
        }))

        current = result.loc[mask, "stats_quality_flag"].fillna("").astype(str)
        result.loc[mask, "stats_quality_flag"] = np.where(
            current.eq(""), issue, current + "|" + issue
        )
        result.loc[mask, [left, right]] = pd.NA

    audit = pd.concat(audit_parts, ignore_index=True, sort=False) if audit_parts else pd.DataFrame()
    return result, audit


def logical_key(frame: pd.DataFrame) -> pd.Series:
    return (
        safe_numeric(frame["tourney_date"]).astype("Int64").astype("string") + "|"
        + frame["tourney_name"].map(ascii_text).astype("string") + "|"
        + frame["winner_name"].map(normalized_name).astype("string") + "|"
        + frame["loser_name"].map(normalized_name).astype("string") + "|"
        + frame["round"].astype("string").fillna("") + "|"
        + frame["competition_type"].astype("string").fillna("")
    )


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    result["_logical_key"] = logical_key(result)
    result["_row_order"] = np.arange(len(result), dtype=np.int64)
    result["_completeness"] = result[[
        "minutes", "w_svpt", "l_svpt", "w_ace", "l_ace", "winner_rank", "loser_rank"
    ]].notna().sum(axis=1).astype("int8")
    priority = safe_numeric(result.get("source_priority", pd.Series(0, index=result.index))).fillna(0)
    result["_priority"] = priority

    result = result.sort_values(
        ["_logical_key", "_priority", "_completeness", "_row_order"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    duplicate_mask = result.duplicated("_logical_key", keep="first")
    removed = result.loc[duplicate_mask].copy()
    result = result.loc[~duplicate_mask].copy()

    helper_columns = ["_logical_key", "_row_order", "_completeness", "_priority"]
    return result.drop(columns=helper_columns).reset_index(drop=True), removed.drop(columns=helper_columns).reset_index(drop=True)


def repair_match_numbers(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    if "source_match_num" not in result.columns:
        result["source_match_num"] = result["match_num"]

    key = ["tourney_id", "competition_type", "match_num"]
    collision_mask = result.duplicated(key, keep=False) & result["match_num"].notna()
    audit = result.loc[collision_mask, key + ["winner_name", "loser_name", "round"]].copy()

    if not audit.empty:
        for _, indices in result.groupby(["tourney_id", "competition_type"], dropna=False).groups.items():
            group = result.loc[list(indices)]
            dup = group["match_num"].duplicated(keep=False) & group["match_num"].notna()
            if not dup.any():
                continue
            maximum = safe_numeric(group["match_num"]).max()
            next_number = int(maximum) + 1 if pd.notna(maximum) else 1
            for _, same_number in group.loc[dup].groupby("match_num", dropna=False):
                for row_index in same_number.index[1:]:
                    result.at[row_index, "match_num"] = next_number
                    next_number += 1

    result["match_num"] = safe_numeric(result["match_num"]).astype("Int64")
    return result, audit


def build_player_dimension(frame: pd.DataFrame) -> pd.DataFrame:
    players = player_observations(frame)
    players = players.sort_values(
        ["source_origin", "tourney_date"], ascending=[False, False], kind="mergesort"
    )
    return (
        players.drop_duplicates("player_id", keep="first")
        [["player_id", "player_name", "player_ioc"]]
        .sort_values("player_name", kind="mergesort")
        .reset_index(drop=True)
    )


def build_tournament_dimension(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "tourney_id", "tourney_name", "surface", "tourney_level",
        "tourney_level_original", "competition_type", "indoor", "tourney_date",
    ]
    columns = [column for column in columns if column in frame.columns]
    result = frame[columns].copy()
    result = result.sort_values("tourney_date", ascending=False, kind="mergesort")
    return result.drop_duplicates(["tourney_id", "competition_type"], keep="first").reset_index(drop=True)


def atomic_write(frame: pd.DataFrame, output_path: Path, keep_backup: bool) -> Path | None:
    temporary = output_path.with_suffix(".new.parquet")
    backup = output_path.with_suffix(".before_reconciliation.parquet")

    frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
    verification = pd.read_parquet(temporary, columns=[
        "tourney_id", "competition_type", "match_num", "winner_id", "loser_id"
    ])

    if len(verification) != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("La verificación del Parquet temporal no conserva todas las filas.")

    if verification.duplicated(["tourney_id", "competition_type", "match_num"]).any():
        temporary.unlink(missing_ok=True)
        raise RuntimeError("El Parquet temporal contiene claves de partido duplicadas.")

    if verification[["winner_id", "loser_id"]].isna().any().any():
        temporary.unlink(missing_ok=True)
        raise RuntimeError("El Parquet temporal contiene IDs de jugador nulos.")

    if keep_backup and output_path.exists() and not backup.exists():
        shutil.copy2(output_path, backup)

    temporary.replace(output_path)
    return backup if backup.exists() else None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--no-backup", action="store_true")
    parser.add_argument("--max-synthetic-players", type=int, default=15)
    parser.add_argument("--allow-more-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    parquet_path = Path(args.parquet).resolve()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    matches = normalize_core(load_matches(parquet_path))
    profiles_before = build_profiles(player_observations(matches))
    canonical_map, candidates = propose_canonical_map(profiles_before)

    canonical_map.to_csv(report_dir / "player_id_canonical_map.csv", index=False)
    candidates.to_csv(report_dir / "synthetic_player_candidates.csv", index=False)

    matches = apply_canonical_map(matches, canonical_map)
    matches, round_audit = reclassify_qualifying_rounds(matches)
    matches, statistics_audit = clean_impossible_statistics(matches)
    matches, removed_duplicates = deduplicate(matches)
    matches, match_num_audit = repair_match_numbers(matches)

    players = build_player_dimension(matches)
    tournaments = build_tournament_dimension(matches)
    profiles_after = build_profiles(player_observations(matches))
    remaining_synthetic = profiles_after.loc[profiles_after["is_synthetic"]].copy()

    round_audit.to_csv(report_dir / "tml_round_reclassification.csv", index=False)
    statistics_audit.to_csv(report_dir / "tml_invalid_statistics_postprocess.csv", index=False)
    removed_duplicates.to_csv(report_dir / "tml_duplicate_matches_postprocess.csv", index=False)
    match_num_audit.to_csv(report_dir / "tml_match_num_collisions_postprocess.csv", index=False)
    remaining_synthetic.to_csv(report_dir / "remaining_synthetic_players.csv", index=False)
    players.to_csv(report_dir / "player_dimension.csv", index=False)
    tournaments.to_csv(report_dir / "tournament_dimension.csv", index=False)

    synthetic_count = int(remaining_synthetic["player_id"].nunique())
    summary = {
        "output_rows": int(len(matches)),
        "output_columns": int(len(matches.columns)),
        "unique_logical_matches": int(logical_key(matches).nunique()),
        "duplicate_operational_keys": int(matches.duplicated(["tourney_id", "competition_type", "match_num"]).sum()),
        "unique_players": int(players["player_id"].nunique()),
        "unique_tournament_editions": int(tournaments[["tourney_id", "competition_type"]].drop_duplicates().shape[0]),
        "canonical_mappings_applied": int(len(canonical_map)),
        "remaining_synthetic_players": synthetic_count,
        "rounds_reclassified": int(len(round_audit)),
        "invalid_statistics_corrected": int(len(statistics_audit)),
        "duplicates_removed_postprocess": int(len(removed_duplicates)),
        "match_num_collisions_repaired": int(len(match_num_audit)),
    }

    (report_dir / "reconciliation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if synthetic_count > args.max_synthetic_players and not args.allow_more_synthetic:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise RuntimeError(
            f"Quedan {synthetic_count} jugadores sintéticos, por encima del límite "
            f"{args.max_synthetic_players}. Revisa remaining_synthetic_players.csv y "
            "player_id_canonical_map.csv. Usa --allow-more-synthetic únicamente si aceptas el resultado."
        )

    backup = atomic_write(matches, parquet_path, keep_backup=not args.no_backup)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Parquet actualizado: {parquet_path}")
    if backup:
        print(f"Backup: {backup}")
    print(f"Informes: {report_dir}")


if __name__ == "__main__":
    main()
