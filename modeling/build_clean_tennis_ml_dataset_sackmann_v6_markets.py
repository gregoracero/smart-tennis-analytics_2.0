#!/usr/bin/env python3
"""Construye datasets ML temporales, limpios y optimizados para tenis.

Entrada predeterminada
----------------------
data/processed/tennis_matches_with_player_stats.sackmann_v5.parquet

Salidas predeterminadas
-----------------------
data/processed/tennis_ml_no_market.sackmann_v5.parquet
data/processed/tennis_ml_with_market.sackmann_v5.parquet
data/processed/tennis_ml_dataset_report_sackmann_v5.csv
data/processed/tennis_ml_feature_manifest_sackmann_v5.csv
data/processed/tennis_ml_dataset_metadata_sackmann_v5.json

Principios
----------
- Usa exclusivamente filas marcadas como elegibles, salvo modo de auditoria.
- Mantiene splits cronologicos y no realiza imputacion global.
- Usa una lista blanca estricta de variables disponibles antes del partido.
- Incluye contexto conocido antes del partido: superficie, nivel, competicion,
  indoor, best-of y ronda.
- Incluye features globales, por superficie, contexto historico, tendencia Elo
  y calidad reciente de rivales generadas por el feature builder optimizado.
- Conserva IDs, nombres y trazabilidad solo para auditoria, nunca como X.
- Genera datasets no-market y with-market.
- Mantiene un perfil baseline para comparar el conjunto anterior de features
  con el conjunto optimizado sin reconstruir otro Parquet.
- No mezcla player_current_state.parquet con el entrenamiento historico.
- Escribe y valida las salidas de forma atomica.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TARGET = "target_player_1_win"
DATE = "match_date"
SPLIT = "split"
SYNTHETIC_ID_MIN = 90_000_000
FEATURE_SCHEMA_VERSION = "sackmann_v6_temporal_multi_market_targets"

DEFAULT_COMPETITION_TYPES = ["ATP", "ATP_QUALIFYING", "CHALLENGER"]

# Se conservan para auditoria. SAFE_MATCH_CONTEXT_FEATURES es la unica parte
# del contexto que puede entrar en X porque se conoce antes del encuentro.
AUDIT_CONTEXT_COLUMNS = [
    "processing_sequence",
    "tourney_id", "tourney_name", "canonical_tournament_name",
    "tournament_key", "tourney_date", "match_num", "source_match_num",
    "surface", "draw_size", "tourney_level", "tourney_level_original",
    "td_date",
    "competition_type", "indoor", "best_of", "round", DATE,
]
SAFE_MATCH_CONTEXT_FEATURES = [
    "surface", "tourney_level", "competition_type", "indoor", "best_of", "round",
]
TRACEABILITY_COLUMNS = [
    "source_origin", "source_priority", "source_sackmann_file", "source_tml_file",
    "winner_id_tml", "loser_id_tml", "stats_quality_flag", "eligible_for_model",
    "model_exclusion_reason", "source_merged", "source_duplicate_count",
]
PLAYER_IDENTITY_COLUMNS = [
    "player_1_id", "player_2_id", "player_1_name", "player_2_name",
]
# Post-match source columns retained exclusively to construct and audit secondary
# targets. They are never part of no_market_features or market_feature_columns.
# The production market trainer uses them to derive serve-point probabilities,
# set outcomes and total games while preserving the original chronological split.
MARKET_TARGET_SOURCE_COLUMNS = [
    "score",
    "w_svpt", "w_1stWon", "w_2ndWon",
    "l_svpt", "l_1stWon", "l_2ndWon",
]
PLAYER_STATIC_COLUMNS = [
    "player_1_hand", "player_2_hand", "player_1_height", "player_2_height",
    "player_1_ioc", "player_2_ioc", "player_1_age", "player_2_age",
    "player_1_rank", "player_2_rank", "player_1_rank_points",
    "player_2_rank_points",
]

BASELINE_PREMATCH_SUFFIXES = [
    "career_matches_before", "career_wins_before", "career_win_rate_before",
    "surface_matches_before", "surface_wins_before", "surface_win_rate_before",
    "elo_before", "surface_elo_before", "form_last_5_before",
    "form_last_10_before", "avg_minutes_last_5_before", "days_since_last_match",
    "stat_matches_before", "aces_per_service_game_before",
    "double_faults_per_service_game_before", "first_serve_in_pct_before",
    "first_serve_win_pct_before", "second_serve_win_pct_before",
    "service_points_won_pct_before", "break_points_saved_pct_before",
    "return_points_won_pct_before",
]

OPTIMIZED_PREMATCH_SUFFIXES = BASELINE_PREMATCH_SUFFIXES + [
    # Forma, estadisticas y tendencia especificas de la superficie seleccionada.
    "surface_form_last_5_before", "surface_form_last_10_before",
    "surface_stat_matches_before", "surface_service_points_won_pct_before",
    "surface_return_points_won_pct_before",
    "surface_aces_per_service_game_before",
    "surface_double_faults_per_service_game_before",
    "surface_first_serve_in_pct_before",
    "surface_first_serve_win_pct_before",
    "surface_second_serve_win_pct_before",
    "surface_break_points_saved_pct_before",
    "surface_elo_change_last_5_before",
    # Tendencia global y calidad de rivales recientes.
    "elo_change_last_5_before", "avg_opponent_elo_last_5_before",
    "avg_opponent_rank_last_5_before", "quality_adjusted_form_last_5_before",
    # Experiencia historica en el contexto del partido.
    "level_matches_before", "level_wins_before", "level_win_rate_before",
    "round_matches_before", "round_wins_before", "round_win_rate_before",
    "best_of_matches_before", "best_of_wins_before", "best_of_win_rate_before",
]

H2H_COLUMNS = [
    "player_1_h2h_wins_before", "player_2_h2h_wins_before",
    "player_1_h2h_win_rate_before",
]

BASELINE_DIFF_COLUMNS = [
    "diff_career_matches_before", "diff_career_win_rate_before",
    "diff_surface_matches_before", "diff_surface_win_rate_before",
    "diff_elo_before", "diff_surface_elo_before", "diff_form_last_5_before",
    "diff_form_last_10_before", "diff_avg_minutes_last_5_before",
    "diff_days_since_last_match", "diff_aces_per_service_game_before",
    "diff_double_faults_per_service_game_before", "diff_first_serve_in_pct_before",
    "diff_first_serve_win_pct_before", "diff_second_serve_win_pct_before",
    "diff_service_points_won_pct_before", "diff_break_points_saved_pct_before",
    "diff_return_points_won_pct_before", "diff_rank", "diff_rank_points",
]
OPTIMIZED_DIFF_COLUMNS = BASELINE_DIFF_COLUMNS + [
    "diff_surface_form_last_5_before", "diff_surface_form_last_10_before",
    "diff_surface_stat_matches_before",
    "diff_surface_service_points_won_pct_before",
    "diff_surface_return_points_won_pct_before",
    "diff_surface_aces_per_service_game_before",
    "diff_surface_double_faults_per_service_game_before",
    "diff_surface_first_serve_in_pct_before",
    "diff_surface_first_serve_win_pct_before",
    "diff_surface_second_serve_win_pct_before",
    "diff_surface_break_points_saved_pct_before",
    "diff_surface_elo_change_last_5_before", "diff_elo_change_last_5_before",
    "diff_avg_opponent_elo_last_5_before",
    "diff_avg_opponent_rank_last_5_before",
    "diff_quality_adjusted_form_last_5_before", "diff_level_matches_before",
    "diff_level_win_rate_before", "diff_round_matches_before",
    "diff_round_win_rate_before", "diff_best_of_matches_before",
    "diff_best_of_win_rate_before",
]


MOMENTUM_SUFFIXES = [
    "momentum_ewma_30d_before", "momentum_ewma_90d_before",
    "surface_momentum_ewma_90d_before", "momentum_acceleration_before",
    "momentum_volatility_before", "performance_vs_expected_ewma_before",
    "games_won_share_last_5_before", "games_won_share_last_10_before",
    "surface_games_won_share_last_10_before", "sets_won_share_last_10_before",
    "straight_sets_rate_last_10_before", "deciding_set_rate_last_10_before",
    "tiebreak_win_rate_last_20_before",
    "opponent_adjusted_game_dominance_before",
    "surface_opponent_adjusted_game_dominance_before",
]

# Experimental opponent-adjusted Momentum v6 family. The temporal feature
# builder must materialize these columns before this clean-dataset builder runs.
MOMENTUM_V6_SUFFIXES = [
    "result_residual_ewma_30d_before",
    "result_residual_ewma_90d_before",
    "surface_result_residual_ewma_90d_before",
    "game_share_residual_ewma_30d_before",
    "game_share_residual_ewma_90d_before",
    "serve_residual_ewma_30d_before",
    "return_residual_ewma_30d_before",
    "surface_serve_residual_ewma_90d_before",
    "surface_return_residual_ewma_90d_before",
    "dominance_ratio_ewma_30d_before",
    "momentum_effective_sample_before",
    "momentum_stat_effective_sample_before",
    "momentum_confidence_before",
]
WORKLOAD_SUFFIXES = [
    "matches_last_3d_before", "matches_last_7d_before",
    "matches_last_14d_before", "matches_last_28d_before",
    "minutes_last_3d_before", "minutes_last_7d_before",
    "minutes_last_14d_before", "minutes_last_28d_before",
    "minutes_coverage_last_7d_before", "games_last_3d_before",
    "games_last_7d_before", "games_last_14d_before",
    "sets_last_3d_before", "sets_last_7d_before", "sets_last_14d_before",
    "deciding_sets_last_7d_before", "tiebreak_sets_last_7d_before",
    "consecutive_match_days_before", "rest_days_before",
    "rest_days_capped_30_before", "rest_days_squared_before",
    "workload_per_rest_day_before", "same_tournament_matches_before",
    "same_tournament_minutes_before",
    "same_tournament_wins_before", "same_tournament_losses_before",
    "same_tournament_win_rate_smoothed_before",
    "same_tournament_result_residual_before",
    "same_tournament_games_won_share_before",
    "same_tournament_sets_won_share_before",
    "same_tournament_effective_sample_before",
]
DYNAMIC_STATS_SUFFIXES = [
    "service_points_won_ewma_short_before",
    "service_points_won_ewma_long_before",
    "return_points_won_ewma_short_before",
    "return_points_won_ewma_long_before",
    "first_serve_in_ewma_before", "first_serve_win_ewma_before",
    "second_serve_win_ewma_before", "aces_per_service_game_ewma_before",
    "double_faults_per_service_game_ewma_before",
    "break_points_saved_ewma_before",
    "surface_service_points_won_ewma_before",
    "surface_return_points_won_ewma_before",
    "serve_return_strength_ewma_before",
    "surface_serve_return_strength_ewma_before",
    "opponent_adjusted_serve_strength_before",
    "opponent_adjusted_return_strength_before",
    "serve_form_change_before", "return_form_change_before",
]
RANKING_TREND_SUFFIXES = [
    "rank_4w_ago_before", "rank_8w_ago_before", "rank_26w_ago_before",
    "rank_change_4w_before", "rank_change_8w_before", "rank_change_26w_before",
    "rank_points_4w_ago_before", "rank_points_8w_ago_before",
    "rank_points_26w_ago_before", "rank_points_change_4w_before",
    "rank_points_change_8w_before", "rank_points_change_26w_before",
    "elo_change_30d_before", "elo_change_90d_before",
    "surface_elo_change_90d_before", "ranking_vs_elo_disagreement_before",
    "ranking_points_momentum_before", "career_high_rank_before",
    "rank_distance_from_career_high_before",
]
MATCHUP_V5_COLUMNS = [
    "h2h_matches_before", "h2h_games_won_share_player_1_before",
    "surface_h2h_matches_before", "surface_h2h_win_rate_player_1_before",
    "surface_h2h_games_won_share_player_1_before",
    "recent_h2h_win_rate_player_1_before",
    "recent_h2h_games_won_share_player_1_before",
    "common_opponents_count_before", "common_opponent_score_player_1_before",
    "surface_common_opponents_count_before",
    "surface_common_opponent_score_player_1_before",
    "intransitivity_score_player_1_before",
    "surface_intransitivity_score_player_1_before",
]
V5_PLAYER_DIFFERENTIALS = [
    "diff_" + suffix
    for suffix in (
        MOMENTUM_SUFFIXES + MOMENTUM_V6_SUFFIXES + WORKLOAD_SUFFIXES
        + DYNAMIC_STATS_SUFFIXES + RANKING_TREND_SUFFIXES
    )
    if suffix not in {
        "matches_last_3d_before", "matches_last_7d_before",
        "matches_last_14d_before", "matches_last_28d_before",
        "same_tournament_matches_before",
    }
]
V5_DERIVED_COLUMNS = [
    "dynamic_stats_available_player_1", "dynamic_stats_available_player_2",
    "dynamic_stats_available_both", "dynamic_stats_available_count",
    "rank_trend_available_player_1", "rank_trend_available_player_2",
    "rank_trend_available_both", "match_date_exact",
    "intransitivity_shrunk_before",
    "surface_intransitivity_shrunk_before",
    "log_common_opponents_count_before",
    "log_surface_common_opponents_count_before",
]
for _weeks in (4, 8, 26):
    V5_DERIVED_COLUMNS.extend([
        f"player_1_rank_change_{_weeks}w_before_clipped",
        f"player_2_rank_change_{_weeks}w_before_clipped",
        f"diff_rank_change_{_weeks}w_before_clipped",
        f"diff_rank_change_{_weeks}w_before_signed_log",
    ])

PROFILE_FAMILIES = {
    "baseline": [],
    "optimized_v4": [],
    "momentum": ["momentum"],
    "momentum_v6": ["momentum_v6"],
    "workload": ["workload"],
    "dynamic_stats": ["dynamic_stats"],
    "ranking_trends": ["ranking_trends"],
    "matchup": ["matchup"],
    "v5_all": [
        "momentum", "workload", "dynamic_stats", "ranking_trends", "matchup",
    ],
    "v6_all": [
        "momentum", "momentum_v6", "workload", "dynamic_stats",
        "ranking_trends", "matchup",
    ],
}

MARKET_COLUMNS = [
    "player_1_b365_odds", "player_2_b365_odds", "player_1_ps_odds",
    "player_2_ps_odds", "player_1_max_odds", "player_2_max_odds",
    "player_1_avg_odds", "player_2_avg_odds", "player_1_bfe_odds",
    "player_2_bfe_odds",
]
QUALITY_FEATURE_COLUMNS = [
    "min_career_matches_before", "min_stat_matches_before",
    "min_surface_matches_before", "min_surface_stat_matches_before",
    "cold_start_any_player", "surface_cold_start_any_player",
    "stats_reliable_both", "surface_stats_reliable_both",
    "synthetic_player_any",
]
DERIVED_SPORT_FEATURE_COLUMNS = [
    "elo_expected_p1", "surface_elo_expected_p1", "rank_ratio_p1_p2",
    "rank_points_ratio_p1_p2", "absolute_elo_difference",
    "absolute_surface_elo_difference",
    "intransitivity_shrunk_before", "surface_intransitivity_shrunk_before",
    "log_common_opponents_count_before",
    "log_surface_common_opponents_count_before",
] + [column for column in V5_DERIVED_COLUMNS if "rank_change" in column]

FORBIDDEN_PATTERNS = [
    "winner_", "loser_", "score", "w_ace", "l_ace", "w_df", "l_df",
    "w_svpt", "l_svpt", "w_1st", "l_1st", "w_2nd", "l_2nd",
    "w_svgms", "l_svgms", "w_bpsaved", "l_bpsaved", "w_bpfaced",
    "l_bpfaced", "td_winner", "td_loser", "td_w1", "td_l1", "td_wsets",
    "td_lsets", "td_comment", "comment", "odds_matched", "match_score",
    "source_", "id_tml", "exclusion_reason", "eligible_for_model",
]



SECONDARY_TARGET_COLUMNS = [
    "target_player_1_wins_first_set", "target_player_2_wins_first_set",
    "target_player_1_wins_any_set", "target_player_2_wins_any_set",
    "target_total_games", "target_valid_first_set", "target_valid_completed_score",
] + [f"target_over_{str(line).replace('.', '_')}" for line in (18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,36.5,38.5,40.5)]

def _parse_market_score(score: Any, best_of: Any, player_1_won_match: Any) -> dict[str, Any]:
    import re
    empty = {c: np.nan for c in SECONDARY_TARGET_COLUMNS}
    if score is None or pd.isna(score):
        return empty
    text = str(score).strip().upper().replace("–", "-").replace("—", "-")
    if re.search(r"\b(W/O|WO|WALKOVER)\b", text):
        return empty
    retired = bool(re.search(r"\b(RET|ABD|DEF|ABN|INT)\b", text))
    tokens = re.findall(r"(\d{1,2})-(\d{1,2})(?:\(\d{1,2}\))?", text)
    completed_sets=[]
    total_games=0
    for a0,b0 in tokens:
        a,b=int(a0),int(b0)
        complete=((max(a,b)>=6 and abs(a-b)>=2) or (max(a,b)==7 and min(a,b) in {5,6}) or (max(a,b)>=10 and abs(a-b)>=2))
        if complete:
            completed_sets.append((a,b))
            total_games += a+b
    out = dict(empty)
    if completed_sets:
        winner_won_first = int(completed_sets[0][0] > completed_sets[0][1])
        p1_match_winner = int(float(player_1_won_match)) == 1
        first_p1 = winner_won_first if p1_match_winner else 1 - winner_won_first
        out["target_player_1_wins_first_set"] = first_p1
        out["target_player_2_wins_first_set"] = 1-first_p1
        out["target_valid_first_set"] = 1
    try:
        best=int(float(best_of))
    except (TypeError, ValueError):
        best=3
    needed=best//2+1
    winner_sets=sum(a>b for a,b in completed_sets)
    loser_sets=sum(b>a for a,b in completed_sets)
    completed=(not retired and winner_sets>=needed)
    if completed:
        p1_match_winner = int(float(player_1_won_match)) == 1
        out["target_player_1_wins_any_set"] = int(winner_sets>=1) if p1_match_winner else int(loser_sets>=1)
        out["target_player_2_wins_any_set"] = int(loser_sets>=1) if p1_match_winner else int(winner_sets>=1)
        out["target_total_games"] = float(total_games)
        out["target_valid_completed_score"] = 1
        for line in (18.5,19.5,20.5,21.5,22.5,23.5,24.5,25.5,36.5,38.5,40.5):
            out[f"target_over_{str(line).replace('.', '_')}"] = int(total_games > line)
    return out

def add_secondary_targets(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = [_parse_market_score(score, best, target) for score,best,target in zip(frame["score"], frame["best_of"], frame[TARGET])]
    targets = pd.DataFrame(parsed, index=frame.index)
    return pd.concat([frame, targets], axis=1)

def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "data" / "processed").exists():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto")


def player_prematch_columns(suffixes: list[str]) -> list[str]:
    return [
        f"player_{side}_{suffix}"
        for side in (1, 2)
        for suffix in suffixes
    ]


def parquet_columns(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def available(schema: set[str], requested: list[str]) -> list[str]:
    return [column for column in requested if column in schema]


def parse_competition_types(value: str) -> list[str]:
    values = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Debe indicarse al menos una competicion")
    return values


def stable_file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            handle.seek(max(stat.st_size - 1024 * 1024, 0))
            digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns), "edge_sha256": digest.hexdigest(),
    }


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for bookmaker in ("b365", "ps", "max", "avg", "bfe"):
        c1, c2 = f"player_1_{bookmaker}_odds", f"player_2_{bookmaker}_odds"
        if c1 not in result.columns or c2 not in result.columns:
            continue
        odds_1 = pd.to_numeric(result[c1], errors="coerce")
        odds_2 = pd.to_numeric(result[c2], errors="coerce")
        valid = odds_1.gt(1.0) & odds_2.gt(1.0)
        raw_1, raw_2 = 1.0 / odds_1.where(valid), 1.0 / odds_2.where(valid)
        total = raw_1 + raw_2
        result[f"market_{bookmaker}_p1_prob_no_vig"] = (raw_1 / total).astype("float32")
        result[f"market_{bookmaker}_overround"] = (total - 1.0).astype("float32")
    probability_columns = [
        column for column in result.columns if column.endswith("_p1_prob_no_vig")
    ]
    if probability_columns:
        result["market_avg_p1_prob_no_vig"] = result[probability_columns].mean(
            axis=1, skipna=True
        ).astype("float32")
    return result


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def row_min(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    return pd.concat([numeric(frame, left), numeric(frame, right)], axis=1).min(axis=1)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = numerator.notna() & denominator.notna() & denominator.ne(0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def add_quality_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    p1_matches = numeric(result, "player_1_career_matches_before")
    p2_matches = numeric(result, "player_2_career_matches_before")
    p1_stats = numeric(result, "player_1_stat_matches_before")
    p2_stats = numeric(result, "player_2_stat_matches_before")
    result["min_career_matches_before"] = pd.concat(
        [p1_matches, p2_matches], axis=1
    ).min(axis=1).astype("float32")
    result["min_stat_matches_before"] = pd.concat(
        [p1_stats, p2_stats], axis=1
    ).min(axis=1).astype("float32")
    result["cold_start_any_player"] = (p1_matches.lt(5) | p2_matches.lt(5)).astype("int8")
    result["stats_reliable_both"] = (p1_stats.ge(10) & p2_stats.ge(10)).astype("int8")

    if {
        "player_1_surface_matches_before", "player_2_surface_matches_before"
    }.issubset(result.columns):
        p1_surface = numeric(result, "player_1_surface_matches_before")
        p2_surface = numeric(result, "player_2_surface_matches_before")
        result["min_surface_matches_before"] = pd.concat(
            [p1_surface, p2_surface], axis=1
        ).min(axis=1).astype("float32")
        result["surface_cold_start_any_player"] = (
            p1_surface.lt(5) | p2_surface.lt(5)
        ).astype("int8")

    if {
        "player_1_surface_stat_matches_before",
        "player_2_surface_stat_matches_before",
    }.issubset(result.columns):
        p1_surface_stats = numeric(result, "player_1_surface_stat_matches_before")
        p2_surface_stats = numeric(result, "player_2_surface_stat_matches_before")
        result["min_surface_stat_matches_before"] = pd.concat(
            [p1_surface_stats, p2_surface_stats], axis=1
        ).min(axis=1).astype("float32")
        result["surface_stats_reliable_both"] = (
            p1_surface_stats.ge(10) & p2_surface_stats.ge(10)
        ).astype("int8")

    defaults = {
        "min_surface_matches_before": np.nan,
        "min_surface_stat_matches_before": np.nan,
        "surface_cold_start_any_player": 1,
        "surface_stats_reliable_both": 0,
    }
    for column, value in defaults.items():
        if column not in result.columns:
            result[column] = value

    if {"player_1_id", "player_2_id"}.issubset(result.columns):
        p1_id, p2_id = numeric(result, "player_1_id"), numeric(result, "player_2_id")
        result["synthetic_player_any"] = (
            p1_id.ge(SYNTHETIC_ID_MIN) | p2_id.ge(SYNTHETIC_ID_MIN)
        ).astype("int8")
    else:
        result["synthetic_player_any"] = np.int8(0)
    return result


def add_stable_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Transformaciones deterministas conocidas antes del partido."""
    result = frame.copy()
    if "diff_elo_before" in result.columns:
        diff = numeric(result, "diff_elo_before")
        result["elo_expected_p1"] = (
            1.0 / (1.0 + np.power(10.0, -diff / 400.0))
        ).astype("float32")
        result["absolute_elo_difference"] = diff.abs().astype("float32")
    if "diff_surface_elo_before" in result.columns:
        diff = numeric(result, "diff_surface_elo_before")
        result["surface_elo_expected_p1"] = (
            1.0 / (1.0 + np.power(10.0, -diff / 400.0))
        ).astype("float32")
        result["absolute_surface_elo_difference"] = diff.abs().astype("float32")
    if {"player_1_rank", "player_2_rank"}.issubset(result.columns):
        result["rank_ratio_p1_p2"] = safe_ratio(
            numeric(result, "player_1_rank"), numeric(result, "player_2_rank")
        ).clip(0, 20).astype("float32")
    if {"player_1_rank_points", "player_2_rank_points"}.issubset(result.columns):
        result["rank_points_ratio_p1_p2"] = safe_ratio(
            numeric(result, "player_1_rank_points") + 1.0,
            numeric(result, "player_2_rank_points") + 1.0,
        ).clip(0, 20).astype("float32")
    return result



def profile_v5_columns(profile: str) -> tuple[list[str], list[str], list[str]]:
    families = set(PROFILE_FAMILIES[profile])
    suffixes: list[str] = []
    matchup: list[str] = []
    derived: list[str] = []
    if "momentum" in families:
        suffixes += MOMENTUM_SUFFIXES
    if "momentum_v6" in families:
        suffixes += MOMENTUM_V6_SUFFIXES
    if "workload" in families:
        suffixes += WORKLOAD_SUFFIXES
    if "dynamic_stats" in families:
        suffixes += DYNAMIC_STATS_SUFFIXES
        derived += [c for c in V5_DERIVED_COLUMNS if c.startswith("dynamic_stats_")]
    if "ranking_trends" in families:
        suffixes += RANKING_TREND_SUFFIXES
        derived += [c for c in V5_DERIVED_COLUMNS if "rank_" in c]
    if "matchup" in families:
        matchup += MATCHUP_V5_COLUMNS
        derived += [
            "intransitivity_shrunk_before",
            "surface_intransitivity_shrunk_before",
            "log_common_opponents_count_before",
            "log_surface_common_opponents_count_before",
        ]
    if families:
        derived.append("match_date_exact")
    return list(dict.fromkeys(suffixes)), list(dict.fromkeys(matchup)), list(dict.fromkeys(derived))


def add_v5_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derivadas prepartido estables, indicadores de cobertura y shrinkage."""
    result = frame.copy()
    p1_service = "player_1_service_points_won_ewma_long_before"
    p2_service = "player_2_service_points_won_ewma_long_before"
    if {p1_service, p2_service}.issubset(result.columns):
        a1, a2 = result[p1_service].notna(), result[p2_service].notna()
        result["dynamic_stats_available_player_1"] = a1.astype("int8")
        result["dynamic_stats_available_player_2"] = a2.astype("int8")
        result["dynamic_stats_available_both"] = (a1 & a2).astype("int8")
        result["dynamic_stats_available_count"] = (a1.astype("int8") + a2.astype("int8")).astype("int8")
    p1_rank = "player_1_rank_change_8w_before"
    p2_rank = "player_2_rank_change_8w_before"
    if {p1_rank, p2_rank}.issubset(result.columns):
        a1, a2 = result[p1_rank].notna(), result[p2_rank].notna()
        result["rank_trend_available_player_1"] = a1.astype("int8")
        result["rank_trend_available_player_2"] = a2.astype("int8")
        result["rank_trend_available_both"] = (a1 & a2).astype("int8")
    if "td_date" in result.columns:
        result["match_date_exact"] = pd.to_datetime(result["td_date"], errors="coerce").notna().astype("int8")
    else:
        result["match_date_exact"] = np.int8(0)
    if {"intransitivity_score_player_1_before", "common_opponents_count_before"}.issubset(result.columns):
        score = numeric(result, "intransitivity_score_player_1_before")
        support = numeric(result, "common_opponents_count_before").clip(lower=0)
        result["intransitivity_shrunk_before"] = (score * support / (support + 3.0)).astype("float32")
        result["log_common_opponents_count_before"] = np.log1p(support).astype("float32")
    if {"surface_intransitivity_score_player_1_before", "surface_common_opponents_count_before"}.issubset(result.columns):
        score = numeric(result, "surface_intransitivity_score_player_1_before")
        support = numeric(result, "surface_common_opponents_count_before").clip(lower=0)
        result["surface_intransitivity_shrunk_before"] = (score * support / (support + 3.0)).astype("float32")
        result["log_surface_common_opponents_count_before"] = np.log1p(support).astype("float32")
    for weeks in (4, 8, 26):
        p1 = f"player_1_rank_change_{weeks}w_before"
        p2 = f"player_2_rank_change_{weeks}w_before"
        diff = f"diff_rank_change_{weeks}w_before"
        if p1 in result.columns:
            result[f"{p1}_clipped"] = numeric(result, p1).clip(-500, 500).astype("float32")
        if p2 in result.columns:
            result[f"{p2}_clipped"] = numeric(result, p2).clip(-500, 500).astype("float32")
        if diff in result.columns:
            values = numeric(result, diff)
            result[f"{diff}_clipped"] = values.clip(-500, 500).astype("float32")
            result[f"{diff}_signed_log"] = (np.sign(values) * np.log1p(np.abs(values))).astype("float32")
    return result

def assign_temporal_split(frame: pd.DataFrame, train_end: str, validation_end: str) -> pd.Series:
    dates = pd.to_datetime(frame[DATE], errors="coerce")
    train_cut, validation_cut = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if validation_cut <= train_cut:
        raise ValueError("validation_end debe ser posterior a train_end")
    return pd.Series(
        np.select(
            [dates.le(train_cut), dates.le(validation_cut)],
            ["train", "validation"], default="test",
        ),
        index=frame.index, dtype="string",
    )


def validate_feature_whitelist(feature_columns: list[str], dataset_name: str) -> None:
    allowed_non_before = set(
        PLAYER_STATIC_COLUMNS + SAFE_MATCH_CONTEXT_FEATURES + H2H_COLUMNS
        + OPTIMIZED_DIFF_COLUMNS + MARKET_COLUMNS + QUALITY_FEATURE_COLUMNS
        + DERIVED_SPORT_FEATURE_COLUMNS + MATCHUP_V5_COLUMNS + V5_DERIVED_COLUMNS
        + V5_PLAYER_DIFFERENTIALS
    )
    leaking = []
    for column in feature_columns:
        is_player_prematch = (
            column.startswith(("player_1_", "player_2_"))
            and (
                column.endswith("_before")
                or column.endswith("_days_since_last_match")
            )
        )
        is_derived_market = (
            column.startswith("market_")
            and (
                column.endswith("_p1_prob_no_vig")
                or column.endswith("_overround")
            )
        )
        if is_player_prematch or is_derived_market or column in allowed_non_before:
            continue
        # Lista blanca estricta: cualquier columna restante se rechaza.
        leaking.append(column)
    if leaking:
        raise ValueError(
            f"{dataset_name}: columnas no permitidas o posible leakage: {sorted(leaking)}"
        )


def audit_dataset(frame: pd.DataFrame, feature_columns: list[str], dataset_name: str) -> None:
    if TARGET not in frame.columns:
        raise ValueError(f"{dataset_name}: falta target {TARGET}")
    target_values = set(frame[TARGET].dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{dataset_name}: target no binario: {target_values}")
    if frame[DATE].isna().any():
        raise ValueError(f"{dataset_name}: match_date nulos")
    if frame[SPLIT].isna().any() or not set(frame[SPLIT].dropna().unique()).issubset({"train", "validation", "test"}):
        raise ValueError(f"{dataset_name}: split nulo o invalido")
    duplicates = int(frame.duplicated(["tourney_id", "competition_type", "match_num"]).sum())
    if duplicates:
        raise ValueError(f"{dataset_name}: {duplicates:,} claves operacionales duplicadas")
    if frame["processing_sequence"].isna().any():
        raise ValueError(f"{dataset_name}: processing_sequence nulo")
    numeric_features = [c for c in feature_columns if pd.api.types.is_numeric_dtype(frame[c])]
    if numeric_features:
        values = frame[numeric_features].to_numpy(dtype="float64", na_value=np.nan)
        infinite = int(np.isinf(values).sum())
        if infinite:
            raise ValueError(f"{dataset_name}: {infinite:,} valores infinitos en features")
    validate_feature_whitelist(feature_columns, dataset_name)
    target_source_leakage = sorted(set(feature_columns) & set(MARKET_TARGET_SOURCE_COLUMNS))
    if target_source_leakage:
        raise ValueError(
            f"{dataset_name}: fuentes post-partido incluidas accidentalmente en X: "
            f"{target_source_leakage}"
        )
    available_target_sources = [
        column for column in MARKET_TARGET_SOURCE_COLUMNS if column in frame.columns
    ]
    if available_target_sources:
        score_coverage = frame["score"].notna().mean() if "score" in frame else 0.0
        stat_coverage = frame[[
            column for column in MARKET_TARGET_SOURCE_COLUMNS if column != "score"
        ]].notna().all(axis=1).mean()
        if score_coverage <= 0.0:
            raise ValueError(f"{dataset_name}: score sin cobertura para targets secundarios")
        if stat_coverage <= 0.0:
            raise ValueError(f"{dataset_name}: estadisticas de servicio sin cobertura")

    forbidden_explicit = set(
        TRACEABILITY_COLUMNS + PLAYER_IDENTITY_COLUMNS + [TARGET, SPLIT, DATE]
    )
    forbidden_explicit.update(
        column for column in AUDIT_CONTEXT_COLUMNS
        if column not in SAFE_MATCH_CONTEXT_FEATURES
    )
    overlap = sorted(set(feature_columns) & forbidden_explicit)
    if overlap:
        raise ValueError(f"{dataset_name}: contexto prohibido incluido en X: {overlap}")

    duplicates = frame.duplicated(["tourney_id", "competition_type", "match_num"])
    if duplicates.any():
        raise ValueError(f"{dataset_name}: {int(duplicates.sum()):,} claves duplicadas")
    split_values = set(frame[SPLIT].dropna().unique().tolist())
    if not split_values.issubset({"train", "validation", "test"}):
        raise ValueError(f"{dataset_name}: splits no validos: {split_values}")
    ordered = frame.sort_values([DATE, "tourney_id", "competition_type", "match_num"], kind="mergesort")
    if not ordered.index.equals(frame.index):
        raise ValueError(f"{dataset_name}: dataset no ordenado cronologicamente")


def optimize_types(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    # Only binary targets and flags are converted to nullable Int8. The total
    # number of games is a continuous/regression target and must not be cast to
    # Int8, otherwise pandas raises on valid values outside the Int8 flag
    # contract or on non-equivalent floating representations.
    binary_target_columns = {
        "target_player_1_wins_first_set",
        "target_player_2_wins_first_set",
        "target_player_1_wins_any_set",
        "target_player_2_wins_any_set",
        "target_valid_first_set",
        "target_valid_completed_score",
        *{
            f"target_over_{str(line).replace('.', '_')}"
            for line in (
                18.5, 19.5, 20.5, 21.5, 22.5, 23.5,
                24.5, 25.5, 36.5, 38.5, 40.5,
            )
        },
    }
    continuous_target_columns = {"target_total_games"}

    integer_flags = {
        TARGET,
        *binary_target_columns,
        "cold_start_any_player",
        "surface_cold_start_any_player",
        "stats_reliable_both",
        "surface_stats_reliable_both",
        "synthetic_player_any",
        "dynamic_stats_available_player_1",
        "dynamic_stats_available_player_2",
        "dynamic_stats_available_both",
        "dynamic_stats_available_count",
        "rank_trend_available_player_1",
        "rank_trend_available_player_2",
        "rank_trend_available_both",
        "match_date_exact",
    }

    for column in result.columns:
        if column in (DATE, SPLIT):
            continue

        if column in set(MARKET_TARGET_SOURCE_COLUMNS) - {"score"}:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).astype("float32")
            continue

        if column in integer_flags:
            values = pd.to_numeric(result[column], errors="coerce")

            # Fail explicitly if a supposed binary target/flag contains values
            # other than 0, 1 or missing. This gives a useful column name instead
            # of pandas' generic unsafe-cast exception.
            if column != "dynamic_stats_available_count":
                invalid = values.notna() & ~values.isin([0, 1])
                if invalid.any():
                    invalid_values = sorted(
                        values.loc[invalid].dropna().unique().tolist()
                    )
                    raise ValueError(
                        f"{column}: se esperaba un indicador binario 0/1, "
                        f"pero contiene valores no validos: {invalid_values[:20]}"
                    )
            else:
                invalid = values.notna() & ~values.isin([0, 1, 2])
                if invalid.any():
                    invalid_values = sorted(
                        values.loc[invalid].dropna().unique().tolist()
                    )
                    raise ValueError(
                        f"{column}: se esperaba un contador 0/1/2, "
                        f"pero contiene valores no validos: {invalid_values[:20]}"
                    )

            result[column] = values.astype("Int8")

        elif column in continuous_target_columns:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).astype("Float32")

        elif pd.api.types.is_float_dtype(result[column]):
            result[column] = result[column].astype("float32")

        elif pd.api.types.is_integer_dtype(result[column]):
            # Preserve IDs, numeric dates and counters without unsafe downcast.
            continue

        elif (
            pd.api.types.is_object_dtype(result[column])
            or isinstance(result[column].dtype, pd.CategoricalDtype)
        ):
            result[column] = result[column].astype("string")

    return result
def feature_family(column: str) -> str:
    if column in SAFE_MATCH_CONTEXT_FEATURES:
        return "match_context"
    if column in PLAYER_STATIC_COLUMNS:
        return "player_static"
    if column in MARKET_COLUMNS or column.startswith("market_"):
        return "market"
    if column in QUALITY_FEATURE_COLUMNS:
        return "quality"
    if column in DERIVED_SPORT_FEATURE_COLUMNS:
        return "derived_sport"
    if column in H2H_COLUMNS:
        return "h2h"
    if column in MATCHUP_V5_COLUMNS:
        return "matchup_v5"
    if any(column.endswith(suffix) for suffix in MOMENTUM_V6_SUFFIXES):
        return "momentum_v6"
    if column in V5_DERIVED_COLUMNS:
        return "v5_derived"
    lowered = column.lower()
    if "surface_" in lowered:
        return "surface_history"
    if any(token in lowered for token in ("level_", "round_", "best_of_")):
        return "context_history"
    if any(token in lowered for token in ("opponent_", "quality_adjusted")):
        return "opponent_quality"
    if "elo_change" in lowered:
        return "elo_trend"
    if column.startswith("diff_"):
        return "difference"
    if column.endswith("_before"):
        return "global_history"
    return "other"


def market_orientation_contract(feature_columns: list[str]) -> dict[str, Any]:
    feature_set = set(feature_columns)
    paired = []
    signed = []
    reciprocal = []
    complemented = []
    symmetric = []
    unresolved = []
    for column in feature_columns:
        if column.startswith("player_1_"):
            other = "player_2_" + column[len("player_1_"):]
            (paired if other in feature_set else unresolved).append(
                {"player_1": column, "player_2": other} if other in feature_set else column
            )
        elif column.startswith("player_2_"):
            continue
        elif column.startswith("diff_"):
            signed.append(column)
        elif column.endswith("_p1_p2"):
            reciprocal.append(column)
        elif column in {"elo_expected_p1", "surface_elo_expected_p1"}:
            complemented.append(column)
        else:
            symmetric.append(column)
    return {"schema_version": 1, "paired_player_features": paired,
            "signed_features": signed, "reciprocal_features": reciprocal,
            "complemented_features": complemented, "symmetric_features": symmetric,
            "unresolved_directional_features": unresolved}


def make_manifest(
    no_market: pd.DataFrame,
    market: pd.DataFrame,
    no_market_features: list[str],
    market_features: list[str],
    baseline_features: list[str],
) -> pd.DataFrame:
    no_market_set, market_set = set(no_market_features), set(market_features)
    baseline_set = set(baseline_features)
    audit_context_set = set(AUDIT_CONTEXT_COLUMNS)
    rows = []
    for column in market.columns:
        if column == TARGET or column in SECONDARY_TARGET_COLUMNS:
            role = "target"
        elif column in MARKET_TARGET_SOURCE_COLUMNS:
            role = "market_target_source"
        elif column == SPLIT:
            role = "split"
        elif column in PLAYER_IDENTITY_COLUMNS:
            role = "identity"
        elif column in TRACEABILITY_COLUMNS:
            role = "traceability"
        elif column in audit_context_set and column not in SAFE_MATCH_CONTEXT_FEATURES:
            role = "audit_context"
        elif column in market_set and column not in no_market_set:
            role = "market_feature"
        elif column in no_market_set:
            role = "sport_feature"
        else:
            role = "excluded"
        rows.append({
            "column": column, "dtype": str(market[column].dtype), "role": role,
            "feature_family": feature_family(column),
            "allowed_for_training": role in {"sport_feature", "market_feature"},
            "baseline_feature": column in baseline_set,
            "optimized_feature": column in no_market_set and column not in baseline_set,
            "available_no_market": column in no_market.columns,
            "available_with_market": column in market.columns,
        })
    return pd.DataFrame(rows)


def write_parquet_atomically(frame: pd.DataFrame, output_path: Path) -> None:
    temporary = output_path.with_suffix(".new.parquet")
    backup = output_path.with_suffix(".before_rebuild.parquet")
    temporary.unlink(missing_ok=True)
    frame.to_parquet(
        temporary, index=False, engine="pyarrow", compression="zstd",
        row_group_size=100_000,
    )
    verification = pd.read_parquet(
        temporary,
        columns=["tourney_id", "competition_type", "match_num", DATE, SPLIT, TARGET],
        engine="pyarrow",
    )
    if len(verification) != len(frame):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"La escritura temporal de {output_path.name} perdio filas")
    duplicates = int(verification.duplicated(
        ["tourney_id", "competition_type", "match_num"]
    ).sum())
    if duplicates:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{output_path.name}: {duplicates:,} claves duplicadas")
    if verification[[DATE, SPLIT, TARGET]].isna().any().any():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{output_path.name}: fecha, split o target nulos")
    if output_path.exists() and not backup.exists():
        shutil.copy2(output_path, backup)
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    root = find_project_root(Path(__file__).resolve().parent)
    processed = root / "data" / "processed"
    parser = argparse.ArgumentParser(
        description="Crea datasets ML optimizados, temporales y sin leakage"
    )
    parser.add_argument("--input", default=str(processed / "tennis_matches_with_player_stats.sackmann_v5.parquet"))
    parser.add_argument("--feature-builder-manifest", default=str(processed / "tennis_match_feature_manifest.sackmann_v5.json"))
    parser.add_argument("--output-no-market", default=str(processed / "tennis_ml_no_market.sackmann_v5.parquet"))
    parser.add_argument("--output-market", default=str(processed / "tennis_ml_with_market.sackmann_v5.parquet"))
    parser.add_argument("--report", default=str(processed / "tennis_ml_dataset_report_sackmann_v5.csv"))
    parser.add_argument("--manifest", default=str(processed / "tennis_ml_feature_manifest_sackmann_v5.csv"))
    parser.add_argument("--metadata", default=str(processed / "tennis_ml_dataset_metadata_sackmann_v5.json"))
    parser.add_argument("--min-date", default="2001-01-01")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--validation-end", default="2025-12-31")
    parser.add_argument("--min-career-matches", type=int, default=5)
    parser.add_argument(
        "--competition-types", type=parse_competition_types,
        default=DEFAULT_COMPETITION_TYPES,
    )
    parser.add_argument("--include-ineligible", action="store_true")
    parser.add_argument("--only-atp-level", action="store_true")
    parser.add_argument(
        "--feature-profile", choices=list(PROFILE_FAMILIES), default="v5_all",
        help=(
            "Perfil para baseline, ablaciones por familia, v5 completo o "
            "Momentum v6 experimental."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_no_market = Path(args.output_no_market).resolve()
    output_market = Path(args.output_market).resolve()
    report_path = Path(args.report).resolve()
    manifest_path = Path(args.manifest).resolve()
    metadata_path = Path(args.metadata).resolve()
    builder_manifest_path = Path(args.feature_builder_manifest).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    schema = parquet_columns(input_path)
    v5_suffixes, selected_matchup, selected_v5_derived = profile_v5_columns(args.feature_profile)
    use_v4_optimized = args.feature_profile != "baseline"
    selected_suffixes = list(dict.fromkeys(
        (OPTIMIZED_PREMATCH_SUFFIXES if use_v4_optimized else BASELINE_PREMATCH_SUFFIXES)
        + v5_suffixes
    ))
    selected_diffs = list(dict.fromkeys(
        (OPTIMIZED_DIFF_COLUMNS if use_v4_optimized else BASELINE_DIFF_COLUMNS)
        + [c for c in V5_PLAYER_DIFFERENTIALS if c.removeprefix("diff_") in v5_suffixes]
    ))
    requested = list(dict.fromkeys(
        AUDIT_CONTEXT_COLUMNS + TRACEABILITY_COLUMNS + PLAYER_IDENTITY_COLUMNS
        + PLAYER_STATIC_COLUMNS + player_prematch_columns(selected_suffixes)
        + H2H_COLUMNS + selected_matchup + selected_diffs + MARKET_COLUMNS
        + MARKET_TARGET_SOURCE_COLUMNS + [TARGET]
    ))
    selected = available(schema, requested)

    essential = {
        DATE, TARGET, "processing_sequence", "tourney_id", "match_num", "surface", "competition_type",
        "eligible_for_model", "player_1_career_matches_before",
        "player_2_career_matches_before", "player_1_stat_matches_before",
        "player_2_stat_matches_before", "player_1_id", "player_2_id",
        *MARKET_TARGET_SOURCE_COLUMNS,
    }
    missing_core = sorted(essential - set(selected))
    if missing_core:
        raise ValueError(f"Faltan columnas esenciales: {missing_core}")

    if use_v4_optimized:
        expected_optimized = set(
            player_prematch_columns(selected_suffixes)
            + selected_diffs + selected_matchup
        )
        missing_optimized = sorted(expected_optimized - schema)
        if missing_optimized:
            raise ValueError(
                "El feature builder no ha generado todas las columnas optimizadas. "
                f"Faltan: {missing_optimized}"
            )

    print(f"Leyendo {len(selected)} de {len(schema)} columnas: {input_path}")
    frame = pd.read_parquet(input_path, columns=selected, engine="pyarrow")
    input_rows = len(frame)
    frame[DATE] = pd.to_datetime(frame[DATE], errors="coerce")
    if frame[DATE].isna().any():
        raise ValueError(f"Hay {int(frame[DATE].isna().sum()):,} fechas nulas")

    frame["competition_type"] = frame["competition_type"].astype("string").str.strip().str.upper()
    for column in ("surface", "tourney_level", "round"):
        if column in frame.columns:
            frame[column] = frame[column].astype("string").str.strip()
    if "source_origin" in frame.columns:
        frame["source_origin"] = frame["source_origin"].astype("string").str.strip().str.upper()

    frame = frame.loc[frame[DATE].ge(pd.Timestamp(args.min_date))].copy()
    competition_types = ["ATP"] if args.only_atp_level else args.competition_types
    frame = frame.loc[frame["competition_type"].isin(competition_types)].copy()

    eligible_mask = frame["eligible_for_model"].fillna(False).astype(bool)
    ineligible_rows = int((~eligible_mask).sum())
    if not args.include_ineligible:
        frame = frame.loc[eligible_mask].copy()
    rows_after_date_competition_eligibility = len(frame)

    frame = add_secondary_targets(frame)
    frame = add_quality_features(frame)
    frame = add_stable_derived_features(frame)
    frame = add_v5_derived_features(frame)
    insufficient = frame["min_career_matches_before"].lt(args.min_career_matches)
    insufficient_rows = int(insufficient.sum())
    frame = frame.loc[~insufficient].copy()
    frame[SPLIT] = assign_temporal_split(frame, args.train_end, args.validation_end)
    split_counts = frame[SPLIT].value_counts()
    missing_splits = [name for name in ("train", "validation", "test") if int(split_counts.get(name, 0)) == 0]
    if missing_splits:
        raise ValueError(f"Splits temporales vacios: {missing_splits}")
    train_max = frame.loc[frame[SPLIT].eq("train"), DATE].max()
    validation_min = frame.loc[frame[SPLIT].eq("validation"), DATE].min()
    validation_max = frame.loc[frame[SPLIT].eq("validation"), DATE].max()
    test_min = frame.loc[frame[SPLIT].eq("test"), DATE].min()
    if not (train_max < validation_min <= validation_max < test_min):
        raise RuntimeError(
            "Solapamiento temporal detectado: "
            f"train_max={train_max}, validation=[{validation_min}, {validation_max}], test_min={test_min}"
        )
    frame = frame.sort_values(
        [DATE, "tourney_id", "competition_type", "match_num"],
        kind="mergesort",
    ).reset_index(drop=True)

    audit_context = available(set(frame.columns), AUDIT_CONTEXT_COLUMNS)
    trace = available(set(frame.columns), TRACEABILITY_COLUMNS)
    identity = available(set(frame.columns), PLAYER_IDENTITY_COLUMNS)
    market_target_sources = available(set(frame.columns), MARKET_TARGET_SOURCE_COLUMNS)
    missing_market_target_sources = sorted(
        set(MARKET_TARGET_SOURCE_COLUMNS) - set(market_target_sources)
    )
    if missing_market_target_sources:
        raise ValueError(
            "Faltan columnas post-partido necesarias para construir targets de "
            f"sets/juegos: {missing_market_target_sources}"
        )
    player_features = available(
        set(frame.columns),
        PLAYER_STATIC_COLUMNS + player_prematch_columns(selected_suffixes)
        + H2H_COLUMNS + selected_matchup + selected_diffs + QUALITY_FEATURE_COLUMNS
        + DERIVED_SPORT_FEATURE_COLUMNS + selected_v5_derived
        + SAFE_MATCH_CONTEXT_FEATURES,
    )
    no_market_features = list(dict.fromkeys(player_features))
    audit_only_context = [
        column for column in audit_context if column not in SAFE_MATCH_CONTEXT_FEATURES
    ]
    no_market_columns = list(dict.fromkeys(
        audit_only_context + trace + identity + market_target_sources + SECONDARY_TARGET_COLUMNS
        + no_market_features + [SPLIT, TARGET]
    ))
    no_market = frame[no_market_columns].copy()

    raw_market = available(set(frame.columns), MARKET_COLUMNS)
    market_columns = list(dict.fromkeys(
        audit_only_context + trace + identity + market_target_sources + SECONDARY_TARGET_COLUMNS
        + no_market_features + raw_market + [SPLIT, TARGET]
    ))
    market = add_market_features(frame[market_columns].copy())
    market_probability_columns = [
        column for column in market.columns if column.endswith("_p1_prob_no_vig")
    ]
    market_feature_columns = list(dict.fromkeys(
        no_market_features + raw_market
        + [column for column in market.columns if column.startswith("market_")]
    ))
    market_rows_before_coverage_filter = len(market)
    if market_probability_columns:
        market = market.loc[market[market_probability_columns].notna().any(axis=1)].copy()
    else:
        market = market.iloc[0:0].copy()

    no_market = optimize_types(no_market)
    market = optimize_types(market)
    audit_dataset(no_market, no_market_features, "no_market")
    audit_dataset(market, market_feature_columns, "with_market")

    for path in (output_no_market, output_market, report_path, manifest_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet_atomically(no_market, output_no_market)
    write_parquet_atomically(market, output_market)

    baseline_features = available(
        set(no_market.columns),
        PLAYER_STATIC_COLUMNS + player_prematch_columns(BASELINE_PREMATCH_SUFFIXES)
        + H2H_COLUMNS + BASELINE_DIFF_COLUMNS + QUALITY_FEATURE_COLUMNS
        + SAFE_MATCH_CONTEXT_FEATURES,
    )
    manifest = make_manifest(
        no_market, market, no_market_features, market_feature_columns,
        baseline_features,
    )
    manifest.to_csv(manifest_path, index=False)

    report_rows = []
    for dataset_name, data, features in (
        ("no_market", no_market, no_market_features),
        ("with_market", market, market_feature_columns),
    ):
        for split_name in ("train", "validation", "test"):
            group = data.loc[data[SPLIT].eq(split_name)]
            report_rows.append({
                "dataset": dataset_name, "feature_profile": args.feature_profile,
                "feature_families": PROFILE_FAMILIES[args.feature_profile],
                "v5_feature_families": PROFILE_FAMILIES[args.feature_profile],
                "split": split_name, "rows": int(len(group)),
                "target_rate": round(float(group[TARGET].mean()), 6) if len(group) else np.nan,
                "first_date": group[DATE].min() if len(group) else pd.NaT,
                "last_date": group[DATE].max() if len(group) else pd.NaT,
                "columns": int(len(data.columns)), "model_features": int(len(features)),
                "baseline_features": int(len(baseline_features)),
                "optimized_incremental_features": int(len(set(no_market_features) - set(baseline_features))),
                "tml_rows": int(group.get("source_origin", pd.Series("", index=group.index)).astype("string").eq("TML").sum()),
                "merged_source_rows": int(group.get("source_origin", pd.Series("", index=group.index)).astype("string").eq("SACKMANN_TML").sum()),
                "synthetic_player_rows": int(group.get("synthetic_player_any", pd.Series(0, index=group.index)).fillna(0).astype(bool).sum()),
            })
    report = pd.DataFrame(report_rows)
    report.to_csv(report_path, index=False)

    builder_manifest = {}
    if builder_manifest_path.exists():
        builder_manifest = json.loads(builder_manifest_path.read_text(encoding="utf-8-sig"))
    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_profile": args.feature_profile,
        "feature_families": PROFILE_FAMILIES[args.feature_profile],
        "v5_feature_families": PROFILE_FAMILIES[args.feature_profile],
        "momentum_v6_suffixes": (
            MOMENTUM_V6_SUFFIXES
            if "momentum_v6" in PROFILE_FAMILIES[args.feature_profile]
            else []
        ),
        "input": str(input_path),
        "input_fingerprint": stable_file_fingerprint(input_path),
        "feature_builder_manifest": str(builder_manifest_path),
        "feature_builder_metadata": builder_manifest,
        "integrated_reconciliation": builder_manifest.get("integrated_reconciliation", {}),
        "input_rows": int(input_rows), "target": TARGET, "date_column": DATE,
        "min_date": args.min_date, "train_end": args.train_end,
        "validation_end": args.validation_end,
        "test_start": str((pd.Timestamp(args.validation_end) + pd.Timedelta(days=1)).date()),
        "data_max_date": str(frame[DATE].max().date()),
        "min_career_matches": int(args.min_career_matches),
        "competition_types": competition_types,
        "include_ineligible": bool(args.include_ineligible),
        "ineligible_rows_before_filter": ineligible_rows,
        "rows_after_date_competition_eligibility": rows_after_date_competition_eligibility,
        "insufficient_history_rows_removed": insufficient_rows,
        "no_market_rows": int(len(no_market)),
        "market_rows_before_coverage_filter": int(market_rows_before_coverage_filter),
        "market_rows": int(len(market)),
        "no_market_feature_columns": no_market_features,
        "market_feature_columns": market_feature_columns,
        "baseline_no_market_feature_columns": baseline_features,
        "optimized_incremental_feature_columns": sorted(set(no_market_features) - set(baseline_features)),
        "safe_match_context_feature_columns": SAFE_MATCH_CONTEXT_FEATURES,
        "context_columns": audit_only_context,
        "traceability_columns": trace,
        "identity_columns": identity,
        "market_target_source_columns": market_target_sources,
        "secondary_target_columns": SECONDARY_TARGET_COLUMNS,
        "market_orientation_contract": market_orientation_contract(no_market_features),
        "market_target_source_policy": (
            "Post-match score and raw serve totals are retained only for secondary "
            "target construction/audit. They are excluded from every feature list."
        ),
        "market_target_source_coverage": {
            "score_rows": int(frame["score"].notna().sum()),
            "complete_serve_stat_rows": int(frame[[
                "w_svpt", "w_1stWon", "w_2ndWon",
                "l_svpt", "l_1stWon", "l_2ndWon",
            ]].notna().all(axis=1).sum()),
        },
        "market_probability_columns": market_probability_columns,
        "excluded_by_design": (
            "Raw winner/loser fields, duration and unrelated in-match statistics are "
            "excluded. score and six raw serve totals are retained only as non-feature "
            "secondary-target sources; IDs, names, source metadata, current-state and "
            "other post-match fields remain excluded from X."
        ),
        "current_state_policy": (
            "player_current_state.parquet is reserved for inference and is never "
            "joined into the historical training dataset."
        ),
        "imputation_policy": (
            "No global imputation. CatBoost or train-only transformations handle missing values."
        ),
        "split_boundaries": {
            "min_date": str(args.min_date),
            "train_end": str(args.train_end),
            "validation_start": str(pd.Timestamp(args.train_end) + pd.Timedelta(days=1)),
            "validation_end": str(args.validation_end),
            "test_start": str(pd.Timestamp(args.validation_end) + pd.Timedelta(days=1)),
            "data_max_date": str(frame[DATE].max()),
        },
        "split_policy": (
            "Chronological and disjoint: train <= train_end; "
            "train_end < validation <= validation_end; test > validation_end."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + report.to_string(index=False))
    print(f"\nFilas entrada: {input_rows:,}")
    print(f"No market: {len(no_market):,}; features: {len(no_market_features):,}")
    print(f"With market: {len(market):,}; features: {len(market_feature_columns):,}")
    print(f"Perfil: {args.feature_profile}")
    print(
        "Target sources: "
        f"score={int(frame['score'].notna().sum()):,}; "
        f"serve_stats={int(frame[['w_svpt', 'w_1stWon', 'w_2ndWon', 'l_svpt', 'l_1stWon', 'l_2ndWon']].notna().all(axis=1).sum()):,}"
    )
    print(f"Sin mercado: {output_no_market}")
    print(f"Con mercado: {output_market}")
    print(f"Informe: {report_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
