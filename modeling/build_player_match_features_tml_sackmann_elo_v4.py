#!/usr/bin/env python3
"""Construye features cronologicas prepartido y estado actual por jugador.

Entradas
-------
data/processed/jeff_sackmann_with_odds.parquet

Salidas
-------
data/processed/tennis_matches_with_player_stats.sackmann_v5.parquet
data/processed/player_current_state.sackmann_v5.parquet
data/processed/tennis_matches_with_player_stats.sackmann_v5_report.csv
data/processed/tennis_match_feature_manifest.sackmann_v5.json

Garantias
---------
- Todas las features de cada partido se capturan antes de actualizar estados.
- No usa informacion futura ni el resultado actual para construir sus features.
- Separa estado global, estado por superficie y estado por contexto.
- Genera un estado postpartido actual para inferencia, sin fingir datos en vivo.
- No incluye carga fisica de 7/14 dias, porque exige actualizacion frecuente.
- Escritura atomica y validacion de claves, IDs y esquema.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import psutil
except ImportError:
    psutil = None


class ProgressReporter:
    def __init__(self, total, every_rows=10000, every_seconds=5.0, enabled=True):
        self.total=max(int(total),1); self.every_rows=max(int(every_rows),1); self.every_seconds=max(float(every_seconds),0.25); self.enabled=enabled
        self.started=time.perf_counter(); self.last_print=self.started; self.last_row=0; self.process=psutil.Process() if psutil else None
    @staticmethod
    def duration(seconds):
        if seconds is None or not np.isfinite(seconds): return "--:--:--"
        h,r=divmod(max(int(seconds),0),3600); m,sec=divmod(r,60); return f"{h:02d}:{m:02d}:{sec:02d}"
    def memory(self):
        try: return f"{self.process.memory_info().rss/(1024**3):.2f} GiB" if self.process else "n/a"
        except Exception: return "n/a"
    def stage(self,message):
        if self.enabled: print(f"\n[{self.duration(time.perf_counter()-self.started)}] {message}",flush=True)
    def update(self,completed,match_date=None,players=None,force=False):
        if not self.enabled: return
        now=time.perf_counter(); due=completed==self.total or completed-self.last_row>=self.every_rows or now-self.last_print>=self.every_seconds
        if not force and not due: return
        elapsed=max(now-self.started,1e-9); rate=completed/elapsed; eta=(self.total-completed)/rate if rate else None
        date="-" if match_date is None or pd.isna(match_date) else str(pd.Timestamp(match_date).date()); players_text="-" if players is None else f"{players:,}"
        print(f"[{completed/self.total*100:6.2f}%] {completed:,}/{self.total:,} | {rate:,.0f} filas/s | transcurrido {self.duration(elapsed)} | ETA {self.duration(eta)} | fecha {date} | jugadores {players_text} | RAM {self.memory()}",flush=True)
        self.last_print=now; self.last_row=completed

# Aproximacion transparente a la metodologia publicada por Jeff Sackmann.
# El codigo exacto de Tennis Abstract no es publico. Estos parametros siguen
# sus principios documentados: baja entrada para circuitos inferiores, K que
# disminuye con la experiencia, intercambio de suma cero, penalizacion por
# ausencias y mezcla 50/50 del Elo global con el Elo puro de superficie.
ELO_ENTRY_RATINGS = {
    "ATP": 1500.0,
    "ATP_QUALIFYING": 1350.0,
    "CHALLENGER": 1250.0,
    "FUTURES": 1200.0,
}
ELO_ENTRY_DEFAULT = 1250.0
ELO_K_NUMERATOR = 250.0
ELO_K_OFFSET = 5.0
ELO_K_EXPONENT = 0.40
ELO_SURFACE_BLEND = 0.50
ABSENCE_MIN_DAYS = 56
ABSENCE_MAX_DAYS = 365
ABSENCE_MIN_PENALTY = 100.0
ABSENCE_MAX_PENALTY = 150.0
COMEBACK_K_MATCHES = 20
COMEBACK_K_MAX_MULTIPLIER = 1.50
RECENT_WINDOW = 10
TREND_WINDOW = 5
OPPONENT_WINDOW = 5
MIN_DATE = pd.Timestamp("1968-01-01")
SYNTHETIC_ID_MIN = 90_000_000
MANUAL_PLAYER_ID_ALIASES = {210084: 210150, 94974615: 210150}
IDENTITY_HEIGHT_TOLERANCE_CM = 4.0
IDENTITY_AGE_TOLERANCE_YEARS = 1.25
IDENTITY_MAX_OVERLAP_DAYS = 120
RECONCILIATION_SOURCE_PRIORITY = {"SACKMANN_TML": 400, "SACKMANN": 300, "TML": 200, "UNKNOWN": 0}
RECONCILIATION_STAT_COLUMNS = ["minutes","w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced","l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced"]
RECONCILIATION_ODDS_COLUMNS = ["td_b365w","td_b365l","td_psw","td_psl","td_maxw","td_maxl","td_avgw","td_avgl","td_bfew","td_bfel"]
KNOWN_SURFACES = ("Hard", "Clay", "Grass", "Carpet", "Unknown")

SURFACE_MAP = {
    "hard": "Hard", "clay": "Clay", "grass": "Grass",
    "carpet": "Carpet", "unknown": "Unknown",
}
COMPETITION_ORDER = {
    "ATP_QUALIFYING": 10, "ATP": 20, "CHALLENGER": 30, "FUTURES": 40,
}
ROUND_ORDER = {
    "Q1": 10, "Q2": 20, "Q3": 30, "R128": 40, "R64": 50,
    "R32": 60, "R16": 70, "QF": 80, "SF": 90, "F": 100,
    "RR": 110, "BR": 120,
}
WALKOVER_VALUES = {"W/O", "WO", "WALKOVER"}

IDENTITY_COLUMNS = {
    "id": ("winner_id", "loser_id"), "name": ("winner_name", "loser_name"),
    "hand": ("winner_hand", "loser_hand"), "height": ("winner_ht", "loser_ht"),
    "ioc": ("winner_ioc", "loser_ioc"), "age": ("winner_age", "loser_age"),
    "rank": ("winner_rank", "loser_rank"),
    "rank_points": ("winner_rank_points", "loser_rank_points"),
}
ODDS_PAIRS = [
    ("b365", "td_b365w", "td_b365l"), ("ps", "td_psw", "td_psl"),
    ("max", "td_maxw", "td_maxl"), ("avg", "td_avgw", "td_avgl"),
    ("bfe", "td_bfew", "td_bfel"),
]

GLOBAL_FEATURES = [
    "career_matches_before", "career_wins_before", "career_win_rate_before",
    "elo_before", "form_last_5_before", "form_last_10_before",
    "avg_minutes_last_5_before", "days_since_last_match",
    "stat_matches_before", "aces_per_service_game_before",
    "double_faults_per_service_game_before", "first_serve_in_pct_before",
    "first_serve_win_pct_before", "second_serve_win_pct_before",
    "service_points_won_pct_before", "break_points_saved_pct_before",
    "return_points_won_pct_before", "elo_change_last_5_before",
    "avg_opponent_elo_last_5_before", "avg_opponent_rank_last_5_before",
    "quality_adjusted_form_last_5_before",
]
SURFACE_FEATURES = [
    "surface_matches_before", "surface_wins_before", "surface_win_rate_before",
    "surface_elo_before", "surface_elo_raw_before", "surface_form_last_5_before",
    "surface_form_last_10_before", "surface_stat_matches_before",
    "surface_service_points_won_pct_before",
    "surface_return_points_won_pct_before",
    "surface_aces_per_service_game_before",
    "surface_double_faults_per_service_game_before",
    "surface_first_serve_in_pct_before",
    "surface_first_serve_win_pct_before",
    "surface_second_serve_win_pct_before",
    "surface_break_points_saved_pct_before",
    "surface_elo_change_last_5_before",
]
CONTEXT_FEATURES = [
    "level_matches_before", "level_wins_before", "level_win_rate_before",
    "round_matches_before", "round_wins_before", "round_win_rate_before",
    "best_of_matches_before", "best_of_wins_before", "best_of_win_rate_before",
]
FEATURE_NAMES = GLOBAL_FEATURES + SURFACE_FEATURES + CONTEXT_FEATURES
DIFFERENTIAL_FEATURES = [feature for feature in FEATURE_NAMES if feature not in {
    "career_wins_before", "surface_wins_before", "level_wins_before",
    "round_wins_before", "best_of_wins_before",
}]

REQUIRED_COLUMNS = [
    "tourney_id", "tourney_name", "tourney_date", "match_num", "source_match_num",
    "surface", "draw_size", "tourney_level", "tourney_level_original",
    "competition_type", "indoor", "best_of", "round", "score",
    "winner_id", "winner_name", "winner_hand", "winner_ht", "winner_ioc",
    "winner_age", "winner_rank", "winner_rank_points", "loser_id", "loser_name",
    "loser_hand", "loser_ht", "loser_ioc", "loser_age", "loser_rank",
    "loser_rank_points", "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn",
    "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace",
    "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms",
    "l_bpSaved", "l_bpFaced", "source_origin", "source_priority",
    "source_sackmann_file", "source_tml_file", "winner_id_tml", "loser_id_tml",
    "stats_quality_flag", "odds_matched", "td_date", "td_b365w", "td_b365l",
    "td_psw", "td_psl", "td_maxw", "td_maxl", "td_avgw", "td_avgl",
    "td_bfew", "td_bfel", "tournament_key", "canonical_tournament_name",
]
MANDATORY_COLUMNS = {
    "tourney_id", "tourney_name", "tourney_date", "match_num", "surface",
    "competition_type", "round", "score", "winner_id", "winner_name",
    "loser_id", "loser_name", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points", "minutes", "w_ace", "w_df", "w_svpt",
    "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
}

@dataclass
class StatsAccumulator:
    matches: int = 0
    aces: float = 0.0
    double_faults: float = 0.0
    service_points: float = 0.0
    first_serves_in: float = 0.0
    first_serve_points_won: float = 0.0
    second_serve_points_won: float = 0.0
    service_games: float = 0.0
    break_points_saved: float = 0.0
    break_points_faced: float = 0.0
    return_points_won: float = 0.0
    return_points_played: float = 0.0

@dataclass
class ContextRecord:
    matches: int = 0
    wins: int = 0

@dataclass
class PlayerState:
    matches: int = 0
    wins: int = 0
    losses: int = 0
    elo: float = math.nan
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    surface_elo: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: math.nan))
    recent_results: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    recent_minutes: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    surface_recent_results: dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=RECENT_WINDOW))
    )
    elo_history: deque = field(default_factory=lambda: deque(maxlen=TREND_WINDOW + 1))
    surface_elo_history: dict[str, deque] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=TREND_WINDOW + 1))
    )
    opponent_elos: deque = field(default_factory=lambda: deque(maxlen=OPPONENT_WINDOW))
    opponent_ranks: deque = field(default_factory=lambda: deque(maxlen=OPPONENT_WINDOW))
    quality_results: deque = field(default_factory=lambda: deque(maxlen=OPPONENT_WINDOW))
    last_date: pd.Timestamp | None = None
    stats: StatsAccumulator = field(default_factory=StatsAccumulator)
    surface_stats: dict[str, StatsAccumulator] = field(
        default_factory=lambda: defaultdict(StatsAccumulator)
    )
    level_records: dict[str, ContextRecord] = field(
        default_factory=lambda: defaultdict(ContextRecord)
    )
    round_records: dict[str, ContextRecord] = field(
        default_factory=lambda: defaultdict(ContextRecord)
    )
    best_of_records: dict[str, ContextRecord] = field(
        default_factory=lambda: defaultdict(ContextRecord)
    )
    player_id: int | None = None
    player_name: str | None = None
    player_hand: str | None = None
    player_height: float = math.nan
    player_ioc: str | None = None
    player_age: float = math.nan
    player_rank: float = math.nan
    player_rank_points: float = math.nan
    last_surface: str | None = None
    comeback_k_matches_remaining: int = 0
    last_absence_penalty: float = 0.0
    last_absence_days: int = 0


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "data" / "processed").exists():
            return candidate
    raise FileNotFoundError("No se pudo localizar la raiz del proyecto")


def safe_float(value: Any) -> float:
    if value is None or pd.isna(value):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or math.isnan(float(denominator)) or denominator <= 0:
        return math.nan
    return float(numerator / denominator)


def normalized_surface(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return SURFACE_MAP.get(str(value).strip().casefold(), "Unknown")


def normalized_context(value: Any, missing: str = "UNKNOWN") -> str:
    if value is None or pd.isna(value):
        return missing
    text = str(value).strip().upper()
    return text or missing


def is_walkover(score: Any) -> bool:
    if score is None or pd.isna(score):
        return False
    return str(score).strip().upper().replace(" ", "") in WALKOVER_VALUES


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def initial_elo(competition_type: str) -> float:
    return float(ELO_ENTRY_RATINGS.get(competition_type, ELO_ENTRY_DEFAULT))


def dynamic_k(matches_played: int) -> float:
    """K decreciente usado en implementaciones publicas de tennis Elo.

    K = 250 / (matches + 5) ** 0.4
    """
    return float(
        ELO_K_NUMERATOR
        / ((max(int(matches_played), 0) + ELO_K_OFFSET) ** ELO_K_EXPONENT)
    )


def comeback_multiplier(state: PlayerState) -> float:
    remaining = max(int(state.comeback_k_matches_remaining), 0)
    if remaining <= 0:
        return 1.0
    fraction = remaining / COMEBACK_K_MATCHES
    return float(1.0 + (COMEBACK_K_MAX_MULTIPLIER - 1.0) * fraction)


def ensure_elo_initialized(state: PlayerState, surface: str, competition_type: str) -> None:
    entry = initial_elo(competition_type)
    if math.isnan(state.elo):
        state.elo = entry
    if math.isnan(state.surface_elo[surface]):
        state.surface_elo[surface] = entry


def absence_penalty(days_away: int) -> float:
    if days_away < ABSENCE_MIN_DAYS:
        return 0.0
    clipped = min(max(days_away, ABSENCE_MIN_DAYS), ABSENCE_MAX_DAYS)
    fraction = (clipped - ABSENCE_MIN_DAYS) / (ABSENCE_MAX_DAYS - ABSENCE_MIN_DAYS)
    return float(
        ABSENCE_MIN_PENALTY
        + fraction * (ABSENCE_MAX_PENALTY - ABSENCE_MIN_PENALTY)
    )


def apply_absence_adjustment(state: PlayerState, current_date: pd.Timestamp) -> None:
    """Aplica la penalizacion una vez, justo antes del partido de regreso."""
    if state.last_date is None or pd.isna(state.last_date) or pd.isna(current_date):
        return
    days_away = max(int((pd.Timestamp(current_date) - pd.Timestamp(state.last_date)).days), 0)
    penalty = absence_penalty(days_away)
    if penalty <= 0:
        state.last_absence_penalty = 0.0
        state.last_absence_days = 0
        return
    state.elo -= penalty
    for surface_name, rating in list(state.surface_elo.items()):
        if not math.isnan(rating):
            state.surface_elo[surface_name] = rating - penalty
    state.comeback_k_matches_remaining = COMEBACK_K_MATCHES
    state.last_absence_penalty = penalty
    state.last_absence_days = days_away


def blended_surface_elo(state: PlayerState, surface: str) -> float:
    raw = state.surface_elo[surface]
    if math.isnan(state.elo):
        return raw
    if math.isnan(raw):
        return state.elo
    return float((1.0 - ELO_SURFACE_BLEND) * state.elo + ELO_SURFACE_BLEND * raw)


def days_since(last_date: pd.Timestamp | None, current_date: pd.Timestamp) -> float:
    if last_date is None or pd.isna(last_date) or pd.isna(current_date):
        return math.nan
    return float(max((pd.Timestamp(current_date) - pd.Timestamp(last_date)).total_seconds() / 86400.0, 0.0))


def recent_rate(values: deque, window: int) -> float:
    selected = list(values)[-window:]
    return float(sum(selected) / len(selected)) if selected else math.nan


def recent_average(values: deque, window: int) -> float:
    clean = [float(value) for value in list(values)[-window:] if not pd.isna(value)]
    return float(np.mean(clean)) if clean else math.nan


def recent_change(values: deque, current: float) -> float:
    history = list(values)
    if not history:
        return math.nan
    return float(current - history[0])


def stats_snapshot(stats: StatsAccumulator, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}stat_matches_before": float(stats.matches),
        f"{prefix}aces_per_service_game_before": safe_div(stats.aces, stats.service_games),
        f"{prefix}double_faults_per_service_game_before": safe_div(stats.double_faults, stats.service_games),
        f"{prefix}first_serve_in_pct_before": safe_div(stats.first_serves_in, stats.service_points),
        f"{prefix}first_serve_win_pct_before": safe_div(stats.first_serve_points_won, stats.first_serves_in),
        f"{prefix}second_serve_win_pct_before": safe_div(
            stats.second_serve_points_won, stats.service_points - stats.first_serves_in
        ),
        f"{prefix}service_points_won_pct_before": safe_div(
            stats.first_serve_points_won + stats.second_serve_points_won,
            stats.service_points,
        ),
        f"{prefix}break_points_saved_pct_before": safe_div(
            stats.break_points_saved, stats.break_points_faced
        ),
        f"{prefix}return_points_won_pct_before": safe_div(
            stats.return_points_won, stats.return_points_played
        ),
    }


def context_snapshot(record: ContextRecord, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_matches_before": float(record.matches),
        f"{prefix}_wins_before": float(record.wins),
        f"{prefix}_win_rate_before": safe_div(float(record.wins), float(record.matches)),
    }


def snapshot(
    state: PlayerState,
    surface: str,
    level: str,
    round_name: str,
    best_of: str,
    current_date: pd.Timestamp,
) -> dict[str, float]:
    result = {
        "career_matches_before": float(state.matches),
        "career_wins_before": float(state.wins),
        "career_win_rate_before": safe_div(float(state.wins), float(state.matches)),
        "surface_matches_before": float(state.surface_matches[surface]),
        "surface_wins_before": float(state.surface_wins[surface]),
        "surface_win_rate_before": safe_div(
            float(state.surface_wins[surface]), float(state.surface_matches[surface])
        ),
        "elo_before": float(state.elo),
        "surface_elo_before": blended_surface_elo(state, surface),
        "surface_elo_raw_before": float(state.surface_elo[surface]),
        "form_last_5_before": recent_rate(state.recent_results, 5),
        "form_last_10_before": recent_rate(state.recent_results, 10),
        "surface_form_last_5_before": recent_rate(state.surface_recent_results[surface], 5),
        "surface_form_last_10_before": recent_rate(state.surface_recent_results[surface], 10),
        "avg_minutes_last_5_before": recent_average(state.recent_minutes, 5),
        "days_since_last_match": days_since(state.last_date, current_date),
        "elo_change_last_5_before": recent_change(state.elo_history, state.elo),
        "surface_elo_change_last_5_before": recent_change(
            state.surface_elo_history[surface], state.surface_elo[surface]
        ),
        "avg_opponent_elo_last_5_before": recent_average(state.opponent_elos, 5),
        "avg_opponent_rank_last_5_before": recent_average(state.opponent_ranks, 5),
        "quality_adjusted_form_last_5_before": recent_average(state.quality_results, 5),
    }
    result.update(stats_snapshot(state.stats))
    result.update(stats_snapshot(state.surface_stats[surface], "surface_"))
    result.update(context_snapshot(state.level_records[level], "level"))
    result.update(context_snapshot(state.round_records[round_name], "round"))
    result.update(context_snapshot(state.best_of_records[best_of], "best_of"))
    return result


def update_stats_accumulator(
    stats: StatsAccumulator,
    *,
    ace: float,
    double_fault: float,
    service_points: float,
    first_in: float,
    first_won: float,
    second_won: float,
    service_games: float,
    bp_saved: float,
    bp_faced: float,
    opponent_service_points: float,
    opponent_first_won: float,
    opponent_second_won: float,
) -> None:
    required = [service_points, first_in, first_won, second_won, service_games]
    valid_service = (
        all(not math.isnan(value) for value in required)
        and service_points > 0 and service_games > 0
        and 0 <= first_in <= service_points
        and 0 <= first_won <= first_in
        and 0 <= second_won <= service_points - first_in
    )
    if valid_service:
        stats.matches += 1
        stats.aces += 0.0 if math.isnan(ace) else max(ace, 0.0)
        stats.double_faults += 0.0 if math.isnan(double_fault) else max(double_fault, 0.0)
        stats.service_points += service_points
        stats.first_serves_in += first_in
        stats.first_serve_points_won += first_won
        stats.second_serve_points_won += second_won
        stats.service_games += service_games
        if not math.isnan(bp_saved) and not math.isnan(bp_faced) and 0 <= bp_saved <= bp_faced:
            stats.break_points_saved += bp_saved
            stats.break_points_faced += bp_faced

    valid_return = (
        not math.isnan(opponent_service_points)
        and not math.isnan(opponent_first_won)
        and not math.isnan(opponent_second_won)
        and opponent_service_points > 0
        and opponent_first_won >= 0
        and opponent_second_won >= 0
        and opponent_first_won + opponent_second_won <= opponent_service_points
    )
    if valid_return:
        stats.return_points_won += (
            opponent_service_points - opponent_first_won - opponent_second_won
        )
        stats.return_points_played += opponent_service_points


def update_player_state(
    state: PlayerState,
    *,
    won: bool,
    surface: str,
    level: str,
    round_name: str,
    best_of: str,
    match_date: pd.Timestamp,
    minutes: float,
    opponent_elo_before: float,
    opponent_rank: float,
    expected_probability: float,
    identity: dict[str, Any],
    service_stats: dict[str, float],
    return_stats: dict[str, float],
) -> None:
    state.elo_history.append(float(state.elo))
    state.surface_elo_history[surface].append(float(state.surface_elo[surface]))
    state.matches += 1
    if state.comeback_k_matches_remaining > 0:
        state.comeback_k_matches_remaining -= 1
    state.wins += int(won)
    state.losses += int(not won)
    state.surface_matches[surface] += 1
    state.surface_wins[surface] += int(won)
    state.recent_results.append(int(won))
    state.surface_recent_results[surface].append(int(won))
    if not math.isnan(minutes):
        state.recent_minutes.append(minutes)
    state.opponent_elos.append(opponent_elo_before)
    if not math.isnan(opponent_rank):
        state.opponent_ranks.append(opponent_rank)
    state.quality_results.append(float(int(won) - expected_probability))
    state.last_date = pd.Timestamp(match_date)
    state.last_surface = surface

    for record in (
        state.level_records[level], state.round_records[round_name],
        state.best_of_records[best_of],
    ):
        record.matches += 1
        record.wins += int(won)

    update_stats_accumulator(state.stats, **service_stats, **return_stats)
    update_stats_accumulator(state.surface_stats[surface], **service_stats, **return_stats)

    state.player_id = identity.get("id")
    state.player_name = identity.get("name")
    state.player_hand = identity.get("hand")
    state.player_height = identity.get("height", math.nan)
    state.player_ioc = identity.get("ioc")
    state.player_age = identity.get("age", math.nan)
    state.player_rank = identity.get("rank", math.nan)
    state.player_rank_points = identity.get("rank_points", math.nan)


def player_key(player_id: Any, player_name: Any) -> int:
    return int(player_id)


def choose_player_1(tourney_id: Any, competition_type: Any, match_num: Any, winner_id: Any, loser_id: Any) -> bool:
    return int(winner_id) < int(loser_id)



def parse_match_date(frame: pd.DataFrame) -> pd.Series:
    sackmann_date = pd.to_datetime(
        pd.to_numeric(frame["tourney_date"], errors="coerce").astype("Int64").astype(str),
        format="%Y%m%d", errors="coerce",
    )
    if "td_date" in frame.columns:
        return pd.to_datetime(frame["td_date"], errors="coerce").fillna(sackmann_date)
    return sackmann_date


def validate_input(frame: pd.DataFrame) -> None:
    missing = sorted(MANDATORY_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")
    duplicate = frame.duplicated(["tourney_id", "competition_type", "match_num"])
    if duplicate.any():
        raise ValueError(f"Claves de partido duplicadas: {int(duplicate.sum()):,}")
    missing_ids = frame[["winner_id", "loser_id"]].isna().any(axis=1)
    if missing_ids.any():
        raise ValueError(f"Partidos con IDs ausentes: {int(missing_ids.sum()):,}")


def normalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["surface"] = result["surface"].map(normalized_surface).astype("string")
    result["competition_type"] = result["competition_type"].astype("string").str.strip().str.upper()
    result["round"] = result["round"].astype("string").str.strip().str.upper()
    result["tourney_level"] = result.get("tourney_level", pd.Series("U", index=result.index)).astype("string").str.strip().str.upper()
    result["source_origin"] = result.get("source_origin", pd.Series("UNKNOWN", index=result.index)).astype("string").str.strip().str.upper()
    if "indoor" in result.columns:
        result["indoor"] = result["indoor"].astype("boolean")
    return result


def array_value(columns: dict[str, np.ndarray], name: str, index: int, default: Any = None) -> Any:
    values=columns.get(name); return default if values is None else values[index]


def identity_from_columns(columns: dict[str, np.ndarray], prefix: str, index: int) -> dict[str, Any]:
    return {
        "id": int(columns[f"{prefix}_id"][index]), "name": columns[f"{prefix}_name"][index],
        "hand": array_value(columns,f"{prefix}_hand",index), "height": safe_float(array_value(columns,f"{prefix}_ht",index,math.nan)),
        "ioc": array_value(columns,f"{prefix}_ioc",index), "age": safe_float(array_value(columns,f"{prefix}_age",index,math.nan)),
        "rank": safe_float(columns[f"{prefix}_rank"][index]), "rank_points": safe_float(columns[f"{prefix}_rank_points"][index]),
    }



def write_snapshot_to_arrays(arrays, side, index, state, surface, level, round_name, best_of, current_date):
    prefix=f"player_{side}_"
    def put(name,value): arrays[prefix+name][index]=value
    sm=state.surface_matches[surface]; sw=state.surface_wins[surface]
    put("career_matches_before",state.matches); put("career_wins_before",state.wins); put("career_win_rate_before",safe_div(state.wins,state.matches))
    put("surface_matches_before",sm); put("surface_wins_before",sw); put("surface_win_rate_before",safe_div(sw,sm))
    put("elo_before",state.elo); put("surface_elo_before",blended_surface_elo(state,surface)); put("surface_elo_raw_before",state.surface_elo[surface])
    put("form_last_5_before",recent_rate(state.recent_results,5)); put("form_last_10_before",recent_rate(state.recent_results,10))
    put("surface_form_last_5_before",recent_rate(state.surface_recent_results[surface],5)); put("surface_form_last_10_before",recent_rate(state.surface_recent_results[surface],10))
    put("avg_minutes_last_5_before",recent_average(state.recent_minutes,5)); put("days_since_last_match",days_since(state.last_date,current_date))
    put("elo_change_last_5_before",recent_change(state.elo_history,state.elo)); put("surface_elo_change_last_5_before",recent_change(state.surface_elo_history[surface],state.surface_elo[surface]))
    put("avg_opponent_elo_last_5_before",recent_average(state.opponent_elos,5)); put("avg_opponent_rank_last_5_before",recent_average(state.opponent_ranks,5)); put("quality_adjusted_form_last_5_before",recent_average(state.quality_results,5))
    def stats(st,p=""):
        put(p+"stat_matches_before",st.matches); put(p+"aces_per_service_game_before",safe_div(st.aces,st.service_games)); put(p+"double_faults_per_service_game_before",safe_div(st.double_faults,st.service_games))
        put(p+"first_serve_in_pct_before",safe_div(st.first_serves_in,st.service_points)); put(p+"first_serve_win_pct_before",safe_div(st.first_serve_points_won,st.first_serves_in)); put(p+"second_serve_win_pct_before",safe_div(st.second_serve_points_won,st.service_points-st.first_serves_in))
        put(p+"service_points_won_pct_before",safe_div(st.first_serve_points_won+st.second_serve_points_won,st.service_points)); put(p+"break_points_saved_pct_before",safe_div(st.break_points_saved,st.break_points_faced)); put(p+"return_points_won_pct_before",safe_div(st.return_points_won,st.return_points_played))
    stats(state.stats); stats(state.surface_stats[surface],"surface_")
    for pfx,rec in (("level",state.level_records[level]),("round",state.round_records[round_name]),("best_of",state.best_of_records[best_of])):
        put(pfx+"_matches_before",rec.matches); put(pfx+"_wins_before",rec.wins); put(pfx+"_win_rate_before",safe_div(rec.wins,rec.matches))



def reconciliation_normalize_text(value: Any) -> str:
    if value is None or pd.isna(value): return ""
    text=unicodedata.normalize("NFKD",str(value)); text="".join(c for c in text if not unicodedata.combining(c)); text=text.casefold().replace("-"," ").replace("'"," ")
    return " ".join(re.sub(r"[^a-z0-9]+"," ",text).split())


def reconciliation_parse_date(frame: pd.DataFrame) -> pd.Series:
    sack=pd.to_datetime(pd.to_numeric(frame["tourney_date"],errors="coerce").astype("Int64").astype(str),format="%Y%m%d",errors="coerce")
    return pd.to_datetime(frame["td_date"],errors="coerce").fillna(sack) if "td_date" in frame else sack


def reconciliation_observations(frame: pd.DataFrame) -> pd.DataFrame:
    parts=[]
    for side,opp in (("winner","loser"),("loser","winner")):
        parts.append(pd.DataFrame({"player_id":pd.to_numeric(frame[f"{side}_id"],errors="coerce"),"name":frame[f"{side}_name"].astype("string"),"opponent_id":pd.to_numeric(frame[f"{opp}_id"],errors="coerce"),"date":frame["_rec_date"],"ioc":frame.get(f"{side}_ioc",pd.Series(pd.NA,index=frame.index)),"hand":frame.get(f"{side}_hand",pd.Series(pd.NA,index=frame.index)),"height":pd.to_numeric(frame.get(f"{side}_ht",pd.Series(np.nan,index=frame.index)),errors="coerce"),"age":pd.to_numeric(frame.get(f"{side}_age",pd.Series(np.nan,index=frame.index)),errors="coerce"),"tml_id":pd.to_numeric(frame.get(f"{side}_id_tml",pd.Series(np.nan,index=frame.index)),errors="coerce")}))
    obs=pd.concat(parts,ignore_index=True); obs=obs.loc[obs.player_id.notna()].copy(); obs.player_id=obs.player_id.astype("int64"); obs.opponent_id=obs.opponent_id.astype("Int64"); obs.tml_id=obs.tml_id.astype("Int64")
    obs["norm_name"]=obs.name.map(reconciliation_normalize_text); obs["ioc_norm"]=obs.ioc.astype("string").str.strip().str.upper().replace({"<NA>":""}); obs["hand_norm"]=obs.hand.astype("string").str.strip().str.upper().replace({"<NA>":""}); return obs


def reconciliation_profiles(obs: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for pid,g in obs.groupby("player_id",sort=False):
        mode=g.norm_name.mode(); vals=lambda c:{str(v) for v in g[c] if pd.notna(v) and str(v) not in {"","<NA>"}}
        rows.append({"player_id":int(pid),"norm_name":mode.iat[0] if not mode.empty else "","rows":len(g),"first":g.date.min(),"last":g.date.max(),"iocs":vals("ioc_norm"),"hands":vals("hand_norm"),"hmin":g.height.min(),"hmax":g.height.max(),"amin":g.age.min(),"amax":g.age.max(),"synthetic":int(pid)>=SYNTHETIC_ID_MIN})
    return pd.DataFrame(rows)


def reconciliation_compatible(a: pd.Series,b: pd.Series,opponents:set[tuple[int,int]],explicit:bool)->bool:
    if a.norm_name=="" or a.norm_name!=b.norm_name or tuple(sorted((int(a.player_id),int(b.player_id)))) in opponents:return False
    set_ok=lambda x,y:not x or not y or bool(x&y); range_ok=lambda a1,a2,b1,b2,t:any(pd.isna(v) for v in (a1,a2,b1,b2)) or max(a1,b1)<=min(a2,b2)+t
    metadata=set_ok(a.iocs,b.iocs) and set_ok(a.hands,b.hands) and range_ok(a.hmin,a.hmax,b.hmin,b.hmax,IDENTITY_HEIGHT_TOLERANCE_CM) and range_ok(a.amin,a.amax,b.amin,b.amax,IDENTITY_AGE_TOLERANCE_YEARS)
    overlap=(min(a.last,b.last)-max(a.first,b.first)).days if pd.notna(a.first) and pd.notna(b.first) else 0
    return metadata and (explicit or overlap<=IDENTITY_MAX_OVERLAP_DAYS)


def reconciliation_aliases(frame:pd.DataFrame)->tuple[dict[int,int],list[dict[str,Any]]]:
    obs=reconciliation_observations(frame); prof=reconciliation_profiles(obs); by=prof.set_index("player_id",drop=False); opponents={tuple(sorted((int(a),int(b)))) for a,b in obs.loc[obs.opponent_id.notna(),["player_id","opponent_id"]].itertuples(index=False,name=None) if int(a)!=int(b)}; mapping={}; audit=[]
    links=obs.loc[obs.tml_id.notna()&obs.player_id.ne(obs.tml_id),["player_id","tml_id"]].drop_duplicates()
    for a,b in links.itertuples(index=False,name=None):
        a,b=int(a),int(b)
        if a not in by.index or b not in by.index:continue
        real=[v for v in (a,b) if v<SYNTHETIC_ID_MIN]; canonical=min(real) if real else min(a,b); alias=b if canonical==a else a
        if reconciliation_compatible(by.loc[alias],by.loc[canonical],opponents,True):mapping[alias]=canonical;audit.append({"alias_id":alias,"canonical_id":canonical,"rule":"explicit_tml_link"})
    for name,g in prof.groupby("norm_name",sort=False):
        if not name or len(g)<2:continue
        real=g.loc[~g.synthetic].sort_values(["rows","player_id"],ascending=[False,True]); synth=g.loc[g.synthetic]
        for _,alias in synth.iterrows():
            if int(alias.player_id) in mapping or real.empty:continue
            candidates=[c for _,c in real.iterrows() if reconciliation_compatible(alias,c,opponents,False)]
            if len(candidates)==1:
                canonical=int(candidates[0].player_id);mapping[int(alias.player_id)]=canonical;audit.append({"alias_id":int(alias.player_id),"canonical_id":canonical,"rule":"synthetic_exact_name_metadata"})
    for alias,canonical in MANUAL_PLAYER_ID_ALIASES.items():
        alias,canonical=int(alias),int(canonical)
        if alias==canonical or alias not in by.index or canonical not in by.index or tuple(sorted((alias,canonical))) in opponents:raise ValueError(f"Alias manual invalido: {alias} -> {canonical}")
        mapping[alias]=canonical;audit.append({"alias_id":alias,"canonical_id":canonical,"rule":"manual_verified_override"})
    def resolve(v):
        seen=set()
        while v in mapping and v not in seen:seen.add(v);v=mapping[v]
        return v
    unique={(int(x["alias_id"]),int(x["canonical_id"]),str(x["rule"])):x for x in audit}
    return {a:resolve(c) for a,c in mapping.items()},list(unique.values())


def reconciliation_apply_aliases(frame:pd.DataFrame,mapping:dict[int,int])->pd.DataFrame:
    out=frame.copy()
    for side in ("winner","loser"):
        for col in (f"{side}_id",f"{side}_id_tml"):
            if col in out:
                values=pd.to_numeric(out[col],errors="coerce");out[col]=values.map(lambda v:mapping.get(int(v),int(v)) if pd.notna(v) else v).astype("Int64")
    return out


def reconciliation_tourney_family(row:pd.Series)->str:
    for col in ("tournament_key","canonical_tournament_name","tourney_id","tourney_name"):
        value=row.get(col)
        if pd.notna(value) and str(value).strip():
            text=reconciliation_normalize_text(value)
            text=re.sub(r"\b(ch|challenger|masters|open|atp)\b"," ",text);text=" ".join(text.split())
            if text:return text
    return ""


def reconciliation_round(value:Any)->str:
    return "" if value is None or pd.isna(value) else str(value).strip().upper()


def reconciliation_score(value:Any)->str:
    if value is None or pd.isna(value):return ""
    text=str(value).upper().replace(" ","").replace("RET.","RET").replace("W/O","WO")
    return re.sub(r"[^A-Z0-9\-()]","",text)


def reconciliation_source_set(value:Any)->set[str]:
    if value is None or pd.isna(value):return set()
    text=str(value).upper(); result=set()
    if "SACKMANN" in text:result.add("SACKMANN")
    if "TML" in text:result.add("TML")
    return result


def reconciliation_cross_source(left:pd.Series,right:pd.Series)->bool:
    a=reconciliation_source_set(left.get("source_origin"));b=reconciliation_source_set(right.get("source_origin"))
    return bool(a and b and (a!=b or len(a|b)>1))


def reconciliation_pair_is_duplicate(left:pd.Series,right:pd.Series)->bool:
    if not reconciliation_cross_source(left,right):return False
    if int(left.winner_id)!=int(right.winner_id):return False
    lf,rf=left._rec_family,right._rec_family
    if not lf or not rf or lf!=rf:return False
    lr,rr=left._rec_round,right._rec_round
    if lr and rr and lr!=rr:return False
    ls,rs=left._rec_score,right._rec_score
    if ls and rs and ls!=rs:return False
    return True


def reconciliation_quality(frame:pd.DataFrame)->pd.Series:
    score=frame.get("source_origin",pd.Series("UNKNOWN",index=frame.index)).astype("string").str.upper().map(RECONCILIATION_SOURCE_PRIORITY).fillna(0).astype(float)
    if "source_priority" in frame:score+=pd.to_numeric(frame.source_priority,errors="coerce").fillna(0)
    stats=[c for c in RECONCILIATION_STAT_COLUMNS if c in frame];odds=[c for c in RECONCILIATION_ODDS_COLUMNS if c in frame]
    if stats:score+=frame[stats].notna().sum(axis=1)*10
    if odds:score+=frame[odds].notna().sum(axis=1)*2
    if "odds_matched" in frame:score+=frame.odds_matched.fillna(False).astype(bool).astype(int)*20
    return score


def reconciliation_coalesce(group:pd.DataFrame)->pd.Series:
    ordered=group.sort_values(["_rec_quality","_rec_row"],ascending=[False,True],kind="mergesort");base=ordered.iloc[0].copy()
    for _,candidate in ordered.iloc[1:].iterrows():
        for col in ordered.columns:
            if (pd.isna(base[col]) or base[col]=="") and pd.notna(candidate[col]) and candidate[col]!="":base[col]=candidate[col]
    origins=sorted(set(ordered.source_origin.dropna().astype(str))) if "source_origin" in ordered else [];base["source_origin"]="SACKMANN_TML" if len(origins)>1 else (origins[0] if origins else "UNKNOWN");return base


def reconcile_input_frame(frame:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    work=frame.copy();work["_rec_date"]=reconciliation_parse_date(work);aliases,audit=reconciliation_aliases(work);work=reconciliation_apply_aliases(work,aliases)
    winner=pd.to_numeric(work.winner_id,errors="coerce");loser=pd.to_numeric(work.loser_id,errors="coerce");work["_rec_low"]=np.minimum(winner,loser).astype("Int64");work["_rec_high"]=np.maximum(winner,loser).astype("Int64");work["_rec_comp"]=work.competition_type.astype("string").str.strip().str.upper();work["_rec_family"]=work.apply(reconciliation_tourney_family,axis=1);work["_rec_round"]=work["round"].map(reconciliation_round);work["_rec_score"]=work["score"].map(reconciliation_score);work["_rec_quality"]=reconciliation_quality(work);work["_rec_row"]=np.arange(len(work),dtype=np.int64)
    same=work._rec_low.eq(work._rec_high);same_rows=int(same.sum());work=work.loc[~same].copy();broad=["_rec_date","_rec_comp","_rec_low","_rec_high"];counts=work.groupby(broad,dropna=False).size();candidate_keys=counts[counts.gt(1)].index;mask=pd.MultiIndex.from_frame(work[broad]).isin(candidate_keys);candidates=work.loc[mask];clean=work.loc[~mask]
    kept=[];merged_groups=0;merged_rows_removed=0;conflict_groups_preserved=0;conflict_rows_preserved=0
    for _,group in candidates.groupby(broad,dropna=False,sort=False):
        indices=list(group.index);parent={i:i for i in indices}
        def find(x):
            while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
            return x
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb:parent[rb]=ra
        rows={i:group.loc[i] for i in indices}
        for pos,a in enumerate(indices):
            for b in indices[pos+1:]:
                if reconciliation_pair_is_duplicate(rows[a],rows[b]):union(a,b)
        clusters=defaultdict(list)
        for i in indices:clusters[find(i)].append(i)
        group_had_conflict=pd.to_numeric(group.winner_id,errors="coerce").nunique(dropna=True)>1
        if group_had_conflict:conflict_groups_preserved+=1;conflict_rows_preserved+=len(group)
        for members in clusters.values():
            cluster=group.loc[members]
            if len(cluster)>1:
                kept.append(reconciliation_coalesce(cluster));merged_groups+=1;merged_rows_removed+=len(cluster)-1
            else:kept.append(cluster.iloc[0])
    reconciled_candidates=pd.DataFrame(kept) if kept else pd.DataFrame(columns=work.columns);output=pd.concat([clean,reconciled_candidates],ignore_index=True,sort=False);output=output.drop(columns=[c for c in output if c.startswith("_rec_")],errors="ignore")
    if output[["winner_id","loser_id"]].isna().any().any():raise RuntimeError("Reconciliacion produjo IDs nulos")
    if pd.to_numeric(output.winner_id,errors="coerce").eq(pd.to_numeric(output.loser_id,errors="coerce")).any():raise RuntimeError("Reconciliacion dejo same-player")
    if output.duplicated(["tourney_id","competition_type","match_num"]).any():raise RuntimeError("Reconciliacion dejo claves operacionales duplicadas")
    summary={"policy":"conservative_cross_source_same_tournament_round_score","input_rows_before_reconciliation":len(frame),"output_rows_after_reconciliation":len(output),"accepted_identity_aliases":len(aliases),"identity_aliases":audit,"same_player_rows_removed":same_rows,"broad_semantic_candidate_rows":len(candidates),"high_confidence_duplicate_groups_merged":merged_groups,"duplicate_rows_removed":merged_rows_removed,"conflicting_broad_groups_preserved":conflict_groups_preserved,"conflicting_rows_preserved":conflict_rows_preserved,"semantic_conflict_rows_removed":0,"total_rows_removed":len(frame)-len(output)}
    print("\nReconciliacion integrada conservadora:");[print(f"  {k}: {v}") for k,v in summary.items() if k!="identity_aliases"];return output,summary


def build_current_state(states: dict[str, PlayerState], data_max_date: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in states.values():
        if state.player_id is None:
            continue
        global_stats = stats_snapshot(state.stats)
        row: dict[str, Any] = {
            "player_id": state.player_id, "player_name": state.player_name,
            "player_hand": state.player_hand, "player_height": state.player_height,
            "player_ioc": state.player_ioc, "player_age": state.player_age,
            "player_rank": state.player_rank, "player_rank_points": state.player_rank_points,
            "state_as_of_date": data_max_date, "last_match_date": state.last_date,
            "last_surface": state.last_surface,
            "days_since_last_recorded_match_at_data_max": days_since(state.last_date, data_max_date),
            "career_matches": state.matches, "career_wins": state.wins,
            "career_win_rate": safe_div(state.wins, state.matches), "elo": state.elo,
            "last_absence_penalty": state.last_absence_penalty,
            "last_absence_days": state.last_absence_days,
            "comeback_k_matches_remaining": state.comeback_k_matches_remaining,
            "form_last_5": recent_rate(state.recent_results, 5),
            "form_last_10": recent_rate(state.recent_results, 10),
            "avg_minutes_last_5": recent_average(state.recent_minutes, 5),
            "elo_change_last_5": recent_change(state.elo_history, state.elo),
            "avg_opponent_elo_last_5": recent_average(state.opponent_elos, 5),
            "avg_opponent_rank_last_5": recent_average(state.opponent_ranks, 5),
            "quality_adjusted_form_last_5": recent_average(state.quality_results, 5),
            "synthetic_player": int(state.player_id >= SYNTHETIC_ID_MIN),
        }
        row.update({key.removesuffix("_before"): value for key, value in global_stats.items()})
        for surface in KNOWN_SURFACES:
            key = surface.casefold()
            surface_stats = stats_snapshot(state.surface_stats[surface], "surface_")
            row.update({
                f"{key}_matches": state.surface_matches[surface],
                f"{key}_wins": state.surface_wins[surface],
                f"{key}_win_rate": safe_div(state.surface_wins[surface], state.surface_matches[surface]),
                f"{key}_elo": blended_surface_elo(state, surface),
                f"{key}_elo_raw": state.surface_elo[surface],
                f"{key}_form_last_5": recent_rate(state.surface_recent_results[surface], 5),
                f"{key}_form_last_10": recent_rate(state.surface_recent_results[surface], 10),
                f"{key}_elo_change_last_5": recent_change(state.surface_elo_history[surface], state.surface_elo[surface]),
            })
            for column, value in surface_stats.items():
                clean = column.replace("surface_", "").removesuffix("_before")
                row[f"{key}_{clean}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["player_name", "player_id"], kind="mergesort").reset_index(drop=True)


def build_features(frame, *, progress_every_rows=10000, progress_every_seconds=5.0, show_progress=True):
    reporter=ProgressReporter(len(frame),progress_every_rows,progress_every_seconds,show_progress)
    reporter.stage("Fase 1/6: validando y normalizando la entrada")
    validate_input(frame)
    frame = normalize_input(frame)
    frame["match_date"] = parse_match_date(frame)
    frame["_original_row"] = np.arange(len(frame), dtype=np.int32)
    frame["_competition_order"] = frame["competition_type"].map(COMPETITION_ORDER).fillna(999).astype("int16")
    frame["_round_order"] = frame["round"].map(ROUND_ORDER).fillna(999).astype("int16")
    reporter.stage("Fase 2/6: ordenando cronologicamente los partidos")
    frame = frame.sort_values(
        ["match_date", "tourney_id", "_competition_order", "_round_order", "match_num", "_original_row"],
        kind="mergesort", na_position="last",
    ).reset_index(drop=True)
    frame["processing_sequence"] = np.arange(len(frame),dtype=np.int32)

    reporter.stage("Fase 3/6: inicializando estados y matrices de features")
    states: dict[int, PlayerState] = defaultdict(PlayerState)
    h2h = defaultdict(lambda: [0, 0])
    arrays = {
        f"player_{side}_{feature}": np.full(len(frame), np.nan, dtype=np.float32)
        for side in (1, 2) for feature in FEATURE_NAMES
    }
    arrays["player_1_h2h_wins_before"] = np.zeros(len(frame), dtype=np.float32)
    arrays["player_2_h2h_wins_before"] = np.zeros(len(frame), dtype=np.float32)
    arrays["player_1_h2h_win_rate_before"] = np.full(len(frame), np.nan, dtype=np.float32)

    p1_is_winner = np.zeros(len(frame), dtype=bool)
    eligible = np.ones(len(frame), dtype=bool)
    reasons = np.full(len(frame), "", dtype=object)
    loop_columns = list(
        MANDATORY_COLUMNS
        | {
            "match_date",
            "winner_hand",
            "winner_ht",
            "winner_ioc",
            "winner_age",
            "loser_hand",
            "loser_ht",
            "loser_ioc",
            "loser_age",
            "tourney_level",
            "best_of",
        }
    )
    columns = {
        name: frame[name].to_numpy(copy=False)
        for name in loop_columns if name in frame.columns
    }
    
    required_loop_columns = {
        "match_date",
        "winner_id",
        "loser_id",
        "winner_name",
        "loser_name",
        "surface",
        "tourney_level",
        "round",
        "best_of",
        "tourney_id",
        "competition_type",
        "match_num",
    }

    missing_loop_columns = sorted(
        required_loop_columns
        - set(columns)
    )

    if missing_loop_columns:
        raise ValueError(
            "Faltan columnas requeridas para procesar "
            f"los partidos: {missing_loop_columns}"
        )

    reporter.stage("Fase 4/6: calculando features prepartido")
    reporter.update(0,force=True)
    for i in range(len(frame)):
        winner_id, loser_id = columns["winner_id"][i], columns["loser_id"][i]
        winner_name, loser_name = columns["winner_name"][i], columns["loser_name"][i]
        surface=columns["surface"][i]; level=columns["tourney_level"][i]; round_name=columns["round"][i]
        best_value=columns["best_of"][i]; best_of="UNKNOWN" if pd.isna(best_value) else str(best_value)
        raw_date = columns["match_date"][i]
        missing_date = pd.isna(raw_date)
        match_date = MIN_DATE if missing_date else pd.Timestamp(raw_date)

        winner_key, loser_key = player_key(winner_id, winner_name), player_key(loser_id, loser_name)
        competition_type = normalized_context(columns["competition_type"][i])

        if is_walkover(columns["score"][i]):
            eligible[i], reasons[i] = False, "walkover"
        elif missing_date:
            eligible[i], reasons[i] = False, "missing_match_date"
        elif winner_key == loser_key:
            eligible[i], reasons[i] = False, "same_player_identity"

        if not eligible[i]:
            p1_is_winner[i]=int(winner_id)<int(loser_id); reporter.update(i+1,match_date=raw_date,players=len(states)); continue
        winner_state, loser_state = states[winner_key], states[loser_key]
        ensure_elo_initialized(winner_state,surface,competition_type); ensure_elo_initialized(loser_state,surface,competition_type)

        winner_as_p1 = choose_player_1(
            columns["tourney_id"][i], columns["competition_type"][i],
            columns["match_num"][i], winner_id, loser_id,
        )
        p1_is_winner[i] = winner_as_p1
        p1_state,p2_state=((winner_state,loser_state) if winner_as_p1 else (loser_state,winner_state))
        write_snapshot_to_arrays(arrays,1,i,p1_state,surface,level,round_name,best_of,match_date)
        write_snapshot_to_arrays(arrays,2,i,p2_state,surface,level,round_name,best_of,match_date)

        pair_key = tuple(sorted((winner_key, loser_key)))
        record = h2h[pair_key]
        winner_prior = record[0] if winner_key == pair_key[0] else record[1]
        loser_prior = record[1] if winner_key == pair_key[0] else record[0]
        p1_h2h, p2_h2h = (
            (winner_prior, loser_prior) if winner_as_p1 else (loser_prior, winner_prior)
        )
        total_h2h = p1_h2h + p2_h2h
        arrays["player_1_h2h_wins_before"][i] = p1_h2h
        arrays["player_2_h2h_wins_before"][i] = p2_h2h
        arrays["player_1_h2h_win_rate_before"][i] = p1_h2h / total_h2h if total_h2h else np.nan


        winner_elo_before, loser_elo_before = winner_state.elo, loser_state.elo
        win_expected = expected_score(winner_elo_before, loser_elo_before)
        global_k = 0.5 * (
            dynamic_k(winner_state.matches) * comeback_multiplier(winner_state)
            + dynamic_k(loser_state.matches) * comeback_multiplier(loser_state)
        )
        global_change = global_k * (1.0 - win_expected)
        winner_state.elo += global_change
        loser_state.elo -= global_change

        winner_surface_before, loser_surface_before = (
            winner_state.surface_elo[surface], loser_state.surface_elo[surface]
        )
        surface_expected = expected_score(winner_surface_before, loser_surface_before)
        surface_k = 0.5 * (
            dynamic_k(winner_state.surface_matches[surface]) * comeback_multiplier(winner_state)
            + dynamic_k(loser_state.surface_matches[surface]) * comeback_multiplier(loser_state)
        )
        surface_change = surface_k * (1.0 - surface_expected)
        winner_state.surface_elo[surface] += surface_change
        loser_state.surface_elo[surface] -= surface_change

        winner_identity = identity_from_columns(columns, "winner", i)
        loser_identity = identity_from_columns(columns, "loser", i)
        minutes = safe_float(columns["minutes"][i])
        winner_service = {
            "ace": safe_float(columns["w_ace"][i]),
            "double_fault": safe_float(columns["w_df"][i]),
            "service_points": safe_float(columns["w_svpt"][i]),
            "first_in": safe_float(columns["w_1stIn"][i]),
            "first_won": safe_float(columns["w_1stWon"][i]),
            "second_won": safe_float(columns["w_2ndWon"][i]),
            "service_games": safe_float(columns["w_SvGms"][i]),
            "bp_saved": safe_float(columns["w_bpSaved"][i]),
            "bp_faced": safe_float(columns["w_bpFaced"][i]),
        }
        loser_service = {
            "ace": safe_float(columns["l_ace"][i]),
            "double_fault": safe_float(columns["l_df"][i]),
            "service_points": safe_float(columns["l_svpt"][i]),
            "first_in": safe_float(columns["l_1stIn"][i]),
            "first_won": safe_float(columns["l_1stWon"][i]),
            "second_won": safe_float(columns["l_2ndWon"][i]),
            "service_games": safe_float(columns["l_SvGms"][i]),
            "bp_saved": safe_float(columns["l_bpSaved"][i]),
            "bp_faced": safe_float(columns["l_bpFaced"][i]),
        }
        winner_return = {
            "opponent_service_points": loser_service["service_points"],
            "opponent_first_won": loser_service["first_won"],
            "opponent_second_won": loser_service["second_won"],
        }
        loser_return = {
            "opponent_service_points": winner_service["service_points"],
            "opponent_first_won": winner_service["first_won"],
            "opponent_second_won": winner_service["second_won"],
        }
        update_player_state(
            winner_state, won=True, surface=surface, level=level,
            round_name=round_name, best_of=best_of, match_date=match_date,
            minutes=minutes, opponent_elo_before=loser_elo_before,
            opponent_rank=loser_identity["rank"], expected_probability=win_expected,
            identity=winner_identity, service_stats=winner_service,
            return_stats=winner_return,
        )
        update_player_state(
            loser_state, won=False, surface=surface, level=level,
            round_name=round_name, best_of=best_of, match_date=match_date,
            minutes=minutes, opponent_elo_before=winner_elo_before,
            opponent_rank=winner_identity["rank"],
            expected_probability=1.0 - win_expected, identity=loser_identity,
            service_stats=loser_service, return_stats=loser_return,
        )
        if winner_key == pair_key[0]:
            record[0] += 1
        else:
            record[1] += 1
        reporter.update(i+1,match_date=raw_date,players=len(states))

    reporter.stage("Fase 5/6: materializando columnas")
    generated={**arrays,"target_player_1_win":p1_is_winner.astype("int8"),"eligible_for_model":eligible,"model_exclusion_reason":pd.Series(reasons,dtype="string").replace("",pd.NA).to_numpy()}
    for suffix,(wc,lc) in IDENTITY_COLUMNS.items():
        if wc in frame and lc in frame:
            generated[f"player_1_{suffix}"]=np.where(p1_is_winner,frame[wc].to_numpy(),frame[lc].to_numpy()); generated[f"player_2_{suffix}"]=np.where(p1_is_winner,frame[lc].to_numpy(),frame[wc].to_numpy())
    for op,wc,lc in ODDS_PAIRS:
        if wc in frame and lc in frame:
            generated[f"player_1_{op}_odds"]=np.where(p1_is_winner,frame[wc].to_numpy(),frame[lc].to_numpy()); generated[f"player_2_{op}_odds"]=np.where(p1_is_winner,frame[lc].to_numpy(),frame[wc].to_numpy())
    for feature in DIFFERENTIAL_FEATURES: generated[f"diff_{feature}"]=(arrays[f"player_1_{feature}"]-arrays[f"player_2_{feature}"]).astype("float32")
    generated["diff_rank"]=(pd.to_numeric(generated["player_1_rank"],errors="coerce")-pd.to_numeric(generated["player_2_rank"],errors="coerce")).astype("float32")
    generated["diff_rank_points"]=(pd.to_numeric(generated["player_1_rank_points"],errors="coerce")-pd.to_numeric(generated["player_2_rank_points"],errors="coerce")).astype("float32")
    frame=pd.concat([frame,pd.DataFrame(generated,index=frame.index)],axis=1)

    frame = frame.drop(columns=["_original_row", "_competition_order", "_round_order"])
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(frame[column].dtype):
            frame[column] = frame[column].astype("string")

    data_max_date = pd.to_datetime(frame["match_date"], errors="coerce").max()
    current_state = build_current_state(states, data_max_date)
    return frame, current_state


def write_parquet_atomically(
    result: pd.DataFrame,
    output_path: Path,
    verification_columns: list[str],
    *,
    keep_backup: bool = False,
) -> None:
    temporary = output_path.with_suffix(".new.parquet")
    backup = output_path.with_suffix(".before_rebuild.parquet")
    temporary.unlink(missing_ok=True)
    result.to_parquet(
        temporary, index=False, engine="pyarrow", compression="zstd",
        row_group_size=100_000,
    )
    verification = pd.read_parquet(temporary, columns=verification_columns, engine="pyarrow")
    if len(verification) != len(result):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"La escritura temporal perdio filas: {output_path}")
    if keep_backup and output_path.exists() and not backup.exists():
        shutil.copy2(output_path, backup)
    temporary.replace(output_path)
    print(f"Creado: {output_path}")


def validate_features(result: pd.DataFrame) -> None:
    if result.duplicated(["tourney_id","competition_type","match_num"]).any(): raise RuntimeError("Claves duplicadas")
    if result[["player_1_id","player_2_id"]].isna().any().any(): raise RuntimeError("IDs orientados nulos")
    eligible=result["eligible_for_model"].fillna(False).astype(bool); x=result.loc[eligible]
    p1=pd.to_numeric(x["player_1_id"],errors="coerce"); p2=pd.to_numeric(x["player_2_id"],errors="coerce")
    if (~p1.lt(p2)).any(): raise RuntimeError("Orientacion invalida en filas elegibles")
    expected=p1.eq(pd.to_numeric(x["winner_id"],errors="coerce")).astype("int8")
    actual=pd.to_numeric(x["target_player_1_win"],errors="coerce").astype("int8")
    if (expected!=actual).any(): raise RuntimeError("Target inconsistente en filas elegibles")
    if x["match_date"].isna().any(): raise RuntimeError("Fecha nula en fila elegible")


def validate_current_state(current_state: pd.DataFrame) -> None:
    if current_state.empty: raise RuntimeError("Estado actual vacio")
    if current_state["player_id"].isna().any() or current_state["player_id"].duplicated().any(): raise RuntimeError("IDs invalidos en estado actual")
    elo=pd.to_numeric(current_state["elo"],errors="coerce")
    if elo.isna().any() or not np.isfinite(elo).all(): raise RuntimeError("Elo nulo o no finito")
    if elo.min()<0: raise RuntimeError(f"Elo minimo no plausible: {elo.min():.1f}")
    if elo.max()<1900: raise RuntimeError(f"Elo maximo no plausible: {elo.max():.1f}")



def create_report(
    result: pd.DataFrame,
    current_state: pd.DataFrame,
    input_rows: int,
    report_path: Path,
) -> None:
    eligible = result["eligible_for_model"].fillna(False).astype(bool)
    report = pd.DataFrame({
        "metric": [
            "rows", "columns", "generated_feature_columns", "eligible_for_model_rows",
            "excluded_from_model_rows", "current_state_players", "new_surface_features",
            "new_context_features", "new_opponent_quality_features", "first_match_date",
            "last_match_date", "duplicate_operational_keys", "player_1_id_nulls",
            "player_2_id_nulls", "elo_min", "elo_median", "elo_mean", "elo_max",
        ],
        "value": [
            input_rows, len(result.columns),
            len([c for c in result.columns if c.startswith(("player_1_", "player_2_", "diff_"))]),
            int(eligible.sum()), int((~eligible).sum()), len(current_state),
            len(SURFACE_FEATURES), len(CONTEXT_FEATURES), 3,
            str(result["match_date"].min()), str(result["match_date"].max()),
            int(result.duplicated(["tourney_id", "competition_type", "match_num"]).sum()),
            int(result["player_1_id"].isna().sum()), int(result["player_2_id"].isna().sum()),
            float(pd.to_numeric(current_state["elo"],errors="coerce").min()), float(pd.to_numeric(current_state["elo"],errors="coerce").median()),
            float(pd.to_numeric(current_state["elo"],errors="coerce").mean()), float(pd.to_numeric(current_state["elo"],errors="coerce").max()),
        ],
    })
    report.to_csv(report_path, index=False)
    print("\n" + report.to_string(index=False))
    print(f"\nInforme: {report_path}")


def parse_args() -> argparse.Namespace:
    root = find_project_root(Path(__file__).resolve().parent)
    processed = root / "data" / "processed"
    parser = argparse.ArgumentParser(description="Genera features cronologicas optimizadas")
    parser.add_argument("--input", default=str(processed / "jeff_sackmann_with_odds.parquet"))
    parser.add_argument("--output", default=str(processed / "tennis_matches_with_player_stats.sackmann_v5.parquet"))
    parser.add_argument("--current-state-output", default=str(processed / "player_current_state.sackmann_v5.parquet"))
    parser.add_argument("--report", default=str(processed / "tennis_matches_with_player_stats.sackmann_v5_report.csv"))
    parser.add_argument("--feature-manifest", default=str(processed / "tennis_match_feature_manifest.sackmann_v5.json"))
    parser.add_argument("--progress-every-rows",type=int,default=10000)
    parser.add_argument("--progress-every-seconds",type=float,default=5.0)
    parser.add_argument("--no-progress",action="store_true")
    parser.add_argument(
        "--keep-backup",
        action="store_true",
        help=(
            "Conserva una copia .before_rebuild.parquet si el destino ya existe. "
            "Por defecto solo se usa escritura atomica y no se deja un backup persistente."
        ),
    )
    return parser.parse_args()


def main() -> None:
    total_started=time.perf_counter()
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    current_state_path = Path(args.current_state_output).resolve()
    report_path = Path(args.report).resolve()
    manifest_path = Path(args.feature_manifest).resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(input_path)
    available = set(parquet.schema_arrow.names)
    selected = [column for column in REQUIRED_COLUMNS if column in available]
    print(f"Leyendo: {input_path}")
    frame = pd.read_parquet(input_path, columns=selected, engine="pyarrow")
    raw_input_rows=len(frame);print(f"Partidos cargados: {raw_input_rows:,}; columnas: {len(selected):,}")
    print("\nFase 0/6: reconciliando identidades y partidos", flush=True)
    frame, reconciliation_summary = reconcile_input_frame(frame)
    input_rows = len(frame)
    print(f"Partidos tras reconciliacion: {input_rows:,}", flush=True)

    result,current_state=build_features(frame,progress_every_rows=args.progress_every_rows,progress_every_seconds=args.progress_every_seconds,show_progress=not args.no_progress)
    del frame
    gc.collect()
    validate_features(result)
    validate_current_state(current_state)

    write_parquet_atomically(
        result, output_path,
        ["tourney_id", "competition_type", "match_num", "player_1_id",
         "player_2_id", "match_date", "eligible_for_model"],
        keep_backup=args.keep_backup,
    )
    write_parquet_atomically(
        current_state, current_state_path,
        ["player_id", "state_as_of_date", "last_match_date"],
        keep_backup=args.keep_backup,
    )
    create_report(result, current_state, input_rows, report_path)

    manifest = {
        "global_features": GLOBAL_FEATURES,
        "surface_features": SURFACE_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "differential_features": DIFFERENTIAL_FEATURES,
        "schema_version": "sackmann_v5",
        "match_features_output": str(output_path),
        "current_state_output": str(current_state_path),
        "report_output": str(report_path),
        "feature_manifest_output": str(manifest_path),
        "integrated_reconciliation": reconciliation_summary,
        "elo_methodology": {
            "implementation": "Sackmann-inspired transparent approximation",
            "entry_ratings": ELO_ENTRY_RATINGS,
            "k_formula": "250 / (matches + 5) ** 0.4",
            "zero_sum_match_updates": True,
            "surface_blend": ELO_SURFACE_BLEND,
            "historical_absence_penalty_applied": False,
            "inactivity_feature": "days_since_last_match",
            "absence_policy": "No Elo deduction in historical builder",
        },
        "data_max_date": str(pd.to_datetime(result["match_date"]).max()),
        "notes": [
            "All match features are pre-match and leakage-safe.",
            "player_current_state is post-match as of the latest recorded data date.",
            "No short-window physical workload features are generated.",
            "Tennis Abstract production code is not public; Elo parameters are a documented approximation.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print(f"Proceso completo en {ProgressReporter.duration(time.perf_counter()-total_started)}")


if __name__ == "__main__":
    main()
