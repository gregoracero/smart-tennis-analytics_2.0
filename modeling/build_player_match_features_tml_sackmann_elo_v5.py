#!/usr/bin/env python3
"""Builder cronologico v5 para Smart Tennis Analytics.

Amplia el builder v4 sin modificarlo y genera cinco familias nuevas:
1. Momentum y margen ajustado.
2. Carga, descanso y fatiga.
3. Servicio y devolucion dinamicos mediante EWMA.
4. Tendencias de ranking y fuerza latente.
5. Matchup, oponentes comunes e intransitividad aproximada.

El builder v4 sigue siendo la fuente de identidad, reconciliacion, Elo base,
estadisticas acumuladas, contexto y escritura segura. Esta version ejecuta v4
en memoria y realiza una segunda pasada cronologica estrictamente prepartido.

Salidas por defecto
-------------------
data/processed/tennis_matches_with_player_stats.sackmann_v5.parquet
data/processed/player_current_state.sackmann_v5.parquet
data/processed/player_pair_current_state.sackmann_v5.parquet
data/processed/tennis_matches_with_player_stats.sackmann_v5_report.csv
data/processed/tennis_match_feature_manifest.sackmann_v5.json

Garantia anti-leakage
---------------------
Para cada partido se sigue siempre este orden:
    snapshot -> escritura de features -> actualizacion con el partido actual.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCHEMA_VERSION = 5
MAX_ACTIVITY_DAYS = 196
EWMA_HALFLIFE_SHORT_DAYS = 30.0
EWMA_HALFLIFE_LONG_DAYS = 90.0
PAIR_RECENT_WINDOW = 5
COMMON_OPPONENT_LOOKBACK_DAYS = 365
COMMON_OPPONENT_MAX = 40
MIN_DATE = pd.Timestamp("1968-01-01")

MOMENTUM_FEATURES = [
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
WORKLOAD_FEATURES = [
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
]
DYNAMIC_STATS_FEATURES = [
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
RANKING_TREND_FEATURES = [
    "rank_4w_ago_before", "rank_8w_ago_before", "rank_26w_ago_before",
    "rank_change_4w_before", "rank_change_8w_before",
    "rank_change_26w_before", "rank_points_4w_ago_before",
    "rank_points_8w_ago_before", "rank_points_26w_ago_before",
    "rank_points_change_4w_before", "rank_points_change_8w_before",
    "rank_points_change_26w_before", "elo_change_30d_before",
    "elo_change_90d_before", "surface_elo_change_90d_before",
    "ranking_vs_elo_disagreement_before", "ranking_points_momentum_before",
    "career_high_rank_before", "rank_distance_from_career_high_before",
]
PLAYER_V5_FEATURES = (
    MOMENTUM_FEATURES + WORKLOAD_FEATURES + DYNAMIC_STATS_FEATURES
    + RANKING_TREND_FEATURES
)
MATCHUP_FEATURES = [
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
NO_DIFFERENCE = {
    "matches_last_3d_before", "matches_last_7d_before",
    "matches_last_14d_before", "matches_last_28d_before",
    "same_tournament_matches_before",
}
PLAYER_V5_DIFFERENTIALS = [f for f in PLAYER_V5_FEATURES if f not in NO_DIFFERENCE]


class ColumnRowView:
    """Vista ligera de una fila sin materializar frame.values como object."""

    __slots__ = ("_columns", "_index")

    def __init__(self, columns: dict[str, Any], index: int) -> None:
        self._columns = columns
        self._index = index

    def __getitem__(self, name: str) -> Any:
        return self._columns[name][self._index]

    def get(self, name: str, default: Any = None) -> Any:
        values = self._columns.get(name)
        return default if values is None else values[self._index]


def load_v4(path: Path):
    spec = importlib.util.spec_from_file_location("smart_tennis_builder_v4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo importar el builder v4: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite(value: Any) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else math.nan
    except (TypeError, ValueError):
        return math.nan


def safe_div(a: Any, b: Any) -> float:
    a, b = finite(a), finite(b)
    return a / b if math.isfinite(a) and math.isfinite(b) and b > 0 else math.nan


def clip01(value: Any) -> float:
    value = finite(value)
    return float(np.clip(value, 0.0, 1.0)) if math.isfinite(value) else math.nan


@dataclass
class ParsedScore:
    completed: bool = False
    retired: bool = False
    walkover: bool = False
    winner_sets: int = 0
    loser_sets: int = 0
    winner_games: int = 0
    loser_games: int = 0
    tiebreak_sets: int = 0
    winner_tiebreak_sets: int = 0
    loser_tiebreak_sets: int = 0
    deciding_set: int = 0
    straight_sets: int = 0


def parse_score(score: Any, best_of: Any = None) -> ParsedScore:
    if score is None or pd.isna(score):
        return ParsedScore()
    text = str(score).strip().upper().replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", " ", text)
    if re.search(r"\b(W/O|WO|WALKOVER)\b", compact):
        return ParsedScore(walkover=True)
    retired = bool(re.search(r"\b(RET|ABD|DEF|ABN|INT)\b", compact))
    tokens = re.findall(r"(\d{1,2})-(\d{1,2})(?:\((\d{1,2})\))?", compact)
    if not tokens:
        return ParsedScore(retired=retired)
    winner_sets = loser_sets = winner_games = loser_games = 0
    tb = wtb = ltb = 0
    for first, second, tie in tokens:
        a, b = int(first), int(second)
        winner_games += a
        loser_games += b
        set_complete = (
            (max(a, b) >= 6 and abs(a - b) >= 2)
            or (max(a, b) == 7 and min(a, b) in {5, 6})
            or (max(a, b) >= 10 and abs(a - b) >= 2)
        )
        if set_complete and a > b:
            winner_sets += 1
        elif set_complete and b > a:
            loser_sets += 1
        if tie or (max(a, b) == 7 and min(a, b) == 6):
            tb += 1
            if a > b:
                wtb += 1
            elif b > a:
                ltb += 1
    best = int(finite(best_of)) if math.isfinite(finite(best_of)) else 3
    sets_needed = best // 2 + 1
    completed = not retired and winner_sets >= sets_needed
    return ParsedScore(
        completed=completed,
        retired=retired,
        winner_sets=winner_sets,
        loser_sets=loser_sets,
        winner_games=winner_games,
        loser_games=loser_games,
        tiebreak_sets=tb,
        winner_tiebreak_sets=wtb,
        loser_tiebreak_sets=ltb,
        deciding_set=int(completed and loser_sets == sets_needed - 1),
        straight_sets=int(completed and loser_sets == 0),
    )


@dataclass
class ActivityRecord:
    date: pd.Timestamp
    tourney_id: str
    surface: str
    won: int
    minutes: float
    games_for: float
    games_against: float
    sets_for: float
    sets_against: float
    deciding_set: int
    tiebreak_sets: int
    tiebreak_sets_won: int
    expected: float
    performance: float
    game_share: float
    opponent_id: int
    opponent_elo: float


@dataclass
class StatRecord:
    date: pd.Timestamp
    surface: str
    service_points_won: float
    return_points_won: float
    first_serve_in: float
    first_serve_win: float
    second_serve_win: float
    aces_per_service_game: float
    double_faults_per_service_game: float
    break_points_saved: float
    adjusted_serve: float
    adjusted_return: float


@dataclass
class RatingRecord:
    date: pd.Timestamp
    rank: float
    rank_points: float
    elo: float
    surface: str
    surface_elo: float


@dataclass
class EnrichedPlayerState:
    activity: deque = field(default_factory=lambda: deque(maxlen=160))
    stats: deque = field(default_factory=lambda: deque(maxlen=120))
    ratings: deque = field(default_factory=lambda: deque(maxlen=160))
    career_high_rank: float = math.nan
    player_id: int | None = None
    player_name: str | None = None
    latest_surface: str | None = None


@dataclass
class PairRecord:
    player_low_id: int
    player_high_id: int
    matches: int = 0
    low_wins: int = 0
    low_games: float = 0.0
    high_games: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=PAIR_RECENT_WINDOW))
    last_date: pd.Timestamp | None = None

    def orientation(self, player_1_id: int) -> tuple[float, float, float]:
        p1_low = player_1_id == self.player_low_id
        p1_wins = self.low_wins if p1_low else self.matches - self.low_wins
        p1_games = self.low_games if p1_low else self.high_games
        total_games = self.low_games + self.high_games
        recent = list(self.recent)
        if not p1_low:
            recent = [(1 - won, 1 - share) for won, share in recent]
        return (
            safe_div(p1_wins, self.matches),
            safe_div(p1_games, total_games),
            safe_div(sum(x[0] for x in recent), len(recent)),
        )


@dataclass
class OpponentPerformance:
    date: pd.Timestamp
    won: int
    game_share: float
    performance: float
    surface: str


def weighted_mean(records: Iterable[Any], current_date: pd.Timestamp, getter, half_life_days: float) -> float:
    values, weights = [], []
    for record in records:
        value = finite(getter(record))
        if not math.isfinite(value):
            continue
        age = max((current_date - record.date).total_seconds() / 86400.0, 0.0)
        weight = 0.5 ** (age / half_life_days)
        values.append(value)
        weights.append(weight)
    return float(np.average(values, weights=weights)) if values else math.nan


def recent_values(records: Iterable[Any], getter, n: int) -> list[float]:
    values = [finite(getter(r)) for r in list(records)[-n:]]
    return [v for v in values if math.isfinite(v)]


def mean_or_nan(values: Iterable[Any]) -> float:
    clean = [finite(v) for v in values if math.isfinite(finite(v))]
    return float(np.mean(clean)) if clean else math.nan


def rate_or_nan(values: Iterable[Any]) -> float:
    return mean_or_nan(values)


def within_days(records: Iterable[Any], current_date: pd.Timestamp, days: int) -> list[Any]:
    return [r for r in records if 0 <= (current_date - r.date).total_seconds() / 86400.0 < days]


def value_at_or_before(records: Iterable[RatingRecord], cutoff: pd.Timestamp, field_name: str) -> float:
    candidates = [r for r in records if r.date <= cutoff]
    if not candidates:
        return math.nan
    return finite(getattr(candidates[-1], field_name))


def elo_at_or_before(records: Iterable[RatingRecord], cutoff: pd.Timestamp, surface: str | None = None) -> float:
    candidates = [r for r in records if r.date <= cutoff and (surface is None or r.surface == surface)]
    if not candidates:
        return math.nan
    return finite(candidates[-1].surface_elo if surface else candidates[-1].elo)


def ewma_stats(records: Iterable[StatRecord], current_date: pd.Timestamp, field_name: str, half_life: float, surface: str | None = None) -> float:
    chosen = [r for r in records if surface is None or r.surface == surface]
    return weighted_mean(chosen, current_date, lambda x: getattr(x, field_name), half_life)


def consecutive_match_days(records: Iterable[ActivityRecord], current_date: pd.Timestamp) -> int:
    days = sorted({r.date.normalize() for r in records if r.date < current_date})
    if not days:
        return 0
    expected = current_date.normalize() - pd.Timedelta(days=1)
    count = 0
    for day in reversed(days):
        if day == expected:
            count += 1
            expected -= pd.Timedelta(days=1)
        elif day < expected:
            break
    return count


def player_snapshot(state: EnrichedPlayerState, current_date: pd.Timestamp, surface: str, tourney_id: str, current_rank: float, current_points: float, current_elo: float, current_surface_elo: float) -> dict[str, float]:
    activity = list(state.activity)
    surface_activity = [r for r in activity if r.surface == surface]
    performance_short = weighted_mean(activity, current_date, lambda r: r.performance, EWMA_HALFLIFE_SHORT_DAYS)
    performance_long = weighted_mean(activity, current_date, lambda r: r.performance, EWMA_HALFLIFE_LONG_DAYS)
    recent_performance = recent_values(activity, lambda r: r.performance, 10)
    games5 = recent_values(activity, lambda r: r.game_share, 5)
    games10 = recent_values(activity, lambda r: r.game_share, 10)
    surf_games10 = recent_values(surface_activity, lambda r: r.game_share, 10)
    sets10 = recent_values(activity, lambda r: safe_div(r.sets_for, r.sets_for + r.sets_against), 10)

    out: dict[str, float] = {
        "momentum_ewma_30d_before": performance_short,
        "momentum_ewma_90d_before": performance_long,
        "surface_momentum_ewma_90d_before": weighted_mean(surface_activity, current_date, lambda r: r.performance, EWMA_HALFLIFE_LONG_DAYS),
        "momentum_acceleration_before": performance_short - performance_long if math.isfinite(performance_short) and math.isfinite(performance_long) else math.nan,
        "momentum_volatility_before": float(np.std(recent_performance, ddof=0)) if recent_performance else math.nan,
        "performance_vs_expected_ewma_before": weighted_mean(activity, current_date, lambda r: r.won - r.expected, EWMA_HALFLIFE_LONG_DAYS),
        "games_won_share_last_5_before": mean_or_nan(games5),
        "games_won_share_last_10_before": mean_or_nan(games10),
        "surface_games_won_share_last_10_before": mean_or_nan(surf_games10),
        "sets_won_share_last_10_before": mean_or_nan(sets10),
        "straight_sets_rate_last_10_before": rate_or_nan(r.straight_sets if hasattr(r, "straight_sets") else int(r.sets_against == 0 and r.sets_for > 0) for r in activity[-10:]),
        "deciding_set_rate_last_10_before": rate_or_nan(r.deciding_set for r in activity[-10:]),
        "tiebreak_win_rate_last_20_before": safe_div(sum(r.tiebreak_sets_won for r in activity[-20:]), sum(r.tiebreak_sets for r in activity[-20:])),
        "opponent_adjusted_game_dominance_before": weighted_mean(activity, current_date, lambda r: (r.game_share - 0.5) + 0.5 * (r.won - r.expected), EWMA_HALFLIFE_LONG_DAYS),
        "surface_opponent_adjusted_game_dominance_before": weighted_mean(surface_activity, current_date, lambda r: (r.game_share - 0.5) + 0.5 * (r.won - r.expected), EWMA_HALFLIFE_LONG_DAYS),
    }

    for days in (3, 7, 14, 28):
        recs = within_days(activity, current_date, days)
        out[f"matches_last_{days}d_before"] = float(len(recs))
        minutes = [r.minutes for r in recs if math.isfinite(r.minutes)]
        out[f"minutes_last_{days}d_before"] = float(sum(minutes)) if minutes else math.nan
        if days in (3, 7, 14):
            out[f"games_last_{days}d_before"] = float(sum(r.games_for + r.games_against for r in recs))
            out[f"sets_last_{days}d_before"] = float(sum(r.sets_for + r.sets_against for r in recs))
    rec7 = within_days(activity, current_date, 7)
    out["minutes_coverage_last_7d_before"] = safe_div(sum(math.isfinite(r.minutes) for r in rec7), len(rec7))
    out["deciding_sets_last_7d_before"] = float(sum(r.deciding_set for r in rec7))
    out["tiebreak_sets_last_7d_before"] = float(sum(r.tiebreak_sets for r in rec7))
    out["consecutive_match_days_before"] = float(consecutive_match_days(activity, current_date))
    rest = (current_date - activity[-1].date).total_seconds() / 86400.0 if activity else math.nan
    out["rest_days_before"] = max(rest, 0.0) if math.isfinite(rest) else math.nan
    out["rest_days_capped_30_before"] = min(out["rest_days_before"], 30.0) if math.isfinite(out["rest_days_before"]) else math.nan
    out["rest_days_squared_before"] = out["rest_days_capped_30_before"] ** 2 if math.isfinite(out["rest_days_capped_30_before"]) else math.nan
    min7 = out.get("minutes_last_7d_before", math.nan)
    out["workload_per_rest_day_before"] = min7 / max(out["rest_days_before"], 1.0) if math.isfinite(min7) and math.isfinite(out["rest_days_before"]) else math.nan
    same = [r for r in activity if r.tourney_id == tourney_id]
    out["same_tournament_matches_before"] = float(len(same))
    same_minutes = [r.minutes for r in same if math.isfinite(r.minutes)]
    out["same_tournament_minutes_before"] = float(sum(same_minutes)) if same_minutes else math.nan

    for target, field_name, half_life, surf in (
        ("service_points_won_ewma_short_before", "service_points_won", 30.0, None),
        ("service_points_won_ewma_long_before", "service_points_won", 90.0, None),
        ("return_points_won_ewma_short_before", "return_points_won", 30.0, None),
        ("return_points_won_ewma_long_before", "return_points_won", 90.0, None),
        ("first_serve_in_ewma_before", "first_serve_in", 90.0, None),
        ("first_serve_win_ewma_before", "first_serve_win", 90.0, None),
        ("second_serve_win_ewma_before", "second_serve_win", 90.0, None),
        ("aces_per_service_game_ewma_before", "aces_per_service_game", 90.0, None),
        ("double_faults_per_service_game_ewma_before", "double_faults_per_service_game", 90.0, None),
        ("break_points_saved_ewma_before", "break_points_saved", 90.0, None),
        ("surface_service_points_won_ewma_before", "service_points_won", 90.0, surface),
        ("surface_return_points_won_ewma_before", "return_points_won", 90.0, surface),
        ("surface_first_serve_win_ewma_before", "first_serve_win", 90.0, surface),
        ("surface_second_serve_win_ewma_before", "second_serve_win", 90.0, surface),
        ("opponent_adjusted_serve_strength_before", "adjusted_serve", 90.0, None),
        ("opponent_adjusted_return_strength_before", "adjusted_return", 90.0, None),
    ):
        out[target] = ewma_stats(state.stats, current_date, field_name, half_life, surf)
    serve_short, serve_long = out["service_points_won_ewma_short_before"], out["service_points_won_ewma_long_before"]
    return_short, return_long = out["return_points_won_ewma_short_before"], out["return_points_won_ewma_long_before"]
    out["serve_return_strength_ewma_before"] = serve_long + return_long - 1.0 if math.isfinite(serve_long) and math.isfinite(return_long) else math.nan
    ss, sr = out["surface_service_points_won_ewma_before"], out["surface_return_points_won_ewma_before"]
    out["surface_serve_return_strength_ewma_before"] = ss + sr - 1.0 if math.isfinite(ss) and math.isfinite(sr) else math.nan
    out["serve_form_change_before"] = serve_short - serve_long if math.isfinite(serve_short) and math.isfinite(serve_long) else math.nan
    out["return_form_change_before"] = return_short - return_long if math.isfinite(return_short) and math.isfinite(return_long) else math.nan

    for weeks in (4, 8, 26):
        cutoff = current_date - pd.Timedelta(weeks=weeks)
        old_rank = value_at_or_before(state.ratings, cutoff, "rank")
        old_points = value_at_or_before(state.ratings, cutoff, "rank_points")
        out[f"rank_{weeks}w_ago_before"] = old_rank
        out[f"rank_change_{weeks}w_before"] = old_rank - current_rank if math.isfinite(old_rank) and math.isfinite(current_rank) else math.nan
        out[f"rank_points_{weeks}w_ago_before"] = old_points
        out[f"rank_points_change_{weeks}w_before"] = current_points - old_points if math.isfinite(old_points) and math.isfinite(current_points) else math.nan
    elo30 = elo_at_or_before(state.ratings, current_date - pd.Timedelta(days=30))
    elo90 = elo_at_or_before(state.ratings, current_date - pd.Timedelta(days=90))
    surf90 = elo_at_or_before(state.ratings, current_date - pd.Timedelta(days=90), surface)
    out["elo_change_30d_before"] = current_elo - elo30 if math.isfinite(current_elo) and math.isfinite(elo30) else math.nan
    out["elo_change_90d_before"] = current_elo - elo90 if math.isfinite(current_elo) and math.isfinite(elo90) else math.nan
    out["surface_elo_change_90d_before"] = current_surface_elo - surf90 if math.isfinite(current_surface_elo) and math.isfinite(surf90) else math.nan
    out["ranking_vs_elo_disagreement_before"] = math.log1p(current_rank) - ((2000.0 - current_elo) / 400.0) if math.isfinite(current_rank) and math.isfinite(current_elo) and current_rank > 0 else math.nan
    p8 = out["rank_points_change_8w_before"]
    p26 = out["rank_points_change_26w_before"]
    out["ranking_points_momentum_before"] = p8 - p26 / 3.25 if math.isfinite(p8) and math.isfinite(p26) else math.nan
    out["career_high_rank_before"] = state.career_high_rank
    out["rank_distance_from_career_high_before"] = current_rank - state.career_high_rank if math.isfinite(current_rank) and math.isfinite(state.career_high_rank) else math.nan
    return out


def pair_key(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((int(a), int(b))))


def pair_snapshot(player_1_id: int, player_2_id: int, surface: str, pair_records: dict, pair_surface_records: dict, opponent_history: dict[int, dict[int, deque]], surface_opponent_history: dict[str, dict[int, dict[int, deque]]], current_date: pd.Timestamp) -> dict[str, float]:
    key = pair_key(player_1_id, player_2_id)
    record: PairRecord | None = pair_records.get(key)
    surface_record: PairRecord | None = pair_surface_records[surface].get(key)
    result = {name: math.nan for name in MATCHUP_FEATURES}
    result["h2h_matches_before"] = float(record.matches if record else 0)
    if record:
        win_rate, games, recent_win = record.orientation(player_1_id)
        result["h2h_games_won_share_player_1_before"] = games
        result["recent_h2h_win_rate_player_1_before"] = recent_win
        recent = list(record.recent)
        if player_1_id != record.player_low_id:
            recent = [(1 - w, 1 - g) for w, g in recent]
        result["recent_h2h_games_won_share_player_1_before"] = mean_or_nan(g for _, g in recent)
    result["surface_h2h_matches_before"] = float(surface_record.matches if surface_record else 0)
    if surface_record:
        win_rate, games, _ = surface_record.orientation(player_1_id)
        result["surface_h2h_win_rate_player_1_before"] = win_rate
        result["surface_h2h_games_won_share_player_1_before"] = games

    def common_score(history_map):
        h1, h2 = history_map.get(player_1_id, {}), history_map.get(player_2_id, {})
        common = list(set(h1).intersection(h2))[:COMMON_OPPONENT_MAX]
        diffs, cycles = [], []
        for opponent in common:
            r1 = [r for r in h1[opponent] if 0 <= (current_date - r.date).days <= COMMON_OPPONENT_LOOKBACK_DAYS]
            r2 = [r for r in h2[opponent] if 0 <= (current_date - r.date).days <= COMMON_OPPONENT_LOOKBACK_DAYS]
            if not r1 or not r2:
                continue
            p1 = mean_or_nan(r.performance for r in r1)
            p2 = mean_or_nan(r.performance for r in r2)
            if math.isfinite(p1) and math.isfinite(p2):
                diffs.append(p1 - p2)
            a_beats_c = mean_or_nan(r.won for r in r1)
            b_beats_c = mean_or_nan(r.won for r in r2)
            if math.isfinite(a_beats_c) and math.isfinite(b_beats_c):
                cycles.append(abs(a_beats_c - b_beats_c) * (1.0 if (a_beats_c - 0.5) * (b_beats_c - 0.5) < 0 else 0.25))
        return len(diffs), mean_or_nan(diffs), mean_or_nan(cycles)

    n, score, intrans = common_score(opponent_history)
    result["common_opponents_count_before"] = float(n)
    result["common_opponent_score_player_1_before"] = score
    result["intransitivity_score_player_1_before"] = intrans
    ns, scores, intranss = common_score(surface_opponent_history[surface])
    result["surface_common_opponents_count_before"] = float(ns)
    result["surface_common_opponent_score_player_1_before"] = scores
    result["surface_intransitivity_score_player_1_before"] = intranss
    return result


def match_stat_record(row: pd.Series, prefix: str, opponent_prefix: str, surface: str, date: pd.Timestamp) -> StatRecord | None:
    svpt = finite(row.get(f"{prefix}_svpt"))
    first_in = finite(row.get(f"{prefix}_1stIn"))
    first_won = finite(row.get(f"{prefix}_1stWon"))
    second_won = finite(row.get(f"{prefix}_2ndWon"))
    service_games = finite(row.get(f"{prefix}_SvGms"))
    opponent_svpt = finite(row.get(f"{opponent_prefix}_svpt"))
    opponent_first_won = finite(row.get(f"{opponent_prefix}_1stWon"))
    opponent_second_won = finite(row.get(f"{opponent_prefix}_2ndWon"))
    if not math.isfinite(svpt) or svpt <= 0:
        return None
    service_won = safe_div(first_won + second_won, svpt)
    return_won = safe_div(opponent_svpt - opponent_first_won - opponent_second_won, opponent_svpt)
    return StatRecord(
        date=date, surface=surface,
        service_points_won=service_won,
        return_points_won=return_won,
        first_serve_in=safe_div(first_in, svpt),
        first_serve_win=safe_div(first_won, first_in),
        second_serve_win=safe_div(second_won, svpt - first_in),
        aces_per_service_game=safe_div(row.get(f"{prefix}_ace"), service_games),
        double_faults_per_service_game=safe_div(row.get(f"{prefix}_df"), service_games),
        break_points_saved=safe_div(row.get(f"{prefix}_bpSaved"), row.get(f"{prefix}_bpFaced")),
        adjusted_serve=service_won - (1.0 - return_won) if math.isfinite(service_won) and math.isfinite(return_won) else math.nan,
        adjusted_return=return_won - (1.0 - service_won) if math.isfinite(service_won) and math.isfinite(return_won) else math.nan,
    )


def update_pair(record: PairRecord, winner_id: int, parsed: ParsedScore, date: pd.Timestamp) -> None:
    record.matches += 1
    low_won = int(winner_id == record.player_low_id)
    record.low_wins += low_won
    winner_games, loser_games = float(parsed.winner_games), float(parsed.loser_games)
    if low_won:
        record.low_games += winner_games
        record.high_games += loser_games
        share = safe_div(winner_games, winner_games + loser_games)
    else:
        record.low_games += loser_games
        record.high_games += winner_games
        share = safe_div(loser_games, winner_games + loser_games)
    if math.isfinite(share):
        record.recent.append((low_won, share))
    record.last_date = date


def build_v5_features(base: pd.DataFrame, current_state_v4: pd.DataFrame, progress_every: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = base.sort_values("processing_sequence", kind="mergesort").reset_index(drop=True).copy()
    player_states: dict[int, EnrichedPlayerState] = defaultdict(EnrichedPlayerState)
    pair_records: dict[tuple[int, int], PairRecord] = {}
    pair_surface_records: dict[str, dict[tuple[int, int], PairRecord]] = defaultdict(dict)
    opponent_history: dict[int, dict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=12)))
    surface_opponent_history: dict[str, dict[int, dict[int, deque]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: deque(maxlen=12))))
    player_arrays = {f"player_{side}_{feature}": np.full(len(frame), np.nan, np.float32) for side in (1, 2) for feature in PLAYER_V5_FEATURES}
    matchup_arrays = {feature: np.full(len(frame), np.nan, np.float32) for feature in MATCHUP_FEATURES}

    # iterrows() accede a frame.values y puede materializar todo el DataFrame
    # heterogeneo como una matriz object. Con cerca de un millon de filas esto
    # requiere mas de 1 GiB adicional. Las columnas se conservan separadas y
    # cada fila se expone mediante una vista ligera.
    row_columns = {
        name: frame[name].array
        for name in frame.columns
    }

    for i in range(len(frame)):
        row = ColumnRowView(row_columns, i)
        if not bool(row.get("eligible_for_model", False)):
            continue
        date = pd.Timestamp(row["match_date"])
        surface = str(row["surface"])
        tourney_id = str(row["tourney_id"])
        p1_id, p2_id = int(row["player_1_id"]), int(row["player_2_id"])
        p1_state, p2_state = player_states[p1_id], player_states[p2_id]
        p1_state.player_id, p2_state.player_id = p1_id, p2_id
        p1_state.player_name, p2_state.player_name = str(row["player_1_name"]), str(row["player_2_name"])
        p1_elo, p2_elo = finite(row["player_1_elo_before"]), finite(row["player_2_elo_before"])
        p1_selo, p2_selo = finite(row["player_1_surface_elo_before"]), finite(row["player_2_surface_elo_before"])
        for side, state, rank, points, elo, selo in (
            (1, p1_state, finite(row["player_1_rank"]), finite(row["player_1_rank_points"]), p1_elo, p1_selo),
            (2, p2_state, finite(row["player_2_rank"]), finite(row["player_2_rank_points"]), p2_elo, p2_selo),
        ):
            snap = player_snapshot(state, date, surface, tourney_id, rank, points, elo, selo)
            for feature, value in snap.items():
                if f"player_{side}_{feature}" in player_arrays:
                    player_arrays[f"player_{side}_{feature}"][i] = finite(value)
        pair_snap = pair_snapshot(p1_id, p2_id, surface, pair_records, pair_surface_records, opponent_history, surface_opponent_history, date)
        for feature, value in pair_snap.items():
            matchup_arrays[feature][i] = finite(value)

        winner_id = int(row["winner_id"])
        loser_id = int(row["loser_id"])
        p1_won = int(row["target_player_1_win"])
        parsed = parse_score(row.get("score"), row.get("best_of"))
        total_games = parsed.winner_games + parsed.loser_games
        winner_share = safe_div(parsed.winner_games, total_games)
        loser_share = 1.0 - winner_share if math.isfinite(winner_share) else math.nan
        expected_p1 = 1.0 / (1.0 + 10.0 ** ((p2_elo - p1_elo) / 400.0)) if math.isfinite(p1_elo) and math.isfinite(p2_elo) else 0.5
        expected_p2 = 1.0 - expected_p1
        minutes = finite(row.get("minutes"))
        for pid, oid, state, won, expected, games_for, games_against, sets_for, sets_against, tb_won in (
            (p1_id, p2_id, p1_state, p1_won, expected_p1, parsed.winner_games if p1_won else parsed.loser_games, parsed.loser_games if p1_won else parsed.winner_games, parsed.winner_sets if p1_won else parsed.loser_sets, parsed.loser_sets if p1_won else parsed.winner_sets, parsed.winner_tiebreak_sets if p1_won else parsed.loser_tiebreak_sets),
            (p2_id, p1_id, p2_state, 1 - p1_won, expected_p2, parsed.loser_games if p1_won else parsed.winner_games, parsed.winner_games if p1_won else parsed.loser_games, parsed.loser_sets if p1_won else parsed.winner_sets, parsed.winner_sets if p1_won else parsed.loser_sets, parsed.loser_tiebreak_sets if p1_won else parsed.winner_tiebreak_sets),
        ):
            game_share = safe_div(games_for, games_for + games_against)
            performance = (won - expected) + (game_share - 0.5 if math.isfinite(game_share) else 0.0)
            activity_record = ActivityRecord(date, tourney_id, surface, won, minutes, games_for, games_against, sets_for, sets_against, parsed.deciding_set, parsed.tiebreak_sets, tb_won, expected, performance, game_share, oid, p2_elo if pid == p1_id else p1_elo)
            state.activity.append(activity_record)
            hist = OpponentPerformance(date, won, game_share, performance, surface)
            opponent_history[pid][oid].append(hist)
            surface_opponent_history[surface][pid][oid].append(hist)
            prefix = "w" if pid == winner_id else "l"
            opponent_prefix = "l" if prefix == "w" else "w"
            stat = match_stat_record(row, prefix, opponent_prefix, surface, date)
            if stat:
                state.stats.append(stat)
            current_rank = finite(row["player_1_rank"] if pid == p1_id else row["player_2_rank"])
            current_points = finite(row["player_1_rank_points"] if pid == p1_id else row["player_2_rank_points"])
            current_elo = p1_elo if pid == p1_id else p2_elo
            current_selo = p1_selo if pid == p1_id else p2_selo
            state.ratings.append(RatingRecord(date, current_rank, current_points, current_elo, surface, current_selo))
            if math.isfinite(current_rank) and current_rank > 0:
                state.career_high_rank = min(state.career_high_rank, current_rank) if math.isfinite(state.career_high_rank) else current_rank
            state.latest_surface = surface

        key = pair_key(p1_id, p2_id)
        if key not in pair_records:
            pair_records[key] = PairRecord(*key)
        if key not in pair_surface_records[surface]:
            pair_surface_records[surface][key] = PairRecord(*key)
        if parsed.completed and total_games > 0:
            update_pair(pair_records[key], winner_id, parsed, date)
            update_pair(pair_surface_records[surface][key], winner_id, parsed, date)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"[v5 enrichment] {i + 1:,}/{len(frame):,}", flush=True)

    generated = {**player_arrays, **matchup_arrays}
    for feature in PLAYER_V5_DIFFERENTIALS:
        generated[f"diff_{feature}"] = (player_arrays[f"player_1_{feature}"] - player_arrays[f"player_2_{feature}"]).astype(np.float32)
    frame = pd.concat([frame, pd.DataFrame(generated, index=frame.index)], axis=1)

    data_max_date = pd.to_datetime(frame["match_date"], errors="coerce").max()
    current_rows = []
    for pid, state in player_states.items():
        if not state.ratings:
            continue
        last = state.ratings[-1]
        snap = player_snapshot(state, data_max_date + pd.Timedelta(nanoseconds=1), state.latest_surface or last.surface, "__CURRENT__", last.rank, last.rank_points, last.elo, last.surface_elo)
        current_rows.append({"player_id": pid, "v5_state_as_of_date": data_max_date, **{k.removesuffix("_before"): v for k, v in snap.items()}})
    enriched_current = pd.DataFrame(current_rows)
    current = current_state_v4.merge(enriched_current, on="player_id", how="left", validate="one_to_one")

    pair_rows = []
    for key, record in pair_records.items():
        low_rate, low_games, recent_low = record.orientation(record.player_low_id)
        pair_rows.append({
            "player_1_id": record.player_low_id,
            "player_2_id": record.player_high_id,
            "surface": "ALL",
            "h2h_matches": record.matches,
            "player_1_h2h_wins": record.low_wins,
            "player_2_h2h_wins": record.matches - record.low_wins,
            "player_1_h2h_win_rate": low_rate,
            "player_1_games_won_share": low_games,
            "recent_player_1_win_rate": recent_low,
            "state_as_of_date": data_max_date,
        })
    for surface, records in pair_surface_records.items():
        for key, record in records.items():
            low_rate, low_games, recent_low = record.orientation(record.player_low_id)
            pair_rows.append({
                "player_1_id": record.player_low_id,
                "player_2_id": record.player_high_id,
                "surface": surface,
                "h2h_matches": record.matches,
                "player_1_h2h_wins": record.low_wins,
                "player_2_h2h_wins": record.matches - record.low_wins,
                "player_1_h2h_win_rate": low_rate,
                "player_1_games_won_share": low_games,
                "recent_player_1_win_rate": recent_low,
                "state_as_of_date": data_max_date,
            })
    pair_current = pd.DataFrame(pair_rows).sort_values(["player_1_id", "player_2_id", "surface"], kind="mergesort").reset_index(drop=True) if pair_rows else pd.DataFrame()
    return frame, current, pair_current


def validate_v5(frame: pd.DataFrame, current: pd.DataFrame, pair_current: pd.DataFrame) -> None:
    required = {f"player_{side}_{feature}" for side in (1, 2) for feature in PLAYER_V5_FEATURES} | set(MATCHUP_FEATURES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Faltan features v5: {missing}")
    if frame.duplicated(["tourney_id", "competition_type", "match_num"]).any():
        raise RuntimeError("Claves duplicadas en salida v5")
    eligible = frame["eligible_for_model"].fillna(False).astype(bool)
    if frame.loc[eligible, ["player_1_id", "player_2_id"]].isna().any().any():
        raise RuntimeError("IDs nulos en filas elegibles v5")
    if current.empty or current["player_id"].duplicated().any():
        raise RuntimeError("Estado actual v5 invalido")
    if not pair_current.empty and pair_current.duplicated(["player_1_id", "player_2_id", "surface"]).any():
        raise RuntimeError("Estado de pares v5 duplicado")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Construye features Sackmann/TML v5")
    parser.add_argument("--v4-builder", default=str(here / "build_player_match_features_tml_sackmann_elo_v4.py"))
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--current-state-output")
    parser.add_argument("--pair-state-output")
    parser.add_argument("--report")
    parser.add_argument("--feature-manifest")
    parser.add_argument("--progress-every-rows", type=int, default=10000)
    parser.add_argument("--progress-every-seconds", type=float, default=5.0)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    v4 = load_v4(Path(args.v4_builder).resolve())
    root = v4.find_project_root(Path(__file__).resolve().parent)
    processed = root / "data" / "processed"
    input_path = Path(args.input or processed / "jeff_sackmann_with_odds.parquet").resolve()
    output_path = Path(args.output or processed / "tennis_matches_with_player_stats.sackmann_v5.parquet").resolve()
    current_path = Path(args.current_state_output or processed / "player_current_state.sackmann_v5.parquet").resolve()
    pair_path = Path(args.pair_state_output or processed / "player_pair_current_state.sackmann_v5.parquet").resolve()
    report_path = Path(args.report or processed / "tennis_matches_with_player_stats.sackmann_v5_report.csv").resolve()
    manifest_path = Path(args.feature_manifest or processed / "tennis_match_feature_manifest.sackmann_v5.json").resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    available = set(pq.ParquetFile(input_path).schema_arrow.names)
    selected = [c for c in v4.REQUIRED_COLUMNS if c in available]
    print(f"Leyendo: {input_path}")
    raw = pd.read_parquet(input_path, columns=selected, engine="pyarrow")
    raw_rows = len(raw)
    print(f"Filas originales: {raw_rows:,}")
    reconciled, reconciliation_summary = v4.reconcile_input_frame(raw)
    del raw
    gc.collect()
    base, current_v4 = v4.build_features(
        reconciled,
        progress_every_rows=args.progress_every_rows,
        progress_every_seconds=args.progress_every_seconds,
        show_progress=not args.no_progress,
    )
    print("\nFase v5: enriquecimiento cronologico de las cinco prioridades")
    enriched, current_v5, pair_current = build_v5_features(base, current_v4, args.progress_every_rows)
    validate_v5(enriched, current_v5, pair_current)

    v4.write_parquet_atomically(enriched, output_path, ["tourney_id", "competition_type", "match_num", "player_1_id", "player_2_id", "match_date", "eligible_for_model"])
    v4.write_parquet_atomically(current_v5, current_path, ["player_id", "state_as_of_date", "last_match_date"])
    if not pair_current.empty:
        v4.write_parquet_atomically(pair_current, pair_path, ["player_1_id", "player_2_id", "surface", "state_as_of_date"])

    eligible = enriched["eligible_for_model"].fillna(False).astype(bool)
    report = pd.DataFrame({
        "metric": [
            "schema_version", "raw_input_rows", "rows_after_reconciliation",
            "output_rows", "output_columns", "eligible_rows",
            "current_state_players", "pair_state_rows", "momentum_features",
            "workload_features", "dynamic_stats_features",
            "ranking_trend_features", "matchup_features", "first_match_date",
            "last_match_date",
        ],
        "value": [
            SCHEMA_VERSION, raw_rows, len(reconciled), len(enriched),
            len(enriched.columns), int(eligible.sum()), len(current_v5),
            len(pair_current), len(MOMENTUM_FEATURES), len(WORKLOAD_FEATURES),
            len(DYNAMIC_STATS_FEATURES), len(RANKING_TREND_FEATURES),
            len(MATCHUP_FEATURES), str(enriched["match_date"].min()),
            str(enriched["match_date"].max()),
        ],
    })
    report.to_csv(report_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_builder": str(Path(args.v4_builder).resolve()),
        "input": str(input_path),
        "output": str(output_path),
        "current_state_output": str(current_path),
        "pair_state_output": str(pair_path),
        "momentum_features": MOMENTUM_FEATURES,
        "workload_features": WORKLOAD_FEATURES,
        "dynamic_stats_features": DYNAMIC_STATS_FEATURES,
        "ranking_trend_features": RANKING_TREND_FEATURES,
        "matchup_features": MATCHUP_FEATURES,
        "player_differential_features": PLAYER_V5_DIFFERENTIALS,
        "integrated_reconciliation": reconciliation_summary,
        "anti_leakage_order": "snapshot -> write -> update",
        "score_policy": {
            "completed_matches_update_margin": True,
            "retirements_do_not_update_pair_margin": True,
            "walkovers_excluded_by_v4": True,
        },
        "notes": [
            "All v5 match features are chronological pre-match snapshots.",
            "Workload features are calculated from recorded matches only.",
            "Missing minutes remain missing and have a coverage feature.",
            "Intransitivity is an explainable common-opponent approximation, not a GNN.",
            "Keep v4 artifacts until ablation and temporal validation are complete.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n" + report.to_string(index=False))
    print(f"\nManifest: {manifest_path}")
    print(f"Completado en {(time.perf_counter() - started) / 60:.1f} minutos")


if __name__ == "__main__":
    main()
