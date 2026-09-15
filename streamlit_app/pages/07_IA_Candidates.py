#!/usr/bin/env python3
"""IA Candidates: evidence-first dossiers for upcoming tennis fixtures.

This page reuses the fixture cache and prediction cache created by
06_Upcoming_Matches.py. It never states an injury unless an explicit structured
source is available. Long inactivity, retirements and walkovers are presented as
separate observable facts.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from streamlit_app.config.data_paths import (
    IA_CANDIDATES_DOSSIERS_PATH,
    UPCOMING_MATCHES_CACHE_PATH,
)
from streamlit_app.data_access.players_adapted import (
    get_head_to_head_history,
    get_match_market_probabilities,
    get_player_inference_snapshot,
    get_player_matches,
    get_player_current_tournament_matches,
    get_players_current_tournament_statistics,
    get_player_tournament_history,
    get_players,
)
from streamlit_app.services.match_prediction_service import predict_upcoming_fixture
from streamlit_app.data_access.tournaments_adapted import (
    analyze_tournament_any_set_history,
    analyze_tournament_first_set_history,
    analyze_tournament_games_threshold,
    get_tournament_editions,
)

# Backward-compatible local names retained by the existing cache helpers.
FIXTURE_CACHE = UPCOMING_MATCHES_CACHE_PATH
DOSSIER_CACHE = IA_CANDIDATES_DOSSIERS_PATH
PAGE_SCHEMA_VERSION = 7
DEFAULT_OVER_LINE_BO3 = 21.5
DEFAULT_OVER_LINE_BO5 = 38.5


@dataclass
class Evidence:
    category: str
    label: str
    statement: str
    confidence: str
    sample_size: int | None
    as_of_date: str
    source: str
    details: list[dict[str, Any]]
    severity: str = "info"


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def stable_fixture_key(row: pd.Series | dict[str, Any]) -> str:
    fixture_id = str(row.get("fixture_id", "") or "").strip()
    if fixture_id and fixture_id.casefold() not in {"nan", "none", "<na>"}:
        return f"fixture:{fixture_id}"
    values = (
        row.get("match_date", ""), row.get("scheduled_time", ""),
        row.get("tournament", ""), row.get("round", ""),
        row.get("player_1_name", ""), row.get("player_2_name", ""),
    )
    return "match:" + "|".join(normalize_text(value) for value in values)


def json_safe(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, date)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def load_disk_dossiers() -> dict[str, Any]:
    if not DOSSIER_CACHE.exists():
        return {}
    try:
        payload = json.loads(DOSSIER_CACHE.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_disk_dossiers(value: dict[str, Any]) -> None:
    DOSSIER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = DOSSIER_CACHE.with_suffix(".json.new")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(DOSSIER_CACHE)


@st.cache_data(show_spinner=False, ttl=900)
def load_fixture_cache(path_text: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    path = Path(path_text)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path, engine="pyarrow")
    if "match_date" in frame:
        frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce").dt.date
    return frame


PLAYER_NAME_ALIASES = {
    # Fixture provider full/legal names -> historical canonical names.
    "pedro martinez portero": "pedro martinez",
}


def player_name_candidates(player_name: Any) -> list[str]:
    """Return conservative candidate keys for a fixture player name.

    Exact normalized names are preferred. Explicit aliases are the only
    cross-name resolution mechanism. Open fuzzy matching is intentionally not
    used because it can silently map two different players with similar names.
    """
    normalized = normalize_text(player_name)
    if not normalized:
        return []

    candidates = [normalized]
    explicit_alias = PLAYER_NAME_ALIASES.get(normalized)
    if explicit_alias:
        candidates.append(normalize_text(explicit_alias))

    # Safe provider suffix variants. These do not remove arbitrary surnames.
    for suffix in (" junior", " jr"):
        if normalized.endswith(suffix):
            reduced = normalized[:-len(suffix)].strip()
            if reduced:
                candidates.append(reduced)

    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


@st.cache_data(show_spinner=False, ttl=3600)
def player_lookup() -> dict[str, tuple[int, str]]:
    players = get_players().copy()
    players = players.dropna(subset=["player_id", "player_name"]).copy()

    lookup: dict[str, tuple[int, str]] = {}
    for _, row in players.iterrows():
        player_id = int(row["player_id"])
        canonical_name = str(row["player_name"]).strip()
        normalized_name = normalize_text(canonical_name)
        if normalized_name:
            lookup[normalized_name] = (player_id, canonical_name)

    # Register an alias only if its canonical target really exists in the
    # historical player catalogue. A misspelled target is therefore harmless.
    for provider_name, historical_name in PLAYER_NAME_ALIASES.items():
        provider_key = normalize_text(provider_name)
        historical_key = normalize_text(historical_name)
        historical_player = lookup.get(historical_key)
        if historical_player is not None:
            lookup[provider_key] = historical_player

    return lookup


def resolve_fixture_player(
    fixture_player_name: Any,
    lookup: dict[str, tuple[int, str]],
) -> tuple[tuple[int, str] | None, dict[str, Any]]:
    """Resolve one fixture name and return auditable matching diagnostics."""
    normalized_fixture_name = normalize_text(fixture_player_name)
    candidates = player_name_candidates(fixture_player_name)

    for candidate in candidates:
        resolved = lookup.get(candidate)
        if resolved is None:
            continue
        method = (
            "exact"
            if candidate == normalized_fixture_name
            and candidate not in PLAYER_NAME_ALIASES
            else "explicit_alias"
        )
        return resolved, {
            "fixture_name": str(fixture_player_name),
            "normalized_fixture_name": normalized_fixture_name,
            "matched_candidate": candidate,
            "historical_player_id": int(resolved[0]),
            "historical_player_name": str(resolved[1]),
            "match_method": method,
        }

    return None, {
        "fixture_name": str(fixture_player_name),
        "normalized_fixture_name": normalized_fixture_name,
        "attempted_candidates": candidates,
        "match_method": "not_matched",
    }

def score_games(score: Any) -> float:
    if score is None or pd.isna(score):
        return np.nan
    text = str(score).upper().strip()
    if text in {"", "W/O", "WO", "WALKOVER"}:
        return np.nan
    total = 0
    parsed = 0
    for token in text.split():
        match = re.match(r"^(\d+)-(\d+)(?:\([^)]*\))?", token)
        if match:
            total += int(match.group(1)) + int(match.group(2))
            parsed += 1
    return float(total) if parsed else np.nan


def status_from_score(score: Any) -> str:
    text = str(score or "").upper()
    if any(token in text for token in ("W/O", "WALKOVER", " WO")) or text.strip() == "WO":
        return "walkover"
    if any(token in text for token in ("RET", "RETIRED")):
        return "retirement"
    if any(token in text for token in ("DEF", "DEFAULT")):
        return "default"
    return "played"


def evidence_frame(frame: pd.DataFrame, columns: list[str], limit: int = 10) -> list[dict[str, Any]]:
    selected = [column for column in columns if column in frame]
    if not selected or frame.empty:
        return []
    return json_safe(frame.loc[:, selected].head(limit).to_dict(orient="records"))


def confidence_for_sample(rows: int) -> str:
    if rows >= 20:
        return "strong_sample"
    if rows >= 8:
        return "moderate_sample"
    if rows >= 3:
        return "limited_sample"
    return "insufficient_sample"


def prediction_from_cache(key: str) -> dict[str, Any]:
    cache = st.session_state.get("upcoming_model_predictions", {})
    value = cache.get(key, {}) if isinstance(cache, dict) else {}
    return value if isinstance(value, dict) else {}


def calculate_prediction(fixture: pd.Series, p1_id: int, p2_id: int) -> dict[str, Any]:
    line = DEFAULT_OVER_LINE_BO5 if int(fixture.get("best_of", 3)) == 5 else DEFAULT_OVER_LINE_BO3
    return predict_upcoming_fixture(
        player_1_id=p1_id,
        player_2_id=p2_id,
        surface=str(fixture.get("surface", "Hard")),
        tournament_level=str(fixture.get("tourney_level", "A")),
        round_name=str(fixture.get("round", "R32")),
        best_of=int(fixture.get("best_of", 3)),
        competition_type=str(fixture.get("competition_type", "ATP")),
        indoor=bool(fixture.get("indoor", False)),
        prediction_date=fixture.get("match_date"),
        tournament_id=str(fixture.get("tournament_id", fixture.get("tournament", ""))),
        over_games_line=line,
        market_simulations=1_000,
    )


def player_context_evidence(
    player_id: int,
    player_name: str,
    surface: str,
    fixture_date: Any,
) -> list[Evidence]:
    output: list[Evidence] = []
    reference = pd.to_datetime(fixture_date, errors="coerce")
    history = get_player_matches(player_id).copy()
    if history.empty:
        return [Evidence(
            "coverage", "Sin histórico local",
            f"No hay partidos históricos disponibles para {player_name}.",
            "insufficient_sample", 0, str(fixture_date), "historical_match_dataset", [], "warning"
        )]
    history["match_date"] = pd.to_datetime(history["match_date"], errors="coerce")
    if pd.notna(reference):
        history = history.loc[history["match_date"].lt(reference.normalize())].copy()
    history = history.sort_values("match_date", ascending=False, kind="mergesort")
    recent_90 = history.loc[history["match_date"].ge(reference - pd.Timedelta(days=90))] if pd.notna(reference) else history.head(20)
    surface_recent = recent_90.loc[recent_90.get("surface", pd.Series("", index=recent_90.index)).astype(str).eq(surface)]
    other_recent = recent_90.loc[~recent_90.index.isin(surface_recent.index)]

    if len(surface_recent) < 5:
        statement = (
            f"{player_name} registra {len(surface_recent)} partido(s) en {surface} durante los "
            f"90 días anteriores, frente a {len(other_recent)} en otras superficies."
        )
        output.append(Evidence(
            "surface_experience", "Experiencia reciente limitada en la superficie",
            statement, confidence_for_sample(len(surface_recent)), len(surface_recent),
            str(fixture_date), "historical_match_dataset",
            evidence_frame(surface_recent, ["match_date", "tourney_name", "surface", "opponent_name", "result", "score"]),
            "warning",
        ))

    last_matches = history.head(8).copy()
    if not last_matches.empty:
        last_date = last_matches["match_date"].max()
        days = int((reference.normalize() - last_date.normalize()).days) if pd.notna(reference) and pd.notna(last_date) else None
        if days is not None and days >= 28:
            output.append(Evidence(
                "inactivity", "Periodo sin competir",
                f"El último partido registrado de {player_name} fue hace {days} días. Esto indica inactividad, no acredita por sí solo una lesión.",
                "observed_fact", 1, str(fixture_date), "historical_match_dataset",
                evidence_frame(last_matches, ["match_date", "tourney_name", "surface", "opponent_name", "result", "score"], 3),
                "warning",
            ))

        statuses = last_matches.get("score", pd.Series("", index=last_matches.index)).map(status_from_score)
        events = last_matches.loc[statuses.ne("played")].copy()
        if not events.empty:
            event_labels = statuses.loc[events.index].value_counts().to_dict()
            output.append(Evidence(
                "match_status", "Incidencias registradas recientemente",
                f"En los últimos {len(last_matches)} partidos de {player_name} figuran incidencias de marcador: {event_labels}. No se infiere una lesión sin una fuente explícita.",
                confidence_for_sample(len(last_matches)), len(events), str(fixture_date),
                "historical_score_field",
                evidence_frame(events, ["match_date", "tourney_name", "surface", "opponent_name", "result", "score"]),
                "warning",
            ))

        surfaces = last_matches["surface"].astype(str).tolist() if "surface" in last_matches else []
        if surfaces and surfaces[0] == surface:
            previous_different = next((item for item in surfaces[1:] if item != surface), None)
            same_at_start = 0
            for item in surfaces:
                if item == surface:
                    same_at_start += 1
                else:
                    break
            if previous_different and same_at_start <= 3:
                output.append(Evidence(
                    "surface_switch", "Cambio reciente de superficie",
                    f"{player_name} lleva {same_at_start} partido(s) consecutivo(s) en {surface} tras competir en {previous_different}.",
                    "observed_fact", same_at_start, str(fixture_date), "historical_match_dataset",
                    evidence_frame(last_matches, ["match_date", "tourney_name", "surface", "opponent_name", "result"], 6),
                    "info",
                ))
    return output


def _number(source: Any, *names: str) -> float:
    if source is None:
        return np.nan
    for name in names:
        value = pd.to_numeric(source.get(name, np.nan), errors="coerce")
        if pd.notna(value) and np.isfinite(float(value)):
            return float(value)
    return np.nan


def _arrival_label(score: float) -> str:
    if score >= 2.5:
        return "Muy positivo"
    if score >= 1.0:
        return "Positivo"
    if score <= -2.5:
        return "Muy negativo"
    if score <= -1.0:
        return "Negativo"
    return "Neutral"


def player_arrival_profile(
    player_id: int,
    player_name: str,
    surface: str,
    fixture: pd.Series,
) -> dict[str, Any]:
    """Evalúa el momento sin usar la probabilidad de victoria del modelo."""
    fixture_date = fixture.get("match_date")
    snapshot = get_player_inference_snapshot(
        player_id=player_id,
        surface=surface,
        tourney_level=str(fixture.get("tourney_level", "A")),
        round_name=str(fixture.get("round", "R32")),
        best_of=int(fixture.get("best_of", 3)),
        prediction_date=fixture_date,
        competition_type=str(fixture.get("competition_type", "ATP")),
        indoor=bool(fixture.get("indoor", False)),
    )
    history = get_player_matches(player_id).copy()
    reference = pd.to_datetime(fixture_date, errors="coerce")
    if not history.empty:
        history["match_date"] = pd.to_datetime(history["match_date"], errors="coerce")
        if pd.notna(reference):
            history = history.loc[history["match_date"].lt(reference.normalize())]
        history = history.sort_values("match_date", ascending=False, kind="mergesort")
    surface_history = (
        history.loc[history["surface"].astype(str).eq(surface)].head(10).copy()
        if not history.empty and "surface" in history else pd.DataFrame()
    )
    surface_matches = len(surface_history)
    surface_wins = int(pd.to_numeric(surface_history.get("won"), errors="coerce").fillna(0).sum()) if surface_matches else 0
    observed_surface_form = surface_wins / surface_matches if surface_matches else np.nan

    metrics = {
        "surface_form": _number(snapshot, "surface_form_last_10_before", "surface_form_last_5_before"),
        "observed_surface_last_10": observed_surface_form,
        "elo_change_30d": _number(snapshot, "elo_change_30d_before"),
        "elo_change_90d": _number(snapshot, "elo_change_90d_before"),
        "surface_elo_change_90d": _number(snapshot, "surface_elo_change_90d_before"),
        "result_residual_30d": _number(snapshot, "result_residual_ewma_30d_before"),
        "surface_result_residual_90d": _number(snapshot, "surface_result_residual_ewma_90d_before"),
        "game_share_residual_30d": _number(snapshot, "game_share_residual_ewma_30d_before"),
        "serve_residual_30d": _number(snapshot, "serve_residual_ewma_30d_before"),
        "return_residual_30d": _number(snapshot, "return_residual_ewma_30d_before"),
        "serve_short": _number(snapshot, "service_points_won_ewma_short_before"),
        "serve_long": _number(snapshot, "service_points_won_ewma_long_before"),
        "return_short": _number(snapshot, "return_points_won_ewma_short_before"),
        "return_long": _number(snapshot, "return_points_won_ewma_long_before"),
        "serve_form_change": _number(snapshot, "serve_form_change_before"),
        "return_form_change": _number(snapshot, "return_form_change_before"),
        "matches_last_14d": _number(snapshot, "matches_last_14d_before"),
        "minutes_last_14d": _number(snapshot, "minutes_last_14d_before"),
        "rest_days": _number(snapshot, "rest_days_before", "days_since_last_match"),
        "momentum_confidence": _number(snapshot, "momentum_confidence_before"),
        "momentum_effective_sample": _number(snapshot, "momentum_effective_sample_before"),
    }
    if pd.isna(metrics["surface_form"]):
        metrics["surface_form"] = observed_surface_form
    if pd.isna(metrics["serve_form_change"]) and pd.notna(metrics["serve_short"]) and pd.notna(metrics["serve_long"]):
        metrics["serve_form_change"] = metrics["serve_short"] - metrics["serve_long"]
    if pd.isna(metrics["return_form_change"]) and pd.notna(metrics["return_short"]) and pd.notna(metrics["return_long"]):
        metrics["return_form_change"] = metrics["return_short"] - metrics["return_long"]

    components: list[dict[str, Any]] = []
    score = 0.0
    def add(name: str, value: float, points: float, detail: str) -> None:
        nonlocal score
        if pd.notna(value):
            score += points
            components.append({"factor": name, "value": value, "contribution": points, "detail": detail})

    form = metrics["surface_form"]
    if pd.notna(form):
        points = 1.5 if form >= .70 else .75 if form >= .60 else -1.5 if form <= .30 else -.75 if form <= .40 else 0.0
        add("Forma en superficie", form, points, f"Forma rolling/observada en {surface}: {form:.0%}")
    elo = metrics["surface_elo_change_90d"]
    if pd.isna(elo):
        elo = metrics["elo_change_30d"]
    if pd.notna(elo):
        points = 1.25 if elo >= 25 else .6 if elo >= 10 else -1.25 if elo <= -25 else -.6 if elo <= -10 else 0.0
        add("Progresión Elo", elo, points, f"Variación reciente: {elo:+.1f} puntos")
    residual = metrics["surface_result_residual_90d"]
    if pd.isna(residual):
        residual = metrics["result_residual_30d"]
    if pd.notna(residual):
        points = 1.0 if residual >= .10 else .5 if residual >= .04 else -1.0 if residual <= -.10 else -.5 if residual <= -.04 else 0.0
        add("Rendimiento vs expectativa", residual, points, f"Residual reciente: {residual:+.3f}")
    for label, key in (("Tendencia de servicio", "serve_form_change"), ("Tendencia de resto", "return_form_change")):
        value = metrics[key]
        if pd.notna(value):
            points = .75 if value >= .02 else -.75 if value <= -.02 else 0.0
            add(label, value, points, f"Cambio short vs long: {value:+.3f}")
    rest = metrics["rest_days"]
    if pd.notna(rest):
        congested = rest <= 1 and pd.notna(metrics["matches_last_14d"]) and metrics["matches_last_14d"] >= 4
        points = -.75 if rest >= 35 else -.5 if congested else .25 if 3 <= rest <= 10 else 0.0
        add("Descanso y carga", rest, points, f"{rest:.0f} días de descanso; {metrics['matches_last_14d']:.0f} partidos en 14d" if pd.notna(metrics["matches_last_14d"]) else f"{rest:.0f} días de descanso")

    return {
        "player_id": player_id,
        "player_name": player_name,
        "arrival_score": score,
        "arrival_label": _arrival_label(score),
        "independent_from_win_probability": True,
        "metrics": json_safe(metrics),
        "components": json_safe(components),
        "coverage": {
            "recent_surface_matches": surface_matches,
            "momentum_confidence": json_safe(metrics["momentum_confidence"]),
            "momentum_effective_sample": json_safe(metrics["momentum_effective_sample"]),
        },
        "recent_surface_matches": evidence_frame(surface_history, [
            "match_date", "tourney_name", "competition_type", "surface",
            "opponent_name", "opponent_rank", "result", "score",
            "elo_before", "surface_elo_before",
        ]),
    }


def independent_arrival_tips(first: dict[str, Any], second: dict[str, Any]) -> list[dict[str, Any]]:
    """Tips contextuales que no utilizan probability_player_1/2."""
    tips: list[dict[str, Any]] = []
    difference = float(first["arrival_score"]) - float(second["arrival_score"])
    if abs(difference) >= 1.25:
        leader, trailer = (first, second) if difference > 0 else (second, first)
        tips.append({"severity": "positive", "title": "Ventaja de momento", "statement": f"{leader['player_name']} llega con mejores señales recientes que {trailer['player_name']} por forma en superficie, Elo, rendimiento frente a expectativa, servicio, resto y carga. Esta lectura no usa el % de victoria."})
    else:
        tips.append({"severity": "neutral", "title": "Momento equilibrado", "statement": "La diferencia entre los perfiles recientes no es suficiente para declarar una ventaja independiente clara."})
    for profile in (first, second):
        metrics = profile["metrics"]
        rows = int(profile["coverage"]["recent_surface_matches"])
        if rows < 5:
            tips.append({"severity": "warning", "title": "Muestra de superficie reducida", "statement": f"{profile['player_name']} solo tiene {rows} partido(s) entre los 10 más recientes en esta superficie; la lectura específica es frágil."})
        rest = pd.to_numeric(metrics.get("rest_days"), errors="coerce")
        if pd.notna(rest) and rest >= 35:
            tips.append({"severity": "warning", "title": "Inactividad", "statement": f"{profile['player_name']} llega con {rest:.0f} días sin competir. Es una señal de inactividad, no una afirmación de lesión."})
        serve = pd.to_numeric(metrics.get("serve_form_change"), errors="coerce")
        ret = pd.to_numeric(metrics.get("return_form_change"), errors="coerce")
        if pd.notna(serve) and serve >= .02:
            tips.append({"severity": "positive", "title": "Servicio en progresión", "statement": f"{profile['player_name']} presenta una mejora reciente de servicio de {serve:+.3f}."})
        if pd.notna(ret) and ret <= -.02:
            tips.append({"severity": "warning", "title": "Resto en retroceso", "statement": f"{profile['player_name']} presenta un descenso reciente de resto de {ret:+.3f}."})
    return tips


def _tournament_momentum_label(score: float) -> str:
    if score >= 2.0:
        return "Muy positivo"
    if score >= 0.75:
        return "Positivo"
    if score <= -2.0:
        return "Muy negativo"
    if score <= -0.75:
        return "Negativo"
    return "Neutral"


def current_tournament_profile(
    player_id: int,
    player_name: str,
    fixture: pd.Series,
) -> dict[str, Any]:
    """Current-edition momentum, independent from match win probability."""
    tournament = str(fixture.get("tournament", ""))
    tournament_id = fixture.get("tournament_id")
    fixture_date = fixture.get("match_date")
    stats = get_players_current_tournament_statistics(
        player_1_id=player_id,
        player_2_id=player_id,
        tournament_name=tournament,
        tournament_id=tournament_id,
        as_of_date=fixture_date,
        surface=fixture.get("surface"),
        competition_type=fixture.get("competition_type"),
        maximum_edition_days=28,
    )
    stats = stats.drop_duplicates("player_id") if not stats.empty else stats
    matches = get_player_current_tournament_matches(
        player_id=player_id,
        tournament_name=tournament,
        tournament_id=tournament_id,
        as_of_date=fixture_date,
        surface=fixture.get("surface"),
        competition_type=fixture.get("competition_type"),
        maximum_edition_days=28,
    )
    if stats.empty:
        return {
            "available": False,
            "player_id": player_id,
            "player_name": player_name,
            "reason": "no_completed_matches_in_current_edition",
            "matches": 0,
            "independent_from_win_probability": True,
        }
    row = stats.iloc[0]
    metrics = {
        key: json_safe(row.get(key))
        for key in (
            "matches", "wins", "sets_won", "sets_lost", "games_won",
            "games_lost", "games_played", "service_points_won_pct",
            "return_points_won_pct", "total_points_won_pct",
            "first_serve_in_pct", "first_serve_win_pct",
            "second_serve_win_pct", "break_points_faced",
            "break_points_saved", "break_points_saved_pct",
            "break_points_generated", "break_points_converted",
            "break_point_conversion_rate", "service_hold_rate", "aces",
            "double_faults", "minutes_accumulated", "average_opponent_rank",
            "average_opponent_elo", "point_stat_coverage", "score_coverage",
        )
    }
    score = 0.0
    components: list[dict[str, Any]] = []
    def add(factor: str, value: Any, contribution: float, detail: str) -> None:
        nonlocal score
        number = pd.to_numeric(value, errors="coerce")
        if pd.notna(number):
            score += contribution
            components.append({
                "factor": factor, "value": float(number),
                "contribution": contribution, "detail": detail,
            })

    service = pd.to_numeric(metrics["service_points_won_pct"], errors="coerce")
    if pd.notna(service):
        points = 1.0 if service >= .68 else .5 if service >= .64 else -1.0 if service <= .56 else -.5 if service <= .60 else 0.0
        add("Puntos ganados al servicio", service, points, f"{service:.1%} en la edición actual")
    ret = pd.to_numeric(metrics["return_points_won_pct"], errors="coerce")
    if pd.notna(ret):
        points = 1.0 if ret >= .42 else .5 if ret >= .38 else -1.0 if ret <= .28 else -.5 if ret <= .32 else 0.0
        add("Puntos ganados al resto", ret, points, f"{ret:.1%} en la edición actual")
    hold = pd.to_numeric(metrics["service_hold_rate"], errors="coerce")
    if pd.notna(hold):
        points = .75 if hold >= .85 else -.75 if hold <= .70 else 0.0
        add("Hold rate", hold, points, f"{hold:.1%} de juegos de servicio mantenidos")
    conversion = pd.to_numeric(metrics["break_point_conversion_rate"], errors="coerce")
    if pd.notna(conversion):
        points = .75 if conversion >= .45 else -.5 if conversion <= .25 else 0.0
        add("Conversión de break points", conversion, points, f"{conversion:.1%} de oportunidades convertidas")
    games_won = pd.to_numeric(metrics["games_won"], errors="coerce")
    games_lost = pd.to_numeric(metrics["games_lost"], errors="coerce")
    if pd.notna(games_won) and pd.notna(games_lost) and games_won + games_lost > 0:
        share = games_won / (games_won + games_lost)
        points = .75 if share >= .60 else -.75 if share <= .40 else 0.0
        add("Dominio por juegos", share, points, f"{share:.1%} de los juegos ganados")

    detail_columns = [
        "match_date", "round", "opponent_name", "opponent_rank", "result",
        "score", "minutes", "sets_won", "sets_lost", "games_won",
        "games_lost", "service_points_won_pct", "return_points_won_pct",
        "break_points_faced", "break_points_saved", "break_points_generated",
        "break_points_converted", "service_hold_rate", "total_points_won_pct",
    ]
    return {
        "available": True,
        "player_id": player_id,
        "player_name": player_name,
        "tournament": tournament,
        "matches": int(row.get("matches", 0)),
        "momentum_score": score,
        "momentum_label": _tournament_momentum_label(score),
        "independent_from_win_probability": True,
        "metrics": metrics,
        "components": json_safe(components),
        "match_evidence": evidence_frame(matches, detail_columns, 10),
    }


def current_tournament_tips(
    first: dict[str, Any], second: dict[str, Any]
) -> list[dict[str, Any]]:
    tips: list[dict[str, Any]] = []
    available = [item for item in (first, second) if item.get("available")]
    if not available:
        return [{
            "severity": "neutral", "title": "Sin muestra en la edición",
            "statement": "Ninguno de los jugadores tiene partidos completados registrados en la edición actual.",
        }]
    for profile in available:
        if int(profile.get("matches", 0)) < 2:
            tips.append({
                "severity": "warning", "title": "Muestra muy pequeña",
                "statement": f"{profile['player_name']} solo tiene {profile.get('matches', 0)} partido completado en la edición; los indicadores son descriptivos.",
            })
    if len(available) == 2:
        difference = float(first["momentum_score"]) - float(second["momentum_score"])
        if abs(difference) >= 1.0:
            leader, trailer = (first, second) if difference > 0 else (second, first)
            tips.append({
                "severity": "positive", "title": "Mejor rendimiento en la edición",
                "statement": f"{leader['player_name']} presenta mejores indicadores dentro del torneo que {trailer['player_name']}, considerando servicio, resto, break points y dominio por juegos. No se usa el % de victoria del modelo.",
            })
        else:
            tips.append({
                "severity": "neutral", "title": "Rendimiento de torneo equilibrado",
                "statement": "Los indicadores de la edición actual no muestran una ventaja clara entre ambos jugadores.",
            })
        first_minutes = pd.to_numeric(first["metrics"].get("minutes_accumulated"), errors="coerce")
        second_minutes = pd.to_numeric(second["metrics"].get("minutes_accumulated"), errors="coerce")
        if pd.notna(first_minutes) and pd.notna(second_minutes) and abs(first_minutes - second_minutes) >= 90:
            loaded, fresher = (first, second) if first_minutes > second_minutes else (second, first)
            tips.append({
                "severity": "warning", "title": "Diferencia de carga",
                "statement": f"{loaded['player_name']} acumula {max(first_minutes, second_minutes):.0f} minutos, frente a {min(first_minutes, second_minutes):.0f} de {fresher['player_name']}.",
            })
    return tips


def h2h_evidence(
    p1_id: int,
    p2_id: int,
    p1_name: str,
    p2_name: str,
    surface: str,
    fixture_date: Any,
    probabilities: dict[str, Any],
    over_line: float,
) -> list[Evidence]:
    history = get_head_to_head_history(p1_id, p2_id).copy()
    if history.empty:
        return [Evidence(
            "h2h", "Sin enfrentamientos previos",
            "No existen enfrentamientos directos en el histórico local.",
            "insufficient_sample", 0, str(fixture_date), "historical_match_dataset", [], "info"
        )]
    reference = pd.to_datetime(fixture_date, errors="coerce")
    history["match_date"] = pd.to_datetime(history["match_date"], errors="coerce")
    if pd.notna(reference):
        history = history.loc[history["match_date"].lt(reference.normalize())].copy()
    n = len(history)
    if not n:
        return []
    p1_wins = int(pd.to_numeric(history["winner_id"], errors="coerce").eq(p1_id).sum())
    p2_wins = n - p1_wins
    p1_probability = pd.to_numeric(probabilities.get("probability_player_1"), errors="coerce")
    favorite_name = p1_name if pd.notna(p1_probability) and p1_probability >= 0.5 else p2_name
    favorite_wins = p1_wins if favorite_name == p1_name else p2_wins
    output = [Evidence(
        "h2h", "Historial cara a cara",
        f"{p1_name} lidera {p1_wins}-{p2_wins} frente a {p2_name} en {n} enfrentamiento(s) registrado(s).",
        confidence_for_sample(n), n, str(fixture_date), "historical_match_dataset",
        evidence_frame(history, ["match_date", "tourney_name", "surface", "round", "winner_name", "loser_name", "score"]),
        "info",
    )]
    if favorite_wins == 0 and n > 0:
        probability = (
            p1_probability if favorite_name == p1_name else 1.0 - p1_probability
        )
        probability_text = f"{float(probability):.1%}" if pd.notna(probability) else "una probabilidad superior"
        output.append(Evidence(
            "model_h2h_conflict", "El modelo y el H2H discrepan",
            f"El modelo favorece a {favorite_name} con {probability_text}, pero {favorite_name} no ganó ninguno de los {n} enfrentamientos previos.",
            confidence_for_sample(n), n, str(fixture_date), "model_prediction_plus_h2h",
            evidence_frame(history, ["match_date", "surface", "winner_name", "score"]), "warning"
        ))

    history["total_games"] = history.get("score", pd.Series(index=history.index, dtype="object")).map(score_games)
    scored = history.dropna(subset=["total_games"]).copy()
    if not scored.empty:
        overs = int(scored["total_games"].gt(over_line).sum())
        rate = overs / len(scored)
        surface_scored = scored.loc[scored.get("surface", pd.Series("", index=scored.index)).astype(str).eq(surface)]
        output.append(Evidence(
            "h2h_games", f"Juegos en el H2H sobre {over_line:.1f}",
            f"{overs} de {len(scored)} enfrentamientos con marcador analizable ({rate:.0%}) superaron {over_line:.1f} juegos. En {surface}: {int(surface_scored['total_games'].gt(over_line).sum())} de {len(surface_scored)}.",
            confidence_for_sample(len(scored)), len(scored), str(fixture_date), "historical_score_field",
            evidence_frame(scored, ["match_date", "surface", "score", "total_games"]),
            "info" if len(scored) >= 3 else "warning",
        ))
    return output


def tournament_edition_context(
    tournament: str,
    competition_type: str,
    fixture_date: Any,
) -> list[Evidence]:
    """Summarize comparable historical editions from tournament analytics."""
    try:
        editions = get_tournament_editions(tournament, competition_type)
    except Exception:
        return []
    if editions.empty:
        return []
    reference = pd.to_datetime(fixture_date, errors="coerce")
    if "season" in editions and pd.notna(reference):
        editions = editions.loc[
            pd.to_numeric(editions["season"], errors="coerce").lt(reference.year)
        ].copy()
    if editions.empty:
        return []
    editions = editions.sort_values("season", ascending=False, kind="mergesort")
    latest = editions.iloc[0]
    rows = int(len(editions))
    average_field = pd.to_numeric(editions.get("unique_players"), errors="coerce").mean()
    average_matches = pd.to_numeric(editions.get("matches"), errors="coerce").mean()
    statement = (
        f"Hay {rows} edición(es) históricas comparables de {tournament} para "
        f"{competition_type}. La edición más reciente disponible es {int(latest['season'])}."
    )
    if pd.notna(average_field) and pd.notna(average_matches):
        statement += (
            f" El cuadro histórico medio contiene {average_field:.0f} jugadores "
            f"y {average_matches:.0f} partidos registrados."
        )
    return [Evidence(
        "tournament_context",
        "Contexto histórico de la edición",
        statement,
        confidence_for_sample(rows),
        rows,
        str(fixture_date),
        "tournaments_adapted",
        evidence_frame(
            editions,
            ["season", "tournament_name", "competition_type", "surface", "tournament_level", "matches", "unique_players", "first_match_date", "last_match_date"],
            10,
        ),
        "info",
    )]


def tournament_evidence(
    player_id: int,
    player_name: str,
    tournament: str,
    tournament_id: Any,
    fixture_date: Any,
) -> list[Evidence]:
    history = get_player_tournament_history(
        player_id=player_id,
        tournament_name=tournament,
        tournament_id=tournament_id,
        years=10,
        as_of_date=fixture_date,
        exclude_current_edition=True,
    )
    if history.empty:
        return []
    latest = history.sort_values("season", ascending=False, kind="mergesort").iloc[0]
    return [Evidence(
        "tournament_history", "Historial en el torneo",
        f"En {int(latest['season'])}, {player_name} terminó con el resultado: {latest['outcome']}. El cálculo de puntos defendidos queda pendiente de una tabla oficial por categoría y ronda.",
        confidence_for_sample(len(history)), len(history), str(fixture_date), "canonical_tournament_history",
        evidence_frame(history, ["season", "tournament", "surface", "matches", "wins", "best_round", "outcome", "last_opponent", "last_score"]),
        "info",
    )]


def generate_dossier(fixture: pd.Series, lookup: dict[str, tuple[int, str]]) -> dict[str, Any]:
    key = stable_fixture_key(fixture)
    p1_lookup, p1_resolution = resolve_fixture_player(
        fixture["player_1_name"], lookup
    )
    p2_lookup, p2_resolution = resolve_fixture_player(
        fixture["player_2_name"], lookup
    )
    if p1_lookup is None or p2_lookup is None:
        return {
            "schema_version": PAGE_SCHEMA_VERSION,
            "fixture_key": key,
            "available": False,
            "reason": "player_not_matched",
            "player_resolution": {
                "player_1": p1_resolution,
                "player_2": p2_resolution,
            },
        }
    p1_id, p1_name = p1_lookup
    p2_id, p2_name = p2_lookup
    prediction = prediction_from_cache(key)
    if not prediction:
        prediction = calculate_prediction(fixture, p1_id, p2_id)
    if prediction.get("available") is False:
        model = prediction
    else:
        model = prediction

    surface = str(fixture.get("surface", "Hard"))
    best_of = int(fixture.get("best_of", 3))
    line = DEFAULT_OVER_LINE_BO5 if best_of == 5 else DEFAULT_OVER_LINE_BO3
    arrival_player_1 = player_arrival_profile(p1_id, p1_name, surface, fixture)
    arrival_player_2 = player_arrival_profile(p2_id, p2_name, surface, fixture)
    arrival_tips = independent_arrival_tips(arrival_player_1, arrival_player_2)
    tournament_player_1 = current_tournament_profile(p1_id, p1_name, fixture)
    tournament_player_2 = current_tournament_profile(p2_id, p2_name, fixture)
    tournament_tips = current_tournament_tips(
        tournament_player_1, tournament_player_2
    )
    empirical = get_match_market_probabilities(
        player_1_id=p1_id,
        player_2_id=p2_id,
        surface=surface,
        best_of=best_of,
        over_games_line=line,
        as_of_date=fixture.get("match_date"),
    )
    evidence: list[Evidence] = []
    evidence.extend(player_context_evidence(p1_id, p1_name, surface, fixture.get("match_date")))
    evidence.extend(player_context_evidence(p2_id, p2_name, surface, fixture.get("match_date")))
    evidence.extend(h2h_evidence(
        p1_id, p2_id, p1_name, p2_name, surface, fixture.get("match_date"), model, line
    ))
    evidence.extend(tournament_edition_context(
        str(fixture.get("tournament", "")),
        str(fixture.get("competition_type", "UNKNOWN")),
        fixture.get("match_date"),
    ))
    evidence.extend(tournament_evidence(
        p1_id, p1_name, str(fixture.get("tournament", "")),
        fixture.get("tournament_id"), fixture.get("match_date")
    ))
    evidence.extend(tournament_evidence(
        p2_id, p2_name, str(fixture.get("tournament", "")),
        fixture.get("tournament_id"), fixture.get("match_date")
    ))

    market = model.get("market_probabilities", {}) if isinstance(model, dict) else {}
    try:
        tournament_first_set = analyze_tournament_first_set_history(
            tournament_name=str(fixture.get("tournament", "")),
            competition_type=fixture.get("competition_type"),
            surface=fixture.get("surface"),
            as_of_date=fixture.get("match_date"),
        )
    except Exception as error:
        tournament_first_set = {
            "available": False,
            "reason": "historical_first_set_analysis_error",
            "error": f"{type(error).__name__}: {error}",
        }
    try:
        tournament_any_set = analyze_tournament_any_set_history(
            tournament_name=str(fixture.get("tournament", "")),
            competition_type=fixture.get("competition_type"),
            surface=fixture.get("surface"),
            as_of_date=fixture.get("match_date"),
        )
    except Exception as error:
        tournament_any_set = {
            "available": False,
            "reason": "historical_any_set_analysis_error",
            "error": f"{type(error).__name__}: {error}",
        }
    payload = {
        "schema_version": PAGE_SCHEMA_VERSION,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "fixture_key": key,
        "available": True,
        "fixture": json_safe(fixture.to_dict()),
        "players": {
            "player_1": {"id": p1_id, "name": p1_name},
            "player_2": {"id": p2_id, "name": p2_name},
        },
        "player_resolution": {
            "player_1": p1_resolution,
            "player_2": p2_resolution,
        },
        "prediction": json_safe({
            "probability_player_1": model.get("probability_player_1"),
            "probability_player_2": model.get("probability_player_2"),
            "model_version": model.get("model_version"),
            "market_probabilities": market,
            "market_model_status": market.get("model_status", {}),
            "over_model_status": market.get("over_model_status", {}),
        }),
        "arrival_assessment": {
            "independent_from_model_win_probability": True,
            "player_1": arrival_player_1,
            "player_2": arrival_player_2,
            "tips": arrival_tips,
        },
        "current_tournament_assessment": {
            "independent_from_model_win_probability": True,
            "player_1": tournament_player_1,
            "player_2": tournament_player_2,
            "tips": tournament_tips,
        },
        "empirical_market_evidence": json_safe(empirical),
        "tournament_direct_market_history": json_safe({
            "first_set": tournament_first_set,
            "any_set": tournament_any_set,
        }),
        "evidence": [asdict(item) for item in evidence],
        "guardrails": [
            "An absence or retirement does not prove an injury.",
            "Small H2H samples are descriptive and not predictive guarantees.",
            "Ranking points defended are omitted until an official versioned points table is connected.",
            "Direct market probabilities and structural simulation are different methodologies and are displayed separately.",
            "At-least-one-set probabilities are not complementary because both players can win a set.",
        ],
    }
    return json_safe(payload)


def direct_market_view(market: dict[str, Any]) -> dict[str, Any]:
    """Normalize direct-market output while retaining backward compatibility."""
    return {
        "available": bool(market.get("available")),
        "player_1_first_set_probability": market.get("player_1_first_set_probability"),
        "player_2_first_set_probability": market.get("player_2_first_set_probability"),
        "player_1_win_any_set_probability": market.get(
            "player_1_win_any_set_probability",
            market.get("player_1_win_set_probability"),
        ),
        "player_2_win_any_set_probability": market.get(
            "player_2_win_any_set_probability",
            market.get("player_2_win_set_probability"),
        ),
        "over_probabilities": market.get("over_probabilities", {}),
        "model_status": market.get("model_status", {}),
        "over_model_status": market.get("over_model_status", {}),
        "structural_baseline": market.get("structural_baseline", {}),
        "selected_over_line": market.get("over_games_line"),
        "selected_over_probability": market.get("over_games_probability"),
    }


def market_status_label(value: Any) -> str:
    return {
        "validated_candidate": "Candidato validado",
        "provisional": "Provisional",
        "experimental": "Experimental",
        "not_validated": "No validado",
    }.get(str(value or ""), str(value or "-").replace("_", " ").title())


def answer_direct_market_question(
    question: str,
    dossier: dict[str, Any],
) -> str | None:
    text = normalize_text(question)
    prediction = dossier.get("prediction", {})
    players = dossier.get("players", {})
    market = direct_market_view(prediction.get("market_probabilities", {}) or {})
    p1_name = players.get("player_1", {}).get("name", "Player 1")
    p2_name = players.get("player_2", {}).get("name", "Player 2")
    if any(token in text for token in ("primer set", "first set", "primer parcial")):
        p1 = pd.to_numeric(market.get("player_1_first_set_probability"), errors="coerce")
        p2 = pd.to_numeric(market.get("player_2_first_set_probability"), errors="coerce")
        if pd.isna(p1) or pd.isna(p2):
            return "El modelo directo de primer set no está disponible para este fixture."
        favorite = p1_name if p1 >= p2 else p2_name
        status = market_status_label(market.get("model_status", {}).get("first_set"))
        return (
            f"{p1_name} tiene {p1:.1%} de probabilidad de ganar el primer set y "
            f"{p2_name} {p2:.1%}. El favorito para el primer set es {favorite}. "
            f"Estado del modelo: {status}."
        )
    if any(token in text for token in (
        "ganar un set", "gana un set", "al menos un set", "algún set", "algun set"
    )):
        p1 = pd.to_numeric(market.get("player_1_win_any_set_probability"), errors="coerce")
        p2 = pd.to_numeric(market.get("player_2_win_any_set_probability"), errors="coerce")
        if pd.isna(p1) or pd.isna(p2):
            return "El modelo de ganar al menos un set no está disponible para este fixture."
        status = market_status_label(market.get("model_status", {}).get("win_any_set"))
        note = (
            "Estas probabilidades no son complementarias: ambos jugadores pueden "
            "ganar al menos un set en el mismo partido."
        )
        return (
            f"{p1_name}: {p1:.1%}. {p2_name}: {p2:.1%}. "
            f"Estado del modelo: {status}. {note}"
        )
    line_match = re.search(r"(?:over|mas de|más de)\s*(\d+(?:[.,]\d+)?)", str(question).casefold())
    if line_match:
        requested = float(line_match.group(1).replace(",", "."))
        overs = {
            float(key): pd.to_numeric(value, errors="coerce")
            for key, value in market.get("over_probabilities", {}).items()
        }
        if overs:
            line = min(overs, key=lambda value: abs(value - requested))
            probability = overs[line]
            status = market_status_label(
                market.get("over_model_status", {}).get(f"{line:.1f}")
            )
            return (
                f"La línea directa disponible más cercana es Over {line:.1f}: "
                f"{probability:.1%}. Estado del modelo: {status}."
            )
        selected = pd.to_numeric(market.get("selected_over_probability"), errors="coerce")
        selected_line = market.get("selected_over_line")
        if pd.notna(selected):
            return (
                f"Para este fixture está disponible Over {selected_line}: "
                f"{selected:.1%}. En BO5 esta salida puede proceder del simulador estructural."
            )
        return "No hay una probabilidad Over disponible para este fixture."
    if "suman" in text and any(token in text for token in ("set", "porcentaje", "probabilidad")):
        return (
            "Las probabilidades de ganar al menos un set no tienen que sumar 100%. "
            "En un partido a tres o cinco sets, ambos jugadores pueden ganar algún set. "
            "Las probabilidades de ganar el primer set sí son complementarias."
        )
    if "experimental" in text or "validado" in text or "estado" in text:
        statuses = market.get("over_model_status", {})
        if statuses:
            lines = ", ".join(
                f"Over {line}: {market_status_label(status)}"
                for line, status in statuses.items()
            )
            return (
                "Primer set: "
                f"{market_status_label(market.get('model_status', {}).get('first_set'))}. "
                "Ganar al menos un set: "
                f"{market_status_label(market.get('model_status', {}).get('win_any_set'))}. "
                f"Overs: {lines}."
            )
    return None


def percent(value: Any) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "-" if pd.isna(number) else f"{float(number):.1%}"


def _extract_games_threshold(question: str) -> float | None:
    """Extract 'over/more than/superaron N juegos' from normalized Spanish text."""
    # Preserve decimal separators before general textual normalization.
    raw = unicodedata.normalize("NFKD", str(question or ""))
    raw = "".join(char for char in raw if not unicodedata.combining(char))
    text = raw.casefold()
    text = re.sub(r"[^a-z0-9.,]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    patterns = (
        r"(?:mas de|superaron|supera|over|por encima de)\s+(?:los\s+)?(\d+(?:[.,]\d+)?)\s+(?:juegos|games)",
        r"(?:juegos|games)\s+(?:por encima de|superiores a|mayores de)\s+(\d+(?:[.,]\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def answer_dynamic_tournament_question(
    question: str, dossier: dict[str, Any]
) -> str | None:
    """Run allow-listed tournament analytics for supported chat intents."""
    text = normalize_text(question)
    threshold = _extract_games_threshold(question)
    if threshold is None or not any(token in text for token in (
        "torneo", "challenger", "historico", "historial", "edicion",
    )):
        return None
    fixture = dossier.get("fixture", {})
    tournament = str(fixture.get("tournament", "")).strip()
    if not tournament:
        return "No puedo identificar el torneo del fixture seleccionado."
    result = analyze_tournament_games_threshold(
        tournament_name=tournament,
        threshold=threshold,
        competition_type=fixture.get("competition_type"),
        surface=fixture.get("surface"),
        as_of_date=fixture.get("match_date"),
        evidence_limit=20,
    )
    if not result.get("available"):
        return (
            f"No hay una muestra histórica analizable para {tournament}. "
            f"Motivo técnico: {result.get('reason', 'unknown')}."
        )
    qualifying = int(result.get("qualifying_matches", 0))
    covered = int(result.get("score_covered_matches", 0))
    rate = pd.to_numeric(result.get("observed_rate"), errors="coerce")
    rate_text = f"{float(rate):.1%}" if pd.notna(rate) else "-"
    first = result.get("first_season")
    last = result.get("last_season")
    excluded = int(result.get("excluded_matches", 0))
    names = ", ".join(result.get("matched_historical_names", [])[:5])
    return (
        f"En el histórico disponible de {tournament}, {qualifying} de {covered} "
        f"partidos con marcador analizable superaron los {threshold:g} juegos "
        f"({rate_text}). Como el total de juegos es entero, 'más de {threshold:g}' "
        f"equivale a {result.get('equivalent_minimum_integer_games')} o más. "
        f"Se excluyeron {excluded} partidos sin marcador interpretable, incluidos "
        f"{result.get('excluded_walkovers', 0)} walkovers. Periodo cubierto: "
        f"{first}-{last}. Nombres históricos emparejados: {names or '-'}."
    )


def answer_from_dossier(question: str, dossier: dict[str, Any]) -> str:
    direct_answer = answer_direct_market_question(question, dossier)
    if direct_answer is not None:
        return direct_answer
    dynamic_answer = answer_dynamic_tournament_question(question, dossier)
    if dynamic_answer is not None:
        return dynamic_answer
    text = normalize_text(question)
    evidence = dossier.get("evidence", [])
    categories: set[str] = set()
    if any(token in text for token in ("h2h", "enfrent", "cara a cara")):
        categories.update({"h2h", "model_h2h_conflict", "h2h_games"})
    if any(token in text for token in ("over", "juegos", "21 5", "total")):
        categories.add("h2h_games")
    if any(token in text for token in ("superficie", "clay", "hard", "grass", "gira")):
        categories.update({"surface_experience", "surface_switch"})
    if any(token in text for token in ("torneo actual", "edicion actual", "break point", "bp salv", "bp gener", "puntos en el torneo", "sets en el torneo")):
        current = dossier.get("current_tournament_assessment", {})
        profiles = [current.get("player_1", {}), current.get("player_2", {})]
        lines = []
        for item in profiles:
            if not item.get("available"):
                lines.append(f"{item.get('player_name')}: sin partidos completados registrados en la edición actual.")
                continue
            metrics = item.get("metrics", {})
            lines.append(
                f"{item.get('player_name')}: {item.get('momentum_label')} "
                f"({item.get('matches', 0)} partidos); servicio {percent(metrics.get('service_points_won_pct'))}, "
                f"resto {percent(metrics.get('return_points_won_pct'))}, "
                f"BP salvados {metrics.get('break_points_saved', '-')}/{metrics.get('break_points_faced', '-')}, "
                f"BP convertidos {metrics.get('break_points_converted', '-')}/{metrics.get('break_points_generated', '-')} ."
            )
        lines.extend(tip.get("statement", "") for tip in current.get("tips", [])[:6])
        return "\n\n".join(line for line in lines if line)
    if any(token in text for token in ("momento", "llega", "forma", "elo", "rolling", "servicio", "resto", "tip")):
        arrival = dossier.get("arrival_assessment", {})
        profiles = [arrival.get("player_1", {}), arrival.get("player_2", {})]
        lines = [f"{item.get('player_name')}: {item.get('arrival_label')} ({float(item.get('arrival_score', 0)):+.2f})." for item in profiles if item]
        lines.extend(tip.get("statement", "") for tip in arrival.get("tips", [])[:6])
        return "\n\n".join(line for line in lines if line)
    if any(token in text for token in ("lesion", "retir", "walkover", "inactiv")):
        categories.update({"inactivity", "match_status"})
    if any(token in text for token in ("torneo", "ronda", "puntos", "defiende")):
        categories.update({"tournament_history", "tournament_context"})
    selected = [item for item in evidence if not categories or item.get("category") in categories]
    if not selected:
        return (
            "No encuentro evidencia suficiente en el dossier para responder esa pregunta. "
            "La ausencia de evidencia no debe convertirse en una afirmación."
        )
    lines = [item["statement"] for item in selected[:6]]
    limitations = []
    if any(item.get("confidence") in {"insufficient_sample", "limited_sample"} for item in selected):
        limitations.append("Al menos una conclusión usa una muestra pequeña.")
    if any(item.get("category") in {"inactivity", "match_status"} for item in selected):
        limitations.append("Estos datos no acreditan por sí solos una lesión.")
    return "\n\n".join([*lines, *(f"Limitación: {item}" for item in limitations)])


st.set_page_config(page_title="IA Candidates", page_icon="🤖", layout="wide")
st.title("🤖 IA Candidates")
st.caption(
    "Dossiers prepartido con momento de llegada, evidencia histórica, predicciones y contexto del torneo."
)
st.info(
    "Esta primera versión es evidence-first: genera conclusiones deterministas. "
    "No afirma lesiones sin una fuente estructurada y no calcula puntos defendidos todavía."
)

if "ia_candidate_dossiers" not in st.session_state:
    st.session_state["ia_candidate_dossiers"] = load_disk_dossiers()
st.session_state["ia_candidate_dossiers"] = {
    key: value for key, value in st.session_state["ia_candidate_dossiers"].items()
    if isinstance(value, dict) and value.get("schema_version") == PAGE_SCHEMA_VERSION
}
if "ia_candidate_chat" not in st.session_state:
    st.session_state["ia_candidate_chat"] = {}

fixtures = load_fixture_cache(
    str(FIXTURE_CACHE),
    FIXTURE_CACHE.stat().st_mtime_ns if FIXTURE_CACHE.exists() else 0,
)
if fixtures.empty:
    fixtures = st.session_state.get("upcoming_matches", pd.DataFrame()).copy()
if fixtures.empty:
    st.warning("No hay fixtures disponibles. Carga primero Upcoming Matches.")
    st.stop()

fixtures = fixtures.copy()
fixtures["match_date"] = pd.to_datetime(fixtures["match_date"], errors="coerce").dt.date
handoff = st.session_state.get("ia_candidate_selection")
handoff_fixture = handoff.get("fixture") if isinstance(handoff, dict) else None
if isinstance(handoff_fixture, dict):
    handoff_frame = pd.DataFrame([handoff_fixture])
    handoff_frame["match_date"] = pd.to_datetime(
        handoff_frame["match_date"], errors="coerce"
    ).dt.date
    fixtures = pd.concat([fixtures, handoff_frame], ignore_index=True, sort=False)
fixtures = fixtures.loc[
    fixtures.get("competition_type", "").isin(["ATP", "ATP_QUALIFYING", "CHALLENGER"])
].copy()
fixtures["fixture_key"] = fixtures.apply(stable_fixture_key, axis=1)
if not isinstance(handoff_fixture, dict):
    fixtures = fixtures.loc[fixtures["match_date"].ge(date.today())].copy()
fixtures = fixtures.drop_duplicates("fixture_key", keep="last")
fixtures["label"] = fixtures.apply(
    lambda row: (
        f"{row.get('match_date')} · {row.get('scheduled_time', 'TBD')} · "
        f"{row.get('player_1_name')} vs {row.get('player_2_name')} · {row.get('tournament')}"
    ), axis=1
)
fixtures = fixtures.sort_values(
    [column for column in ("match_date", "scheduled_time", "tournament") if column in fixtures],
    kind="mergesort",
).reset_index(drop=True)

f1, f2, f3 = st.columns(3)
with f1:
    competitions = st.multiselect(
        "Competition",
        ["ATP", "ATP_QUALIFYING", "CHALLENGER"],
        default=["ATP", "ATP_QUALIFYING", "CHALLENGER"],
    )
with f2:
    surfaces = sorted(fixtures.get("surface", pd.Series(dtype="string")).dropna().astype(str).unique())
    selected_surfaces = st.multiselect("Surface", surfaces, default=surfaces)
with f3:
    search = st.text_input("Search", placeholder="Player or tournament")

visible = fixtures.loc[fixtures["competition_type"].isin(competitions)].copy()
if selected_surfaces:
    visible = visible.loc[visible["surface"].astype(str).isin(selected_surfaces)]
if search.strip():
    needle = normalize_text(search)
    visible = visible.loc[visible["label"].map(normalize_text).str.contains(needle, regex=False)]
if visible.empty:
    st.warning("No hay partidos para los filtros actuales.")
    st.stop()

labels = visible["label"].tolist()
handoff_key = str(handoff.get("fixture_key", "")) if isinstance(handoff, dict) else ""
selected_index = 0
if handoff_key:
    matching = visible.index[visible["fixture_key"].astype(str).eq(handoff_key)].tolist()
    if matching:
        selected_label_value = str(visible.loc[matching[0], "label"])
        selected_index = labels.index(selected_label_value)
selected_label = st.selectbox("Partido", labels, index=selected_index)
fixture = visible.loc[visible["label"].eq(selected_label)].iloc[0]
fixture_key = str(fixture["fixture_key"])
dossiers: dict[str, Any] = st.session_state["ia_candidate_dossiers"]
lookup = player_lookup()

left, right = st.columns([1, 2])
with left:
    st.subheader("Fixture")
    st.write(f"**{fixture['player_1_name']} vs {fixture['player_2_name']}**")
    st.write(f"{fixture.get('tournament')} · {fixture.get('surface')} · {fixture.get('round')}")
    st.write(f"{fixture.get('match_date')} · {fixture.get('scheduled_time', 'TBD')}")
    generate = st.button("Generate AI dossier", type="primary", use_container_width=True)
    refresh_dossier = st.button("Regenerate dossier", use_container_width=True)

auto_generate = bool(
    isinstance(handoff, dict)
    and handoff.get("auto_generate")
    and handoff_key == fixture_key
    and fixture_key not in dossiers
)
if generate or refresh_dossier or auto_generate:
    with st.spinner("Consultando histórico, H2H, superficie, torneo y modelo..."):
        dossiers[fixture_key] = generate_dossier(fixture, lookup)
        st.session_state["ia_candidate_dossiers"] = dossiers
        save_disk_dossiers(dossiers)
    if isinstance(handoff, dict):
        updated_handoff = dict(handoff)
        updated_handoff["auto_generate"] = False
        st.session_state["ia_candidate_selection"] = updated_handoff
    st.rerun()

dossier = dossiers.get(fixture_key)
with right:
    st.subheader("Model and market context")
    if not dossier:
        st.caption("Genera el dossier para cargar predicciones y evidencia.")
    elif not dossier.get("available"):
        st.error(f"Dossier unavailable: {dossier.get('reason')}")
        resolution = dossier.get("player_resolution", {})
        if resolution:
            with st.expander("Player matching diagnostics", expanded=True):
                st.json(resolution)
    else:
        prediction = dossier.get("prediction", {})
        market = prediction.get("market_probabilities", {}) or {}
        direct = direct_market_view(market)
        winner_1, winner_2 = st.columns(2)
        winner_1.metric(
            f"{fixture['player_1_name']} wins",
            percent(prediction.get("probability_player_1")),
        )
        winner_2.metric(
            f"{fixture['player_2_name']} wins",
            percent(prediction.get("probability_player_2")),
        )
        st.markdown("**Calibrated direct-market models**")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "P1 first set",
            percent(direct.get("player_1_first_set_probability")),
            help=market_status_label(direct.get("model_status", {}).get("first_set")),
        )
        m2.metric(
            "P2 first set",
            percent(direct.get("player_2_first_set_probability")),
            help=market_status_label(direct.get("model_status", {}).get("first_set")),
        )
        m3.metric(
            "P1 wins >=1 set",
            percent(direct.get("player_1_win_any_set_probability")),
            help=market_status_label(direct.get("model_status", {}).get("win_any_set")),
        )
        m4.metric(
            "P2 wins >=1 set",
            percent(direct.get("player_2_win_any_set_probability")),
            help=market_status_label(direct.get("model_status", {}).get("win_any_set")),
        )
        if direct.get("over_probabilities"):
            over_rows = [
                {
                    "Market": f"Over {line}",
                    "Probability": percent(probability),
                    "Status": market_status_label(
                        direct.get("over_model_status", {}).get(str(line))
                    ),
                }
                for line, probability in direct["over_probabilities"].items()
            ]
            st.dataframe(
                pd.DataFrame(over_rows),
                use_container_width=True,
                hide_index=True,
            )
        else:
            line = direct.get("selected_over_line", market.get("over_games_line"))
            probability = direct.get(
                "selected_over_probability", market.get("over_games_probability")
            )
            if pd.notna(pd.to_numeric(probability, errors="coerce")):
                st.metric(
                    f"Structural Over {line}",
                    percent(probability),
                    help="BO5 or fallback structural simulation",
                )
        structural = direct.get("structural_baseline", {})
        if structural.get("available"):
            with st.expander("Structural serve-point simulation", expanded=False):
                s1, s2 = st.columns(2)
                s1.metric(
                    "Expected total games",
                    f"{pd.to_numeric(structural.get('expected_total_games'), errors='coerce'):.1f}",
                )
                s2.metric(
                    "Simulation match win P1",
                    percent(structural.get("match_win_player_1")),
                )
                scores = structural.get("set_score_probabilities", {})
                if scores:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Set score": score, "Probability": percent(probability)}
                                for score, probability in scores.items()
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

if dossier and dossier.get("available"):
    current_tournament = dossier.get("current_tournament_assessment", {})
    if current_tournament:
        st.subheader("Momentum en el torneo actual")
        st.caption("Partidos completados de la edición actual. No utiliza la probabilidad de victoria del modelo.")
        tournament_left, tournament_right = st.columns(2)
        for container, profile in (
            (tournament_left, current_tournament.get("player_1", {})),
            (tournament_right, current_tournament.get("player_2", {})),
        ):
            with container:
                if not profile.get("available"):
                    st.info(f"{profile.get('player_name', 'Player')}: sin partidos completados registrados en esta edición.")
                    continue
                st.metric(
                    profile.get("player_name", "Player"),
                    profile.get("momentum_label", "-"),
                    delta=f"Score torneo {float(profile.get('momentum_score', 0)):+.2f}",
                )
                metrics = profile.get("metrics", {})
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Partidos", metrics.get("matches", "-"))
                k2.metric("Sets", f"{metrics.get('sets_won', '-')}-{metrics.get('sets_lost', '-')}")
                k3.metric("Juegos", f"{metrics.get('games_won', '-')}-{metrics.get('games_lost', '-')}")
                k4.metric("Minutos", metrics.get("minutes_accumulated", "-"))
                comparison = pd.DataFrame([
                    {"Métrica": "Puntos servicio ganados", "Valor": percent(metrics.get("service_points_won_pct"))},
                    {"Métrica": "Puntos resto ganados", "Valor": percent(metrics.get("return_points_won_pct"))},
                    {"Métrica": "Puntos totales ganados", "Valor": percent(metrics.get("total_points_won_pct"))},
                    {"Métrica": "Hold rate", "Valor": percent(metrics.get("service_hold_rate"))},
                    {"Métrica": "BP salvados", "Valor": f"{metrics.get('break_points_saved', '-')}/{metrics.get('break_points_faced', '-')} ({percent(metrics.get('break_points_saved_pct'))})"},
                    {"Métrica": "BP convertidos", "Valor": f"{metrics.get('break_points_converted', '-')}/{metrics.get('break_points_generated', '-')} ({percent(metrics.get('break_point_conversion_rate'))})"},
                    {"Métrica": "Aces / DF", "Valor": f"{metrics.get('aces', '-')} / {metrics.get('double_faults', '-')}"},
                    {"Métrica": "Ranking medio rivales", "Valor": metrics.get("average_opponent_rank", "-")},
                ])
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                if profile.get("components"):
                    st.dataframe(pd.DataFrame(profile["components"]), use_container_width=True, hide_index=True)
                with st.expander("Evidencia partido a partido"):
                    evidence_rows = profile.get("match_evidence", [])
                    if evidence_rows:
                        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)
                    else:
                        st.caption("Sin evidencia partido a partido disponible.")
        st.markdown("**Tips del torneo actual**")
        for tip in current_tournament.get("tips", []):
            icon = "✅" if tip.get("severity") == "positive" else "⚠️" if tip.get("severity") == "warning" else "ℹ️"
            st.write(f"{icon} **{tip.get('title')}**: {tip.get('statement')}")

    arrival = dossier.get("arrival_assessment", {})
    if arrival:
        st.subheader("Momento de llegada, independiente del % de victoria")
        st.caption("No utiliza probability_player_1/2. Resume superficie, rolling stats, Momentum v6, Elo, servicio, resto, carga y descanso.")
        left_arrival, right_arrival = st.columns(2)
        for container, profile in ((left_arrival, arrival.get("player_1", {})), (right_arrival, arrival.get("player_2", {}))):
            with container:
                st.metric(profile.get("player_name", "Player"), profile.get("arrival_label", "-"), delta=f"Score {float(profile.get('arrival_score', 0)):+.2f}")
                coverage = profile.get("coverage", {})
                confidence = coverage.get("momentum_confidence")
                confidence_text = percent(confidence) if confidence is not None else "-"
                st.caption(f"Muestra superficie: {coverage.get('recent_surface_matches', 0)} · Confianza momentum: {confidence_text}")
                if profile.get("components"):
                    st.dataframe(pd.DataFrame(profile["components"]), use_container_width=True, hide_index=True)
                with st.expander("Últimos partidos en la superficie"):
                    recent = profile.get("recent_surface_matches", [])
                    if recent:
                        st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
                    else:
                        st.caption("Sin muestra reciente en esta superficie.")
        st.markdown("**Tips independientes**")
        for tip in arrival.get("tips", []):
            icon = "✅" if tip.get("severity") == "positive" else "⚠️" if tip.get("severity") == "warning" else "ℹ️"
            st.write(f"{icon} **{tip.get('title')}**: {tip.get('statement')}")
    st.subheader("Evidence-backed conclusions")
    evidence = dossier.get("evidence", [])
    severity_order = {"warning": 0, "info": 1}
    evidence = sorted(evidence, key=lambda item: (severity_order.get(item.get("severity"), 9), item.get("category", "")))
    for index, item in enumerate(evidence):
        icon = "⚠️" if item.get("severity") == "warning" else "ℹ️"
        with st.expander(
            f"{icon} {item.get('label')} · {item.get('confidence')} · n={item.get('sample_size', '-')}",
            expanded=index < 4,
        ):
            st.write(item.get("statement"))
            st.caption(f"Source: {item.get('source')} · As of: {item.get('as_of_date')}")
            details = item.get("details", [])
            if details:
                st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)

    st.subheader("Ask this dossier")
    history = st.session_state["ia_candidate_chat"].setdefault(fixture_key, [])
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    question = st.chat_input("Ej. ¿Cuántos partidos históricos del torneo superaron los 17 juegos?")
    if question:
        answer = answer_from_dossier(question, dossier)
        history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        st.session_state["ia_candidate_chat"][fixture_key] = history
        st.rerun()

    with st.expander("Raw dossier JSON"):
        st.json(dossier)

st.divider()
st.caption(
    "Puntos defendidos: el historial del torneo ya está integrado; falta una tabla oficial versionada de puntos por nivel y ronda. "
    "Lesiones: pendiente de una fuente explícita; nunca se infieren únicamente desde inactividad o retiradas."
)
