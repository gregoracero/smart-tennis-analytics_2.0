#!/usr/bin/env python3
"""Upcoming Matches: today/tomorrow tennis fixtures and Predictor handoff.

Provider: Tennis-API.com Fixtures v2 (RapidAPI).
Configuration (Streamlit secrets or environment):

TENNIS_API_KEY = "..."
TENNIS_API_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
SOFASCORE_API_KEY = "YOUR_SOFASCORE_RAPIDAPI_KEY"
SOFASCORE_API_HOST = "sofascore-scraper-1000-free-calls.p.rapidapi.com"

The page requests ATP singles fixtures by date, classifies ATP main draw,
ATP qualifying and Challenger, lets a user select one fixture, maps players to
the local catalog and stores `upcoming_match_selection` in session_state.
"""
from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd
import requests
import streamlit as st
import numpy as np

from streamlit_app.config.data_paths import (
    ACTIVE_MODEL_PATH,
    PRODUCTION_ARTIFACTS_DIR,
    UPCOMING_CLASSIFICATION_AUDIT_PATH,
    UPCOMING_MATCHES_CACHE_PATH,
)
from streamlit_app.data_access.players_adapted import get_match_market_probabilities
from streamlit_app.services.sofascore_scraper_service import (
    DEFAULT_SOFASCORE_HOST,
    SofaScoreAPIError,
    fetch_fixture_odds as fetch_sofascore_fixture_odds,
    fetch_scheduled_matches as fetch_sofascore_scheduled_matches,
    find_scheduled_match as find_sofascore_scheduled_match,
    normalize_scheduled_result as normalize_sofascore_scheduled_result,
)
from streamlit_app.services.match_prediction_service import (
    fetch_event_details_by_players_date,
    fetch_odds_movements_by_event,
    fetch_prematch_odds_by_event,
    fetch_tournament_results,
    normalize_tournament_result,
    predict_upcoming_fixture,
)

# Backward-compatible local name retained by cache helpers.
CACHE_PATH = UPCOMING_MATCHES_CACHE_PATH
MATCH_PREDICTOR_PAGE = "pages/05_Match_Prediction_adapted.py"
IA_CANDIDATES_PAGE = "pages/07_IA_Candidates.py"
DEFAULT_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
DEFAULT_ODDS_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
ROUND_ALIASES = {
    "qualifying first round": "Q1", "qualifying round 1": "Q1", "q1": "Q1",
    "qualifying second round": "Q2", "qualifying round 2": "Q2", "q2": "Q2",
    "qualifying final": "Q3", "final qualifying": "Q3", "q3": "Q3",
    "round of 128": "R128", "r128": "R128", "round of 64": "R64", "r64": "R64",
    "round of 32": "R32", "r32": "R32", "round of 16": "R16", "r16": "R16",
    "quarterfinal": "QF", "quarter final": "QF", "qf": "QF",
    "semifinal": "SF", "semi final": "SF", "sf": "SF",
    "final": "F", "round robin": "RR", "rr": "RR",
}

PRODUCTION_ROOT = PRODUCTION_ARTIFACTS_DIR
MIN_CONTEXT_RELIABILITY_SAMPLE = 100
PREDICTION_CACHE_SCHEMA_VERSION = 7
SCORE_CACHE_SCHEMA_VERSION = 3
UPCOMING_MARKET_SIMULATIONS = 1_000
DEFAULT_OVER_LINE_BO3 = 22.5
DEFAULT_OVER_LINE_BO5 = 38.5
MAX_BATCH_SECONDS = 90.0



def provider_id_text(value: Any) -> str:
    """Normalize numeric provider IDs so 123, 123.0 and '123' share a key."""
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        return str(int(numeric))
    return str(value or "").strip()

def stable_prediction_key(fixture: pd.Series | dict[str, Any]) -> str:
    """Create one stable key for saving and rendering predictions."""
    get = fixture.get
    fixture_id = str(get("fixture_id", "") or "").strip()
    if fixture_id and fixture_id.casefold() not in {"nan", "none", "<na>"}:
        return f"fixture:{fixture_id}"
    values = (
        get("match_date", ""), get("scheduled_time", ""),
        get("tournament", ""), get("round", ""),
        get("player_1_name", ""), get("player_2_name", ""),
    )
    normalized = [re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip() for value in values]
    return "match:" + "|".join(normalized)


def prediction_error_row(key: str, fixture: pd.Series, status: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction_key": key,
        "player_1": str(fixture.get("player_1_name", "")),
        "player_2": str(fixture.get("player_2_name", "")),
        "tournament": str(fixture.get("tournament", "")),
        "reason": str(status.get("reason", "prediction_unavailable")),
        "error": str(status.get("error", "")),
        "player_1_id": status.get("player_1_id"),
        "player_2_id": status.get("player_2_id"),
    }


@st.cache_data(show_spinner=False, ttl=3600)
def load_reliability_profile(
    active_mtime_ns: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carga el perfil correspondiente al modelo activo y valida su contrato."""
    del active_mtime_ns

    if not ACTIVE_MODEL_PATH.exists():
        return pd.DataFrame(), {
            "available": False,
            "reason": "active_model_missing",
        }

    active = json.loads(
        ACTIVE_MODEL_PATH.read_text(
            encoding="utf-8-sig"
        )
    )
    artifact_directory = (
        PRODUCTION_ROOT
        / str(active["artifact_directory"])
    ).resolve()
    production_resolved = PRODUCTION_ROOT.resolve()
    if (
        artifact_directory != production_resolved
        and production_resolved
        not in artifact_directory.parents
    ):
        return pd.DataFrame(), {
            "available": False,
            "reason": "artifact_directory_outside_production",
        }

    profile_path = (
        artifact_directory
        / "reliability_profile.parquet"
    )
    if not profile_path.exists():
        return pd.DataFrame(), {
            "available": False,
            "reason": "reliability_profile_missing",
            "path": str(profile_path),
        }

    profile = pd.read_parquet(
        profile_path,
        engine="pyarrow",
    )
    required = {
        "model_version",
        "profile_scope",
        "surface",
        "competition_type",
        "probability_bin_left",
        "probability_bin_right",
        "sample_size",
        "mean_predicted_probability",
        "observed_win_rate",
        "calibration_gap",
        "confidence_lower",
        "confidence_upper",
        "reliability_label",
    }
    missing = sorted(required - set(profile.columns))
    if missing:
        return pd.DataFrame(), {
            "available": False,
            "reason": "reliability_profile_incomplete",
            "missing": missing,
        }

    active_version = str(
        active.get("active_model_version", "")
    )
    versions = set(
        profile["model_version"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    if versions != {active_version}:
        return pd.DataFrame(), {
            "available": False,
            "reason": "model_version_mismatch",
            "active_model_version": active_version,
            "profile_versions": sorted(versions),
        }

    for column in (
        "probability_bin_left",
        "probability_bin_right",
        "sample_size",
        "mean_predicted_probability",
        "observed_win_rate",
        "calibration_gap",
        "confidence_lower",
        "confidence_upper",
    ):
        profile[column] = pd.to_numeric(
            profile[column],
            errors="coerce",
        )

    return profile, {
        "available": True,
        "path": str(profile_path),
        "model_version": active_version,
        "champion": active.get("champion"),
        "probability_policy": active.get(
            "probability_policy"
        ),
    }


def favorite_probability_from_cache(
    prediction_key: Any,
    prediction_cache: dict[str, dict[str, Any]],
) -> float:
    item = prediction_cache.get(
        str(prediction_key),
        {},
    )
    try:
        player_1 = float(
            item.get("probability_player_1")
        )
        player_2 = float(
            item.get("probability_player_2")
        )
    except (TypeError, ValueError):
        return np.nan
    if not (
        np.isfinite(player_1)
        and np.isfinite(player_2)
    ):
        return np.nan
    return float(max(player_1, player_2))


def reliability_candidates(
    profile: pd.DataFrame,
    probability: float,
    surface: str,
    competition_type: str,
) -> list[tuple[str, pd.DataFrame, int]]:
    """Devuelve candidatos en orden de especificidad y muestra minima."""
    if profile.empty or not np.isfinite(probability):
        return []

    # El borde derecho es exclusivo salvo para la banda final, donde 1.0 entra.
    is_final_probability = bool(
        np.isclose(
            probability,
            1.0,
            rtol=0.0,
            atol=1e-12,
        )
    )

    lower_bound_matches = (
        profile[
            "probability_bin_left"
        ].le(
            probability
        )
    )

    upper_bound_matches = (
        profile[
            "probability_bin_right"
        ].gt(
            probability
        )
    )

    final_band_matches = (
        profile[
            "probability_bin_right"
        ].ge(
            1.0
        )
        & is_final_probability
    )

    band = profile.loc[
        lower_bound_matches
        & (
            upper_bound_matches
            | final_band_matches
        )
    ].copy()

    return [
        (
            "SURFACE_COMPETITION",
            band.loc[
                band["profile_scope"].eq(
                    "SURFACE_COMPETITION"
                )
                & band["surface"].astype(str).eq(
                    str(surface)
                )
                & band["competition_type"].astype(str).eq(
                    str(competition_type)
                )
            ],
            MIN_CONTEXT_RELIABILITY_SAMPLE,
        ),
        (
            "COMPETITION",
            band.loc[
                band["profile_scope"].eq(
                    "COMPETITION"
                )
                & band["competition_type"].astype(str).eq(
                    str(competition_type)
                )
            ],
            MIN_CONTEXT_RELIABILITY_SAMPLE,
        ),
        (
            "SURFACE",
            band.loc[
                band["profile_scope"].eq(
                    "SURFACE"
                )
                & band["surface"].astype(str).eq(
                    str(surface)
                )
            ],
            MIN_CONTEXT_RELIABILITY_SAMPLE,
        ),
        (
            "GLOBAL",
            band.loc[
                band["profile_scope"].eq(
                    "GLOBAL"
                )
            ],
            1,
        ),
    ]


def lookup_reliability(
    profile: pd.DataFrame,
    probability: float,
    surface: str,
    competition_type: str,
) -> dict[str, Any]:
    if profile.empty:
        return {
            "available": False,
            "reason": "profile_unavailable",
        }
    if not np.isfinite(probability):
        return {
            "available": False,
            "reason": "prediction_unavailable",
        }

    for scope, candidate, minimum_sample in reliability_candidates(
        profile,
        probability,
        surface,
        competition_type,
    ):
        if candidate.empty:
            continue
        row = candidate.sort_values(
            "sample_size",
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        sample_size = int(row["sample_size"])
        if sample_size < minimum_sample:
            continue
        return {
            "available": True,
            "profile_scope": scope,
            "probability_band": str(
                row["probability_band"]
            ),
            "sample_size": sample_size,
            "mean_predicted_probability": float(
                row["mean_predicted_probability"]
            ),
            "observed_win_rate": float(
                row["observed_win_rate"]
            ),
            "calibration_gap": float(
                row["calibration_gap"]
            ),
            "confidence_lower": float(
                row["confidence_lower"]
            ),
            "confidence_upper": float(
                row["confidence_upper"]
            ),
            "reliability_label": str(
                row["reliability_label"]
            ),
            "surface": str(surface),
            "competition_type": str(
                competition_type
            ),
        }

    return {
        "available": False,
        "reason": "no_eligible_profile",
    }


def reliability_display(value: Any) -> str:
    if not isinstance(value, dict) or not value.get(
        "available"
    ):
        return "-"
    return (
        f"{value['reliability_label']} · "
        f"n={value['sample_size']:,}"
    )


st.set_page_config(page_title="Upcoming Matches", page_icon="📅", layout="wide")
st.title("📅 Upcoming Matches")
st.caption("Partidos ATP, qualifying y Challenger de hoy y mañana, listos para enviar a Match Prediction.")


def is_itf_tournament(
    tournament: str,
    rank: str = "",
    event_type: str = "",
) -> bool:
    """Detecta torneos ITF y Futures que no deben entrar en el modelo."""

    text = normalize_text(
        f"{tournament} {rank} {event_type}"
    )

    explicit_itf_terms = (
        "itf",
        "futures",
        "world tennis tour",
        "mens world tennis tour",
        "women world tennis tour",
    )

    if any(
        term in text
        for term in explicit_itf_terms
    ):
        return True

    # Torneos masculinos ITF actuales.
    if re.search(
        r"\bm\s*15\b",
        text,
    ):
        return True

    if re.search(
        r"\bm\s*25\b",
        text,
    ):
        return True

    # Variantes sin espacio por si la normalización cambia.
    if re.search(
        r"\bm15\b|\bm25\b",
        text,
    ):
        return True

    # Torneos femeninos ITF, por compatibilidad futura.
    if re.search(
        r"\bw\s*15\b"
        r"|\bw\s*25\b"
        r"|\bw\s*35\b"
        r"|\bw\s*50\b"
        r"|\bw\s*75\b"
        r"|\bw\s*100\b",
        text,
    ):
        return True

    return False


def is_supported_fixture(
    row: pd.Series,
) -> bool:
    if row.get(
        "competition_type"
    ) not in {
        "ATP",
        "ATP_QUALIFYING",
        "CHALLENGER",
    }:
        return False

    tournament_text = normalize_text(row.get("tournament", ""))
    if "challenger" not in tournament_text and is_itf_tournament(
        tournament=str(row.get("tournament", "")),
        rank=str(row.get("provider_rank", "")),
        event_type=str(row.get("provider_event_type", "")),
    ):
        return False

    return True



def secret_or_env(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def first_value(value: Any, paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        current = value
        found = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found and current not in (None, ""):
            return current
    return default


def objects_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "result", "results", "fixtures", "matches", "items"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
        if isinstance(candidate, dict):
            nested = objects_from_payload(candidate)
            if nested:
                return nested
    return []


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def contains_doubles_separator(
    player_name: Any,
) -> bool:
    """Detecta equipos de dobles representados en un único nombre."""

    text = str(
        player_name or ""
    ).strip()

    return bool(
        re.search(
            r"\s*/\s*"
            r"|\s+&\s+"
            r"|\s+\+\s+",
            text,
        )
    )


def is_doubles_fixture(
    row: pd.Series,
) -> bool:
    """Determina si una fila corresponde a un partido de dobles.

    Se utilizan tanto los campos explícitos del proveedor como
    patrones presentes en los nombres de los participantes.
    """

    player_group = normalize_text(
        row.get(
            "player_group",
            "",
        )
    )

    event_type = normalize_text(
        row.get(
            "provider_event_type",
            "",
        )
    )

    tournament = normalize_text(
        row.get(
            "tournament",
            "",
        )
    )

    if any(
        term in player_group
        for term in (
            "double",
            "doubles",
            "team",
        )
    ):
        return True

    if any(
        term in event_type
        for term in (
            "double",
            "doubles",
        )
    ):
        return True

    if any(
        term in tournament
        for term in (
            "double",
            "doubles",
        )
    ):
        return True

    if contains_doubles_separator(
        row.get(
            "player_1_name",
            "",
        )
    ):
        return True

    if contains_doubles_separator(
        row.get(
            "player_2_name",
            "",
        )
    ):
        return True

    return False

def normalize_surface(
    value: Any,
) -> str:
    """Normaliza la superficie recibida desde el proveedor."""

    text = normalize_text(
        value
    )

    if any(
        term in text
        for term in (
            "clay",
            "red clay",
            "green clay",
            "tierra",
        )
    ):
        return "Clay"

    if any(
        term in text
        for term in (
            "grass",
            "cesped",
            "lawn",
        )
    ):
        return "Grass"

    if "carpet" in text:
        return "Carpet"

    if any(
        term in text
        for term in (
            "hard",
            "hardcourt",
            "hard court",
            "acrylic",
            "indoor hard",
            "outdoor hard",
        )
    ):
        return "Hard"

    # El modelo solo admite las superficies declaradas.
    # Hard es el fallback conservador usado por la página.
    return "Hard"


def normalize_round(
    value: Any,
    tournament: str = "",
    tournament_level: str = "",
) -> str:
    """Normaliza la ronda considerando la estructura del torneo.

    Los valores genéricos First, Second, Third y Fourth dependen
    del tamaño del cuadro. En un Grand Slam no representan las
    mismas rondas que en un Challenger.
    """

    text = normalize_text(value)

    tournament_text = normalize_text(
        tournament
    )

    level = str(
        tournament_level
        or ""
    ).strip().upper()

    exact_mapping = {
        "qualifying first round": "Q1",
        "qualifying round 1": "Q1",
        "first qualifying round": "Q1",
        "q1": "Q1",

        "qualifying second round": "Q2",
        "qualifying round 2": "Q2",
        "second qualifying round": "Q2",
        "q2": "Q2",

        "qualifying final": "Q3",
        "final qualifying": "Q3",
        "final qualifying round": "Q3",
        "q3": "Q3",

        "round of 128": "R128",
        "round 128": "R128",
        "r128": "R128",

        "round of 64": "R64",
        "round 64": "R64",
        "r64": "R64",

        "round of 32": "R32",
        "round 32": "R32",
        "r32": "R32",

        "round of 16": "R16",
        "round 16": "R16",
        "r16": "R16",

        "quarterfinal": "QF",
        "quarter final": "QF",
        "quarter finals": "QF",
        "quarterfinals": "QF",
        "qf": "QF",

        "semifinal": "SF",
        "semi final": "SF",
        "semi finals": "SF",
        "semifinals": "SF",
        "sf": "SF",

        "final": "F",

        "round robin": "RR",
        "group stage": "RR",
        "rr": "RR",
    }

    if text in exact_mapping:
        return exact_mapping[text]

    for source, target in exact_mapping.items():
        if source in text:
            return target

    grand_slam_names = (
        "australian open",
        "roland garros",
        "french open",
        "wimbledon",
        "us open",
        "u s open",
        "united states open",
    )

    is_grand_slam = (
        level == "G"
        or any(
            name in tournament_text
            for name in grand_slam_names
        )
    )

    if is_grand_slam:
        grand_slam_mapping = {
            "first": "R128",
            "first round": "R128",
            "second": "R64",
            "second round": "R64",
            "third": "R32",
            "third round": "R32",
            "fourth": "R16",
            "fourth round": "R16",
        }

        if text in grand_slam_mapping:
            return grand_slam_mapping[text]

    standard_mapping = {
        "first": "R32",
        "first round": "R32",
        "second": "R16",
        "second round": "R16",
        "third": "QF",
        "third round": "QF",
        "fourth": "SF",
        "fourth round": "SF",
    }

    return standard_mapping.get(
        text,
        "R32",
    )


def classify_competition(
    tournament: str,
    rank: str,
    round_code: str,
    event_type: str = "",
    rank_id: Any = None,
) -> str:
    """Clasifica ATP, qualifying, Challenger e ITF.

    El proveedor agrupa Challenger e ITF > $10K bajo un mismo
    provider_rank. Por eso, la clasificación debe priorizar el
    nombre específico del torneo sobre la categoría genérica.
    """

    tournament_text = normalize_text(
        tournament
    )

    rank_text = normalize_text(
        rank
    )

    event_type_text = normalize_text(
        event_type
    )

    combined_text = normalize_text(
        f"{tournament} {rank} {event_type}"
    )

    numeric_rank_id = pd.to_numeric(
        rank_id,
        errors="coerce",
    )

    # ------------------------------------------------------
    # 1. ITF INEQUÍVOCO POR NOMBRE
    # ------------------------------------------------------
    # Torneos masculinos World Tennis Tour:
    # M15, M25 y variantes como M25+H.
    if re.search(
        r"(^|\s)m\s*(15|25)(\s|$)",
        tournament_text,
    ):
        return "ITF"

    if re.search(
        r"(^|\s)m\s*(15|25)\s*h(\s|$)",
        tournament_text,
    ):
        return "ITF"

    # Compatibilidad futura con torneos femeninos.
    if re.search(
        r"(^|\s)w\s*"
        r"(15|25|35|50|75|100)"
        r"(\s|$)",
        tournament_text,
    ):
        return "ITF"

    if any(
        term in tournament_text
        for term in (
            "futures",
            "world tennis tour",
            "mens world tennis tour",
            "women world tennis tour",
        )
    ):
        return "ITF"

    # ------------------------------------------------------
    # 2. CHALLENGER POR NOMBRE DEL TORNEO
    # ------------------------------------------------------
    # Esta comprobación debe venir antes de buscar "itf" en
    # provider_rank, porque el proveedor utiliza:
    # "Challengers/ITF tournaments > $10K".
    if "challenger" in tournament_text:
        return "CHALLENGER"

    if re.search(
        r"\bch\s*(50|75|100|125|175)\b",
        tournament_text,
    ):
        return "CHALLENGER"

    if re.search(
        r"\bchallenger\s*"
        r"(50|75|100|125|175)\b",
        combined_text,
    ):
        return "CHALLENGER"

    if any(
        term in combined_text
        for term in (
            "atp challenger",
            "challenger tour",
        )
    ):
        return "CHALLENGER"

    # ------------------------------------------------------
    # 3. ITF POR CATEGORÍA DEL PROVEEDOR
    # ------------------------------------------------------
    # rank_id=0 identifica normalmente ITF de nivel inferior.
    if (
        pd.notna(numeric_rank_id)
        and int(numeric_rank_id) == 0
    ):
        return "ITF"

    # La palabra ITF solo se utiliza después de descartar
    # que el nombre específico sea Challenger.
    if (
        "itf" in rank_text
        or "itf" in event_type_text
    ):
        return "ITF"

    # ------------------------------------------------------
    # 4. ATP QUALIFYING
    # ------------------------------------------------------
    if (
        str(round_code).upper()
        in {
            "Q1",
            "Q2",
            "Q3",
        }
        or "qualifying" in combined_text
        or "qualification" in combined_text
    ):
        return "ATP_QUALIFYING"

    # ------------------------------------------------------
    # 5. ATP MAIN DRAW
    # ------------------------------------------------------
    return "ATP"


def infer_tourney_level(
    tournament: str,
    rank: str,
    competition: str,
) -> str:
    text = normalize_text(
        f"{tournament} {rank}"
    )

    if competition == "CHALLENGER":
            return "C"
    
    
    if competition == "ITF":
        return "ITF"

    

    if (
        "grand slam" in text
        or any(
            name in text
            for name in (
                "australian open",
                "roland garros",
                "french open",
                "wimbledon",
                "us open",
                "u s open",
                "united states open",
            )
        )
    ):
        return "G"

    if (
        "masters 1000" in text
        or "atp 1000" in text
    ):
        return "M"

    if "finals" in text:
        return "F"

    if (
        "davis cup" in text
        or "united cup" in text
    ):
        return "D"

    if competition in {
        "ATP",
        "ATP_QUALIFYING",
    }:
        return "A"

    return "UNKNOWN"


def infer_best_of(level: str, round_code: str, competition: str) -> int:
    return 5 if level == "G" and competition == "ATP" and not round_code.startswith("Q") else 3


def fixture_to_row(item: dict[str, Any], requested_date: date) -> dict[str, Any] | None:
    
    order_of_play = first_value(
        item,
        (
            "orderOfPlay",
            "order_of_play",
            "courtOrder",
            "court_order",
            "matchNumber",
            "match_number",
            "order",
        ),
        None,
    )
    
    
    
    player_1 = first_value(item, (
        "player1.name", "player_1.name", "players.0.name", "home.name",
        "firstPlayer.name", "playerOne.name", "player1", "player_1",
    ))
    player_2 = first_value(item, (
        "player2.name", "player_2.name", "players.1.name", "away.name",
        "secondPlayer.name", "playerTwo.name", "player2", "player_2",
    ))
    if not player_1 or not player_2 or isinstance(player_1, dict) or isinstance(player_2, dict):
        return None
    tournament = str(first_value(item, (
        "tournament.name", "event.name", "competition.name", "tournamentName", "tournament",
    ), "Unknown tournament"))
    rank = str(
        first_value(
            item,
            (
                "tournament.rank.name",
                "tournament.rank.title",
                "tournament.rank.description",
                "rank.name",
                "rank.title",
                "category.name",
                "category.title",
                "level",
            ),
            "",
        )
    )
    rank_id = first_value(
        item,
        (
            "tournament.rank.id",
            "tournament.rank.rankId",
            "tournament.rankId",
            "rank.id",
            "rankId",
            "category.id",
        ),
        None,
    )

    event_type = str(
        first_value(
            item,
            (
                "eventType.name",
                "event_type.name",
                "eventType",
                "event_type",
                "tourType",
                "tour_type",
                "category.name",
                "category.title",
            ),
            "",
        )
    )
    
    player_group = str(
        first_value(
            item,
            (
                "playerGroup",
                "player_group",
                "group",
                "matchType",
                "match_type",
                "type",
                "eventType.playerGroup",
                "event_type.player_group",
            ),
            "",
        )
    )
    round_name = str(
        first_value(
            item,
            (
                "round.name",
                "round.title",
                "roundName",
                "round",
            ),
            "",
        )
    )

    # Primera clasificación provisional. Sirve para determinar
    # el nivel antes de interpretar First, Second, Third o Fourth.
    preliminary_round_code = (
        normalize_round(
            round_name,
            tournament=tournament,
        )
    )

    competition = classify_competition(
        tournament=tournament,
        rank=rank,
        round_code=preliminary_round_code,
        event_type=event_type,
        rank_id=rank_id,
    )

    level = infer_tourney_level(
        tournament=tournament,
        rank=rank,
        competition=competition,
    )

    # Normalización definitiva con el nivel ya conocido.
    round_code = normalize_round(
        round_name,
        tournament=tournament,
        tournament_level=level,
    )
    surface = normalize_surface(first_value(item, (
        "tournament.court.name", "tournament.court.surface", "court.name", "surface", "court",
    ), "Hard"))
    start = first_value(item, ("startTime", "start_time", "scheduledAt", "scheduled_at", "date", "event_date"))
    start_ts = pd.to_datetime(start, errors="coerce", utc=True)
    if pd.isna(start_ts):
        start_ts = pd.Timestamp(requested_date)
    else:
        start_ts = start_ts.tz_convert("Europe/Madrid").tz_localize(None)
    return {
        "provider_player_1_id": first_value(
            item,
            ("player1.id", "player_1.id", "players.0.id", "home.id", "firstPlayer.id"),
            None,
        ),
        "provider_player_2_id": first_value(
            item,
            ("player2.id", "player_2.id", "players.1.id", "away.id", "secondPlayer.id"),
            None,
        ),
        "provider_round_id": first_value(
            item,
            ("round.id", "roundId", "round_id"),
            None,
        ),
        "fixture_id": str(
            first_value(
                item,
                (
                    "id",
                    "fixture_id",
                    "match_id",
                    "key",
                ),
                "",
            )
        ),
        "match_date": start_ts.date(),
        "scheduled_time": (
            start_ts.strftime("%H:%M")
            if start
            else "TBD"
        ),
        "tournament": tournament,
        "tournament_id": str(
            first_value(
                item,
                (
                    "tournament.id",
                    "event.id",
                    "competition.id",
                    "tournamentId",
                ),
                tournament,
            )
        ),
        "competition_type": competition,
        "order_of_play": pd.to_numeric(
            order_of_play,
            errors="coerce",
        ),
        "provider_rank": rank,
        "provider_rank_id": rank_id,
        "provider_event_type": event_type,
        "tourney_level": level,
        "round": round_code,
        "round_source": round_name,
        "surface": surface,
        "player_group": player_group,
        "indoor": bool(
            "indoor"
            in normalize_text(
                first_value(
                    item,
                    (
                        "tournament.court.name",
                        "court.name",
                        "environment",
                    ),
                    "",
                )
            )
        ),
        "best_of": infer_best_of(
            level,
            round_code,
            competition,
        ),
        "player_1_name": str(
            player_1
        ).strip(),
        "player_2_name": str(
            player_2
        ).strip(),
        "provider": "tennis-api.com",
    }


@st.cache_data(show_spinner=False, ttl=900)
def fetch_fixtures_for_date(date_text: str, api_key: str, host: str) -> pd.DataFrame:
    requested_date = date.fromisoformat(date_text)
    rows: list[dict[str, Any]] = []
    page = 1
    while page <= 20:
        url = f"https://{host}/tennis/v2/atp/fixtures/{date_text}"
        response = requests.get(
            url,
            headers={"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": host},
            params={
                "include": "round,tournament,tournament.court,tournament.rank,tournament.country",
                "filter": "PlayerGroup:singles",
                "pageSize": 100,
                "pageNo": page,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        objects = objects_from_payload(payload)
        for item in objects:
            row = fixture_to_row(item, requested_date)
            if row:
                rows.append(row)
        has_next = bool(first_value(payload, ("meta.hasNextPage", "hasNextPage", "pagination.hasNextPage"), False))
        if not has_next or not objects:
            break
        page += 1
    return pd.DataFrame(rows).drop_duplicates(
        subset=["match_date", "tournament", "player_1_name", "player_2_name"], keep="last"
    ) if rows else pd.DataFrame()


def fetch_window(
    start_date: date,
    days: int,
    api_key: str,
    host: str,
) -> pd.DataFrame:
    frames = [
        fetch_fixtures_for_date(
            str(
                start_date
                + timedelta(
                    days=offset
                )
            ),
            api_key,
            host,
        )
        for offset in range(
            days
        )
    ]

    frames = [
        frame
        for frame in frames
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    all_fixtures = pd.concat(
        frames,
        ignore_index=True,
    )
    
    all_fixtures["is_doubles"] = (
        all_fixtures.apply(
            is_doubles_fixture,
            axis=1,
        )
    )

    # Conservar temporalmente el conjunto completo para
    # poder auditar cómo clasifica el proveedor.
    audit_path = UPCOMING_CLASSIFICATION_AUDIT_PATH

    all_fixtures.to_parquet(
        audit_path,
        index=False,
    )

    allowed_competitions = {
        "ATP",
        "ATP_QUALIFYING",
        "CHALLENGER",
    }

    result = all_fixtures.loc[
        all_fixtures[
            "competition_type"
        ].isin(
            allowed_competitions
        )
        & ~all_fixtures[
            "is_doubles"
        ]
    ].copy()
    
    

    result["scheduled_sort"] = (
        pd.to_datetime(
            result.get(
                "scheduled_datetime"
            ),
            errors="coerce",
        )
    )

    result = (
        result.sort_values(
            [
                "match_date",
                "tournament",
                "scheduled_sort",
                "player_1_name",
                "player_2_name",
            ],
            ascending=[
                True,
                True,
                True,
                True,
                True,
            ],
            kind="mergesort",
            na_position="last",
        )
        .drop(
            columns=[
                "scheduled_sort",
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    return result


def save_cache(frame: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(fetched_at_utc=pd.Timestamp.now(tz="UTC").isoformat()).to_parquet(CACHE_PATH, index=False)


def load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame()

    result = pd.read_parquet(
        CACHE_PATH
    )

    if result.empty:
        return result

    result["is_doubles"] = (
        result.apply(
            is_doubles_fixture,
            axis=1,
        )
    )

    allowed_competitions = {
        "ATP",
        "ATP_QUALIFYING",
        "CHALLENGER",
    }

    result = result.loc[
        result[
            "competition_type"
        ].isin(
            allowed_competitions
        )
        & ~result[
            "is_doubles"
        ]
    ].copy()

    return result.reset_index(
        drop=True
    )


PLAYER_NAME_ALIASES = {
    "pedro martinez portero": "pedro martinez",
}


def canonical_player_lookup_key(value: Any) -> str:
    normalized = normalize_text(value)
    return normalize_text(PLAYER_NAME_ALIASES.get(normalized, normalized))


def player_catalog() -> pd.DataFrame:
    from streamlit_app.data_access.players_adapted import get_players
    players = get_players().copy()
    players["normalized_name"] = players["player_name"].map(normalize_text)
    return players.drop_duplicates("normalized_name")


def resolve_player_id(name: str, players: pd.DataFrame) -> int | None:
    match = players.loc[
        players["normalized_name"].eq(canonical_player_lookup_key(name))
    ]
    return None if match.empty else int(match.iloc[0]["player_id"])



reliability_profile, reliability_metadata = load_reliability_profile(
    ACTIVE_MODEL_PATH.stat().st_mtime_ns
    if ACTIVE_MODEL_PATH.exists()
    else 0
)

if not reliability_metadata.get("available"):
    st.warning(
        "Historical reliability is unavailable: "
        f"{reliability_metadata.get('reason', 'unknown')}"
    )

api_key = secret_or_env("TENNIS_API_KEY")
host = secret_or_env("TENNIS_API_HOST", DEFAULT_HOST) or DEFAULT_HOST
odds_host = secret_or_env("TENNIS_ODDS_API_HOST", DEFAULT_ODDS_HOST) or DEFAULT_ODDS_HOST
sofascore_api_key = secret_or_env("SOFASCORE_API_KEY")
sofascore_host = (
    secret_or_env("SOFASCORE_API_HOST", DEFAULT_SOFASCORE_HOST)
    or DEFAULT_SOFASCORE_HOST
)

with st.container(border=True):
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        start_date = st.date_input("From", value=date.today())
    with c2:
        horizon = st.selectbox("Calendar", [2, 3, 7], index=0, format_func=lambda value: f"{value} days")
    with c3:
        refresh = st.button("Refresh fixtures", type="primary", use_container_width=True)
    with c4:
        load_scores = st.button("Load scores", use_container_width=True)
    with c5:
        use_cache = st.button("Use local cache", use_container_width=True)

if "upcoming_matches" not in st.session_state:
    st.session_state.upcoming_matches = load_cache()

if refresh:
    if not api_key:
        st.error("TENNIS_API_KEY is not configured in .streamlit/secrets.toml or the environment.")
    else:
        try:
            with st.spinner("Downloading ATP fixtures..."):
                fixtures = fetch_window(start_date, horizon, api_key, host)
            st.session_state.upcoming_matches = fixtures
            if not fixtures.empty:
                save_cache(fixtures)
        except requests.RequestException as error:
            st.error(f"Fixtures API error: {error}")

if use_cache:
    st.session_state.upcoming_matches = load_cache()
if st.session_state.get("upcoming_score_cache_schema") != SCORE_CACHE_SCHEMA_VERSION:
    st.session_state["upcoming_score_cache"] = {}
    st.session_state["upcoming_loaded_score_tournaments"] = set()
    st.session_state["upcoming_score_request_audit"] = []
    st.session_state["upcoming_score_cache_schema"] = SCORE_CACHE_SCHEMA_VERSION
if "upcoming_score_cache" not in st.session_state:
    st.session_state["upcoming_score_cache"] = {}
if "upcoming_odds_movement_cache" not in st.session_state:
    st.session_state["upcoming_odds_movement_cache"] = {}
if "upcoming_loaded_score_tournaments" not in st.session_state:
    st.session_state["upcoming_loaded_score_tournaments"] = set()
if "upcoming_score_request_audit" not in st.session_state:
    st.session_state["upcoming_score_request_audit"] = []
if "upcoming_sofascore_scheduled_cache" not in st.session_state:
    st.session_state["upcoming_sofascore_scheduled_cache"] = {}
score_cache: dict[str, dict[str, Any]] = st.session_state["upcoming_score_cache"]
sofascore_scheduled_cache: dict[str, dict[str, Any]] = st.session_state[
    "upcoming_sofascore_scheduled_cache"
]
loaded_score_tournaments: set[str] = set(
    st.session_state["upcoming_loaded_score_tournaments"]
)
movement_cache: dict[str, dict[str, Any]] = st.session_state["upcoming_odds_movement_cache"]
fixtures = st.session_state.upcoming_matches.copy()

def get_sofascore_scheduled(date_value: Any) -> dict[str, Any]:
    """Return one SofaScore scheduled payload per date and Streamlit session."""
    date_key = pd.to_datetime(date_value, errors="raise").date().isoformat()
    if date_key not in sofascore_scheduled_cache:
        sofascore_scheduled_cache[date_key] = fetch_sofascore_scheduled_matches(
            date_key,
            api_key=sofascore_api_key or "",
            host=sofascore_host,
        )
        st.session_state["upcoming_sofascore_scheduled_cache"] = dict(
            sofascore_scheduled_cache
        )
    return sofascore_scheduled_cache[date_key]


def score_name_keys(value: Any) -> set[str]:
    """Conservative name variants used only for provider result reconciliation."""
    normalized = normalize_text(value)
    if not normalized:
        return set()
    tokens = normalized.split()
    keys = {normalized}
    if len(tokens) >= 2:
        keys.add(f"{tokens[0]} {tokens[-1]}")
        keys.add(tokens[-1])
    alias = PLAYER_NAME_ALIASES.get(normalized)
    if alias:
        keys.add(normalize_text(alias))
    return {key for key in keys if key}


def result_matches_fixture(result: dict[str, Any], fixture: pd.Series) -> bool:
    result_1 = score_name_keys(result.get("player_1_name"))
    result_2 = score_name_keys(result.get("player_2_name"))
    fixture_1 = score_name_keys(fixture.get("player_1_name"))
    fixture_2 = score_name_keys(fixture.get("player_2_name"))
    direct = bool(result_1 & fixture_1) and bool(result_2 & fixture_2)
    reversed_order = bool(result_1 & fixture_2) and bool(result_2 & fixture_1)
    return direct or reversed_order


def score_cache_key(row: pd.Series | dict[str, Any]) -> str:
    return f"fixture:{stable_prediction_key(row)}"


def cache_normalized_result_for_fixtures(
    normalized: dict[str, Any],
    tournament_fixtures: pd.DataFrame,
) -> int:
    """Attach one provider result directly to every matching visible fixture."""
    if not normalized.get("player_1_name") or not normalized.get("player_2_name"):
        return 0
    matches = 0
    for _, fixture in tournament_fixtures.iterrows():
        if not result_matches_fixture(normalized, fixture):
            continue
        score_cache[score_cache_key(fixture)] = normalized
        matches += 1
    return matches


def load_sofascore_scores_for_tournament(
    season_id: str, tournament_fixtures: pd.DataFrame,
) -> tuple[int, set[str]]:
    """Load finished scores from cached SofaScore scheduled responses."""
    if not sofascore_api_key:
        raise SofaScoreAPIError("SOFASCORE_API_KEY is not configured.")
    count = 0
    used_dates: set[str] = set()
    for _, fixture in tournament_fixtures.iterrows():
        date_key = pd.to_datetime(fixture["match_date"], errors="raise").date().isoformat()
        scheduled = get_sofascore_scheduled(date_key)
        used_dates.add(date_key)
        resolved = find_sofascore_scheduled_match(
            scheduled, player_1_name=str(fixture["player_1_name"]),
            player_2_name=str(fixture["player_2_name"]),
            tournament=str(fixture["tournament"]),
        )
        if not resolved.get("available"):
            continue
        normalized = normalize_sofascore_scheduled_result(
            resolved, player_1_name=str(fixture["player_1_name"]),
            player_2_name=str(fixture["player_2_name"]),
        )
        if normalized.get("status_type") != "finished":
            continue
        names = sorted((normalize_text(normalized.get("player_1_name")),
                        normalize_text(normalized.get("player_2_name"))))
        if names[0] and names[1]:
            score_cache[f"{season_id}|{'|'.join(names)}"] = normalized
            score_cache[score_cache_key(fixture)] = normalized
            count += 1
    return count, used_dates

if load_scores:
    if not api_key:
        st.session_state["upcoming_api_notice"] = {
            "level": "error", "message": "TENNIS_API_KEY is not configured."
        }
    elif fixtures.empty:
        st.session_state["upcoming_api_notice"] = {
            "level": "warning", "message": "Load fixtures before loading scores."
        }
    else:
        result_errors: list[str] = []
        loaded_results = 0
        matched_fixtures = 0
        quota_exhausted = False
        tennis_api_disabled_for_click = False
        tennis_api_requests_issued = 0
        sofascore_dates_requested: set[str] = set()
        today_value = pd.Timestamp.today().date()
        eligible_fixtures = fixtures.loc[
            pd.to_datetime(fixtures["match_date"], errors="coerce").dt.date.le(today_value)
        ].copy()
        # Build exactly one request row per provider tournament/season id.
        # Do not deduplicate by the display name because the same season may be
        # represented by more than one tournament label in the fixture payload.
        groups = eligible_fixtures[["tournament_id", "tournament"]].copy()
        groups["season_id"] = groups["tournament_id"].map(provider_id_text)
        groups = (
            groups.loc[groups["season_id"].ne("")]
            .sort_values(["season_id", "tournament"], kind="mergesort")
            .drop_duplicates(subset=["season_id"], keep="first")
            .reset_index(drop=True)
        )
        pending_groups = groups.loc[
            ~groups["season_id"].isin(loaded_score_tournaments)
        ].copy()
        if pending_groups.empty:
            st.session_state["upcoming_api_notice"] = {
                "level": "info",
                "message": "All eligible tournament scores are already cached for this session.",
            }
        else:
            progress = st.progress(0.0, text="Loading tournament scores...")
            requested_this_click: set[str] = set()
            request_audit: list[dict[str, Any]] = list(
                st.session_state["upcoming_score_request_audit"]
            )
            for position, (_, tournament_row) in enumerate(pending_groups.iterrows(), start=1):
                season_id = str(tournament_row["season_id"])
                # Defensive guard: even if upstream data becomes duplicated,
                # never issue a second HTTP request for the same season id.
                if season_id in requested_this_click:
                    continue
                requested_this_click.add(season_id)
                request_started = pd.Timestamp.now(tz="UTC").isoformat()
                fallback_rows = eligible_fixtures.loc[
                    eligible_fixtures["tournament_id"].map(provider_id_text).eq(season_id)
                ].copy()
                if tennis_api_disabled_for_click:
                    try:
                        fallback_count, fallback_dates = load_sofascore_scores_for_tournament(
                            season_id, fallback_rows
                        )
                        sofascore_dates_requested.update(fallback_dates)
                        loaded_score_tournaments.add(season_id)
                        loaded_results += fallback_count
                        request_audit.append({
                            "season_id": season_id, "tournament": str(tournament_row["tournament"]),
                            "requested_at_utc": request_started,
                            "status": "SOFASCORE_FALLBACK_AFTER_429",
                            "result_records": fallback_count,
                            "fallback_dates": sorted(fallback_dates),
                            "tennis_api_request_issued": False,
                        })
                    except Exception as fallback_error:
                        result_errors.append(
                            f"{tournament_row['tournament']}: SofaScore fallback failed: "
                            f"{type(fallback_error).__name__}: {fallback_error}"
                        )
                    continue
                try:
                    tennis_api_requests_issued += 1
                    raw_results = fetch_tournament_results(
                        season_id=season_id, api_key=api_key, host=host, tour_type="atp"
                    )
                    tournament_visible_matches = 0
                    normalized_record_count = 0
                    for raw_result in raw_results:
                        normalized = normalize_tournament_result(raw_result)
                        names = sorted((
                            normalize_text(normalized.get("player_1_name")),
                            normalize_text(normalized.get("player_2_name")),
                        ))
                        if not names[0] or not names[1]:
                            continue
                        normalized_record_count += 1
                        score_cache[f"{season_id}|{'|'.join(names)}"] = normalized
                        tournament_visible_matches += cache_normalized_result_for_fixtures(
                            normalized,
                            fallback_rows,
                        )

                    if normalized_record_count == 0 and sofascore_api_key:
                        fallback_count, fallback_dates = load_sofascore_scores_for_tournament(
                            season_id,
                            fallback_rows,
                        )
                        sofascore_dates_requested.update(fallback_dates)
                        loaded_results += fallback_count
                        if fallback_count > 0:
                            loaded_score_tournaments.add(season_id)
                        request_audit.append({
                            "season_id": season_id,
                            "tournament": str(tournament_row["tournament"]),
                            "requested_at_utc": request_started,
                            "status": "SOFASCORE_FALLBACK_AFTER_EMPTY_200",
                            "result_records": int(len(raw_results)),
                            "normalized_records": 0,
                            "fallback_records": fallback_count,
                            "fallback_dates": sorted(fallback_dates),
                            "tennis_api_request_issued": True,
                        })
                    else:
                        if normalized_record_count > 0:
                            loaded_score_tournaments.add(season_id)
                        request_audit.append({
                            "season_id": season_id,
                            "tournament": str(tournament_row["tournament"]),
                            "requested_at_utc": request_started,
                            "status": (
                                "SUCCESS" if normalized_record_count
                                else "EMPTY_OR_UNPARSEABLE_RESULTS"
                            ),
                            "result_records": int(len(raw_results)),
                            "normalized_records": normalized_record_count,
                            "visible_matches": tournament_visible_matches,
                        })
                        loaded_results += normalized_record_count
                except (PermissionError, requests.HTTPError) as error:
                    status_code = getattr(getattr(error, "response", None), "status_code", None)
                    is_forbidden = isinstance(error, PermissionError) or status_code == 403
                    is_rate_limited = status_code == 429
                    if (is_forbidden or is_rate_limited) and sofascore_api_key:
                        if is_rate_limited:
                            quota_exhausted = True
                            tennis_api_disabled_for_click = True
                        try:
                            fallback_count, fallback_dates = load_sofascore_scores_for_tournament(
                                season_id, fallback_rows
                            )
                            sofascore_dates_requested.update(fallback_dates)
                            loaded_score_tournaments.add(season_id)
                            loaded_results += fallback_count
                            request_audit.append({
                                "season_id": season_id,
                                "tournament": str(tournament_row["tournament"]),
                                "requested_at_utc": request_started,
                                "status": ("SOFASCORE_FALLBACK_AFTER_429" if is_rate_limited
                                           else "SOFASCORE_FALLBACK_AFTER_403"),
                                "result_records": fallback_count,
                                "fallback_dates": sorted(fallback_dates),
                                "tennis_api_request_issued": True,
                            })
                        except Exception as fallback_error:
                            result_errors.append(
                                f"{tournament_row['tournament']}: Tennis API HTTP {status_code or 403} "
                                f"and SofaScore fallback failed: {type(fallback_error).__name__}: "
                                f"{fallback_error}"
                            )
                        continue
                    request_audit.append({
                        "season_id": season_id, "tournament": str(tournament_row["tournament"]),
                        "requested_at_utc": request_started,
                        "status": f"HTTP_{status_code or 'FORBIDDEN'}", "result_records": 0,
                        "tennis_api_request_issued": True,
                    })
                    if is_rate_limited:
                        quota_exhausted = True
                        result_errors.append(
                            "Tennis API quota reached and SOFASCORE_API_KEY is missing."
                        )
                        break
                    result_errors.append(
                        f"{tournament_row['tournament']}: HTTP {status_code or '?'}"
                    )
                except Exception as error:
                    request_audit.append({
                        "season_id": season_id,
                        "tournament": str(tournament_row["tournament"]),
                        "requested_at_utc": request_started,
                        "status": type(error).__name__,
                        "result_records": 0,
                    })
                    result_errors.append(
                        f"{tournament_row['tournament']}: {type(error).__name__}: {error}"
                    )
                progress.progress(
                    position / max(len(pending_groups), 1),
                    text=f"Tournament {position}/{len(pending_groups)} · {tournament_row['tournament']}",
                )
                # Basic allows 4 requests/second. Stay comfortably below it.
                time.sleep(0.55)
            progress.empty()
            for _, fixture_row in fixtures.iterrows():
                season_id = provider_id_text(fixture_row.get("tournament_id"))
                names = sorted((
                    normalize_text(fixture_row.get("player_1_name")),
                    normalize_text(fixture_row.get("player_2_name")),
                ))
                direct_key = score_cache_key(fixture_row)
                legacy_key = f"{season_id}|{'|'.join(names)}"
                matched_fixtures += int(
                    direct_key in score_cache or legacy_key in score_cache
                )
            st.session_state["upcoming_score_cache"] = dict(score_cache)
            st.session_state["upcoming_loaded_score_tournaments"] = set(
                loaded_score_tournaments
            )
            st.session_state["upcoming_score_request_audit"] = request_audit[-200:]
            message = (
                "Score loading completed. This action loads statuses and final scores, "
                "not bookmaker odds. "
                f"{tennis_api_requests_issued} Tennis API request(s) "
                f"issued for {len(requested_this_click)} unique tournament(s); "
                f"{len(sofascore_dates_requested)} SofaScore date payload(s) used; "
                f"{len(loaded_score_tournaments)} tournament(s) cached, "
                f"{loaded_results} new result record(s), "
                f"{matched_fixtures} visible fixture match(es)."
            )
            if quota_exhausted and sofascore_api_key:
                message += (
                    " Tennis API returned HTTP 429. SofaScore fallback was activated for "
                    "the affected and remaining tournaments; no further Tennis API requests "
                    "were issued during this click."
                )
            elif quota_exhausted:
                message += (
                    " Tennis API returned HTTP 429 and SOFASCORE_API_KEY is not configured. "
                    "Further requests were stopped."
                )
            elif result_errors:
                message += " Errors: " + "; ".join(result_errors[:3])
            current_audit = request_audit[-max(len(requested_this_click), 1):]
            empty_result_tournaments = sum(
                item.get("status") in {
                    "EMPTY_OR_UNPARSEABLE_RESULTS",
                    "SOFASCORE_FALLBACK_AFTER_EMPTY_200",
                }
                for item in current_audit
            )
            if empty_result_tournaments:
                message += (
                    f" Tennis API returned an empty/unusable HTTP 200 response for "
                    f"{empty_result_tournaments} tournament(s)."
                )
                if sofascore_dates_requested:
                    message += " SofaScore scheduled fallback was attempted by date."
                elif not sofascore_api_key:
                    message += " SOFASCORE_API_KEY is not configured, so fallback was unavailable."
            if loaded_results > 0 and matched_fixtures == 0:
                message += (
                    " Results were returned but none matched the visible fixtures. "
                    "Check provider names and tournament IDs in the score request audit."
                )
            st.session_state["upcoming_api_notice"] = {
                "level": "warning" if result_errors or matched_fixtures == 0 else "success",
                "message": message,
            }
    st.rerun()

notice = st.session_state.pop("upcoming_api_notice", None)
if isinstance(notice, dict):
    level = str(notice.get("level", "info"))
    getattr(st, level if level in {"success", "warning", "error", "info"} else "info")(
        str(notice.get("message", ""))
    )

if fixtures.empty:
    st.info("No fixtures loaded. Configure the API key and select Refresh fixtures.")
    with st.expander("API configuration"):
        st.code('TENNIS_API_KEY = "YOUR_RAPIDAPI_KEY"\nTENNIS_API_HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"', language="toml")
    st.stop()

filter_1, filter_2, filter_3 = st.columns(3)
with filter_1:
    competition_filter = st.multiselect(
        "Competition", ["ATP", "ATP_QUALIFYING", "CHALLENGER"],
        default=["ATP", "ATP_QUALIFYING", "CHALLENGER"],
    )
with filter_2:
    tournament_options = sorted(fixtures["tournament"].dropna().unique().tolist())
    tournament_filter = st.multiselect("Tournament", tournament_options)
with filter_3:
    date_options = sorted(fixtures["match_date"].dropna().unique().tolist())
    date_filter = st.multiselect("Date", date_options, default=date_options)
batch_filter_1, batch_filter_2 = st.columns([1, 2])
with batch_filter_1:
    calculation_batch_size = st.selectbox(
        "Matches per calculation batch",
        options=[5, 10, 20, 40],
        index=1,
        help=(
            "Controls how many unresolved matches are calculated in each run. "
            "All filtered matches remain visible in the daily tables."
        ),
    )
with batch_filter_2:
    st.caption(
        f"Upcoming Matches uses {UPCOMING_MARKET_SIMULATIONS:,} Monte Carlo "
        f"simulations per fixture and returns control after about "
        f"{MAX_BATCH_SECONDS:.0f} seconds. All fixtures remain visible while "
        "probabilities are calculated progressively."
    )
visible = fixtures.loc[
    fixtures["competition_type"].isin(competition_filter)
    & fixtures["match_date"].isin(date_filter)
].copy()
if tournament_filter:
    visible = visible.loc[visible["tournament"].isin(tournament_filter)]

st.metric("Available singles fixtures", len(visible))
if visible.empty:
    st.warning("No matches satisfy the current filters.")
    st.stop()

# Safety filter. It also protects against old local cache files.
visible = visible.loc[
    ~visible.apply(is_doubles_fixture, axis=1)
].copy()

visible["fixture_selection_key"] = (
    visible["match_date"].astype(str)
    + "|"
    + visible["tournament"].astype(str)
    + "|"
    + visible["player_1_name"].astype(str)
    + "|"
    + visible["player_2_name"].astype(str)
)

fixture_ids = visible.get(
    "fixture_id",
    pd.Series("", index=visible.index, dtype="string"),
).astype("string").fillna("").str.strip()
visible["prediction_key"] = visible.apply(stable_prediction_key, axis=1).astype("string")

# Use the real scheduled timestamp when present; otherwise combine date/time.
if "scheduled_datetime" in visible.columns:
    visible["scheduled_sort"] = pd.to_datetime(
        visible["scheduled_datetime"], errors="coerce"
    )
else:
    visible["scheduled_sort"] = pd.to_datetime(
        visible["match_date"].astype(str)
        + " "
        + visible["scheduled_time"].fillna("TBD").replace(
            {"TBD": "23:59", "": "23:59"}
        ),
        errors="coerce",
    )

if "order_of_play" not in visible.columns:
    visible["order_of_play"] = np.nan
visible["order_of_play"] = pd.to_numeric(
    visible["order_of_play"], errors="coerce"
)

visible = visible.sort_values(
    [
        "match_date",
        "tournament",
        "order_of_play",
        "scheduled_sort",
        "player_1_name",
        "player_2_name",
    ],
    kind="mergesort",
    na_position="last",
).reset_index(drop=True)

if st.session_state.get("upcoming_prediction_cache_schema") != PREDICTION_CACHE_SCHEMA_VERSION:
    st.session_state["upcoming_model_predictions"] = {}
    st.session_state["upcoming_prediction_status"] = {}
    st.session_state["upcoming_historical_evidence"] = {}
    st.session_state["upcoming_prediction_cache_schema"] = PREDICTION_CACHE_SCHEMA_VERSION
if "upcoming_model_predictions" not in st.session_state:
    st.session_state["upcoming_model_predictions"] = {}
if "upcoming_prediction_status" not in st.session_state:
    st.session_state["upcoming_prediction_status"] = {}
if "upcoming_historical_evidence" not in st.session_state:
    st.session_state["upcoming_historical_evidence"] = {}
prediction_cache: dict[str, dict[str, Any]] = st.session_state["upcoming_model_predictions"]
prediction_status: dict[str, dict[str, Any]] = st.session_state["upcoming_prediction_status"]
historical_evidence_cache: dict[str, dict[str, Any]] = st.session_state[
    "upcoming_historical_evidence"
]
visible_keys = set(visible["prediction_key"].astype(str))
calculated_count = sum(key in prediction_cache for key in visible_keys)
failed_count = sum(
    key in prediction_status
    and not bool(prediction_status[key].get("available"))
    for key in visible_keys
)
resolved_keys = set(prediction_cache) | set(prediction_status)
pending_keys = [
    key for key in visible["prediction_key"].astype(str)
    if key not in resolved_keys
]
pending_count = len(pending_keys)

control_1, control_2, control_3 = st.columns([1.5, 1, 2.5])
with control_1:
    calculate_probabilities = st.button(
        "Calculate next batch" if resolved_keys else "Calculate model probabilities",
        type="primary",
        use_container_width=True,
        disabled=visible.empty or pending_count == 0,
    )
with control_2:
    clear_probabilities = st.button(
        "Clear probabilities",
        use_container_width=True,
        disabled=not prediction_cache and not prediction_status,
    )
with control_3:
    st.caption(
        f"Calculated: {calculated_count}/{len(visible)} · "
        f"Failed/unavailable: {failed_count} · Pending: {pending_count}"
    )
    diagnostic_items = [
        item.get("feature_diagnostics", {})
        for item in prediction_cache.values()
        if isinstance(item, dict)
    ]
    approximate_count = sum(
        int(item.get("missing_numeric_count", 0) > 0)
        for item in diagnostic_items
        if isinstance(item, dict)
    )
    if approximate_count:
        st.caption(
            f"{approximate_count} prediction(s) use model-native missing-value "
            "handling for unavailable historical state features."
        )
if clear_probabilities:
    st.session_state["upcoming_model_predictions"] = {}
    st.session_state["upcoming_prediction_status"] = {}
    st.session_state["upcoming_historical_evidence"] = {}
    st.session_state.pop("upcoming_last_batch", None)
    st.session_state.pop("upcoming_open_model_details", None)
    st.session_state["upcoming_prediction_cache_schema"] = PREDICTION_CACHE_SCHEMA_VERSION
    st.rerun()

if calculate_probabilities:
    players_for_prediction = player_catalog()
    local_player_lookup = {
        str(row["normalized_name"]): int(row["player_id"])
        for _, row in players_for_prediction.iterrows()
    }

    resolved_keys = set(prediction_cache) | set(prediction_status)
    unresolved = visible.loc[
        ~visible["prediction_key"].astype(str).isin(resolved_keys)
    ].head(int(calculation_batch_size)).copy()

    total_matches = len(unresolved)
    batch_started_at = time.perf_counter()
    processed_in_batch = 0
    stopped_by_time_limit = False

    progress = st.progress(
        0.0,
        text=f"Calculating up to {total_matches} unresolved matches...",
    )

    for position, (_, fixture) in enumerate(unresolved.iterrows(), start=1):
        elapsed_seconds = time.perf_counter() - batch_started_at
        if position > 1 and elapsed_seconds >= MAX_BATCH_SECONDS:
            stopped_by_time_limit = True
            break

        prediction_key = str(fixture["prediction_key"])
        player_1_id = local_player_lookup.get(
            canonical_player_lookup_key(fixture["player_1_name"])
        )
        player_2_id = local_player_lookup.get(
            canonical_player_lookup_key(fixture["player_2_name"])
        )

        if player_1_id is None or player_2_id is None:
            prediction_status[prediction_key] = {
                "available": False,
                "reason": "player_not_matched",
                "player_1_id": player_1_id,
                "player_2_id": player_2_id,
                "player_1_name": str(fixture["player_1_name"]),
                "player_2_name": str(fixture["player_2_name"]),
                "tournament": str(fixture["tournament"]),
            }
        else:
            try:
                result = predict_upcoming_fixture(
                    player_1_id=player_1_id,
                    player_2_id=player_2_id,
                    surface=str(fixture["surface"]),
                    tournament_level=str(fixture["tourney_level"]),
                    round_name=str(fixture["round"]),
                    best_of=int(fixture["best_of"]),
                    competition_type=str(fixture["competition_type"]),
                    indoor=bool(fixture["indoor"]),
                    prediction_date=fixture["match_date"],
                    tournament_id=str(
                        fixture.get("tournament_id", fixture["tournament"])
                    ),
                    tournament_name=str(fixture["tournament"]),
                    over_games_line=float(
                        DEFAULT_OVER_LINE_BO5
                        if int(fixture["best_of"]) == 5
                        else DEFAULT_OVER_LINE_BO3
                    ),
                    market_simulations=UPCOMING_MARKET_SIMULATIONS,
                )
                if not isinstance(result, dict):
                    raise TypeError(
                        f"Prediction service returned {type(result).__name__}, expected dict"
                    )

                prediction_status[prediction_key] = {
                    **result,
                    "player_1_id": player_1_id,
                    "player_2_id": player_2_id,
                    "player_1_name": str(fixture["player_1_name"]),
                    "player_2_name": str(fixture["player_2_name"]),
                    "tournament": str(fixture["tournament"]),
                }

                if result.get("available"):
                    probability_1 = pd.to_numeric(
                        result.get("probability_player_1"), errors="coerce"
                    )
                    probability_2 = pd.to_numeric(
                        result.get("probability_player_2"), errors="coerce"
                    )
                    if pd.isna(probability_1) or pd.isna(probability_2):
                        raise ValueError(
                            "Available prediction has missing winner probabilities"
                        )

                    model_market_probabilities = result.get(
                        "market_probabilities", {}
                    )
                    prediction_cache[prediction_key] = {
                        "probability_player_1": float(probability_1),
                        "probability_player_2": float(probability_2),
                        "model_version": result.get("model_version"),
                        "champion": result.get("champion"),
                        "probability_policy": result.get("probability_policy"),
                        "feature_diagnostics": result.get(
                            "feature_diagnostics", {}
                        ),
                        "market_probabilities": model_market_probabilities,
                        "historical_market_evidence": None,
                        "predicted_at_utc": pd.Timestamp.now(
                            tz="UTC"
                        ).isoformat(),
                    }
            except Exception as error:
                prediction_cache.pop(prediction_key, None)
                prediction_status[prediction_key] = {
                    "available": False,
                    "reason": "prediction_error",
                    "error": f"{type(error).__name__}: {error}",
                    "player_1_id": player_1_id,
                    "player_2_id": player_2_id,
                    "player_1_name": str(fixture["player_1_name"]),
                    "player_2_name": str(fixture["player_2_name"]),
                    "tournament": str(fixture["tournament"]),
                }

        processed_in_batch += 1

        # Persist after every fixture so a later failure never loses progress.
        st.session_state["upcoming_model_predictions"] = dict(prediction_cache)
        st.session_state["upcoming_prediction_status"] = dict(prediction_status)
        st.session_state[
            "upcoming_prediction_cache_schema"
        ] = PREDICTION_CACHE_SCHEMA_VERSION

        progress.progress(
            processed_in_batch / max(total_matches, 1),
            text=(
                f"Processed {processed_in_batch}/{total_matches} in this batch · "
                f"{time.perf_counter() - batch_started_at:.1f}s elapsed"
            ),
        )

    progress.empty()
    st.session_state["upcoming_model_predictions"] = dict(prediction_cache)
    st.session_state["upcoming_prediction_status"] = dict(prediction_status)
    st.session_state[
        "upcoming_prediction_cache_schema"
    ] = PREDICTION_CACHE_SCHEMA_VERSION
    st.session_state["upcoming_last_batch"] = {
        "processed": processed_in_batch,
        "requested": total_matches,
        "stopped_by_time_limit": stopped_by_time_limit,
        "elapsed_seconds": float(time.perf_counter() - batch_started_at),
    }
    st.rerun()

last_batch = st.session_state.get("upcoming_last_batch")
if isinstance(last_batch, dict) and last_batch.get("processed"):
    message = (
        f"Last batch processed {last_batch['processed']} match(es) in "
        f"{last_batch.get('elapsed_seconds', 0.0):.1f}s."
    )
    if last_batch.get("stopped_by_time_limit"):
        message += " The batch stopped at the time limit; run the next batch to continue."
    st.info(message)

visible_prediction_errors = []
for _, fixture in visible.iterrows():
    key = str(fixture["prediction_key"])
    status = prediction_status.get(key, {})
    if status and not status.get("available"):
        visible_prediction_errors.append(prediction_error_row(key, fixture, status))
if visible_prediction_errors:
    with st.expander(
        f"Prediction errors and unavailable matches ({len(visible_prediction_errors)})",
        expanded=(not prediction_cache),
    ):
        st.caption(
            "These matches did not produce a model result. The reason explains why "
            "their probability columns remain empty."
        )
        st.dataframe(pd.DataFrame(visible_prediction_errors), use_container_width=True, hide_index=True)


if "upcoming_odds_cache" not in st.session_state:
    st.session_state["upcoming_odds_cache"] = {}
odds_cache: dict[str, dict[str, Any]] = st.session_state["upcoming_odds_cache"]
def cached_probability(prediction_key: Any, side: int) -> float:
    item = prediction_cache.get(str(prediction_key), {})
    value = item.get(f"probability_player_{side}")
    try:
        probability = float(value)
        return probability * 100.0 if np.isfinite(probability) else np.nan
    except (TypeError, ValueError):
        return np.nan


def cached_fair_odds(prediction_key: Any, side: int) -> float:
    """Cuota decimal justa del modelo, sin margen: 1 / probability."""
    item = prediction_cache.get(str(prediction_key), {})
    value = item.get(f"probability_player_{side}")
    try:
        probability = float(value)
        if not np.isfinite(probability) or probability <= 0.0:
            return np.nan
        return 1.0 / probability
    except (TypeError, ValueError):
        return np.nan


visible["model_probability_player_1"] = visible["prediction_key"].map(
    lambda key: cached_probability(key, 1)
)
visible["model_fair_odds_player_1"] = visible["prediction_key"].map(
    lambda key: cached_fair_odds(key, 1)
)
visible["model_probability_player_2"] = visible["prediction_key"].map(
    lambda key: cached_probability(key, 2)
)

visible["model_fair_odds_player_2"] = visible["prediction_key"].map(
    lambda key: cached_fair_odds(key, 2)
)

def cached_market_probability(prediction_key: Any, field: str) -> float:
    item = prediction_cache.get(str(prediction_key), {})
    market = item.get("market_probabilities", {}) if isinstance(item, dict) else {}
    value = pd.to_numeric(market.get(field), errors="coerce")
    return float(value * 100.0) if pd.notna(value) and np.isfinite(value) else np.nan

visible["player_1_first_set_probability"] = visible["prediction_key"].map(
    lambda key: cached_market_probability(key, "player_1_first_set_probability")
)
visible["player_2_first_set_probability"] = visible["prediction_key"].map(
    lambda key: cached_market_probability(key, "player_2_first_set_probability")
)
visible["player_1_win_set_probability"] = visible["prediction_key"].map(
    lambda key: cached_market_probability(key, "player_1_win_any_set_probability")
)
visible["player_2_win_set_probability"] = visible["prediction_key"].map(
    lambda key: cached_market_probability(key, "player_2_win_any_set_probability")
)
visible["over_games_probability"] = visible["prediction_key"].map(lambda key: cached_market_probability(key, "over_games_probability"))
visible["under_games_probability"] = visible["prediction_key"].map(lambda key: cached_market_probability(key, "under_games_probability"))

def cached_over_probability(prediction_key: Any, line: float) -> float:
    item = prediction_cache.get(str(prediction_key), {})
    market = item.get("market_probabilities", {}) if isinstance(item, dict) else {}
    probabilities = market.get("over_probabilities", {}) if isinstance(market, dict) else {}
    if not isinstance(probabilities, dict):
        return np.nan
    value = pd.to_numeric(probabilities.get(f"{float(line):.1f}"), errors="coerce")
    return float(value * 100.0) if pd.notna(value) and np.isfinite(value) else np.nan

OVER_LINES = (18.5, 19.5, 20.5, 21.5, 22.5)
for over_line in OVER_LINES:
    visible[f"over_{str(over_line).replace('.', '_')}_probability"] = visible["prediction_key"].map(
        lambda key, line=over_line: cached_over_probability(key, line)
    )
visible["favorite_model_probability"] = visible["prediction_key"].map(
    lambda key: favorite_probability_from_cache(
        key,
        prediction_cache,
    )
)
visible["reliability_details"] = visible.apply(
    lambda row: lookup_reliability(
        profile=reliability_profile,
        probability=float(row["favorite_model_probability"])
        if pd.notna(row["favorite_model_probability"])
        else np.nan,
        surface=str(row["surface"]),
        competition_type=str(row["competition_type"]),
    ),
    axis=1,
)
visible["model_reliability"] = visible["reliability_details"].map(
    reliability_display
)

def _score_for_fixture(row: pd.Series) -> dict[str, Any]:
    direct = score_cache.get(score_cache_key(row), {})
    if isinstance(direct, dict) and direct:
        return direct
    season_id = provider_id_text(row.get("tournament_id", ""))
    names = sorted((
        normalize_text(row.get("player_1_name")),
        normalize_text(row.get("player_2_name")),
    ))
    legacy = score_cache.get(f"{season_id}|{'|'.join(names)}", {})
    return legacy if isinstance(legacy, dict) else {}

visible["score_details"] = visible.apply(_score_for_fixture, axis=1)
visible["match_status"] = visible["score_details"].map(lambda item: str(item.get("status", "Upcoming")))
visible["match_score"] = visible["score_details"].map(lambda item: str(item.get("score", "")) or "-")

def _cached_odds(key: Any, side: int) -> float:
    return pd.to_numeric(odds_cache.get(str(key), {}).get(f"odds_player_{side}"), errors="coerce")

def _no_vig_probability(key: Any, side: int) -> float:
    item = odds_cache.get(str(key), {})
    o1 = pd.to_numeric(item.get("odds_player_1"), errors="coerce")
    o2 = pd.to_numeric(item.get("odds_player_2"), errors="coerce")
    if pd.isna(o1) or pd.isna(o2) or o1 <= 1 or o2 <= 1:
        return np.nan
    raw1, raw2 = 1.0 / float(o1), 1.0 / float(o2)
    total = raw1 + raw2
    return (raw1 if side == 1 else raw2) / total if total > 0 else np.nan

visible["market_odds_player_1"] = visible["prediction_key"].map(lambda key: _cached_odds(key, 1))
visible["market_odds_player_2"] = visible["prediction_key"].map(lambda key: _cached_odds(key, 2))
visible["edge_player_1"] = visible.apply(
    lambda row: row["model_probability_player_1"] / 100.0 - _no_vig_probability(row["prediction_key"], 1)
    if pd.notna(row["model_probability_player_1"]) else np.nan, axis=1
)
visible["edge_player_2"] = visible.apply(
    lambda row: row["model_probability_player_2"] / 100.0 - _no_vig_probability(row["prediction_key"], 2)
    if pd.notna(row["model_probability_player_2"]) else np.nan, axis=1
)

# ---------------------------------------------------------------------------
# On-demand bookmaker odds
# ---------------------------------------------------------------------------
def _walk_json(value: Any):
    """Yield every dictionary contained in a JSON-like payload."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _decimal_odd(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number > 1.0 else np.nan


def _extract_price(item: dict[str, Any]) -> float:
    for key in ("odds", "odd", "price", "decimal", "decimalOdds", "value"):
        if key in item:
            price = _decimal_odd(item.get(key))
            if np.isfinite(price):
                return price
    return np.nan


def _extract_label(item: dict[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "name", "label", "selection", "outcome", "participant", "player",
        "runner", "title", "shortName", "short_name", "side",
    ):
        value = item.get(key)
        if isinstance(value, dict):
            value = first_value(value, ("name", "label", "title"), "")
        if value not in (None, ""):
            values.append(str(value))
    return " ".join(values)


def _extract_bookmaker(item: dict[str, Any]) -> str:
    for key in ("bookmakerName", "bookmaker_name", "bookmaker", "provider", "source"):
        value = item.get(key)
        if isinstance(value, dict):
            value = first_value(value, ("name", "title"), "")
        if value not in (None, ""):
            return str(value)
    return ""


def parse_match_winner_odds(
    payload: Any,
    player_1_name: str,
    player_2_name: str,
) -> dict[str, Any]:
    """Extract the best decimal match-winner price for each player.

    The provider has changed response nesting between endpoint versions. The
    parser therefore walks all outcome objects and matches player labels, while
    also accepting conventional Player 1 / Player 2 and Home / Away labels.
    """
    normalized_1 = normalize_text(player_1_name)
    normalized_2 = normalize_text(player_2_name)
    candidates_1: list[tuple[float, str]] = []
    candidates_2: list[tuple[float, str]] = []

    aliases_1 = {"1", "p1", "player 1", "player1", "home", "participant 1"}
    aliases_2 = {"2", "p2", "player 2", "player2", "away", "participant 2"}

    for item in _walk_json(payload):
        price = _extract_price(item)
        if not np.isfinite(price):
            continue
        label = normalize_text(_extract_label(item))
        if not label:
            continue
        bookmaker = _extract_bookmaker(item)
        if normalized_1 and (normalized_1 in label or label in normalized_1 or label in aliases_1):
            candidates_1.append((float(price), bookmaker))
        elif normalized_2 and (normalized_2 in label or label in normalized_2 or label in aliases_2):
            candidates_2.append((float(price), bookmaker))

    best_1 = max(candidates_1, default=(np.nan, ""), key=lambda value: value[0])
    best_2 = max(candidates_2, default=(np.nan, ""), key=lambda value: value[0])
    return {
        "available": bool(np.isfinite(best_1[0]) or np.isfinite(best_2[0])),
        "odds_player_1": best_1[0],
        "odds_player_2": best_2[0],
        "bookmaker_player_1": best_1[1],
        "bookmaker_player_2": best_2[1],
    }


def _safe_provider_result(
    value: Any,
    *,
    reason: str,
    source: str,
) -> dict[str, Any]:
    """Guarantee a dictionary contract for every provider integration."""
    if isinstance(value, dict):
        return value
    return {
        "available": False,
        "reason": reason,
        "source": source,
        "returned_type": type(value).__name__,
    }


def _resolve_event(
    fixture: pd.Series,
    api_key_value: str,
    host_value: str,
) -> dict[str, Any]:
    result = fetch_event_details_by_players_date(
        player_1_name=str(fixture["player_1_name"]),
        player_2_name=str(fixture["player_2_name"]),
        match_date=fixture["match_date"],
        api_key=api_key_value,
        host=host_value,
    )
    return _safe_provider_result(
        result,
        reason="tennis_event_resolution_returned_none",
        source="tennis_api",
    )


def fetch_fixture_odds(
    fixture: pd.Series,
    api_key_value: str,
    host_value: str,
) -> dict[str, Any]:
    """Use Tennis API first and SofaScore after HTTP 403 or HTTP 429."""
    try:
        event = _resolve_event(fixture, api_key_value, host_value)
        if not event.get("available"):
            return event
        tennis_result = fetch_prematch_odds_by_event(
            event_id=str(event["event_id"]),
            player_1_name=str(fixture["player_1_name"]),
            player_2_name=str(fixture["player_2_name"]),
            api_key=api_key_value,
            host=host_value,
        )
        return _safe_provider_result(
            tennis_result,
            reason="tennis_odds_returned_none",
            source="tennis_api",
        )
    except (PermissionError, requests.HTTPError) as tennis_error:
        status_code = getattr(
            getattr(tennis_error, "response", None), "status_code", None
        )
        is_forbidden = isinstance(tennis_error, PermissionError) or status_code == 403
        is_rate_limited = status_code == 429
        if not (is_forbidden or is_rate_limited):
            raise
        if not sofascore_api_key:
            return {
                "available": False,
                "reason": "tennis_api_fallback_key_missing",
                "status_code": status_code,
                "error": str(tennis_error),
                "source": "tennis_api",
            }
        scheduled = _safe_provider_result(
            get_sofascore_scheduled(fixture["match_date"]),
            reason="sofascore_scheduled_returned_none",
            source="sofascore_scraper",
        )
        if not scheduled.get("available", True):
            return scheduled
        result = fetch_sofascore_fixture_odds(
            player_1_name=str(fixture["player_1_name"]),
            player_2_name=str(fixture["player_2_name"]),
            match_date=fixture["match_date"],
            tournament=str(fixture["tournament"]),
            api_key=sofascore_api_key,
            host=sofascore_host,
            scheduled_payload=scheduled,
        )
        result = _safe_provider_result(
            result,
            reason="sofascore_odds_returned_none",
            source="sofascore_scraper",
        )
        return {
            **result,
            "fallback_trigger": (
                "tennis_api_http_429" if is_rate_limited
                else "tennis_api_http_403"
            ),
            "tennis_api_error": str(tennis_error),
        }
    except Exception as error:
        return {
            "available": False,
            "reason": "odds_provider_error",
            "error": f"{type(error).__name__}: {error}",
            "source": "odds_integration",
        }


def _sofascore_movements(
    fixture: pd.Series,
    odds_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "available": bool(odds_result.get("available")),
        "movements": [
            {
                "Player": fixture["player_1_name"],
                "Opening odds": odds_result.get("opening_odds_player_1"),
                "Current odds": odds_result.get("odds_player_1"),
                "Direction": odds_result.get("movement_player_1"),
            },
            {
                "Player": fixture["player_2_name"],
                "Opening odds": odds_result.get("opening_odds_player_2"),
                "Current odds": odds_result.get("odds_player_2"),
                "Direction": odds_result.get("movement_player_2"),
            },
        ],
        "source": "sofascore_scraper_cached_odds",
        "reason": odds_result.get("reason"),
    }


def fetch_fixture_movements(
    fixture: pd.Series,
    api_key_value: str,
    host_value: str,
) -> dict[str, Any]:
    key = str(fixture["prediction_key"])
    cached = _safe_provider_result(
        odds_cache.get(key, {}),
        reason="cached_odds_returned_none",
        source="upcoming_odds_cache",
    )
    if cached.get("source") == "sofascore_scraper_odds_fallback":
        return _sofascore_movements(fixture, cached)
    try:
        event = _resolve_event(fixture, api_key_value, host_value)
        if not event.get("available"):
            return event
        tennis_result = fetch_odds_movements_by_event(
            event_id=str(event["event_id"]),
            api_key=api_key_value,
            host=host_value,
        )
        return _safe_provider_result(
            tennis_result,
            reason="tennis_movements_returned_none",
            source="tennis_api",
        )
    except (PermissionError, requests.HTTPError) as tennis_error:
        status_code = getattr(
            getattr(tennis_error, "response", None), "status_code", None
        )
        is_forbidden = isinstance(tennis_error, PermissionError) or status_code == 403
        is_rate_limited = status_code == 429
        if not (is_forbidden or is_rate_limited):
            raise
        if not sofascore_api_key:
            return {
                "available": False,
                "reason": "tennis_api_fallback_key_missing",
                "status_code": status_code,
                "error": str(tennis_error),
                "source": "tennis_api",
            }
        scheduled = _safe_provider_result(
            get_sofascore_scheduled(fixture["match_date"]),
            reason="sofascore_scheduled_returned_none",
            source="sofascore_scraper",
        )
        if not scheduled.get("available", True):
            return scheduled
        fallback = _safe_provider_result(
            fetch_sofascore_fixture_odds(
                player_1_name=str(fixture["player_1_name"]),
                player_2_name=str(fixture["player_2_name"]),
                match_date=fixture["match_date"],
                tournament=str(fixture["tournament"]),
                api_key=sofascore_api_key,
                host=sofascore_host,
                scheduled_payload=scheduled,
            ),
            reason="sofascore_odds_returned_none",
            source="sofascore_scraper",
        )
        odds_cache[key] = fallback
        st.session_state["upcoming_odds_cache"] = dict(odds_cache)
        result = _sofascore_movements(fixture, fallback)
        result["source"] = "sofascore_scraper_odds_fallback"
        result["fallback_trigger"] = (
            "tennis_api_http_429" if is_rate_limited
            else "tennis_api_http_403"
        )
        return result
    except Exception as error:
        return {
            "available": False,
            "reason": "movement_provider_error",
            "error": f"{type(error).__name__}: {error}",
            "source": "odds_integration",
        }

def odds_display(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.2f}" if np.isfinite(number) else "-"


def prediction_display(value: Any, percentage: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    if not percentage:
        return f"{number:.2f}"
    if number >= 99.95:
        return ">99.9%"
    if number <= 0.05:
        return "<0.1%"
    return f"{number:.1f}%"


players_for_actions = player_catalog()
local_player_lookup = {
    str(row["normalized_name"]): int(row["player_id"])
    for _, row in players_for_actions.iterrows()
}


def make_match_selection(fixture: pd.Series) -> tuple[dict[str, Any] | None, str | None]:
    player_1_id = local_player_lookup.get(canonical_player_lookup_key(fixture["player_1_name"]))
    player_2_id = local_player_lookup.get(canonical_player_lookup_key(fixture["player_2_name"]))
    if player_1_id is None or player_2_id is None:
        return None, "One or both provider names could not be matched to the local player catalog."

    prediction_item = prediction_cache.get(str(fixture["prediction_key"]), {})
    odds_item = odds_cache.get(str(fixture["prediction_key"]), {})
    selection = {
        "player_1_id": player_1_id,
        "player_2_id": player_2_id,
        "player_1_name": fixture["player_1_name"],
        "player_2_name": fixture["player_2_name"],
        "surface": fixture["surface"],
        "tournament_level": fixture["tourney_level"],
        "round": fixture["round"],
        "best_of": int(fixture["best_of"]),
        "competition_type": fixture["competition_type"],
        "indoor": bool(fixture["indoor"]),
        "tournament": fixture["tournament"],
        "tournament_id": str(fixture.get("tournament_id", fixture["tournament"])),
        "tournament_history_name": fixture["tournament"],
        "tournament_history_id": str(fixture.get("tournament_id", fixture["tournament"])),
        "tournament_history_years": 10,
        "match_date": str(fixture["match_date"]),
        "scheduled_time": fixture["scheduled_time"],
        "model_probability_player_1": prediction_item.get("probability_player_1"),
        "model_probability_player_2": prediction_item.get("probability_player_2"),
        "model_fair_odds_player_1": cached_fair_odds(fixture["prediction_key"], 1),
        "model_fair_odds_player_2": cached_fair_odds(fixture["prediction_key"], 2),
        "market_odds_player_1": odds_item.get("odds_player_1"),
        "market_odds_player_2": odds_item.get("odds_player_2"),
        "market_bookmaker_player_1": odds_item.get("bookmaker_player_1"),
        "market_bookmaker_player_2": odds_item.get("bookmaker_player_2"),
        "model_version": prediction_item.get("model_version"),
        "champion": prediction_item.get("champion"),
        "probability_policy": prediction_item.get("probability_policy"),
        "feature_diagnostics": prediction_item.get("feature_diagnostics", {}),
        "market_probabilities": prediction_item.get("market_probabilities", {}),
        "historical_market_evidence": historical_evidence_cache.get(
            str(fixture["prediction_key"]),
            {},
        ),
        "reliability": fixture.get("reliability_details")
        if isinstance(fixture.get("reliability_details"), dict)
        else None,
    }
    return selection, None


# ---------------------------------------------------------------------------
# Day and tournament sections with interactive sortable match rows
# ---------------------------------------------------------------------------
def _match_finished(status: Any) -> bool:
    normalized = normalize_text(status)
    return any(token in normalized for token in ("ended", "finished", "completed", "final"))


def _score_total_games(score: Any) -> float:
    text = str(score or "").upper().strip()
    if not text or any(token in text for token in ("W/O", "WALKOVER")):
        return np.nan
    total = 0
    parsed = 0
    for token in text.split():
        match = re.match(r"^(\d+)-(\d+)(?:\([^)]*\))?", token)
        if match:
            total += int(match.group(1)) + int(match.group(2))
            parsed += 1
    return float(total) if parsed else np.nan


def _winner_side(fixture: pd.Series) -> int | None:
    details = fixture.get("score_details", {})
    if not isinstance(details, dict):
        return None
    winner = normalize_text(details.get("winner_name", ""))
    p1 = normalize_text(fixture.get("player_1_name", ""))
    p2 = normalize_text(fixture.get("player_2_name", ""))
    if winner and winner == p1:
        return 1
    if winner and winner == p2:
        return 2
    return None


def _green_if(value: str, favorable: bool) -> str:
    if not favorable:
        return value
    return f'<span style="color:#14833b;font-weight:700">{value}</span>'


def _write_html(container: Any, value: str) -> None:
    container.markdown(value, unsafe_allow_html=True)


def _paired_value(first: str, second: str, favorable: bool = False) -> str:
    value = f"P1 {first}<br>P2 {second}"
    return _green_if(value, favorable)


sort_options = {
    "Time": "scheduled_sort",
    "Status": "match_status",
    "Round": "round",
    "Player 1": "player_1_name",
    "Player 2": "player_2_name",
    "P1 % Win": "model_probability_player_1",
    "P2 % Win": "model_probability_player_2",
    "P1 Fair": "model_fair_odds_player_1",
    "P2 Fair": "model_fair_odds_player_2",
    "P1 Odds": "market_odds_player_1",
    "P2 Odds": "market_odds_player_2",
    "P1 Edge": "edge_player_1",
    "P2 Edge": "edge_player_2",
    "O18.5": "over_18_5_probability",
    "O19.5": "over_19_5_probability",
    "O20.5": "over_20_5_probability",
    "O21.5": "over_21_5_probability",
    "O22.5": "over_22_5_probability",
}

for day_position, (match_date, date_frame) in enumerate(
    visible.groupby("match_date", sort=True)
):
    date_frame = date_frame.reset_index(drop=True).copy()
    formatted_date = pd.to_datetime(match_date, errors="coerce")
    date_label = (
        formatted_date.strftime("%A, %d %B %Y")
        if pd.notna(formatted_date)
        else str(match_date)
    )
    calculated_for_day = int(date_frame["model_probability_player_1"].notna().sum())
    st.subheader(
        f"📅 {date_label} · {len(date_frame)} matches · "
        f"{calculated_for_day} calculated"
    )

    for tournament_position, (
        tournament_name,
        tournament_frame,
    ) in enumerate(
        date_frame.groupby(
            "tournament",
            sort=True,
        )
    ):
        tournament_frame = (
            tournament_frame
            .reset_index(drop=True)
            .copy()
        )

        surface_values = sorted(
            tournament_frame["surface"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        surface_label = (
            " / ".join(surface_values)
            if surface_values
            else "Unknown surface"
        )

        ended_count = int(
            tournament_frame[
                "match_status"
            ]
            .map(_match_finished)
            .sum()
        )

        # Debe definirse antes de utilizarse en cualquier widget.
        safe_tournament = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            str(tournament_name),
        ).strip("_")

        # Protección para nombres que solo contengan símbolos.
        if not safe_tournament:
            safe_tournament = "unknown_tournament"

        with st.expander(
            (
                f"🎾 {tournament_name} · "
                f"{surface_label} · "
                f"{len(tournament_frame)} matches · "
                f"{ended_count} ended"
            ),
            expanded=(
                day_position == 0
                and tournament_position == 0
            ),
        ):
            render_tournament_actions = st.toggle(
                (
                    "Load interactive actions "
                    "for this tournament"
                ),
                value=False,
                key=(
                    "render_tournament_"
                    f"{match_date}_"
                    f"{safe_tournament}"
                ),
                help=(
                    "The default table is sortable and "
                    "contains every match. Enable this "
                    "only when Odds, Model, IA or Match "
                    "actions are needed."
                ),
            )

            if not render_tournament_actions:
                preview = tournament_frame.copy()

            # Continúa aquí el código de la tabla ligera.

                preview["Players"] = (
                    "P1 "
                    + preview["player_1_name"].astype(str)
                    + " | P2 "
                    + preview["player_2_name"].astype(str)
                )

                preview["% Win"] = preview.apply(
                    lambda row: (
                        "P1 "
                        + prediction_display(
                            row.get(
                                "model_probability_player_1"
                            ),
                            True,
                        )
                        + " | P2 "
                        + prediction_display(
                            row.get(
                                "model_probability_player_2"
                            ),
                            True,
                        )
                    ),
                    axis=1,
                )

                preview["Fair"] = preview.apply(
                    lambda row: (
                        "P1 "
                        + prediction_display(
                            row.get(
                                "model_fair_odds_player_1"
                            )
                        )
                        + " | P2 "
                        + prediction_display(
                            row.get(
                                "model_fair_odds_player_2"
                            )
                        )
                    ),
                    axis=1,
                )

                preview["Odds"] = preview.apply(
                    lambda row: (
                        "P1 "
                        + odds_display(
                            row.get("market_odds_player_1")
                        )
                        + " | P2 "
                        + odds_display(
                            row.get("market_odds_player_2")
                        )
                    ),
                    axis=1,
                )

                preview["Edge"] = preview.apply(
                    lambda row: (
                        "P1 "
                        + (
                            prediction_display(
                                float(
                                    row["edge_player_1"]
                                )
                                * 100.0,
                                True,
                            )
                            if pd.notna(
                                row.get("edge_player_1")
                            )
                            else "-"
                        )
                        + " | P2 "
                        + (
                            prediction_display(
                                float(
                                    row["edge_player_2"]
                                )
                                * 100.0,
                                True,
                            )
                            if pd.notna(
                                row.get("edge_player_2")
                            )
                            else "-"
                        )
                    ),
                    axis=1,
                )

                for preview_line in OVER_LINES:
                    source_column = (
                        "over_"
                        + str(preview_line).replace(
                            ".",
                            "_",
                        )
                        + "_probability"
                    )

                    preview[
                        f"O{preview_line:.1f}"
                    ] = preview[source_column].map(
                        lambda value: prediction_display(
                            value,
                            True,
                        )
                    )

                preview["Score"] = preview.apply(
                    lambda row: (
                        str(
                            row.get(
                                "match_score",
                                "-",
                            )
                        )
                        if _match_finished(
                            row.get("match_status")
                        )
                        else "-"
                    ),
                    axis=1,
                )

                preview = preview.rename(
                    columns={
                        "scheduled_time": "Time",
                        "match_status": "Status",
                        "round": "Round",
                    }
                )

                preview_columns = [
                    "Time",
                    "Status",
                    "Round",
                    "Score",
                    "Players",
                    "% Win",
                    "Fair",
                    "Odds",
                    "Edge",
                    "O18.5",
                    "O19.5",
                    "O20.5",
                    "O21.5",
                    "O22.5",
                ]

                st.dataframe(
                    preview.loc[
                        :,
                        preview_columns,
                    ],
                    use_container_width=True,
                    hide_index=True,
                    height=min(
                        44 + 35 * len(preview),
                        460,
                    ),
                )

                continue

            sort_1, sort_2 = st.columns([2, 1])
            safe_tournament = re.sub(r"[^a-zA-Z0-9_]+", "_", str(tournament_name))
            with sort_1:
                selected_sort = st.selectbox(
                    "Sort matches by",
                    list(sort_options),
                    key=f"sort_{match_date}_{safe_tournament}",
                )
            with sort_2:
                descending = st.toggle(
                    "Descending",
                    key=f"sort_desc_{match_date}_{safe_tournament}",
                )
            tournament_frame = tournament_frame.sort_values(
                sort_options[selected_sort],
                ascending=not descending,
                kind="mergesort",
                na_position="last",
            )

            widths = [
                .48, .62, .45, .72, 1.55, .82, .72, .72, .72,
                .62, .62, .62, .62, .62, .72, .72, .72,
            ]
            headers = [
                "Time", "Status", "Round", "Score", "Players",
                "% Win", "Fair", "Odds", "Edge",
                "O18.5", "O19.5", "O20.5", "O21.5", "O22.5",
                "Load Odds", "Model Details", "IA Analysis", "Match Details",
            ]
            # Add one action width because there are four actions.
            widths.append(.72)
            header_columns = st.columns(widths, gap="small")
            for container, label in zip(header_columns, headers):
                container.markdown(f"**{label}**")

            for _, fixture in tournament_frame.iterrows():
                key = str(fixture["prediction_key"])
                safe_key = re.sub(r"[^a-zA-Z0-9_]+", "_", key)
                cols = st.columns(widths, gap="small", vertical_alignment="center")
                finished = _match_finished(fixture.get("match_status"))
                winner_side = _winner_side(fixture) if finished else None
                p1_probability = pd.to_numeric(
                    fixture.get("model_probability_player_1"), errors="coerce"
                )
                p2_probability = pd.to_numeric(
                    fixture.get("model_probability_player_2"), errors="coerce"
                )
                winner_prediction_correct = bool(
                    finished
                    and winner_side in {1, 2}
                    and (
                        (winner_side == 1 and pd.notna(p1_probability) and p1_probability > 50.0)
                        or (winner_side == 2 and pd.notna(p2_probability) and p2_probability > 50.0)
                    )
                )
                total_games = _score_total_games(fixture.get("match_score")) if finished else np.nan

                cols[0].write(str(fixture.get("scheduled_time", "TBD")))
                cols[1].write(str(fixture.get("match_status", "Upcoming")))
                cols[2].write(str(fixture.get("round", "-")))
                cols[3].write(str(fixture.get("match_score", "-")) if finished else "-")
                cols[4].markdown(
                    f"P1 **{fixture.get('player_1_name', '-')}**<br>"
                    f"P2 **{fixture.get('player_2_name', '-')}**",
                    unsafe_allow_html=True,
                )
                _write_html(
                    cols[5],
                    _paired_value(
                        prediction_display(p1_probability, True),
                        prediction_display(p2_probability, True),
                        winner_prediction_correct,
                    ),
                )
                _write_html(
                    cols[6],
                    _paired_value(
                        prediction_display(fixture.get("model_fair_odds_player_1")),
                        prediction_display(fixture.get("model_fair_odds_player_2")),
                    ),
                )
                _write_html(
                    cols[7],
                    _paired_value(
                        odds_display(fixture.get("market_odds_player_1")),
                        odds_display(fixture.get("market_odds_player_2")),
                    ),
                )
                _write_html(
                    cols[8],
                    _paired_value(
                        prediction_display(float(fixture["edge_player_1"]) * 100, True)
                        if pd.notna(fixture.get("edge_player_1")) else "-",
                        prediction_display(float(fixture["edge_player_2"]) * 100, True)
                        if pd.notna(fixture.get("edge_player_2")) else "-",
                    ),
                )

                for column_index, line in enumerate(OVER_LINES, start=9):
                    probability = pd.to_numeric(
                        fixture.get(f"over_{str(line).replace('.', '_')}_probability"),
                        errors="coerce",
                    )
                    favorable = bool(
                        finished
                        and pd.notna(total_games)
                        and pd.notna(probability)
                        and probability > 50.0
                        and total_games > line
                    )
                    _write_html(
                        cols[column_index],
                        _green_if(prediction_display(probability, True), favorable),
                    )

                with cols[14]:
                    if st.button(
                        "Loaded" if odds_cache.get(key, {}).get("available") else "Load",
                        key=f"odds_{safe_key}",
                        use_container_width=True,
                        disabled=not bool(api_key),
                    ):
                        try:
                            with st.spinner("Resolving event and loading pre-match odds..."):
                                odds_result = _safe_provider_result(
                                    fetch_fixture_odds(fixture, api_key, odds_host),
                                    reason="odds_integration_returned_none",
                                    source="upcoming_matches",
                                )
                            odds_cache[key] = odds_result
                            st.session_state["upcoming_odds_cache"] = dict(odds_cache)
                            level = "success" if odds_result.get("available") else "warning"
                            message = (
                                f"Odds loaded: {odds_result.get('odds_player_1', '-')} / "
                                f"{odds_result.get('odds_player_2', '-')}"
                                if odds_result.get("available")
                                else "No usable match-winner odds were returned. "
                                f"Reason: {odds_result.get('reason', 'unavailable')}. "
                                f"Source: {odds_result.get('source', 'unknown')}."
                            )
                            st.session_state["upcoming_api_notice"] = {
                                "level": level, "message": message,
                            }
                        except Exception as error:
                            st.session_state["upcoming_api_notice"] = {
                                "level": "error",
                                "message": f"Odds error: {type(error).__name__}: {error}",
                            }
                        st.rerun()

                with cols[15]:
                    if st.button(
                        "View",
                        key=f"model_{safe_key}",
                        use_container_width=True,
                    ):
                        st.session_state["upcoming_open_model_details"] = (
                            None
                            if st.session_state.get("upcoming_open_model_details") == key
                            else key
                        )
                        st.rerun()

                with cols[16]:
                    if st.button(
                        "Open",
                        key=f"ia_{safe_key}",
                        use_container_width=True,
                    ):
                        selection, error = make_match_selection(fixture)
                        if error:
                            st.session_state["upcoming_action_error"] = error
                        else:
                            st.session_state["upcoming_match_selection"] = selection
                            st.session_state["ia_candidate_selection"] = {
                                "fixture_key": key,
                                "fixture": fixture.to_dict(),
                                "auto_generate": True,
                            }
                            st.switch_page(IA_CANDIDATES_PAGE)

                with cols[17]:
                    if st.button(
                        "Open",
                        key=f"details_{safe_key}",
                        use_container_width=True,
                    ):
                        selection, error = make_match_selection(fixture)
                        if error:
                            st.session_state["upcoming_action_error"] = error
                        else:
                            st.session_state["upcoming_match_selection"] = selection
                            st.switch_page(MATCH_PREDICTOR_PAGE)

                if st.session_state.get("upcoming_open_model_details") == key:
                    model_item = prediction_cache.get(key, {})
                    with st.container(border=True):
                        st.markdown(
                            f"**Model details · {fixture.get('player_1_name')} vs "
                            f"{fixture.get('player_2_name')}**"
                        )
                        if not model_item:
                            st.info(
                                "Model details are unavailable until probabilities are calculated."
                            )
                        else:
                            st.json(model_item, expanded=False)
                st.divider()

if st.session_state.get("upcoming_action_error"):
    st.warning(st.session_state.pop("upcoming_action_error"))

with st.expander("Data source and limitations"):
    st.markdown(
        """
        - Tennis API remains the primary provider. SofaScore Scraper is called only after Tennis API returns HTTP 403 or HTTP 429.
        - SofaScore scheduled responses are cached once per date per Streamlit session and reused for scores, odds and movement.
        - Only ATP singles fixtures are requested.
        - ATP main draw, qualifying and Challenger are classified from tournament rank/name and round.
        - Times are shown in Europe/Madrid when the provider returns a timestamp.
        - The local cache is only a resilience fallback and is overwritten after successful refresh.
        - A fixture can be sent to Match Predictor only when both provider names match local player IDs.
        - Reliability is sourced from the active model's out-of-sample temporal test and uses contextual fallback when a segment is too small.
        - First-set and at-least-one-set probabilities come from calibrated supervised direct-market models. BO3 Over probabilities use cumulative direct models with monotonic correction. Structural set-score and expected-games estimates remain available from the serve-point Monte Carlo baseline.
        - H2H and similar-opponent frequencies remain separate descriptive historical evidence.
        - Similar opponents are selected using available pre-match Elo, surface Elo, rolling serve/return and recent-form features. Missing features reduce comparability.
        - H2H and similar-opponent estimates can have small samples. They do not guarantee an outcome or profitability.
        """
    )
