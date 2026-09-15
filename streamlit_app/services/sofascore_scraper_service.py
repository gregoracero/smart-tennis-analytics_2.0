#!/usr/bin/env python3
"""SofaScore Scraper fallback for Upcoming Matches.

This module is intentionally independent from the model and Tennis API service.
It uses the RapidAPI SofaScore Scraper only when the caller explicitly requests
fallback behavior, normally after Tennis API returns HTTP 403.

Configuration is supplied by the caller:
    SOFASCORE_API_KEY
    SOFASCORE_API_HOST=sofascore-scraper-1000-free-calls.p.rapidapi.com

No API key is stored in this source file.
"""
from __future__ import annotations

import re
import unicodedata
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import requests

DEFAULT_SOFASCORE_HOST = "sofascore-scraper-1000-free-calls.p.rapidapi.com"


class SofaScoreAPIError(RuntimeError):
    """HTTP or payload failure returned by the SofaScore Scraper provider."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _request_json(
    path: str,
    *,
    api_key: str,
    host: str = DEFAULT_SOFASCORE_HOST,
    params: dict[str, Any] | None = None,
    timeout: int = 45,
) -> tuple[Any, dict[str, str]]:
    if not api_key:
        raise SofaScoreAPIError("SOFASCORE_API_KEY is not configured.")
    response = requests.get(
        f"https://{host}{path}",
        headers={
            "Accept": "application/json",
            "X-RapidAPI-Key": api_key,
            "X-RapidAPI-Host": host,
        },
        params=params,
        timeout=timeout,
    )
    if response.status_code >= 400:
        detail = response.text[:500]
        raise SofaScoreAPIError(
            f"SofaScore Scraper HTTP {response.status_code}: {detail}",
            status_code=response.status_code,
        )
    try:
        payload = response.json()
    except ValueError as error:
        raise SofaScoreAPIError("SofaScore Scraper returned non-JSON content.") from error
    quota_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower().startswith("x-ratelimit")
    }
    return payload, quota_headers


def fetch_scheduled_matches(
    match_date: Any,
    *,
    api_key: str,
    host: str = DEFAULT_SOFASCORE_HOST,
    max_pages: int = 10,
) -> dict[str, Any]:
    """Fetch all scheduled tennis events for one date, following pagination."""
    date_text = pd.to_datetime(match_date, errors="raise").date().isoformat()
    tournaments: list[dict[str, Any]] = []
    quota_headers: dict[str, str] = {}
    page = 1
    while page <= max(int(max_pages), 1):
        payload, quota_headers = _request_json(
            "/matches/scheduled",
            api_key=api_key,
            host=host,
            params={"sport": "tennis", "date": date_text, "page": page},
        )
        page_tournaments = payload.get("tournaments", []) if isinstance(payload, dict) else []
        tournaments.extend(
            item for item in page_tournaments if isinstance(item, dict)
        )
        if not bool(payload.get("has_next_page")):
            break
        page += 1
    matches: list[dict[str, Any]] = []
    for group in tournaments:
        tournament = group.get("tournament", {})
        for match in group.get("matches", []):
            if not isinstance(match, dict):
                continue
            enriched = dict(match)
            enriched.setdefault("tournament", tournament)
            matches.append(enriched)
    return {
        "available": True,
        "date": date_text,
        "pages_requested": page,
        "tournaments": tournaments,
        "matches": matches,
        "quota_headers": quota_headers,
        "source": "sofascore_scraper_scheduled",
    }


def _name_pair(match: dict[str, Any]) -> tuple[str, str]:
    home = match.get("home_team", {})
    away = match.get("away_team", {})
    return str(home.get("name", "")).strip(), str(away.get("name", "")).strip()


def _same_player_name(left: str, right: str) -> bool:
    a, b = normalize_text(left), normalize_text(right)
    if not a or not b:
        return False
    if a == b:
        return True
    a_tokens, b_tokens = a.split(), b.split()
    return (
        len(a_tokens) >= 2
        and len(b_tokens) >= 2
        and a_tokens[-1] == b_tokens[-1]
        and a_tokens[0][0] == b_tokens[0][0]
    )


def find_scheduled_match(
    scheduled: dict[str, Any],
    *,
    player_1_name: str,
    player_2_name: str,
    tournament: str = "",
) -> dict[str, Any]:
    """Resolve a fixture in scheduled data and report screen/home orientation."""
    target_tournament = normalize_text(tournament)
    candidates: list[tuple[int, bool, dict[str, Any]]] = []
    for match in scheduled.get("matches", []):
        home, away = _name_pair(match)
        direct = _same_player_name(home, player_1_name) and _same_player_name(away, player_2_name)
        reversed_order = _same_player_name(home, player_2_name) and _same_player_name(away, player_1_name)
        if not direct and not reversed_order:
            continue
        match_tournament = match.get("tournament", {})
        tournament_name = str(match_tournament.get("name", ""))
        group_name = str(match_tournament.get("group", {}).get("name", ""))
        tournament_text = normalize_text(f"{tournament_name} {group_name}")
        tournament_score = int(
            bool(target_tournament)
            and (
                target_tournament in tournament_text
                or tournament_text in target_tournament
            )
        )
        candidates.append((tournament_score, reversed_order, match))
    if not candidates:
        return {
            "available": False,
            "reason": "sofascore_match_not_found",
            "player_1_name": player_1_name,
            "player_2_name": player_2_name,
        }
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, reversed_order, match = candidates[0]
    return {
        "available": True,
        "match_id": str(match.get("id", "")),
        "reversed_order": bool(reversed_order),
        "match": match,
        "source": "sofascore_scraper_scheduled",
    }


def _score_text(match: dict[str, Any]) -> str:
    home_score = match.get("home_score", {}) or {}
    away_score = match.get("away_score", {}) or {}
    home_periods = home_score.get("periods", {}) or {}
    away_periods = away_score.get("periods", {}) or {}
    period_numbers = sorted({
        int(key.removeprefix("period"))
        for key in set(home_periods) | set(away_periods)
        if re.fullmatch(r"period\d+", str(key))
    })
    sets: list[str] = []
    for number in period_numbers:
        home = home_periods.get(f"period{number}")
        away = away_periods.get(f"period{number}")
        if home is None or away is None:
            continue
        sets.append(f"{home}-{away}")
    if sets:
        return " ".join(sets)
    home = home_score.get("display", home_score.get("current"))
    away = away_score.get("display", away_score.get("current"))
    return f"{home}-{away}" if home is not None and away is not None else ""


def normalize_scheduled_result(
    resolved_match: dict[str, Any],
    *,
    player_1_name: str,
    player_2_name: str,
) -> dict[str, Any]:
    """Return status and score in the same player order as the Tennis fixture."""
    match = resolved_match.get("match", resolved_match)
    reversed_order = bool(resolved_match.get("reversed_order"))
    status = match.get("status", {}) or {}
    score = _score_text(match)
    if reversed_order and score:
        reversed_sets: list[str] = []
        for token in score.split():
            parts = token.split("-", 1)
            reversed_sets.append(f"{parts[1]}-{parts[0]}" if len(parts) == 2 else token)
        score = " ".join(reversed_sets)
    winner_code = pd.to_numeric(match.get("winner_code"), errors="coerce")
    winner_name = ""
    if pd.notna(winner_code):
        screen_winner = 1 if int(winner_code) == 1 else 2
        if reversed_order:
            screen_winner = 2 if screen_winner == 1 else 1
        winner_name = player_1_name if screen_winner == 1 else player_2_name
    return {
        "available": True,
        "player_1_name": player_1_name,
        "player_2_name": player_2_name,
        "winner_name": winner_name,
        "status": str(status.get("description", status.get("type", "Upcoming"))),
        "status_type": str(status.get("type", "")),
        "score": score,
        "match_id": str(match.get("id", "")),
        "provider_result": match,
        "source": "sofascore_scraper_scheduled_fallback",
    }


def fractional_to_decimal(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return np.nan
    try:
        return float(Fraction(text)) + 1.0
    except (ValueError, ZeroDivisionError):
        return np.nan


def fetch_match_odds(
    match_id: Any,
    *,
    api_key: str,
    host: str = DEFAULT_SOFASCORE_HOST,
    provider_id: int | None = None,
    featured_only: bool = True,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "match_id": str(match_id),
        "featured_only": str(bool(featured_only)).lower(),
    }
    if provider_id is not None:
        params["provider_id"] = int(provider_id)
    payload, quota_headers = _request_json(
        "/matches/odds",
        api_key=api_key,
        host=host,
        params=params,
    )
    return {
        "available": True,
        "match_id": str(match_id),
        "payload": payload,
        "quota_headers": quota_headers,
        "source": "sofascore_scraper_odds",
    }


def parse_match_winner_odds(
    odds_response: dict[str, Any],
    *,
    reversed_order: bool = False,
) -> dict[str, Any]:
    payload = odds_response.get("payload", odds_response)
    featured = payload.get("featured", {}) if isinstance(payload, dict) else {}
    market = None
    for key in ("default", "full_time"):
        candidate = featured.get(key)
        if isinstance(candidate, dict) and candidate.get("choices"):
            market = candidate
            break
    if market is None:
        for candidate in payload.get("markets", []) if isinstance(payload, dict) else []:
            group = normalize_text(candidate.get("market_group"))
            name = normalize_text(candidate.get("market_name"))
            if ("home away" in group or "full time" in name) and candidate.get("choices"):
                market = candidate
                break
    if market is None:
        return {
            "available": False,
            "reason": "sofascore_match_winner_market_unavailable",
            "match_id": odds_response.get("match_id"),
            "source": "sofascore_scraper_odds",
        }
    choices = {
        str(choice.get("name", "")): choice
        for choice in market.get("choices", [])
        if isinstance(choice, dict)
    }
    home_choice, away_choice = choices.get("1", {}), choices.get("2", {})
    if reversed_order:
        first_choice, second_choice = away_choice, home_choice
    else:
        first_choice, second_choice = home_choice, away_choice
    first = fractional_to_decimal(first_choice.get("fractional_value"))
    second = fractional_to_decimal(second_choice.get("fractional_value"))
    opening_first = fractional_to_decimal(first_choice.get("initial_fractional_value"))
    opening_second = fractional_to_decimal(second_choice.get("initial_fractional_value"))
    available = bool(np.isfinite(first) or np.isfinite(second))
    return {
        "available": available,
        "reason": None if available else "sofascore_prices_unavailable",
        "odds_player_1": first,
        "odds_player_2": second,
        "opening_odds_player_1": opening_first,
        "opening_odds_player_2": opening_second,
        "movement_player_1": first_choice.get("change"),
        "movement_player_2": second_choice.get("change"),
        "bookmaker_player_1": f"SofaScore provider {payload.get('provider_id', '')}".strip(),
        "bookmaker_player_2": f"SofaScore provider {payload.get('provider_id', '')}".strip(),
        "market_name": market.get("market_name"),
        "market_group": market.get("market_group"),
        "market_suspended": bool(market.get("suspended")),
        "match_id": str(payload.get("match_id", odds_response.get("match_id", ""))),
        "source": "sofascore_scraper_odds_fallback",
        "provider_payload": payload,
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }


def fetch_fixture_odds(
    *,
    player_1_name: str,
    player_2_name: str,
    match_date: Any,
    tournament: str,
    api_key: str,
    host: str = DEFAULT_SOFASCORE_HOST,
    scheduled_payload: dict[str, Any] | None = None,
    provider_id: int | None = None,
) -> dict[str, Any]:
    scheduled = scheduled_payload or fetch_scheduled_matches(
        match_date,
        api_key=api_key,
        host=host,
    )
    resolved = find_scheduled_match(
        scheduled,
        player_1_name=player_1_name,
        player_2_name=player_2_name,
        tournament=tournament,
    )
    if not resolved.get("available"):
        return resolved
    odds = fetch_match_odds(
        resolved["match_id"],
        api_key=api_key,
        host=host,
        provider_id=provider_id,
        featured_only=True,
    )
    parsed = parse_match_winner_odds(
        odds,
        reversed_order=bool(resolved.get("reversed_order")),
    )
    return {
        **parsed,
        "sofascore_match": resolved.get("match"),
        "scheduled_source": scheduled.get("source"),
    }
