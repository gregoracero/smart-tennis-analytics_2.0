#!/usr/bin/env python3
"""Comprueba qué endpoints de Tennis API funcionan con el plan Basic."""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


HOST = "tennis-api-atp-wta-itf.p.rapidapi.com"
BASE_URL = f"https://{HOST}"

OUTPUT_PATH = Path(
    "data/processed/tennis_api_basic_endpoint_audit.json"
)


def compact_response(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {
            "response_type": "list",
            "item_count": len(value),
            "sample": value[:1],
        }

    if isinstance(value, dict):
        result: dict[str, Any] = {
            "response_type": "dict",
            "keys": sorted(value.keys()),
        }

        for key in (
            "data",
            "result",
            "results",
            "items",
            "fixtures",
            "events",
        ):
            nested = value.get(key)

            if isinstance(nested, list):
                result["data_key"] = key
                result["item_count"] = len(nested)
                result["sample"] = nested[:1]
                break

            if isinstance(nested, dict):
                result["data_key"] = key
                result["nested_keys"] = sorted(nested.keys())
                break

        return result

    return {
        "response_type": type(value).__name__,
        "sample": str(value)[:500],
    }


def call_endpoint(
    name: str,
    path: str,
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    started = time.perf_counter()

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=30,
        )

        elapsed = time.perf_counter() - started

        try:
            payload = response.json()
        except ValueError:
            payload = response.text[:1000]

        return {
            "name": name,
            "method": "GET",
            "path": path,
            "requested_url": response.url,
            "status_code": response.status_code,
            "elapsed_seconds": round(elapsed, 3),
            "rapidapi_proxy_response": response.headers.get(
                "X-RapidAPI-Proxy-Response"
            ),
            "rate_limit_requests_limit": response.headers.get(
                "X-RateLimit-Requests-Limit"
            ),
            "rate_limit_requests_remaining": response.headers.get(
                "X-RateLimit-Requests-Remaining"
            ),
            "rate_limit_requests_reset": response.headers.get(
                "X-RateLimit-Requests-Reset"
            ),
            "available_on_basic": response.status_code == 200,
            "interpretation": interpret_status(response.status_code),
            "response_summary": compact_response(payload),
        }

    except requests.RequestException as error:
        return {
            "name": name,
            "method": "GET",
            "path": path,
            "status_code": None,
            "available_on_basic": False,
            "interpretation": "network_or_request_error",
            "error": f"{type(error).__name__}: {error}",
        }


def interpret_status(status_code: int) -> str:
    if status_code == 200:
        return "available"

    if status_code == 400:
        return "endpoint_reached_but_parameters_invalid"

    if status_code == 401:
        return "authentication_failed"

    if status_code == 403:
        return "possibly_blocked_by_plan_or_host"

    if status_code == 404:
        return "endpoint_reached_but_resource_not_found"

    if status_code == 429:
        return "rate_or_quota_limit_reached"

    if status_code >= 500:
        return "provider_server_error"

    return "unexpected_status"


def main() -> None:
    api_key = os.getenv("TENNIS_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Configura TENNIS_API_KEY como variable de entorno."
        )

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": HOST,
    }

    today = date.today().isoformat()

    # Sustituir estos valores por IDs conocidos obtenidos de fixtures.
    player_1_id = os.getenv("TENNIS_TEST_PLAYER_1_ID", "")
    player_2_id = os.getenv("TENNIS_TEST_PLAYER_2_ID", "")
    season_id = os.getenv("TENNIS_TEST_SEASON_ID", "")
    event_id = os.getenv("TENNIS_TEST_EVENT_ID", "")
    player_1_name = os.getenv("TENNIS_TEST_PLAYER_1_NAME", "")
    player_2_name = os.getenv("TENNIS_TEST_PLAYER_2_NAME", "")
    match_date = os.getenv("TENNIS_TEST_MATCH_DATE", today)

    endpoints: list[tuple[str, str, dict[str, Any] | None]] = [
        (
            "fixtures_by_date",
            f"/tennis/v2/atp/fixtures/{today}",
            {
                "include": (
                    "round,tournament,tournament.court,"
                    "tournament.rank,tournament.country"
                ),
                "filter": "PlayerGroup:singles",
                "pageSize": 10,
                "pageNo": 1,
            },
        ),
        (
            "tournament_calendar",
            f"/tennis/v2/atp/tournament/calendar/{date.today().year}",
            {
                "since": today,
                "pageSize": 10,
                "pageNo": 1,
            },
        ),
        (
            "live_events",
            "/tennis/v2/extend/api/events/live",
            None,
        ),
    ]

    if season_id:
        endpoints.append(
            (
                "tournament_results",
                f"/tennis/v2/atp/tournament/results/{season_id}",
                None,
            )
        )

    if player_1_id and player_2_id:
        endpoints.append(
            (
                "h2h_info",
                (
                    "/tennis/v2/atp/h2h/info/"
                    f"{player_1_id}/{player_2_id}"
                ),
                None,
            )
        )

    if player_1_name and player_2_name:
        endpoints.append(
            (
                "event_by_players_and_date",
                (
                    "/tennis/v2/extend/api/event/get/"
                    f"{quote(player_1_name, safe='')}/"
                    f"{quote(player_2_name, safe='')}/"
                    f"{match_date}"
                ),
                None,
            )
        )

    if event_id:
        endpoints.extend(
            [
                (
                    "pre_match_odds",
                    (
                        "/tennis/v2/extend/api/odds/"
                        f"pre-match/{event_id}"
                    ),
                    None,
                ),
                (
                    "compare_match_winner_odds",
                    (
                        "/tennis/v2/extend/api/odds/"
                        f"compare/{event_id}"
                    ),
                    {"market_id": 1},
                ),
            ]
        )

    results: list[dict[str, Any]] = []

    for position, (name, path, params) in enumerate(
        endpoints,
        start=1,
    ):
        print(
            f"[{position}/{len(endpoints)}] Probando {name}...",
            flush=True,
        )

        result = call_endpoint(
            name=name,
            path=path,
            headers=headers,
            params=params,
        )

        results.append(result)

        print(
            f"  HTTP {result.get('status_code')} "
            f"-> {result.get('interpretation')}",
            flush=True,
        )

        # Muy por debajo del límite Basic de 4 solicitudes por segundo.
        time.sleep(0.5)

    report = {
        "generated_at_utc": (
            __import__("datetime")
            .datetime.now(
                __import__("datetime").timezone.utc
            )
            .isoformat()
        ),
        "host": HOST,
        "plan_under_test": "Basic",
        "documented_plan_limits": {
            "requests_per_day": 50,
            "hard_limit": True,
            "requests_per_second": 4,
        },
        "request_count": len(results),
        "available_count": sum(
            item.get("status_code") == 200
            for item in results
        ),
        "forbidden_count": sum(
            item.get("status_code") == 403
            for item in results
        ),
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = OUTPUT_PATH.with_suffix(
        OUTPUT_PATH.suffix + ".new"
    )

    temporary.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        OUTPUT_PATH,
    )

    print(f"\nInforme: {OUTPUT_PATH}")

    for item in results:
        print(
            f"{item['name']:<30} "
            f"{str(item.get('status_code')):<5} "
            f"{item.get('interpretation')}"
        )


if __name__ == "__main__":
    main()
