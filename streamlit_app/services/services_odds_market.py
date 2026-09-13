#!/usr/bin/env python3
"""Servicio local para ejecutar OddsHarvester y analizar cuotas de tenis.

Este módulo mantiene el scraping completamente separado de Streamlit. Ejecuta
OddsHarvester como un proceso externo, normaliza su salida JSON/CSV, crea un
Parquet canónico y expone funciones de lectura y diagnóstico.

No intenta eludir bloqueos, captchas ni controles de acceso. Si OddsPortal o
OddsHarvester rechazan la ejecución, el servicio devuelve un error explícito.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STAGING_DIR = PROJECT_ROOT / "data" / "staging" / "oddsharvester"
CACHE_PATH = PROCESSED_DIR / "upcoming_market_odds.oddsharvester.parquet"
RUNS_PATH = PROCESSED_DIR / "upcoming_market_odds.oddsharvester_runs.parquet"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MARKET = "match_winner"
SUPPORTED_FORMATS = {"json", "csv"}


@dataclass
class HarvesterRun:
    available: bool
    status: str
    command: list[str]
    target_date: str
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    return_code: int | None
    output_file: str | None
    rows_raw: int
    rows_normalized: int
    stdout: str
    stderr: str
    error: str | None = None


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _safe_float(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("value", "price", "odds", "odd", "decimal"):
            if key in value:
                return _safe_float(value[key])
        return np.nan
    try:
        number = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and number > 1.0 else np.nan


def _first_value(record: dict[str, Any], paths: Iterable[str], default: Any = None) -> Any:
    for path in paths:
        current: Any = record
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


def _flatten_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "results", "matches", "events", "items", "records"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _flatten_records(value)
            if nested:
                return nested
    if payload:
        return [payload]
    return []


def _outcome_name(item: dict[str, Any]) -> str:
    return str(_first_value(item, ("name", "label", "outcome", "selection", "participant"), ""))


def _outcome_price(item: dict[str, Any]) -> float:
    return _safe_float(_first_value(item, ("odds", "odd", "price", "value", "decimal_odds")))


def _collect_outcomes(value: Any) -> list[tuple[str, float, str]]:
    """Collect (selection, decimal odds, bookmaker) from heterogeneous output."""
    found: list[tuple[str, float, str]] = []

    def walk(node: Any, bookmaker: str = "") -> None:
        if isinstance(node, dict):
            current_bookmaker = str(
                _first_value(node, ("bookmaker", "bookmaker_name", "bookie", "provider"), bookmaker)
                or bookmaker
            )
            label = _outcome_name(node)
            price = _outcome_price(node)
            if label and np.isfinite(price):
                found.append((label, float(price), current_bookmaker))
            for key, child in node.items():
                if key not in {"name", "label", "outcome", "selection", "participant", "odds", "odd", "price", "value"}:
                    walk(child, current_bookmaker)
        elif isinstance(node, list):
            for child in node:
                walk(child, bookmaker)

    walk(value)
    return found


def _player_names(record: dict[str, Any]) -> tuple[str, str]:
    player_1 = _first_value(
        record,
        (
            "player_1_name", "player1_name", "home_name", "home_team",
            "home", "participant1", "player1.name", "player_1.name",
        ),
        "",
    )
    player_2 = _first_value(
        record,
        (
            "player_2_name", "player2_name", "away_name", "away_team",
            "away", "participant2", "player2.name", "player_2.name",
        ),
        "",
    )
    if isinstance(player_1, dict):
        player_1 = _first_value(player_1, ("name", "label"), "")
    if isinstance(player_2, dict):
        player_2 = _first_value(player_2, ("name", "label"), "")

    if not player_1 or not player_2:
        match_name = str(_first_value(record, ("match", "event", "name", "title"), ""))
        parts = re.split(r"\s+(?:vs\.?|v\.?|-)\s+", match_name, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            player_1 = player_1 or parts[0]
            player_2 = player_2 or parts[1]
    return str(player_1).strip(), str(player_2).strip()


def _direct_prices(record: dict[str, Any]) -> tuple[float, float]:
    first = _safe_float(
        _first_value(
            record,
            (
                "odds_player_1", "player_1_odds", "home_odds", "home_odd",
                "best_home_odds", "best_odds_home", "odds.home", "odds.0",
            ),
        )
    )
    second = _safe_float(
        _first_value(
            record,
            (
                "odds_player_2", "player_2_odds", "away_odds", "away_odd",
                "best_away_odds", "best_odds_away", "odds.away", "odds.1",
            ),
        )
    )
    return first, second


def _prices_for_players(record: dict[str, Any], player_1: str, player_2: str) -> tuple[float, float, str, str, int]:
    first, second = _direct_prices(record)
    outcomes = _collect_outcomes(record)
    p1 = normalize_text(player_1)
    p2 = normalize_text(player_2)
    candidates_1: list[tuple[float, str]] = []
    candidates_2: list[tuple[float, str]] = []
    for label, price, bookmaker in outcomes:
        normalized = normalize_text(label)
        if p1 and (normalized == p1 or p1 in normalized or normalized in p1):
            candidates_1.append((price, bookmaker))
        elif p2 and (normalized == p2 or p2 in normalized or normalized in p2):
            candidates_2.append((price, bookmaker))
    if not np.isfinite(first) and candidates_1:
        first = max(candidates_1, key=lambda item: item[0])[0]
    if not np.isfinite(second) and candidates_2:
        second = max(candidates_2, key=lambda item: item[0])[0]
    book_1 = max(candidates_1, default=(np.nan, ""), key=lambda item: item[0])[1]
    book_2 = max(candidates_2, default=(np.nan, ""), key=lambda item: item[0])[1]
    bookmakers = {book for _, _, book in outcomes if book}
    return first, second, book_1, book_2, len(bookmakers)


def normalize_oddsharvester_records(
    records: list[dict[str, Any]],
    target_date: str,
    source_file: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for position, record in enumerate(records):
        player_1, player_2 = _player_names(record)
        if not player_1 or not player_2:
            continue
        odds_1, odds_2, bookmaker_1, bookmaker_2, bookmaker_count = _prices_for_players(
            record, player_1, player_2
        )
        event_date = pd.to_datetime(
            _first_value(record, ("start_time", "commence_time", "kickoff", "date", "match_date"), target_date),
            errors="coerce",
            utc=True,
        )
        match_date = event_date.date().isoformat() if pd.notna(event_date) else target_date
        tournament = str(
            _first_value(record, ("league", "tournament", "competition", "league_name", "tournament_name"), "")
        )
        source_url = str(_first_value(record, ("url", "match_url", "event_url", "link"), ""))
        key_material = "|".join(
            sorted((normalize_text(player_1), normalize_text(player_2)))
            + [match_date, normalize_text(tournament)]
        )
        event_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:24]
        overround = np.nan
        no_vig_1 = np.nan
        no_vig_2 = np.nan
        if np.isfinite(odds_1) and np.isfinite(odds_2):
            raw_1, raw_2 = 1.0 / odds_1, 1.0 / odds_2
            overround = raw_1 + raw_2
            if overround > 0:
                no_vig_1, no_vig_2 = raw_1 / overround, raw_2 / overround
        rows.append(
            {
                "event_key": event_key,
                "match_date": match_date,
                "scheduled_time_utc": event_date.isoformat() if pd.notna(event_date) else None,
                "tournament": tournament,
                "player_1_name": player_1,
                "player_2_name": player_2,
                "normalized_player_1_name": normalize_text(player_1),
                "normalized_player_2_name": normalize_text(player_2),
                "market": DEFAULT_MARKET,
                "odds_player_1": odds_1,
                "odds_player_2": odds_2,
                "bookmaker_player_1": bookmaker_1,
                "bookmaker_player_2": bookmaker_2,
                "bookmaker_count": bookmaker_count,
                "market_overround": overround,
                "no_vig_probability_player_1": no_vig_1,
                "no_vig_probability_player_2": no_vig_2,
                "provider": "oddsportal_via_oddsharvester",
                "source_url": source_url,
                "source_file": str(source_file),
                "source_row": position,
                "fetched_at_utc": fetched_at,
                "has_complete_match_winner_odds": bool(np.isfinite(odds_1) and np.isfinite(odds_2)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.sort_values(
            ["match_date", "tournament", "player_1_name", "player_2_name"],
            kind="mergesort",
        )
        .drop_duplicates("event_key", keep="last")
        .reset_index(drop=True)
    )


def _read_output(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return _flatten_records(json.loads(path.read_text(encoding="utf-8-sig")))
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Unsupported OddsHarvester output: {path}")


def _resolve_output(base: Path, output_format: str) -> Path | None:
    candidates = [
        base,
        base.with_suffix(f".{output_format}"),
        Path(str(base) + f".{output_format}"),
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    matches = sorted(base.parent.glob(f"{base.name}*.{output_format}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _executable_prefix() -> list[str]:
    configured = os.getenv("ODDSHARVESTER_COMMAND", "").strip()
    if configured:
        return configured.split()
    executable = shutil.which("oddsharvester")
    if executable:
        return [executable]
    python_executable = os.getenv("ODDSHARVESTER_PYTHON", "").strip()
    if python_executable:
        return [python_executable, "-m", "oddsharvester"]
    return [sys.executable, "-m", "oddsharvester"]


def installation_diagnostics() -> dict[str, Any]:
    prefix = _executable_prefix()
    result: dict[str, Any] = {
        "command_prefix": prefix,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "available": False,
    }
    try:
        completed = subprocess.run(
            [*prefix, "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        result.update(
            {
                "available": completed.returncode == 0,
                "return_code": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        )
    except Exception as error:
        result["error"] = str(error)
    return result


def run_upcoming_tennis_odds(
    target_date: date,
    *,
    preview_only: bool = True,
    headless: bool = True,
    output_format: str = "json",
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    target_bookmaker: str | None = None,
    timezone_name: str = "Europe/Madrid",
) -> dict[str, Any]:
    if output_format not in SUPPORTED_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(SUPPORTED_FORMATS)}")
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target_text = target_date.strftime("%Y%m%d")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_base = STAGING_DIR / f"tennis_upcoming_{target_text}_{run_id}"
    prefix = _executable_prefix()
    command = [
        *prefix,
        "upcoming",
        "--sport", "tennis",
        "--date", target_text,
        "--market", DEFAULT_MARKET,
        "--format", output_format,
        "--output", str(output_base),
        "--timezone", timezone_name,
    ]
    if headless:
        command.append("--headless")
    if preview_only:
        command.append("--preview-only")
    if target_bookmaker:
        command.extend(["--target-bookmaker", target_bookmaker])

    started = datetime.now(timezone.utc)
    return_code: int | None = None
    stdout = ""
    stderr = ""
    error: str | None = None
    output_file: Path | None = None
    records: list[dict[str, Any]] = []
    normalized = pd.DataFrame()
    status = "FAILED"
    available = False

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(timeout_seconds),
            check=False,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        output_file = _resolve_output(output_base, output_format)
        if return_code != 0:
            error = f"OddsHarvester exited with code {return_code}."
        elif output_file is None:
            error = "OddsHarvester completed but no output file was found."
        else:
            records = _read_output(output_file)
            normalized = normalize_oddsharvester_records(records, target_date.isoformat(), output_file)
            status = "SUCCESS" if not normalized.empty else "EMPTY"
            available = not normalized.empty
            if not normalized.empty:
                _merge_cache(normalized)
    except subprocess.TimeoutExpired as timeout_error:
        status = "TIMEOUT"
        error = f"OddsHarvester exceeded {timeout_seconds} seconds."
        stdout = str(timeout_error.stdout or "")
        stderr = str(timeout_error.stderr or "")
    except Exception as run_error:
        error = str(run_error)

    finished = datetime.now(timezone.utc)
    run = HarvesterRun(
        available=available,
        status=status,
        command=command,
        target_date=target_date.isoformat(),
        started_at_utc=started.isoformat(),
        finished_at_utc=finished.isoformat(),
        duration_seconds=(finished - started).total_seconds(),
        return_code=return_code,
        output_file=str(output_file) if output_file else None,
        rows_raw=len(records),
        rows_normalized=len(normalized),
        stdout=stdout[-12000:],
        stderr=stderr[-12000:],
        error=error,
    )
    _append_run(run)
    return {**asdict(run), "data": normalized}


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _merge_cache(new_rows: pd.DataFrame) -> None:
    if CACHE_PATH.exists():
        previous = pd.read_parquet(CACHE_PATH, engine="pyarrow")
        combined = pd.concat([previous, new_rows], ignore_index=True, sort=False)
    else:
        combined = new_rows.copy()
    combined["fetched_at_utc"] = pd.to_datetime(combined["fetched_at_utc"], errors="coerce", utc=True)
    combined = (
        combined.sort_values("fetched_at_utc", kind="mergesort")
        .drop_duplicates("event_key", keep="last")
        .reset_index(drop=True)
    )
    combined["fetched_at_utc"] = combined["fetched_at_utc"].astype("string")
    _atomic_write_parquet(combined, CACHE_PATH)


def _append_run(run: HarvesterRun) -> None:
    row = asdict(run)
    row["command"] = json.dumps(row["command"], ensure_ascii=False)
    frame = pd.DataFrame([row])
    if RUNS_PATH.exists():
        frame = pd.concat([pd.read_parquet(RUNS_PATH, engine="pyarrow"), frame], ignore_index=True, sort=False)
    _atomic_write_parquet(frame.tail(200).reset_index(drop=True), RUNS_PATH)


def load_market_odds_cache(target_date: date | None = None) -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(CACHE_PATH, engine="pyarrow")
    if target_date is not None and "match_date" in frame.columns:
        frame = frame.loc[frame["match_date"].astype(str).eq(target_date.isoformat())].copy()
    return frame.reset_index(drop=True)


def load_run_history(limit: int = 20) -> pd.DataFrame:
    if not RUNS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(RUNS_PATH, engine="pyarrow").tail(int(limit)).iloc[::-1].reset_index(drop=True)


def clear_market_odds_cache() -> None:
    CACHE_PATH.unlink(missing_ok=True)


def summarize_market_odds(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty:
        return {
            "matches": 0,
            "complete_matches": 0,
            "tournaments": 0,
            "bookmakers": 0,
            "median_overround": np.nan,
            "last_refresh_utc": None,
        }
    bookmaker_values = set()
    for column in ("bookmaker_player_1", "bookmaker_player_2"):
        if column in frame:
            bookmaker_values.update(frame[column].dropna().astype(str).loc[lambda s: s.ne("")].tolist())
    fetched = pd.to_datetime(frame.get("fetched_at_utc"), errors="coerce", utc=True)
    return {
        "matches": int(len(frame)),
        "complete_matches": int(frame.get("has_complete_match_winner_odds", pd.Series(False, index=frame.index)).fillna(False).sum()),
        "tournaments": int(frame.get("tournament", pd.Series(dtype="string")).nunique(dropna=True)),
        "bookmakers": len(bookmaker_values),
        "median_overround": pd.to_numeric(frame.get("market_overround"), errors="coerce").median(),
        "last_refresh_utc": fetched.max().isoformat() if fetched.notna().any() else None,
    }
