#!/usr/bin/env python3
"""Página de laboratorio para ejecutar y revisar OddsHarvester.

Esta página no modifica Upcoming Matches ni el modelo de predicción. Ejecuta un
job local, normaliza los resultados y permite inspeccionar cuotas, probabilidades
sin margen, cobertura y logs.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, APP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from streamlit_app.services.services_odds_market import (
    CACHE_PATH,
    RUNS_PATH,
    clear_market_odds_cache,
    installation_diagnostics,
    load_market_odds_cache,
    load_run_history,
    run_upcoming_tennis_odds,
    summarize_market_odds,
)

st.set_page_config(
    page_title="Upcoming Odds Analysis",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Upcoming Odds Analysis")
st.caption(
    "Laboratorio independiente para recopilar y revisar cuotas pre-match de tenis "
    "mediante OddsHarvester y OddsPortal."
)
st.warning(
    "OddsHarvester automatiza un navegador y extrae información de un sitio externo. "
    "Utiliza esta página únicamente si el uso cumple las condiciones del sitio y la "
    "normativa aplicable. La página no intenta superar bloqueos, captchas ni controles "
    "de acceso."
)


def percentage(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.1%}" if np.isfinite(number) else "-"


def decimal_value(value: Any, decimals: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{number:.{decimals}f}" if np.isfinite(number) else "-"


with st.expander("1. Installation and runtime diagnostics", expanded=True):
    if st.button("Check OddsHarvester installation", use_container_width=False):
        st.session_state["odds_installation_diagnostics"] = installation_diagnostics()

    diagnostics = st.session_state.get("odds_installation_diagnostics")
    if diagnostics:
        if diagnostics.get("available"):
            st.success("OddsHarvester is available to this Streamlit process.")
        else:
            st.error("OddsHarvester is not available to this Streamlit process.")
        diagnostic_1, diagnostic_2 = st.columns(2)
        diagnostic_1.code(
            " ".join(str(value) for value in diagnostics.get("command_prefix", [])),
            language="text",
        )
        diagnostic_2.code(str(diagnostics.get("python_executable", "")), language="text")
        if diagnostics.get("stderr"):
            st.code(str(diagnostics["stderr"]), language="text")
        if diagnostics.get("error"):
            st.error(str(diagnostics["error"]))

    st.markdown(
        r"""
        Recommended isolated environment on Windows:

        ```powershell
        py -3.12 -m venv .venv-oddsharvester
        .\.venv-oddsharvester\Scripts\Activate.ps1
        python -m pip install --upgrade pip
        pip install oddsharvester
        playwright install chromium
        ```

        If Streamlit uses another virtual environment, set the environment variable
        `ODDSHARVESTER_PYTHON` to the Python executable of the isolated environment:

        ```powershell
        $env:ODDSHARVESTER_PYTHON = "$PWD\.venv-oddsharvester\Scripts\python.exe"
        ```
        """
    )

st.subheader("2. Run collection job")
with st.container(border=True):
    input_1, input_2, input_3, input_4 = st.columns(4)
    with input_1:
        target_date = st.date_input(
            "Target date",
            value=date.today(),
            min_value=date.today() - timedelta(days=7),
            max_value=date.today() + timedelta(days=14),
        )
    with input_2:
        preview_only = st.toggle(
            "Preview only",
            value=True,
            help="Faster mode. Prefer this for initial testing and best-price analysis.",
        )
    with input_3:
        headless = st.toggle("Headless browser", value=True)
    with input_4:
        output_format = st.selectbox("Raw output", ["json", "csv"], index=0)

    advanced_1, advanced_2, advanced_3 = st.columns(3)
    with advanced_1:
        timeout_seconds = st.number_input(
            "Timeout, seconds",
            min_value=60,
            max_value=1200,
            value=300,
            step=30,
        )
    with advanced_2:
        target_bookmaker = st.text_input(
            "Target bookmaker, optional",
            placeholder="Leave empty for all/best available",
        )
    with advanced_3:
        timezone_name = st.text_input("Reference timezone", value="Europe/Madrid")

    run_clicked = st.button(
        "Run OddsHarvester job",
        type="primary",
        use_container_width=True,
    )

if run_clicked:
    with st.spinner(
        "Launching the browser job. This may take several minutes. Do not start a second run."
    ):
        result = run_upcoming_tennis_odds(
            target_date=target_date,
            preview_only=preview_only,
            headless=headless,
            output_format=output_format,
            timeout_seconds=int(timeout_seconds),
            target_bookmaker=target_bookmaker.strip() or None,
            timezone_name=timezone_name.strip() or "Europe/Madrid",
        )
    st.session_state["last_odds_harvester_run"] = result
    if result.get("status") == "SUCCESS":
        st.success(
            f"Collection completed: {result['rows_normalized']} normalized matches "
            f"in {result['duration_seconds']:.1f} seconds."
        )
    elif result.get("status") == "EMPTY":
        st.info("The run completed but no normalized tennis matches were found.")
    else:
        st.error(result.get("error") or f"Run status: {result.get('status')}")

last_run = st.session_state.get("last_odds_harvester_run")
if last_run:
    with st.expander("Last job details", expanded=last_run.get("status") != "SUCCESS"):
        details = {
            key: value
            for key, value in last_run.items()
            if key not in {"data", "stdout", "stderr", "command"}
        }
        st.json(details)
        st.code(" ".join(last_run.get("command", [])), language="powershell")
        log_1, log_2 = st.columns(2)
        with log_1:
            st.caption("Standard output")
            st.code(last_run.get("stdout") or "No stdout", language="text")
        with log_2:
            st.caption("Standard error")
            st.code(last_run.get("stderr") or "No stderr", language="text")

st.subheader("3. Normalized market data")
cache = load_market_odds_cache(target_date)
summary = summarize_market_odds(cache)
metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
metric_1.metric("Matches", summary["matches"])
metric_2.metric("Complete prices", summary["complete_matches"])
metric_3.metric("Tournaments", summary["tournaments"])
metric_4.metric("Bookmakers", summary["bookmakers"])
metric_5.metric("Median overround", percentage(summary["median_overround"] - 1.0) if pd.notna(summary["median_overround"]) else "-")

control_1, control_2, control_3 = st.columns([2, 2, 1])
with control_1:
    tournament_options = sorted(cache.get("tournament", pd.Series(dtype="string")).dropna().astype(str).unique().tolist())
    tournament_filter = st.multiselect("Tournament", tournament_options)
with control_2:
    search_text = st.text_input("Search player", placeholder="Player name")
with control_3:
    complete_only = st.toggle("Complete only", value=True)

visible = cache.copy()
if tournament_filter and not visible.empty:
    visible = visible.loc[visible["tournament"].astype(str).isin(tournament_filter)]
if search_text.strip() and not visible.empty:
    query = search_text.strip().casefold()
    mask = (
        visible["player_1_name"].astype(str).str.casefold().str.contains(query, regex=False)
        | visible["player_2_name"].astype(str).str.casefold().str.contains(query, regex=False)
    )
    visible = visible.loc[mask]
if complete_only and not visible.empty:
    visible = visible.loc[visible["has_complete_match_winner_odds"].fillna(False)]

if visible.empty:
    st.info("No cached matches satisfy the selected date and filters.")
else:
    display = visible.copy()
    display["P1 No-vig"] = display["no_vig_probability_player_1"].map(percentage)
    display["P2 No-vig"] = display["no_vig_probability_player_2"].map(percentage)
    display["Overround"] = display["market_overround"].map(
        lambda value: percentage(value - 1.0) if pd.notna(value) else "-"
    )
    display["Odds P1"] = display["odds_player_1"].map(decimal_value)
    display["Odds P2"] = display["odds_player_2"].map(decimal_value)
    display_columns = [
        "match_date",
        "tournament",
        "player_1_name",
        "Odds P1",
        "bookmaker_player_1",
        "P1 No-vig",
        "player_2_name",
        "Odds P2",
        "bookmaker_player_2",
        "P2 No-vig",
        "Overround",
        "bookmaker_count",
        "fetched_at_utc",
    ]
    st.dataframe(
        display[[column for column in display_columns if column in display.columns]],
        hide_index=True,
        use_container_width=True,
    )

    csv_bytes = visible.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name=f"upcoming_odds_analysis_{target_date.isoformat()}.csv",
        mime="text/csv",
    )

with st.expander("Raw normalized rows"):
    st.dataframe(visible, hide_index=True, use_container_width=True)

st.subheader("4. Run history and storage")
st.caption(f"Market cache: {CACHE_PATH}")
st.caption(f"Run history: {RUNS_PATH}")
history = load_run_history(20)
if history.empty:
    st.info("No OddsHarvester runs have been recorded yet.")
else:
    history_columns = [
        "target_date", "status", "duration_seconds", "return_code",
        "rows_raw", "rows_normalized", "started_at_utc", "error",
    ]
    st.dataframe(
        history[[column for column in history_columns if column in history.columns]],
        hide_index=True,
        use_container_width=True,
    )

with st.expander("Maintenance", expanded=False):
    st.warning("Clearing the cache removes normalized market rows but keeps raw staging files and run history.")
    if st.button("Clear normalized market cache"):
        clear_market_odds_cache()
        st.session_state.pop("last_odds_harvester_run", None)
        st.rerun()

with st.expander("Method and limitations"):
    st.markdown(
        """
        - This page is an isolated laboratory. It does not update Upcoming Matches or model features.
        - `Preview only` is preferred for initial testing because it asks OddsHarvester for a smaller result.
        - Odds are normalized to decimal values above 1.0. Two complete prices are required to calculate no-vig probabilities and overround.
        - The highest detected price is retained when multiple outcome prices can be attributed to the same player.
        - Name, tournament and date normalization can fail when the upstream page changes its schema.
        - Empty results are valid when no matching events are published. Browser blocks, captchas and access restrictions are returned as errors and are not bypassed.
        - Market prices are observations from an external source. They do not imply a guaranteed outcome or financial return.
        """
    )
