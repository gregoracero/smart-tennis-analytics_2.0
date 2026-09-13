#!/usr/bin/env python3
"""Reconstruccion completa Sackmann + TML con actualizacion diaria local.

Flujo
-----
1. Lee los historicos TML de temporada.
2. Descubre en data/raw/tml_v2/daily_refresh los CSV diarios:
   - ongoing_tourneys.csv
   - ch_ongoing_tourney.csv o challenger_ongoing_tourneys.csv
   - 2026_atp_quali.csv, normalmente dentro de atp_quali/
3. Consolida historico + diario por familia, con prioridad para el CSV diario.
4. Deduplica conservadoramente solo partidos con mismo resultado y firma fuerte.
5. Ejecuta build_jeff_sackmann_with_odds_tml.py con los CSV consolidados.
6. Ejecuta postprocess_sackmann_tml_integrated_parquet.py.
7. Valida, genera informes, hace backup y promociona atomicamente.

Es idempotente: ejecutar varias veces con los mismos archivos produce la misma
consolidacion. No modifica el Parquet oficial si cualquier etapa falla.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow.parquet as pq


DAILY_FILE_PATTERNS = {
    "main": (
        "ongoing_tourneys.csv",
        "atp_ongoing_tourneys.csv",
    ),
    "challenger": (
        "ch_ongoing_tourney.csv",
        "challenger_ongoing_tourneys.csv",
        "ch_ongoing_tourneys.csv",
    ),
    "qualifying": (
        "*_atp_quali.csv",
        "atp_quali.csv",
    ),
}

COLUMN_ALIASES = {
    "date": (
        "date", "match_date", "td_date", "tourney_date", "tournament_date",
    ),
    "tournament_id": (
        "tourney_id", "tournament_id", "event_id", "tourneyid",
    ),
    "tournament_name": (
        "tourney_name", "tournament", "tournament_name", "event", "event_name",
    ),
    "round": (
        "round", "round_name", "match_round",
    ),
    "winner_id": (
        "winner_id", "winnerid", "w_id", "player_1_id",
    ),
    "loser_id": (
        "loser_id", "loserid", "l_id", "player_2_id",
    ),
    "winner_name": (
        "winner_name", "winner", "w_name", "player_1_name",
    ),
    "loser_name": (
        "loser_name", "loser", "l_name", "player_2_name",
    ),
    "score": (
        "score", "match_score", "result",
    ),
    "match_num": (
        "match_num", "match_number", "source_match_num", "match_id",
    ),
}


@dataclass
class ConsolidationResult:
    family: str
    output: Path
    historical_rows: int
    daily_rows: int
    combined_rows_before_dedup: int
    output_rows: int
    duplicate_rows_removed: int
    conflicting_signature_groups_preserved: int
    daily_files: list[str]
    columns: list[str]


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (
            (candidate / "data" / "raw").exists()
            and (candidate / "data" / "processed").exists()
            and (candidate / "ingestion").exists()
        ):
            return candidate
    raise FileNotFoundError(
        "No se encontro la raiz del proyecto con data/raw, data/processed e ingestion"
    )


def run_command(command: list[str], cwd: Path, label: str) -> None:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=cwd, check=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def canonical_column_map(columns: Iterable[str]) -> dict[str, str]:
    normalized = {normalize_column_name(column): column for column in columns}
    result: dict[str, str] = {}
    for semantic, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                result[semantic] = normalized[alias]
                break
    return result


def read_csv_robust(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                on_bad_lines="error",
            )
            frame.columns = [str(column).strip() for column in frame.columns]
            return frame
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")
    raise RuntimeError(f"No se pudo leer {path}: {' | '.join(errors)}")


def validate_daily_frame(frame: pd.DataFrame, path: Path, family: str) -> None:
    if frame.empty:
        raise ValueError(f"Archivo diario vacio: {path}")
    mapping = canonical_column_map(frame.columns)
    missing_names = not (
        {"winner_name", "loser_name"}.issubset(mapping)
        or {"winner_id", "loser_id"}.issubset(mapping)
    )
    if missing_names:
        raise ValueError(
            f"{family}: {path.name} no contiene ganador/perdedor reconocibles. "
            f"Columnas: {list(frame.columns)}"
        )
    if "tournament_name" not in mapping and "tournament_id" not in mapping:
        raise ValueError(
            f"{family}: {path.name} no contiene torneo reconocible. "
            f"Columnas: {list(frame.columns)}"
        )


def discover_daily_files(daily_dir: Path, family: str) -> list[Path]:
    if not daily_dir.exists():
        return []
    matches: list[Path] = []
    for pattern in DAILY_FILE_PATTERNS[family]:
        matches.extend(daily_dir.rglob(pattern))
    # Excluye outputs temporales y conserva orden estable.
    unique = {
        path.resolve()
        for path in matches
        if path.is_file() and not path.name.startswith(".") and ".consolidated." not in path.name
    }
    return sorted(unique, key=lambda path: str(path).casefold())


def normalized_date(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce").astype("Int64").astype("string")
    parsed_numeric = pd.to_datetime(numeric, format="%Y%m%d", errors="coerce")
    parsed_generic = pd.to_datetime(text, errors="coerce", dayfirst=False)
    parsed = parsed_numeric.fillna(parsed_generic)
    return parsed.dt.strftime("%Y-%m-%d").fillna(text.fillna(""))


def normalized_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].map(normalize_text).astype("string")


def make_signatures(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Devuelve firma amplia y firma de resultado.

    La firma amplia identifica fecha/torneo/ronda/pareja no ordenada. La firma
    de resultado anade ganador y perdedor en su orientacion real. Solo se
    deduplican firmas de resultado iguales. Si la firma amplia coincide pero el
    ganador cambia, las filas se conservan como conflicto potencial.
    """
    mapping = canonical_column_map(frame.columns)
    date = (
        normalized_date(frame[mapping["date"]])
        if "date" in mapping
        else pd.Series("", index=frame.index, dtype="string")
    )
    tournament = normalized_series(
        frame,
        mapping.get("tournament_id") or mapping.get("tournament_name"),
    )
    round_name = normalized_series(frame, mapping.get("round"))

    winner_id = normalized_series(frame, mapping.get("winner_id"))
    loser_id = normalized_series(frame, mapping.get("loser_id"))
    winner_name = normalized_series(frame, mapping.get("winner_name"))
    loser_name = normalized_series(frame, mapping.get("loser_name"))
    winner = winner_id.where(winner_id.ne(""), winner_name)
    loser = loser_id.where(loser_id.ne(""), loser_name)

    low = winner.where(winner.le(loser), loser)
    high = loser.where(winner.le(loser), winner)
    broad = date + "|" + tournament + "|" + round_name + "|" + low + "|" + high
    result = broad + "|W:" + winner + "|L:" + loser
    return broad.astype("string"), result.astype("string")


def align_frames(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    columns: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for column in frame.columns:
            if column not in seen:
                seen.add(column)
                columns.append(column)
    return [frame.reindex(columns=columns) for frame in frames]


def consolidate_family(
    family: str,
    historical_path: Path,
    daily_files: list[Path],
    output_path: Path,
) -> ConsolidationResult:
    historical = read_csv_robust(historical_path)
    historical["_refresh_source_priority"] = 0
    historical["_refresh_source_file"] = str(historical_path)
    historical["_refresh_source_row"] = range(len(historical))

    daily_frames: list[pd.DataFrame] = []
    for position, path in enumerate(daily_files, start=1):
        current = read_csv_robust(path)
        validate_daily_frame(current, path, family)
        current["_refresh_source_priority"] = 100 + position
        current["_refresh_source_file"] = str(path)
        current["_refresh_source_row"] = range(len(current))
        daily_frames.append(current)

    all_frames = align_frames([historical, *daily_frames])
    combined = pd.concat(all_frames, ignore_index=True, sort=False)
    broad, result = make_signatures(combined)
    combined["_refresh_broad_signature"] = broad
    combined["_refresh_result_signature"] = result

    source_result_counts = combined.groupby(
        "_refresh_broad_signature", dropna=False
    )["_refresh_result_signature"].nunique(dropna=False)
    conflicts = int(source_result_counts.gt(1).sum())

    # Diarios ganan a historicos. Dentro de diarios gana el ultimo archivo
    # lexicograficamente y su ultima fila. Solo se colapsa mismo resultado.
    combined = combined.sort_values(
        [
            "_refresh_source_priority",
            "_refresh_source_file",
            "_refresh_source_row",
        ],
        kind="mergesort",
    )
    before = len(combined)
    combined = combined.drop_duplicates(
        "_refresh_result_signature", keep="last"
    ).reset_index(drop=True)
    removed = before - len(combined)

    helper_columns = [
        column for column in combined.columns if column.startswith("_refresh_")
    ]
    output = combined.drop(columns=helper_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    verification = read_csv_robust(output_path)
    if len(verification) != len(output):
        raise RuntimeError(f"La consolidacion de {family} perdio filas al escribir")

    return ConsolidationResult(
        family=family,
        output=output_path,
        historical_rows=len(historical),
        daily_rows=sum(len(frame) for frame in daily_frames),
        combined_rows_before_dedup=before,
        output_rows=len(output),
        duplicate_rows_removed=removed,
        conflicting_signature_groups_preserved=conflicts,
        daily_files=[str(path) for path in daily_files],
        columns=list(output.columns),
    )


def validate_rebuilt_parquet(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    required = {
        "tourney_id", "competition_type", "match_num", "tourney_date", "round",
        "winner_id", "winner_name", "loser_id", "loser_name", "source_origin",
        "tournament_key", "canonical_tournament_name", "tourney_name_original",
    }
    missing = sorted(required - schema)
    if missing:
        raise RuntimeError(f"El Parquet reconstruido no contiene: {missing}")
    columns = [
        "tourney_id", "competition_type", "match_num", "tourney_date", "round",
        "winner_id", "winner_name", "loser_id", "loser_name", "source_origin",
        "tournament_key", "canonical_tournament_name",
    ]
    frame = pd.read_parquet(path, columns=columns, engine="pyarrow")
    operational_key = ["tourney_id", "competition_type", "match_num"]
    confirmed_key = [
        "tourney_id", "competition_type", "tourney_date", "round",
        "winner_id", "loser_id",
    ]
    operational_duplicates = int(frame.duplicated(operational_key).sum())
    same_result_duplicate_rows = int(frame.duplicated(confirmed_key).sum())
    duplicate_rows = frame.loc[frame.duplicated(confirmed_key, keep=False)]
    if duplicate_rows.empty:
        cross_source_groups = 0
        same_source_groups = 0
    else:
        source_counts = duplicate_rows.groupby(
            confirmed_key, dropna=False
        )["source_origin"].nunique()
        cross_source_groups = int(source_counts.gt(1).sum())
        same_source_groups = int(source_counts.eq(1).sum())

    identity_rules = {
        "alexander bublik": 122330,
        "alexander shevchenko": 207686,
        "aleksander shevchenko": 207686,
        "aleksandr shevchenko": 207686,
        "jerome kym": 208843,
        "moez echargui": 121411,
    }
    wrong_identity_rows = 0
    for side in ("winner", "loser"):
        names = frame[f"{side}_name"].map(normalize_text)
        ids = pd.to_numeric(frame[f"{side}_id"], errors="coerce")
        for name, expected_id in identity_rules.items():
            wrong_identity_rows += int((names.eq(name) & ids.ne(expected_id)).sum())

    canonical_name_counts = frame.groupby(
        "tournament_key", dropna=False
    )["canonical_tournament_name"].nunique(dropna=True)
    summary = {
        "rows": int(len(frame)),
        "columns": int(len(schema)),
        "operational_duplicates": operational_duplicates,
        "same_result_duplicate_rows": same_result_duplicate_rows,
        "same_source_duplicate_groups_kept": same_source_groups,
        "cross_source_duplicate_groups": cross_source_groups,
        "winner_id_nulls": int(frame.winner_id.isna().sum()),
        "loser_id_nulls": int(frame.loser_id.isna().sum()),
        "wrong_confirmed_identity_rows": wrong_identity_rows,
        "tournament_keys_with_multiple_canonical_names": int(
            canonical_name_counts.gt(1).sum()
        ),
        "source_origin_counts": {
            str(key): int(value)
            for key, value in frame.source_origin.value_counts(dropna=False).items()
        },
        "competition_type_counts": {
            str(key): int(value)
            for key, value in frame.competition_type.value_counts(dropna=False).items()
        },
        "max_tourney_date": int(
            pd.to_numeric(frame.tourney_date, errors="coerce").max()
        ),
    }
    fatal = {
        "operational_duplicates": operational_duplicates,
        "cross_source_duplicate_groups": cross_source_groups,
        "winner_id_nulls": summary["winner_id_nulls"],
        "loser_id_nulls": summary["loser_id_nulls"],
        "wrong_confirmed_identity_rows": wrong_identity_rows,
        "tournament_keys_with_multiple_canonical_names": summary[
            "tournament_keys_with_multiple_canonical_names"
        ],
    }
    failing = {key: value for key, value in fatal.items() if value != 0}
    if failing:
        raise RuntimeError(f"Validacion del candidato fallida: {failing}")
    return summary


def promote_atomically(staged: Path, official: Path, backup_dir: Path) -> Path | None:
    official.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = None
    if official.exists():
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup = backup_dir / f"{official.stem}.{timestamp}{official.suffix}"
        shutil.copy2(official, backup)
    os.replace(staged, official)
    return backup


def parse_args() -> argparse.Namespace:
    root = find_project_root(Path(__file__).resolve().parent)
    ingestion_v2 = root / "ingestion" / "v2"
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    parser = argparse.ArgumentParser(
        description="Reconstruye Sackmann + TML incorporando CSV diarios ongoing"
    )
    parser.add_argument("--builder", default=str(ingestion_v2 / "build_jeff_sackmann_with_odds_tml.py"))
    parser.add_argument("--postprocessor", default=str(ingestion_v2 / "postprocess_sackmann_tml_integrated_parquet.py"))
    parser.add_argument("--sackmann-dir", default=str(raw / "tennis_atp-master"))
    parser.add_argument("--odds-dir", default=str(raw / "tennis_data_co_uk_v2"))
    parser.add_argument("--tml-main", default=str(raw / "tml_v2" / "current_season" / "2026.csv"))
    parser.add_argument("--tml-qualifying", default=str(raw / "tml_v2" / "current_season" / "2026_atp_quali.csv"))
    parser.add_argument("--tml-challenger", default=str(raw / "tml_v2" / "current_season" / "2026_challenger.csv"))
    parser.add_argument("--daily-refresh-dir", default=str(raw / "tml_v2" / "daily_refresh"))
    parser.add_argument("--require-daily-files", action="store_true", help="Falla si falta alguna de las tres familias diarias")
    parser.add_argument("--tml-player-database", default=str(raw / "tml_v2" / "current_season" / "ATP_Database.csv"))
    parser.add_argument("--atp-players", default=str(raw / "tennis_atp-master" / "atp_players.csv"))
    parser.add_argument("--unmatched-player-policy", choices=["synthetic", "empty", "error"], default="synthetic")
    parser.add_argument("--output", default=str(processed / "jeff_sackmann_with_odds.parquet"))
    parser.add_argument("--report-dir", default=str(processed / "tml_integration_reports"))
    parser.add_argument("--staging-dir", default=str(processed / "rebuild_staging"))
    parser.add_argument("--backup-dir", default=str(processed / "parquet_backups"))
    parser.add_argument("--expected-cross-source-merges", type=int, default=-1)
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--keep-staging-on-success", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = find_project_root(Path(__file__).resolve().parent)
    builder = Path(args.builder).resolve()
    postprocessor = Path(args.postprocessor).resolve()
    official_output = Path(args.output).resolve()
    report_dir = Path(args.report_dir).resolve()
    staging_dir = Path(args.staging_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve()
    daily_dir = Path(args.daily_refresh_dir).resolve()

    for required in (builder, postprocessor):
        if not required.exists():
            raise FileNotFoundError(required)
    staging_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    staged_output = staging_dir / "jeff_sackmann_with_odds.rebuild.parquet"
    staged_reports = staging_dir / "reports"
    consolidated_dir = staging_dir / "daily_consolidated"
    staged_output.unlink(missing_ok=True)
    for directory in (staged_reports, consolidated_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    started = time.time()
    historical_paths = {
        "main": Path(args.tml_main).resolve(),
        "challenger": Path(args.tml_challenger).resolve(),
        "qualifying": Path(args.tml_qualifying).resolve(),
    }
    daily_files = {
        family: discover_daily_files(daily_dir, family)
        for family in ("main", "challenger", "qualifying")
    }
    if args.require_daily_files:
        missing = [family for family, paths in daily_files.items() if not paths]
        if missing:
            raise FileNotFoundError(
                f"Faltan archivos diarios para: {missing} en {daily_dir}"
            )

    print("\n" + "=" * 78)
    print("ETAPA 0/4: consolidacion historica + daily refresh TML")
    print("=" * 78)
    consolidation_results: list[ConsolidationResult] = []
    output_names = {
        "main": "2026.main.consolidated.csv",
        "challenger": "2026.challenger.consolidated.csv",
        "qualifying": "2026.qualifying.consolidated.csv",
    }
    for family in ("main", "challenger", "qualifying"):
        result = consolidate_family(
            family,
            historical_paths[family],
            daily_files[family],
            consolidated_dir / output_names[family],
        )
        consolidation_results.append(result)
        print(
            f"{family}: historico={result.historical_rows:,}; "
            f"diario={result.daily_rows:,}; salida={result.output_rows:,}; "
            f"duplicados eliminados={result.duplicate_rows_removed:,}; "
            f"conflictos conservados={result.conflicting_signature_groups_preserved:,}"
        )

    refresh_manifest = {
        "daily_refresh_dir": str(daily_dir),
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "families": [
            {
                **result.__dict__,
                "output": str(result.output),
                "daily_file_hashes": {
                    path: file_sha256(Path(path)) for path in result.daily_files
                },
            }
            for result in consolidation_results
        ],
    }
    (staged_reports / "daily_refresh_summary.json").write_text(
        json.dumps(refresh_manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    consolidated = {result.family: result.output for result in consolidation_results}
    builder_command = [
        sys.executable,
        str(builder),
        "--sackmann-dir", str(Path(args.sackmann_dir).resolve()),
        "--odds-dir", str(Path(args.odds_dir).resolve()),
        "--tml-main", str(consolidated["main"]),
        "--tml-qualifying", str(consolidated["qualifying"]),
        "--tml-challenger", str(consolidated["challenger"]),
        "--tml-player-database", str(Path(args.tml_player_database).resolve()),
        "--atp-players", str(Path(args.atp_players).resolve()),
        "--unmatched-player-policy", args.unmatched_player_policy,
        "--output", str(staged_output),
        "--report-dir", str(staged_reports),
    ]
    if args.skip_odds:
        builder_command.append("--skip-odds")
    run_command(
        builder_command,
        root,
        "ETAPA 1/4: construccion completa Sackmann + TML diario + cuotas",
    )
    if not staged_output.exists():
        raise RuntimeError("El builder termino sin crear el Parquet temporal")

    postprocess_command = [
        sys.executable,
        str(postprocessor),
        "--parquet", str(staged_output),
        "--report-dir", str(staged_reports),
        "--expected-cross-source-merges", str(args.expected_cross_source_merges),
    ]
    run_command(
        postprocess_command,
        root,
        "ETAPA 2/4: reparacion de identidades, duplicados y torneos",
    )

    print("\n" + "=" * 78)
    print("ETAPA 3/4: validacion final")
    print("=" * 78)
    validation = validate_rebuilt_parquet(staged_output)

    print("\n" + "=" * 78)
    print("ETAPA 4/4: promocion atomica")
    print("=" * 78)
    builder_summary = read_json(staged_reports / "integration_summary.json")
    postprocess_summary = read_json(staged_reports / "upstream_postprocess_summary.json")
    backup = promote_atomically(staged_output, official_output, backup_dir)
    for source in staged_reports.glob("*"):
        if source.is_file():
            shutil.copy2(source, report_dir / source.name)

    final_summary = {
        "status": "success",
        "official_output": str(official_output),
        "backup": str(backup) if backup else None,
        "elapsed_seconds": round(time.time() - started, 2),
        "daily_refresh": refresh_manifest,
        "builder_summary": builder_summary,
        "postprocess_summary": postprocess_summary,
        "final_validation": validation,
    }
    summary_path = report_dir / "full_rebuild_summary.json"
    summary_path.write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    if not args.keep_staging_on_success:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print("\n" + json.dumps(final_summary, indent=2, ensure_ascii=False, default=str))
    print(f"\n[OK] Parquet oficial reconstruido: {official_output}")
    if backup:
        print(f"Backup anterior: {backup}")
    print(f"Resumen: {summary_path}")
    print(
        "\nSiguientes pasos:\n"
        "  python modeling/build_player_match_features_tml_sackmann_elo_v6_markets.py\n"
        "  python modeling/build_clean_tennis_ml_dataset_sackmann_v6_markets.py"
    )


if __name__ == "__main__":
    main()

