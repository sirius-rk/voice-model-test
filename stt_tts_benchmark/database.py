from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from stt_tts_benchmark.models import BenchmarkResult


CREATE_RESULTS_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_results (
    brs_id INTEGER PRIMARY KEY AUTOINCREMENT,
    brs_created_at TEXT NOT NULL,
    brs_page_type TEXT NOT NULL,
    brs_model_name TEXT NOT NULL,
    brs_provider TEXT NOT NULL,
    brs_is_local INTEGER NOT NULL,
    brs_input_text TEXT NOT NULL DEFAULT '',
    brs_input_audio_path TEXT NOT NULL DEFAULT '',
    brs_input_audio_hash TEXT NOT NULL DEFAULT '',
    brs_output_audio_path TEXT NOT NULL DEFAULT '',
    brs_transcript TEXT NOT NULL DEFAULT '',
    brs_normalized_reference TEXT NOT NULL DEFAULT '',
    brs_normalized_transcript TEXT NOT NULL DEFAULT '',
    brs_latency_seconds REAL,
    brs_audio_duration_seconds REAL,
    brs_realtime_factor REAL,
    brs_wer REAL,
    brs_cer REAL,
    brs_tone_score INTEGER,
    brs_intonation_score INTEGER,
    brs_naturalness_score INTEGER,
    brs_pronunciation_score INTEGER,
    brs_notes TEXT NOT NULL DEFAULT '',
    brs_status TEXT NOT NULL,
    brs_error_message TEXT NOT NULL DEFAULT ''
);
"""

INSERT_RESULTS_SQL = """
INSERT INTO benchmark_results (
    brs_created_at,
    brs_page_type,
    brs_model_name,
    brs_provider,
    brs_is_local,
    brs_input_text,
    brs_input_audio_path,
    brs_input_audio_hash,
    brs_output_audio_path,
    brs_transcript,
    brs_normalized_reference,
    brs_normalized_transcript,
    brs_latency_seconds,
    brs_audio_duration_seconds,
    brs_realtime_factor,
    brs_wer,
    brs_cer,
    brs_tone_score,
    brs_intonation_score,
    brs_naturalness_score,
    brs_pronunciation_score,
    brs_notes,
    brs_status,
    brs_error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

CSV_COLUMNS = [
    "id",
    "created_at",
    "page_type",
    "model_name",
    "provider",
    "is_local",
    "input_text",
    "input_audio_path",
    "input_audio_hash",
    "output_audio_path",
    "transcript",
    "normalized_reference",
    "normalized_transcript",
    "latency_seconds",
    "audio_duration_seconds",
    "realtime_factor",
    "wer",
    "cer",
    "tone_score",
    "intonation_score",
    "naturalness_score",
    "pronunciation_score",
    "notes",
    "status",
    "error_message",
]


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(database_path)


def init_db(database_path: Path) -> None:
    with connect(database_path) as conn:
        conn.execute(CREATE_RESULTS_SQL)
        conn.commit()


def _result_values(result: BenchmarkResult) -> tuple[object, ...]:
    row = result.to_row()
    return (
        row["created_at"],
        row["page_type"],
        row["model_name"],
        row["provider"],
        1 if row["is_local"] else 0,
        row["input_text"],
        row["input_audio_path"],
        row["input_audio_hash"],
        row["output_audio_path"],
        row["transcript"],
        row["normalized_reference"],
        row["normalized_transcript"],
        row["latency_seconds"],
        row["audio_duration_seconds"],
        row["realtime_factor"],
        row["wer"],
        row["cer"],
        row["tone_score"],
        row["intonation_score"],
        row["naturalness_score"],
        row["pronunciation_score"],
        row["notes"],
        row["status"],
        row["error_message"],
    )


def insert_results(database_path: Path, results: list[BenchmarkResult]) -> None:
    if not results:
        return
    with connect(database_path) as conn:
        conn.executemany(INSERT_RESULTS_SQL, [_result_values(result) for result in results])
        conn.commit()


def fetch_results(database_path: Path, page_type: str | None = None, limit: int = 200) -> list[dict[str, object]]:
    init_db(database_path)
    where = ""
    params: list[object] = []
    if page_type:
        where = "WHERE brs_page_type = ?"
        params.append(page_type)
    params.append(limit)
    query = f"""
        SELECT
            brs_id,
            brs_created_at,
            brs_page_type,
            brs_model_name,
            brs_provider,
            brs_is_local,
            brs_input_text,
            brs_input_audio_path,
            brs_input_audio_hash,
            brs_output_audio_path,
            brs_transcript,
            brs_normalized_reference,
            brs_normalized_transcript,
            brs_latency_seconds,
            brs_audio_duration_seconds,
            brs_realtime_factor,
            brs_wer,
            brs_cer,
            brs_tone_score,
            brs_intonation_score,
            brs_naturalness_score,
            brs_pronunciation_score,
            brs_notes,
            brs_status,
            brs_error_message
        FROM benchmark_results
        {where}
        ORDER BY brs_id DESC
        LIMIT ?;
    """
    with connect(database_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(zip(CSV_COLUMNS, row, strict=True)) for row in rows]


def delete_results(database_path: Path, result_ids: list[int]) -> None:
    if not result_ids:
        return
    placeholders = ",".join("?" for _ in result_ids)
    with connect(database_path) as conn:
        conn.execute(f"DELETE FROM benchmark_results WHERE brs_id IN ({placeholders})", result_ids)
        conn.commit()


def export_results_csv(database_path: Path, export_dir: Path, page_type: str | None = None) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    suffix = page_type or "all"
    output_path = export_dir / f"benchmark_results_{suffix}.csv"
    rows = fetch_results(database_path, page_type=page_type, limit=100000)
    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
