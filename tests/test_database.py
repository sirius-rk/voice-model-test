from pathlib import Path

from stt_tts_benchmark.database import fetch_results, init_db, insert_results
from stt_tts_benchmark.models import BenchmarkResult


def test_database_insert_and_fetch(tmp_path: Path) -> None:
    database_path = tmp_path / "results.db"
    init_db(database_path)
    insert_results(
        database_path,
        [
            BenchmarkResult(
                page_type="stt",
                model_name="Whisper small",
                provider="faster-whisper",
                is_local=True,
                transcript="hello",
            )
        ],
    )

    rows = fetch_results(database_path, page_type="stt")

    assert len(rows) == 1
    assert rows[0]["model_name"] == "Whisper small"

