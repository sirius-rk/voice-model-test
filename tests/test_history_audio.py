from __future__ import annotations

from pathlib import Path

import stt_tts_benchmark.audio as audio
from stt_tts_benchmark.audio import persist_uploaded_audio
from stt_tts_benchmark.database import fetch_results, init_db, insert_results
from stt_tts_benchmark.history import _audio_display_name, _matches_search
from stt_tts_benchmark.models import BenchmarkResult


class _UploadedFile:
    name = "calls/customer one.wav"

    def getbuffer(self) -> memoryview:
        return memoryview(b"audio bytes")


def test_persist_uploaded_audio_keeps_original_name_and_safe_path(tmp_path: Path) -> None:
    stored_path, original_name = persist_uploaded_audio(_UploadedFile(), tmp_path / "inputs")

    assert original_name == "customer one.wav"
    assert stored_path.name.endswith("_customer_one.wav")
    assert stored_path.read_bytes() == b"audio bytes"


def test_history_stores_and_searches_uploaded_filename(tmp_path: Path) -> None:
    database_path = tmp_path / "results.db"
    init_db(database_path)
    insert_results(
        database_path,
        [
            BenchmarkResult(
                page_type="stt",
                model_name="Qwen3-ASR",
                provider="qwen3-asr-local",
                is_local=True,
                input_audio_path=str(tmp_path / "stored.wav"),
                input_audio_name="customer one.wav",
                transcript="The order is ready.",
            )
        ],
    )

    row = fetch_results(database_path, page_type="stt")[0]
    assert row["input_audio_name"] == "customer one.wav"
    assert _audio_display_name(row, "input_audio_path") == "customer one.wav"
    assert _matches_search(row, "input_audio_path", "order is ready")
    assert _matches_search(row, "input_audio_path", "customer one")


def test_convert_to_mono_wav_uses_safe_ffmpeg_arguments(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"encoded audio")
    commands: list[list[str]] = []

    monkeypatch.setattr(audio.shutil, "which", lambda _: "ffmpeg")

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"wav audio")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    converted = audio.convert_to_mono_wav(source, tmp_path / "inputs", channel="left")

    assert converted.name == "source_left_mono.wav"
    assert converted.read_bytes() == b"wav audio"
    assert not source.exists()
    assert "-ac" in commands[0] and commands[0][commands[0].index("-ac") + 1] == "1"
    assert "-ar" in commands[0] and commands[0][commands[0].index("-ar") + 1] == "16000"
    assert "-af" in commands[0] and commands[0][commands[0].index("-af") + 1] == "pan=mono|c0=FL"


def test_convert_to_mono_wav_rejects_unknown_channel(tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"encoded audio")

    try:
        audio.convert_to_mono_wav(source, tmp_path / "inputs", channel="both")
    except ValueError as exc:
        assert str(exc) == "channel must be one of: mix, left, right"
    else:
        raise AssertionError("Expected invalid channel to raise ValueError")
