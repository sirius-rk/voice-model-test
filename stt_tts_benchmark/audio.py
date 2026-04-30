from __future__ import annotations

import hashlib
import wave
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, OSError):
        return None


def safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return cleaned[:80] or "audio"

