from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from stt_tts_benchmark.audio import wav_duration_seconds


@dataclass
class STTOutput:
    transcript: str
    latency_seconds: float
    audio_duration_seconds: float | None


class WhisperAdapter:
    model_name = "Whisper"
    provider = "faster-whisper"
    is_local = True

    def __init__(self, model_size: str, device: str, compute_type: str, language: str) -> None:
        self.model_size = model_size
        self.device = "auto" if device == "auto" else device
        self.compute_type = compute_type
        self.language = language or None

    def transcribe(self, audio_path: Path) -> STTOutput:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed. Install with: python -m pip install -e .[whisper]") from exc

        model_kwargs = {"device": self.device}
        if self.compute_type != "default":
            model_kwargs["compute_type"] = self.compute_type
        model = WhisperModel(self.model_size, **model_kwargs)

        started_at = time.perf_counter()
        segments, _info = model.transcribe(str(audio_path), language=self.language)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        latency = time.perf_counter() - started_at
        return STTOutput(
            transcript=transcript,
            latency_seconds=latency,
            audio_duration_seconds=wav_duration_seconds(audio_path),
        )

