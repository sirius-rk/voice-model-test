from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stt_tts_benchmark.audio import wav_duration_seconds


@dataclass
class STTOutput:
    transcript: str
    latency_seconds: float
    audio_duration_seconds: float | None
    language: str | None = None


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


class Qwen3ASRAdapter:
    model_name = "Qwen3-ASR"
    provider = "qwen3-asr-local"
    is_local = True
    _model_cache: dict[tuple[str, str, str, int], Any] = {}

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "auto",
        language: str = "auto",
        max_new_tokens: int = 4096,
    ) -> None:
        self.model_id = model_id.strip()
        self.device = device.strip().lower() or "auto"
        self.dtype = dtype.strip().lower() or "auto"
        normalized_language = language.strip()
        self.language = None if normalized_language.lower() in {"", "auto", "none"} else normalized_language
        self.max_new_tokens = max(1, int(max_new_tokens))

    def transcribe(self, audio_path: Path) -> STTOutput:
        model = self._load_model()
        started_at = time.perf_counter()
        results = model.transcribe(audio=str(audio_path), language=self.language)
        if not results:
            raise RuntimeError("Qwen3-ASR returned no transcription result.")
        result = results[0]
        transcript = str(getattr(result, "text", "") or "").strip()
        language = str(getattr(result, "language", "") or "").strip() or None
        return STTOutput(
            transcript=transcript,
            latency_seconds=time.perf_counter() - started_at,
            audio_duration_seconds=wav_duration_seconds(audio_path),
            language=language,
        )

    def _load_model(self) -> Any:
        if not self.model_id:
            raise RuntimeError("Qwen3-ASR model id or local path is empty.")
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-ASR local dependencies are not installed. "
                "Install with: python -m pip install -e .[qwen-asr]"
            ) from exc

        resolved_device = self._resolve_device(torch)
        effective_dtype = self._effective_dtype(resolved_device)
        cache_key = (self.model_id, resolved_device, effective_dtype, self.max_new_tokens)
        if cache_key not in self._model_cache:
            model_kwargs: dict[str, Any] = {
                "device_map": "cuda:0" if resolved_device == "cuda" else "cpu",
                "dtype": getattr(torch, effective_dtype),
                "max_inference_batch_size": 1,
                "max_new_tokens": self.max_new_tokens,
            }
            self._model_cache[cache_key] = Qwen3ASRModel.from_pretrained(self.model_id, **model_kwargs)
        return self._model_cache[cache_key]

    def _resolve_device(self, torch: Any) -> str:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("Qwen3-ASR device must be auto, cpu, or cuda.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Qwen3-ASR CUDA was requested, but torch cannot access a CUDA device.")
        return "cuda" if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()) else "cpu"

    def _effective_dtype(self, resolved_device: str) -> str:
        if resolved_device == "cpu":
            return "float32"
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise RuntimeError("Qwen3-ASR dtype must be auto, float16, bfloat16, or float32.")
        return "bfloat16" if self.dtype == "auto" else self.dtype
