from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime


@dataclass
class BenchmarkResult:
    page_type: str
    model_name: str
    provider: str
    is_local: bool
    input_text: str = ""
    input_audio_path: str = ""
    input_audio_hash: str = ""
    output_audio_path: str = ""
    transcript: str = ""
    normalized_reference: str = ""
    normalized_transcript: str = ""
    latency_seconds: float | None = None
    audio_duration_seconds: float | None = None
    realtime_factor: float | None = None
    wer: float | None = None
    cer: float | None = None
    tone_score: int | None = None
    intonation_score: int | None = None
    naturalness_score: int | None = None
    pronunciation_score: int | None = None
    notes: str = ""
    status: str = "success"
    error_message: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_row(self) -> dict[str, object]:
        return asdict(self)

