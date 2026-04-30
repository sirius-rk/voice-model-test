from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _path_env(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    database_path: Path
    export_dir: Path
    device: str
    language: str
    whisper_model_size: str
    whisper_compute_type: str
    kokoro_voice: str
    kokoro_lang_code: str
    piper_executable: str
    piper_model_path: Path
    qwen3_tts_command: str
    qwen3_tts_workdir: Path | None
    qwen3_tts_model_path: Path
    qwen3_tts_model_id: str
    qwen3_tts_language: str
    qwen3_tts_speaker: str
    qwen3_tts_instruct: str
    gemini_api_key: str
    gemini_tts_model: str
    gemini_tts_voice: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        workdir = os.getenv("QWEN3_TTS_WORKDIR", "").strip()
        return cls(
            output_dir=_path_env("APP_OUTPUT_DIR", "outputs"),
            database_path=_path_env("APP_DATABASE_PATH", "data/results.db"),
            export_dir=_path_env("APP_EXPORT_DIR", "exports"),
            device=os.getenv("APP_DEVICE", "auto"),
            language=os.getenv("APP_LANGUAGE", "zh"),
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "default"),
            kokoro_voice=os.getenv("KOKORO_VOICE", "zf_xiaoxiao"),
            kokoro_lang_code=os.getenv("KOKORO_LANG_CODE", "z"),
            piper_executable=os.getenv("PIPER_EXECUTABLE", "piper"),
            piper_model_path=_path_env("PIPER_MODEL_PATH", "models/piper/voice.onnx"),
            qwen3_tts_command=os.getenv("QWEN3_TTS_COMMAND", "").strip(),
            qwen3_tts_workdir=Path(workdir).expanduser() if workdir else None,
            qwen3_tts_model_path=_path_env("QWEN3_TTS_MODEL_PATH", "models/qwen3-tts"),
            qwen3_tts_model_id=os.getenv("QWEN3_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            qwen3_tts_language=os.getenv("QWEN3_TTS_LANGUAGE", "Chinese"),
            qwen3_tts_speaker=os.getenv("QWEN3_TTS_SPEAKER", "Vivian"),
            qwen3_tts_instruct=os.getenv("QWEN3_TTS_INSTRUCT", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_tts_model=os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts"),
            gemini_tts_voice=os.getenv("GEMINI_TTS_VOICE", "Kore"),
        )
