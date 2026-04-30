from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from stt_tts_benchmark.adapters import GeminiTTSAdapter, KokoroTTSAdapter, PiperTTSAdapter, Qwen3TTSAdapter
from stt_tts_benchmark.audio import safe_filename
from stt_tts_benchmark.config import AppConfig
from stt_tts_benchmark.database import init_db, insert_results
from stt_tts_benchmark.history import render_history
from stt_tts_benchmark.models import BenchmarkResult


DEFAULT_ENGLISH_DIALOGUE = (
    "Hello, thanks for calling. I can help you check your order status today. "
    "Could you please confirm your email address?"
)


st.set_page_config(page_title="TTS Comparison", layout="wide")

config = AppConfig.from_env()
init_db(config.database_path)

st.title("TTS Comparison Benchmark")

with st.sidebar:
    st.subheader("Runtime")
    selected_models = st.multiselect(
        "Models",
        ["Kokoro", "Piper", "Qwen3-TTS", "Gemini TTS"],
        default=["Kokoro", "Piper", "Qwen3-TTS", "Gemini TTS"],
    )
    kokoro_voice = st.text_input("Kokoro voice", value=config.kokoro_voice)
    kokoro_lang_code = st.text_input("Kokoro language code", value=config.kokoro_lang_code)
    piper_model_path = st.text_input("Piper model path", value=str(config.piper_model_path))
    qwen3_command = st.text_area("Qwen3-TTS command template", value=config.qwen3_tts_command, height=120)
    qwen3_model_id = st.text_input("Qwen3-TTS model id", value=config.qwen3_tts_model_id)
    qwen3_language = st.text_input("Qwen3-TTS language", value=config.qwen3_tts_language)
    qwen3_speaker = st.text_input("Qwen3-TTS speaker", value=config.qwen3_tts_speaker)
    qwen3_instruct = st.text_input("Qwen3-TTS instruct", value=config.qwen3_tts_instruct)
    gemini_model = st.text_input("Gemini model", value=config.gemini_tts_model)
    gemini_voice = st.text_input("Gemini voice", value=config.gemini_tts_voice)

dialogue_text = st.text_area("Dialogue text", value=DEFAULT_ENGLISH_DIALOGUE, height=180)

st.subheader("Manual TTS scores")
score_cols = st.columns(4)
tone_score = score_cols[0].slider("Tone", 1, 5, 3)
intonation_score = score_cols[1].slider("Intonation", 1, 5, 3)
naturalness_score = score_cols[2].slider("Naturalness", 1, 5, 3)
pronunciation_score = score_cols[3].slider("Pronunciation", 1, 5, 3)
notes = st.text_area("Notes", height=100)


def _adapter_for(model_name: str):
    if model_name == "Kokoro":
        return KokoroTTSAdapter(voice=kokoro_voice, lang_code=kokoro_lang_code)
    if model_name == "Piper":
        return PiperTTSAdapter(executable=config.piper_executable, model_path=Path(piper_model_path))
    if model_name == "Qwen3-TTS":
        return Qwen3TTSAdapter(
            command_template=qwen3_command,
            workdir=config.qwen3_tts_workdir,
            model_path=config.qwen3_tts_model_path,
            model_id=qwen3_model_id,
            language=qwen3_language,
            speaker=qwen3_speaker,
            instruct=qwen3_instruct,
        )
    if model_name == "Gemini TTS":
        return GeminiTTSAdapter(api_key=config.gemini_api_key, model=gemini_model, voice=gemini_voice)
    raise ValueError(f"Unsupported model: {model_name}")


if st.button("Run TTS comparison", type="primary", disabled=not selected_models or not dialogue_text.strip()):
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    text_slug = safe_filename(dialogue_text[:32])
    results: list[BenchmarkResult] = []
    preview_rows = []

    for model_name in selected_models:
        adapter = _adapter_for(model_name)
        output_path = config.output_dir / "tts" / f"{timestamp}_{safe_filename(model_name)}_{text_slug}.wav"
        try:
            output = adapter.synthesize(dialogue_text, output_path)
            realtime_factor = (
                output.latency_seconds / output.audio_duration_seconds
                if output.audio_duration_seconds and output.audio_duration_seconds > 0
                else None
            )
            result = BenchmarkResult(
                page_type="tts",
                model_name=adapter.model_name,
                provider=adapter.provider,
                is_local=adapter.is_local,
                input_text=dialogue_text,
                output_audio_path=str(output.output_audio_path),
                latency_seconds=output.latency_seconds,
                audio_duration_seconds=output.audio_duration_seconds,
                realtime_factor=realtime_factor,
                tone_score=tone_score,
                intonation_score=intonation_score,
                naturalness_score=naturalness_score,
                pronunciation_score=pronunciation_score,
                notes=notes,
            )
            preview_rows.append(
                {
                    "model": adapter.model_name,
                    "status": "success",
                    "latency_seconds": output.latency_seconds,
                    "audio_duration_seconds": output.audio_duration_seconds,
                    "realtime_factor": realtime_factor,
                    "output_audio_path": str(output.output_audio_path),
                    "error": "",
                }
            )
        except Exception as exc:
            result = BenchmarkResult(
                page_type="tts",
                model_name=adapter.model_name,
                provider=adapter.provider,
                is_local=adapter.is_local,
                input_text=dialogue_text,
                output_audio_path=str(output_path),
                tone_score=tone_score,
                intonation_score=intonation_score,
                naturalness_score=naturalness_score,
                pronunciation_score=pronunciation_score,
                notes=notes,
                status="error",
                error_message=str(exc),
            )
            preview_rows.append(
                {
                    "model": adapter.model_name,
                    "status": "error",
                    "latency_seconds": None,
                    "audio_duration_seconds": None,
                    "realtime_factor": None,
                    "output_audio_path": str(output_path),
                    "error": str(exc),
                }
            )
        results.append(result)

    insert_results(config.database_path, results)
    st.success(f"Saved {len(results)} TTS result rows.")
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)
    for row in preview_rows:
        if row["status"] == "success":
            st.audio(row["output_audio_path"])

render_history(
    database_path=config.database_path,
    export_dir=config.export_dir,
    page_type="tts",
    title="Recent TTS results",
    project_root=Path.cwd(),
    audio_column="output_audio_path",
)
