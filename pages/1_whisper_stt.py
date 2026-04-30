from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from stt_tts_benchmark.adapters import WhisperAdapter
from stt_tts_benchmark.audio import file_sha256
from stt_tts_benchmark.config import AppConfig
from stt_tts_benchmark.database import init_db, insert_results
from stt_tts_benchmark.history import render_history
from stt_tts_benchmark.metrics import character_error_rate, normalize_text, word_error_rate
from stt_tts_benchmark.models import BenchmarkResult


st.set_page_config(page_title="Whisper STT", layout="wide")

config = AppConfig.from_env()
init_db(config.database_path)

st.title("Whisper STT Benchmark")

with st.sidebar:
    st.subheader("Runtime")
    model_size = st.text_input("Whisper model size", value=config.whisper_model_size)
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=["auto", "cpu", "cuda"].index(config.device if config.device in {"auto", "cpu", "cuda"} else "auto"))
    compute_type = st.text_input("Compute type", value=config.whisper_compute_type)
    language = st.text_input("Language", value=config.language)
    st.caption(f"SQLite: {config.database_path}")

uploaded_audio = st.file_uploader("Dialogue audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
reference_text = st.text_area("Reference transcript", height=160)

if st.button("Run Whisper STT", type="primary", disabled=uploaded_audio is None):
    assert uploaded_audio is not None
    suffix = Path(uploaded_audio.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_audio.getbuffer())
        audio_path = Path(temp_file.name)

    adapter = WhisperAdapter(model_size=model_size, device=device, compute_type=compute_type, language=language)
    try:
        output = adapter.transcribe(audio_path)
        normalized_reference = normalize_text(reference_text)
        normalized_transcript = normalize_text(output.transcript)
        wer = word_error_rate(reference_text, output.transcript) if reference_text.strip() else None
        cer = character_error_rate(reference_text, output.transcript) if reference_text.strip() else None
        realtime_factor = (
            output.latency_seconds / output.audio_duration_seconds
            if output.audio_duration_seconds and output.audio_duration_seconds > 0
            else None
        )
        result = BenchmarkResult(
            page_type="stt",
            model_name=f"Whisper {model_size}",
            provider=adapter.provider,
            is_local=True,
            input_audio_path=str(audio_path),
            input_audio_hash=file_sha256(audio_path),
            transcript=output.transcript,
            normalized_reference=normalized_reference,
            normalized_transcript=normalized_transcript,
            latency_seconds=output.latency_seconds,
            audio_duration_seconds=output.audio_duration_seconds,
            realtime_factor=realtime_factor,
            wer=wer,
            cer=cer,
        )
        insert_results(config.database_path, [result])
        st.success("STT result saved.")
        st.text_area("Transcript", value=output.transcript, height=180)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Latency", f"{output.latency_seconds:.2f}s")
        metric_cols[1].metric("Audio duration", f"{output.audio_duration_seconds:.2f}s" if output.audio_duration_seconds else "N/A")
        metric_cols[2].metric("WER", f"{wer:.3f}" if wer is not None else "N/A")
        metric_cols[3].metric("CER", f"{cer:.3f}" if cer is not None else "N/A")
    except Exception as exc:
        result = BenchmarkResult(
            page_type="stt",
            model_name=f"Whisper {model_size}",
            provider=adapter.provider,
            is_local=True,
            input_audio_path=str(audio_path),
            input_audio_hash=file_sha256(audio_path),
            status="error",
            error_message=str(exc),
        )
        insert_results(config.database_path, [result])
        st.error(str(exc))

render_history(
    database_path=config.database_path,
    export_dir=config.export_dir,
    page_type="stt",
    title="Recent STT results",
    project_root=Path.cwd(),
    audio_column="input_audio_path",
)
