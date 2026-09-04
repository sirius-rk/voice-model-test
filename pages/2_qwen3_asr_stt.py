from __future__ import annotations

from pathlib import Path

import streamlit as st

from stt_tts_benchmark.adapters import Qwen3ASRAdapter
from stt_tts_benchmark.audio import convert_to_mono_wav, file_sha256, persist_uploaded_audio
from stt_tts_benchmark.config import AppConfig
from stt_tts_benchmark.database import init_db, insert_results
from stt_tts_benchmark.history import render_history
from stt_tts_benchmark.metrics import character_error_rate, normalize_text, word_error_rate
from stt_tts_benchmark.models import BenchmarkResult


st.set_page_config(page_title="Qwen3-ASR STT", layout="wide")

config = AppConfig.from_env()
init_db(config.database_path)

st.title("Qwen3-ASR STT Benchmark")
st.caption("Local Qwen3-ASR model inference through the Transformers backend.")

with st.sidebar:
    st.subheader("Local model runtime")
    model_id = st.text_input("Model id or local path", value=config.qwen3_asr_model_id)
    device_options = ["auto", "cpu", "cuda"]
    configured_device = config.qwen3_asr_device.lower()
    device = st.selectbox(
        "Device",
        device_options,
        index=device_options.index(configured_device) if configured_device in device_options else 0,
    )
    dtype_options = ["auto", "bfloat16", "float16", "float32"]
    configured_dtype = config.qwen3_asr_dtype.lower()
    dtype = st.selectbox(
        "Precision",
        dtype_options,
        index=dtype_options.index(configured_dtype) if configured_dtype in dtype_options else 0,
        help="CPU inference uses float32 for compatibility.",
    )
    language = st.text_input(
        "Language (Auto or model language name)",
        value=config.qwen3_asr_language,
        help="Use Auto for language detection, or values such as Chinese and English.",
    )
    max_new_tokens = st.number_input(
        "Max new tokens",
        min_value=64,
        max_value=4096,
        value=config.qwen3_asr_max_new_tokens,
        step=64,
    )
    st.caption(f"SQLite: {config.database_path}")

uploaded_audio = st.file_uploader("Dialogue audio", type=["wav", "mp3", "m4a", "flac", "ogg"])
convert_to_mono = st.checkbox(
    "Convert to mono WAV before transcription",
    help="Uses FFmpeg to create a 16 kHz mono PCM WAV. The original filename is retained in history.",
)
channel_labels = {
    "Mix both channels": "mix",
    "Left channel": "left",
    "Right channel": "right",
}
channel_label = st.selectbox(
    "Mono channel",
    list(channel_labels),
    disabled=not convert_to_mono,
    help="Use Left or Right when the stereo recording contains separate speaker feeds. Mixing can cancel speech.",
)
reference_text = st.text_area("Reference transcript", height=160)

if st.button("Run Qwen3-ASR", type="primary", disabled=uploaded_audio is None):
    assert uploaded_audio is not None
    audio_path, audio_name = persist_uploaded_audio(uploaded_audio, config.output_dir / "inputs")

    adapter = Qwen3ASRAdapter(
        model_id=model_id,
        device=device,
        dtype=dtype,
        language=language,
        max_new_tokens=int(max_new_tokens),
    )
    try:
        with st.spinner("Preparing audio, loading the local model, and transcribing..."):
            if convert_to_mono:
                audio_path = convert_to_mono_wav(
                    audio_path,
                    config.output_dir / "inputs",
                    config.ffmpeg_executable,
                    channel=channel_labels[channel_label],
                )
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
            model_name=f"Qwen3-ASR {model_id}",
            provider=adapter.provider,
            is_local=adapter.is_local,
            input_audio_path=str(audio_path),
            input_audio_name=audio_name,
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
        st.success("Qwen3-ASR result saved.")
        if output.language:
            st.caption(f"Detected language: {output.language}")
        st.text_area("Transcript", value=output.transcript, height=180)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Latency", f"{output.latency_seconds:.2f}s")
        metric_cols[1].metric("Audio duration", f"{output.audio_duration_seconds:.2f}s" if output.audio_duration_seconds else "N/A")
        metric_cols[2].metric("WER", f"{wer:.3f}" if wer is not None else "N/A")
        metric_cols[3].metric("CER", f"{cer:.3f}" if cer is not None else "N/A")
    except Exception as exc:
        result = BenchmarkResult(
            page_type="stt",
            model_name=f"Qwen3-ASR {model_id}",
            provider=adapter.provider,
            is_local=adapter.is_local,
            input_audio_path=str(audio_path),
            input_audio_name=audio_name,
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
    key_prefix="qwen3_asr_stt",
)
