from pathlib import Path

import streamlit as st

from stt_tts_benchmark.config import AppConfig
from stt_tts_benchmark.database import init_db


st.set_page_config(
    page_title="STT/TTS Benchmark",
    page_icon="🎙️",
    layout="wide",
)

config = AppConfig.from_env()
init_db(config.database_path)

st.title("STT/TTS Benchmark")
st.caption("Local-first speech recognition and speech synthesis benchmark workspace.")

st.write(
    "Use the pages in the sidebar to compare Whisper STT accuracy, TTS prosody, "
    "and model response speed. Results are persisted to SQLite and can be exported as CSV."
)

col1, col2, col3 = st.columns(3)
col1.metric("Database", str(config.database_path))
col2.metric("Output dir", str(config.output_dir))
col3.metric("Export dir", str(config.export_dir))

st.subheader("Setup")
st.code(
    "\n".join(
        [
            "python -m pip install -e .",
            "python -m pip install -e .[whisper,gemini,kokoro]",
            "streamlit run streamlit_app.py",
        ]
    ),
    language="powershell",
)

st.subheader("Result files")
for path in (config.database_path, config.output_dir, config.export_dir):
    Path(path).mkdir(parents=True, exist_ok=True) if path.suffix == "" else path.parent.mkdir(parents=True, exist_ok=True)
    st.write(f"- `{path}`")

