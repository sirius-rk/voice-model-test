# STT/TTS Benchmark

This project is a Streamlit workspace for comparing speech recognition and speech synthesis models on the same dialogue samples.

## Features

- Whisper STT benchmark page using local `faster-whisper`.
- TTS comparison page for Kokoro, Piper, Qwen3-TTS, and Gemini TTS.
- SQLite result storage plus CSV export.
- History table playback and deletion for saved test rows.
- Accuracy metrics for STT when a reference transcript is provided: WER and CER.
- Manual TTS scoring for tone, intonation, naturalness, pronunciation, and notes.
- Batch inserts for multi-model TTS runs.
- Deleting history rows also removes associated generated or uploaded audio when the file is inside the project directory.

## Setup

Use the repository virtual environment first:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -e .[whisper,gemini,kokoro,piper,qwen]
Copy-Item .env.example .env
```

Qwen3-TTS can be used directly through `qwen-tts` when the package is installed. If you prefer a separate Qwen runtime, set `QWEN3_TTS_COMMAND` to a command template instead.

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

## Configuration

Configuration is read from `.env`.

- `APP_DATABASE_PATH`: SQLite database path, default `data/results.db`.
- `APP_EXPORT_DIR`: CSV export directory, default `exports`.
- `APP_OUTPUT_DIR`: generated audio directory, default `outputs`.
- `WHISPER_MODEL_SIZE`: local Whisper model size or a downloaded faster-whisper model directory.
- `PIPER_EXECUTABLE`: Piper executable name or full path.
- `PIPER_MODEL_PATH`: Piper `.onnx` voice model path. The default English baseline is `models/piper/en_US-lessac-medium.onnx`.
- `QWEN3_TTS_COMMAND`: shell command template for local Qwen3-TTS.
- `QWEN3_TTS_MODEL_PATH`: downloaded local Qwen3-TTS model directory.
- `QWEN3_TTS_MODEL_ID`: Hugging Face model id used when `QWEN3_TTS_MODEL_PATH` is not downloaded locally.
- `QWEN3_TTS_LANGUAGE`, `QWEN3_TTS_SPEAKER`, `QWEN3_TTS_INSTRUCT`: Qwen3-TTS generation controls. The default English baseline uses language `English`, speaker `Ryan`, and a calm customer-service tone instruction.
- `GEMINI_API_KEY`: Gemini API key.

The Qwen3-TTS command template supports `{text}`, `{output_path}`, and `{model_path}` placeholders. Example:

```powershell
python qwen3_tts_cli.py --model {model_path} --text {text} --output {output_path}
```

## Result Storage

The SQLite table is `benchmark_results`, and its columns use the `brs_` prefix. CSV exports are written to `exports/`.

Large model files, voice files, generated audio, databases, and exports are ignored by Git.
