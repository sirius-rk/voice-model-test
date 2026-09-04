# STT/TTS Benchmark

This project is a Streamlit workspace for comparing speech recognition and speech synthesis models on the same dialogue samples.

## Features

- Whisper STT benchmark page using local `faster-whisper`.
- Qwen3-ASR STT benchmark page using a local Transformers model.
- TTS comparison page for Kokoro, Piper, Qwen3-TTS, and Gemini TTS.
- SQLite result storage plus CSV export.
- History search, original uploaded filenames, matching transcripts, playback, and deletion for saved test rows.
- Accuracy metrics for STT when a reference transcript is provided: WER and CER.
- Manual TTS scoring for tone, intonation, naturalness, pronunciation, and notes.
- Batch inserts for multi-model TTS runs.
- Deleting history rows also removes associated generated or uploaded audio when the file is inside the project directory.

## Setup

Use the repository virtual environment first:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -e .[whisper,gemini,kokoro,piper,qwen]
.\.venv\Scripts\python.exe -m pip install -e .[qwen-asr]
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
- `FFMPEG_EXECUTABLE`: FFmpeg executable name or full path, default `ffmpeg`.
- `WHISPER_MODEL_SIZE`: local Whisper model size or a downloaded faster-whisper model directory.
- `QWEN3_ASR_MODEL_ID`: Qwen3-ASR Hugging Face model id or local model directory. The default is `Qwen/Qwen3-ASR-0.6B`.
- `QWEN3_ASR_DEVICE`: `auto`, `cpu`, or `cuda`.
- `QWEN3_ASR_DTYPE`: `auto`, `bfloat16`, `float16`, or `float32`. CPU inference uses `float32`.
- `QWEN3_ASR_LANGUAGE`: `auto` for detection, or a model language name such as `Chinese` or `English`.
- `QWEN3_ASR_MAX_NEW_TOKENS`: maximum generated tokens for one transcription; use a higher value such as `4096` for long recordings.
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

STT uploads are copied to `APP_OUTPUT_DIR/inputs/` using a content-hash filename, while the original uploaded filename is stored separately for history display. Each history entry shows its own transcript and audio player, so audio cannot be confused with another test row. Existing rows created before this field was added fall back to the stored path basename.

Both STT pages include an optional `Convert to mono WAV before transcription` switch. When enabled, FFmpeg converts the upload to 16 kHz mono PCM WAV before the model runs. The `Mono channel` selector supports mixing both channels or selecting only the left/right channel; selecting one channel is useful for stereo recordings that contain separate speaker feeds. The history keeps the original uploaded filename and plays the converted file.

Large model files, voice files, generated audio, databases, and exports are ignored by Git.

### Local Qwen3-ASR

Install the optional local model dependency with:

```bash
.venv/bin/python -m pip install -e '.[qwen-asr]'
```

Open the `Qwen3-ASR STT` page from the Streamlit sidebar. The first transcription downloads the configured model from Hugging Face unless `QWEN3_ASR_MODEL_ID` points to an existing local model directory. The default 0.6B model is selected for lower resource usage; use `Qwen/Qwen3-ASR-1.7B` when the host has sufficient resources.
