from __future__ import annotations

import subprocess
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from stt_tts_benchmark.audio import wav_duration_seconds


@dataclass
class TTSOutput:
    output_audio_path: Path
    latency_seconds: float
    audio_duration_seconds: float | None


class KokoroTTSAdapter:
    model_name = "Kokoro"
    provider = "kokoro"
    is_local = True

    def __init__(self, voice: str, lang_code: str) -> None:
        self.voice = voice
        self.lang_code = lang_code

    def synthesize(self, text: str, output_path: Path) -> TTSOutput:
        try:
            from kokoro import KPipeline
            import soundfile as sf
        except ImportError:
            return self._synthesize_with_pykokoro(text, output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        pipeline = KPipeline(lang_code=self.lang_code)
        generator = pipeline(text, voice=self.voice)
        chunks = []
        sample_rate = 24000
        for _graphemes, _phonemes, audio in generator:
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("Kokoro produced no audio chunks.")
        if len(chunks) == 1:
            merged = chunks[0]
        else:
            import numpy as np

            merged = np.concatenate(chunks)
        sf.write(str(output_path), merged, sample_rate)
        latency = time.perf_counter() - started_at
        return TTSOutput(output_path, latency, wav_duration_seconds(output_path))

    def _synthesize_with_pykokoro(self, text: str, output_path: Path) -> TTSOutput:
        try:
            from pykokoro import GenerationConfig, PipelineConfig, build_pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Kokoro dependencies are not installed. Install with: python -m pip install -e .[kokoro]"
            ) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        lang = _kokoro_language(self.lang_code)
        started_at = time.perf_counter()
        pipeline = build_pipeline(
            config=PipelineConfig(
                voice=self.voice,
                generation=GenerationConfig(lang=lang),
            ),
            eager=True,
        )
        try:
            result = pipeline.run(text)
            result.save_wav(output_path)
        finally:
            pipeline.close()
        latency = time.perf_counter() - started_at
        return TTSOutput(output_path, latency, wav_duration_seconds(output_path))


class PiperTTSAdapter:
    model_name = "Piper"
    provider = "piper"
    is_local = True

    def __init__(self, executable: str, model_path: Path) -> None:
        self.executable = executable
        self.model_path = model_path

    def synthesize(self, text: str, output_path: Path) -> TTSOutput:
        if not self.model_path.exists():
            raise RuntimeError(f"Piper model file does not exist: {self.model_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [_resolve_executable(self.executable), "--model", str(self.model_path), "--output_file", str(output_path)]
        started_at = time.perf_counter()
        completed = subprocess.run(
            command,
            input=text,
            text=True,
            capture_output=True,
            check=False,
        )
        latency = time.perf_counter() - started_at
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Piper command failed."
            raise RuntimeError(message)
        return TTSOutput(output_path, latency, wav_duration_seconds(output_path))


class Qwen3TTSAdapter:
    model_name = "Qwen3-TTS"
    provider = "qwen"
    is_local = True

    def __init__(
        self,
        command_template: str,
        workdir: Path | None,
        model_path: Path,
        model_id: str,
        language: str,
        speaker: str,
        instruct: str,
    ) -> None:
        self.command_template = command_template
        self.workdir = workdir
        self.model_path = model_path
        self.model_id = model_id
        self.language = language
        self.speaker = speaker
        self.instruct = instruct

    def synthesize(self, text: str, output_path: Path) -> TTSOutput:
        if not self.command_template:
            return self._synthesize_with_package(text, output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command_text = self.command_template.format(
            text=_shell_quote(text),
            output_path=_shell_quote(str(output_path)),
            model_path=_shell_quote(str(self.model_path)),
        )
        started_at = time.perf_counter()
        completed = subprocess.run(
            command_text,
            cwd=str(self.workdir) if self.workdir else None,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        latency = time.perf_counter() - started_at
        if completed.returncode != 0:
            message = completed.stderr.strip() or completed.stdout.strip() or "Qwen3-TTS command failed."
            raise RuntimeError(message)
        if not output_path.exists():
            raise RuntimeError(f"Qwen3-TTS command completed but did not create {output_path}")
        return TTSOutput(output_path, latency, wav_duration_seconds(output_path))

    def _synthesize_with_package(self, text: str, output_path: Path) -> TTSOutput:
        try:
            import soundfile as sf
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError("qwen-tts is not installed. Install with: python -m pip install -e .[qwen]") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        model_name_or_path = str(self.model_path) if self.model_path.exists() else self.model_id
        use_cuda = torch.cuda.is_available()
        attention = "flash_attention_2" if use_cuda and _has_flash_attention() else "sdpa"
        started_at = time.perf_counter()
        model = Qwen3TTSModel.from_pretrained(
            model_name_or_path,
            device_map="cuda:0" if use_cuda else "cpu",
            dtype=torch.bfloat16 if use_cuda else torch.float32,
            attn_implementation=attention,
        )
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=self.language,
            speaker=self.speaker,
            instruct=self.instruct or None,
            non_streaming_mode=True,
        )
        sf.write(str(output_path), wavs[0], sample_rate)
        latency = time.perf_counter() - started_at
        return TTSOutput(output_path, latency, wav_duration_seconds(output_path))


class GeminiTTSAdapter:
    model_name = "Gemini TTS"
    provider = "google"
    is_local = False

    def __init__(self, api_key: str, model: str, voice: str) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice

    def synthesize(self, text: str, output_path: Path) -> TTSOutput:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed. Install with: python -m pip install -e .[gemini]") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        client = genai.Client(api_key=self.api_key)
        started_at = time.perf_counter()
        response = client.models.generate_content(
            model=self.model,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                    )
                ),
            ),
        )
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        _write_pcm_wav(output_path, audio_data)
        latency = time.perf_counter() - started_at
        return TTSOutput(output_path, latency, wav_duration_seconds(output_path))


def _write_pcm_wav(output_path: Path, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> None:
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


def _shell_quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def _resolve_executable(executable: str) -> str:
    if Path(executable).exists():
        return executable
    script_path = Path(sys.executable).parent / f"{executable}.exe"
    if script_path.exists():
        return str(script_path)
    return executable


def _has_flash_attention() -> bool:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return False
    return True


def _kokoro_language(lang_code: str) -> str:
    aliases = {
        "a": "en-us",
        "b": "en-gb",
        "e": "es",
        "f": "fr-fr",
        "h": "hi",
        "i": "it",
        "j": "ja",
        "p": "pt",
        "z": "zh",
    }
    return aliases.get(lang_code.lower(), lang_code.lower())
