from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (wave.Error, OSError):
        return None


def safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    return cleaned[:80] or "audio"


def persist_uploaded_audio(uploaded_file: object, directory: Path) -> tuple[Path, str]:
    """Save an uploaded audio file outside the temporary directory.

    The returned display name preserves the user's original filename while the
    stored path uses a content hash and sanitized basename for safe persistence.
    """
    raw_name = str(getattr(uploaded_file, "name", "") or "").replace("\\", "/")
    original_name = raw_name.rsplit("/", 1)[-1].strip() or "audio.wav"
    suffix = Path(original_name).suffix.lower() or ".wav"
    stem = safe_filename(Path(original_name).stem)
    data = bytes(uploaded_file.getbuffer())  # type: ignore[attr-defined]
    if not data:
        raise ValueError("Uploaded audio file is empty.")

    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    stored_path = directory / f"{digest[:16]}_{stem}{suffix}"
    if not stored_path.exists():
        stored_path.write_bytes(data)
    return stored_path, original_name


def convert_to_mono_wav(
    input_path: Path,
    directory: Path,
    executable: str = "ffmpeg",
    channel: str = "mix",
) -> Path:
    """Convert an audio file to 16 kHz mono PCM WAV using FFmpeg.

    ``mix`` downmixes both channels. ``left`` and ``right`` preserve one
    channel, which is important for recordings where each stereo channel has
    a separate speaker or feed.
    """
    if channel not in {"mix", "left", "right"}:
        raise ValueError("channel must be one of: mix, left, right")

    executable = executable.strip() or "ffmpeg"
    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        executable_path = Path(executable).expanduser()
        if executable_path.is_file():
            resolved_executable = str(executable_path)
        else:
            raise RuntimeError(
                f"FFmpeg executable was not found: {executable}. "
                "Install FFmpeg or set FFMPEG_EXECUTABLE to its full path."
            )

    directory.mkdir(parents=True, exist_ok=True)
    output_suffix = "_mono.wav" if channel == "mix" else f"_{channel}_mono.wav"
    output_path = directory / f"{input_path.stem}{output_suffix}"
    channel_args = []
    if channel in {"left", "right"}:
        source_channel = "FL" if channel == "left" else "FR"
        channel_args = ["-af", f"pan=mono|c0={source_channel}"]
    try:
        subprocess.run(
            [
                resolved_executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(input_path),
                *channel_args,
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        output_path.unlink(missing_ok=True)
        details = (exc.stderr or "").strip()
        raise RuntimeError(f"FFmpeg mono conversion failed: {details or 'unknown error'}") from exc

    input_path.unlink(missing_ok=True)
    return output_path
