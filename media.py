"""Media conversion helpers using ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

SAMPLE_RATE = 16_000

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_media_file(path: Path) -> bool:
    return is_video(path) or is_audio(path)


def extract_audio_to_wav(source: Path, dest: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    return dest


def mix_wav_files(sources: list[Path], dest: Path) -> Path:
    if not sources:
        raise ValueError("No audio sources to mix.")
    if len(sources) == 1:
        if sources[0] != dest:
            dest.write_bytes(sources[0].read_bytes())
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    inputs: list[str] = []
    for source in sources:
        inputs.extend(["-i", str(source)])

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *inputs,
        "-filter_complex",
        f"amix=inputs={len(sources)}:duration=longest:dropout_transition=0",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    return dest


def prepare_media_for_transcription(source: Path, work_wav: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Media file not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    extract_audio_to_wav(source, work_wav)
    return work_wav
