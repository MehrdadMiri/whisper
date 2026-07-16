"""Media conversion helpers using ffmpeg."""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

SAMPLE_RATE = 16_000
_MIN_USABLE_AUDIO_SECONDS = 0.25

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_media_file(path: Path) -> bool:
    return is_video(path) or is_audio(path)


def wav_duration_seconds(path: Path) -> float:
    if not path.is_file() or path.stat().st_size < 44:
        return 0.0
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            if rate <= 0:
                return 0.0
            return wav.getnframes() / float(rate)
    except wave.Error:
        return 0.0


def _ffmpeg_has_decoder(name: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-decoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return name in (result.stdout or "")


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def extract_audio_to_wav(source: Path, dest: Path, sample_rate: int = SAMPLE_RATE) -> Path:
    """Extract audio to 16 kHz mono PCM WAV, tolerating corrupt AAC frames."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = (
        "aformat=channel_layouts=stereo,"
        "pan=mono|c0=0.5*c0+0.5*c1,"
        f"aresample={sample_rate}"
    )

    decoder_attempts: list[str | None] = []
    # AudioToolbox AAC recovers slightly more from corrupt macOS meeting exports.
    if source.suffix.lower() in {".mp4", ".m4a", ".mov", ".aac", ".m4v"} and _ffmpeg_has_decoder(
        "aac_at"
    ):
        decoder_attempts.append("aac_at")
    decoder_attempts.append(None)

    last_stderr = ""
    for decoder in decoder_attempts:
        dest.unlink(missing_ok=True)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-err_detect",
            "ignore_err",
            "-fflags",
            "+discardcorrupt+genpts",
        ]
        if decoder:
            cmd.extend(["-c:a", decoder])
        cmd.extend(
            [
                "-i",
                str(source),
                "-vn",
                "-map",
                "0:a:0",
                "-af",
                filter_chain,
                "-c:a",
                "pcm_s16le",
                str(dest),
            ]
        )
        result = _run_ffmpeg(cmd)
        last_stderr = (result.stderr or result.stdout or "").strip()
        if wav_duration_seconds(dest) >= _MIN_USABLE_AUDIO_SECONDS:
            return dest

    detail = last_stderr.splitlines()[-1] if last_stderr else "no usable audio decoded"
    raise RuntimeError(f"Failed to extract audio from {source}: {detail}")


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
    result = _run_ffmpeg(cmd)
    if result.returncode != 0 or wav_duration_seconds(dest) < _MIN_USABLE_AUDIO_SECONDS:
        detail = (result.stderr or result.stdout or "mix failed").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "Failed to mix audio")
    return dest


def probe_duration_seconds(path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (result.stdout or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def prepare_media_for_transcription(source: Path, work_wav: Path) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"Media file not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    extract_audio_to_wav(source, work_wav)
    return work_wav


def extract_wav_segment(
    source: Path,
    dest: Path,
    start_seconds: float,
    end_seconds: float,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Cut a WAV slice [start_seconds, end_seconds) into dest."""
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    duration = end_seconds - start_seconds
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(dest),
    ]
    result = _run_ffmpeg(cmd)
    if result.returncode != 0 or wav_duration_seconds(dest) < _MIN_USABLE_AUDIO_SECONDS:
        detail = (result.stderr or result.stdout or "segment cut failed").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else "Failed to extract WAV segment")
    return dest
