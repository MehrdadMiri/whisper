"""Transcription using faster-whisper (Apple Silicon optimized)."""

from __future__ import annotations

import uuid
from pathlib import Path

from faster_whisper import WhisperModel

from control import conversation_path, new_conversation_id
from media import is_media_file, prepare_media_for_transcription

MODEL_NAME = "large-v3-turbo"
MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_AUDIO = Path("/tmp/recording.wav")

DEVICE = "cpu"
COMPUTE_TYPE = "int8_float16"


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def _load_model(models_dir: Path = MODELS_DIR) -> WhisperModel:
    return WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=str(models_dir),
        cpu_threads=0,
    )


def transcribe(
    audio_path: Path,
    output_path: Path,
    language: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> Path:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _load_model(models_dir=models_dir)
    segments, _info = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=language,
        vad_filter=True,
    )

    lines: list[str] = []
    for segment in segments:
        start = _format_timestamp(segment.start)
        end = _format_timestamp(segment.end)
        text = segment.text.strip()
        if text:
            lines.append(f"[{start} - {end}] {text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def transcribe_media_file(
    source_path: Path,
    output_path: Path | None = None,
    language: str | None = None,
    models_dir: Path = MODELS_DIR,
) -> Path:
    if not is_media_file(source_path):
        raise ValueError(f"Unsupported media file: {source_path}")

    if output_path is None:
        output_path = conversation_path(new_conversation_id())

    work_wav = Path(f"/tmp/gapscribe_convert_{uuid.uuid4().hex}.wav")
    try:
        prepare_media_for_transcription(source_path, work_wav)
        return transcribe(
            audio_path=work_wav,
            output_path=output_path,
            language=language,
            models_dir=models_dir,
        )
    finally:
        work_wav.unlink(missing_ok=True)


def cleanup_recording(audio_path: Path = DEFAULT_AUDIO) -> None:
    if audio_path.exists():
        audio_path.unlink()
