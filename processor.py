"""Translation using faster-whisper (Apple Silicon optimized)."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from faster_whisper import WhisperModel

from control import conversation_path, new_conversation_id
from media import (
    extract_wav_segment,
    is_media_file,
    prepare_media_for_transcription,
    probe_duration_seconds,
    wav_duration_seconds,
)

MODEL_NAME = "large-v3"  # turbo cannot translate; use full multilingual model
MODELS_DIR = Path(__file__).resolve().parent / "models"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_AUDIO = Path("/tmp/recording.wav")

DEVICE = "cpu"
COMPUTE_TYPE_FALLBACKS = ("int8_float16", "int8", "float32")

MINUTE_SECONDS = 60.0
OVERLAP_SECONDS = 5.0
DEFAULT_WORKERS = min(4, max(1, (os.cpu_count() or 2) // 2))


def _prompt_terms(data: dict, key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def load_whisper_prompt(config_path: Path = CONFIG_PATH) -> str | None:
    """Build Whisper initial_prompt from config.json if present (optional)."""
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    terms = (
        _prompt_terms(data, "prompt_snapp")
        + _prompt_terms(data, "prompt_team")
        + _prompt_terms(data, "prompt_team_members")
    )
    if not terms:
        return None
    return " ".join(f"'{term}'" for term in terms)


def _format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    mins, secs = divmod(total, 60)
    return f"{mins:02d}:{secs:02d}"


def _resolve_workers(workers: int | None, job_count: int) -> int:
    if workers is None:
        workers = DEFAULT_WORKERS
    return max(1, min(int(workers), max(1, job_count)))


def _load_model(
    models_dir: Path = MODELS_DIR,
    *,
    num_workers: int = 1,
) -> WhisperModel:
    cpu_count = os.cpu_count() or 4
    cpu_threads = max(1, cpu_count // max(1, num_workers))
    errors: list[str] = []
    for compute_type in COMPUTE_TYPE_FALLBACKS:
        try:
            return WhisperModel(
                MODEL_NAME,
                device=DEVICE,
                compute_type=compute_type,
                download_root=str(models_dir),
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
        except ValueError as exc:
            errors.append(f"{compute_type}: {exc}")
    raise RuntimeError(
        "Unable to load Whisper model with supported compute types: "
        + "; ".join(errors)
    )


def _minute_windows(duration: float) -> list[tuple[int, float, float, float, float]]:
    """Return (minute_index, core_start, core_end, chunk_start, chunk_end)."""
    if duration <= 0:
        return []

    total_minutes = max(1, math.ceil(duration / MINUTE_SECONDS))
    windows: list[tuple[int, float, float, float, float]] = []
    for minute_index in range(total_minutes):
        core_start = minute_index * MINUTE_SECONDS
        core_end = min(duration, (minute_index + 1) * MINUTE_SECONDS)
        if core_end <= core_start:
            continue
        chunk_start = max(0.0, core_start - OVERLAP_SECONDS)
        chunk_end = min(duration, core_end + OVERLAP_SECONDS)
        windows.append((minute_index, core_start, core_end, chunk_start, chunk_end))
    return windows


def _translate_chunk(
    model: WhisperModel,
    chunk_path: Path,
    *,
    language: str | None,
    initial_prompt: str | None,
    chunk_start: float,
    core_start: float,
    core_end: float,
) -> list[str]:
    segments, _info = model.transcribe(
        str(chunk_path),
        task="translate",
        language=language,
        initial_prompt=initial_prompt,
        vad_filter=True,
    )

    lines: list[str] = []
    for segment in segments:
        abs_start = chunk_start + float(segment.start)
        abs_end = chunk_start + float(segment.end)
        midpoint = (abs_start + abs_end) / 2.0
        # Keep only speech whose midpoint falls in this minute's core window.
        if midpoint < core_start or midpoint >= core_end:
            continue
        text = segment.text.strip()
        if not text:
            continue
        lines.append(
            f"[{_format_timestamp(abs_start)} - {_format_timestamp(abs_end)}] {text}"
        )
    return lines


def transcribe(
    audio_path: Path,
    output_path: Path,
    language: str | None = None,
    models_dir: Path = MODELS_DIR,
    workers: int | None = None,
) -> Path:
    """Translate audio minute-by-minute (with 5s overlap) into a markdown file."""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = wav_duration_seconds(audio_path)
    if duration <= 0:
        raise RuntimeError(f"Audio file has no usable duration: {audio_path}")

    windows = _minute_windows(duration)
    worker_count = _resolve_workers(workers, len(windows))
    model = _load_model(models_dir=models_dir, num_workers=worker_count)
    initial_prompt = load_whisper_prompt()
    total_minutes = windows[-1][0] + 1 if windows else 0

    sections: list[str] = [
        "# Translation",
        "",
        f"Source: `{audio_path.name}`",
        f"Duration: {_format_timestamp(duration)}",
        f"Workers: {worker_count}",
        "",
    ]

    work_dir = Path(f"/tmp/gapscribe_chunks_{uuid.uuid4().hex}")
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        chunk_paths: dict[int, Path] = {}
        for minute_index, _core_start, _core_end, chunk_start, chunk_end in windows:
            chunk_path = work_dir / f"minute_{minute_index:04d}.wav"
            extract_wav_segment(audio_path, chunk_path, chunk_start, chunk_end)
            chunk_paths[minute_index] = chunk_path

        print(
            f"Translating {total_minutes} minute(s) with {worker_count} worker(s)...",
            flush=True,
        )
        if initial_prompt:
            print(f"Using Whisper prompt from {CONFIG_PATH.name}", flush=True)

        results: dict[int, list[str]] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    _translate_chunk,
                    model,
                    chunk_paths[minute_index],
                    language=language,
                    initial_prompt=initial_prompt,
                    chunk_start=chunk_start,
                    core_start=core_start,
                    core_end=core_end,
                ): (minute_index, core_start, core_end)
                for minute_index, core_start, core_end, chunk_start, _chunk_end in windows
            }
            for future in as_completed(futures):
                minute_index, core_start, core_end = futures[future]
                results[minute_index] = future.result()
                print(
                    f"Finished minute {minute_index + 1}/{total_minutes} "
                    f"({_format_timestamp(core_start)} - {_format_timestamp(core_end)})",
                    flush=True,
                )

        for minute_index, core_start, core_end, _chunk_start, _chunk_end in windows:
            minute_label = minute_index + 1
            lines = results[minute_index]
            sections.append(
                f"## Minute {minute_label} "
                f"({_format_timestamp(core_start)} - {_format_timestamp(core_end)})"
            )
            sections.append("")
            if lines:
                sections.extend(lines)
            else:
                sections.append("_(no speech detected)_")
            sections.append("")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return output_path


def transcribe_media_file(
    source_path: Path,
    output_path: Path | None = None,
    language: str | None = None,
    models_dir: Path = MODELS_DIR,
    workers: int | None = None,
) -> Path:
    if not is_media_file(source_path):
        raise ValueError(f"Unsupported media file: {source_path}")

    if output_path is None:
        output_path = conversation_path(new_conversation_id())

    work_wav = Path(f"/tmp/gapscribe_convert_{uuid.uuid4().hex}.wav")
    try:
        prepare_media_for_transcription(source_path, work_wav)
        source_duration = probe_duration_seconds(source_path)
        audio_duration = wav_duration_seconds(work_wav)
        if (
            source_duration is not None
            and source_duration > 0
            and audio_duration > 0
            and audio_duration < source_duration * 0.85
        ):
            print(
                "Warning: source media reports "
                f"{source_duration / 60:.1f} min but only "
                f"{audio_duration / 60:.1f} min of audio could be decoded "
                "(corrupt or truncated audio stream).",
                flush=True,
            )
        return transcribe(
            audio_path=work_wav,
            output_path=output_path,
            language=language,
            models_dir=models_dir,
            workers=workers,
        )
    finally:
        work_wav.unlink(missing_ok=True)


def cleanup_recording(audio_path: Path = DEFAULT_AUDIO) -> None:
    if audio_path.exists():
        audio_path.unlink()
