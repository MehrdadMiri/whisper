"""Audio and screen capture device discovery on macOS."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

import sounddevice as sd


@dataclass(frozen=True)
class Microphone:
    id: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool


def list_microphones() -> list[Microphone]:
    devices = sd.query_devices()
    default_in = sd.default.device[0]
    mics: list[Microphone] = []

    for index, info in enumerate(devices):
        if info["max_input_channels"] < 1:
            continue
        mics.append(
            Microphone(
                id=index,
                name=str(info["name"]),
                channels=int(info["max_input_channels"]),
                sample_rate=float(info["default_samplerate"]),
                is_default=index == default_in,
            )
        )
    return mics


def get_default_mic_id() -> int:
    default_in = sd.default.device[0]
    if default_in is None or int(default_in) < 0:
        mics = list_microphones()
        if not mics:
            raise RuntimeError("No microphone input devices found.")
        return mics[0].id
    return int(default_in)


def _ffmpeg_device_listing() -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        capture_output=True,
        text=True,
    )
    return result.stderr


def find_system_audio_device() -> str | None:
    """Return an avfoundation system-audio device name when a loopback device exists."""
    listing = _ffmpeg_device_listing()
    in_audio = False
    for line in listing.splitlines():
        if "AVFoundation video devices" in line:
            in_audio = False
            continue
        if "AVFoundation audio devices" in line:
            in_audio = True
            continue
        if not in_audio:
            continue
        match = re.search(r"\[\d+\]\s+(.+)", line)
        if not match:
            continue
        name = match.group(1).strip()
        lowered = name.lower()
        if any(token in lowered for token in ("blackhole", "loopback", "soundflower")):
            return name
    return None


def find_screen_capture_input() -> tuple[str, bool] | None:
    """Return (avfoundation input, has_system_audio) for screen capture."""
    listing = _ffmpeg_device_listing()
    screen_name = None
    audio_name = find_system_audio_device()

    in_video = False
    for line in listing.splitlines():
        if "AVFoundation video devices" in line:
            in_video = True
            continue
        if "AVFoundation audio devices" in line:
            in_video = False
            continue
        if not in_video:
            continue
        match = re.search(r"\[\d+\]\s+(.+)", line)
        if not match:
            continue
        name = match.group(1).strip()
        if "capture screen" in name.lower():
            screen_name = name
            break

    if screen_name is None:
        return None
    if audio_name:
        return f"{screen_name}:{audio_name}", True
    return f"{screen_name}:none", False
