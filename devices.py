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


def find_screen_capture_input() -> str | None:
    """Return an avfoundation input string for screen (+ system audio when available)."""
    listing = _ffmpeg_device_listing()
    screen_name = None
    audio_name = None

    in_video = False
    in_audio = False
    for line in listing.splitlines():
        if "AVFoundation video devices" in line:
            in_video, in_audio = True, False
            continue
        if "AVFoundation audio devices" in line:
            in_video, in_audio = False, True
            continue

        match = re.search(r"\[(\d+)\]\s+(.+)", line)
        if not match:
            continue

        index, name = match.group(1), match.group(2).strip()
        if in_video and "capture screen" in name.lower():
            screen_name = name
        if in_audio and audio_name is None:
            lowered = name.lower()
            if any(token in lowered for token in ("blackhole", "loopback", "soundflower")):
                audio_name = name

    if screen_name is None:
        return None
    if audio_name:
        return f"{screen_name}:{audio_name}"
    return f"{screen_name}:none"
