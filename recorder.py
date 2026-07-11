"""Audio and optional screen recording for GapScribe."""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile
import sounddevice as sd

from control import SessionConfig, clear_session_files, write_state
from devices import find_screen_capture_input, get_default_mic_id, list_microphones
from media import SAMPLE_RATE, mix_wav_files

CHANNELS = 1
DTYPE = "float32"
RECORDING_PATH = Path("/tmp/recording.wav")
MIC_WAV_PATH = Path("/tmp/gapscribe_mics.wav")
SCREEN_VIDEO_PATH = Path("/tmp/gapscribe_screen.mp4")
SCREEN_AUDIO_PATH = Path("/tmp/gapscribe_screen.wav")


class MicTrack:
    def __init__(self, device_id: int) -> None:
        self.device_id = device_id
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.lock = threading.Lock()

    def _callback(self, indata: np.ndarray, _frames: int, _time, status) -> None:
        if status:
            print(f"Mic {self.device_id} status: {status}", file=sys.stderr)
        mono = indata.mean(axis=1, keepdims=True) if indata.shape[1] > 1 else indata
        with self.lock:
            self.frames.append(mono.copy())

    def start(self) -> None:
        self.frames = []
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def to_pcm(self) -> np.ndarray:
        with self.lock:
            if not self.frames:
                return np.zeros(0, dtype=np.int16)
            audio = np.concatenate(self.frames, axis=0)[:, 0]
        pcm = np.clip(audio, -1.0, 1.0)
        return (pcm * 32767).astype(np.int16)


class ScreenCapture:
    def __init__(self, output_path: Path = SCREEN_VIDEO_PATH) -> None:
        self.output_path = output_path
        self.process: subprocess.Popen[str] | None = None
        self.input_spec: str | None = find_screen_capture_input()

    @property
    def available(self) -> bool:
        return self.input_spec is not None

    def start(self) -> None:
        if not self.input_spec:
            raise RuntimeError(
                "Screen capture is unavailable. Grant Screen Recording permission in "
                "System Settings → Privacy & Security → Screen Recording."
            )
        if self.output_path.exists():
            self.output_path.unlink()

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "avfoundation",
            "-capture_cursor",
            "1",
            "-capture_mouse_clicks",
            "1",
            "-i",
            self.input_spec,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self) -> Path | None:
        if self.process is None:
            return None
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            return self.output_path
        return None


class SessionRecorder:
    def __init__(
        self,
        output_path: Path = RECORDING_PATH,
        mic_ids: list[int] | None = None,
        screen_enabled: bool = False,
    ) -> None:
        self.output_path = output_path
        self.config = SessionConfig(
            enabled_mic_ids=mic_ids or [get_default_mic_id()],
            screen_enabled=screen_enabled,
        )
        self.config.save()
        self.mic_labels = {mic.id: mic.name for mic in list_microphones()}
        self.tracks: dict[int, MicTrack] = {}
        self.screen = ScreenCapture()
        self._running = False
        self._start_time: float | None = None
        self._control_lock = threading.Lock()

    @property
    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def _sync_mics(self) -> None:
        enabled = set(self.config.enabled_mic_ids)
        for mic_id in list(self.tracks):
            if mic_id not in enabled:
                self.tracks[mic_id].stop()
                del self.tracks[mic_id]

        for mic_id in enabled:
            if mic_id not in self.tracks:
                track = MicTrack(mic_id)
                track.start()
                self.tracks[mic_id] = track

    def _sync_screen(self) -> None:
        if self.config.screen_enabled and self.screen.process is None:
            try:
                self.screen.start()
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                self.config.screen_enabled = False
                self.config.save()
        elif not self.config.screen_enabled and self.screen.process is not None:
            self.screen.stop()

    def _apply_control_updates(self) -> None:
        with self._control_lock:
            latest = SessionConfig.load()
            changed = (
                latest.enabled_mic_ids != self.config.enabled_mic_ids
                or latest.screen_enabled != self.config.screen_enabled
            )
            if not changed:
                return
            self.config = latest
        self._sync_mics()
        self._sync_screen()

    def start(self) -> None:
        if self._running:
            raise RuntimeError("Recording is already in progress.")
        self._running = True
        self._start_time = time.monotonic()
        self._sync_mics()
        self._sync_screen()

    def stop(self) -> Path:
        if not self._running:
            raise RuntimeError("No active recording to stop.")

        self._running = False

        screen_path = None
        if self.config.screen_enabled:
            screen_path = self.screen.stop()

        mic_sources: list[Path] = []
        for mic_id, track in list(self.tracks.items()):
            pcm = track.to_pcm()
            track.stop()
            if pcm.size == 0:
                continue
            mic_path = Path(f"/tmp/gapscribe_mic_{mic_id}.wav")
            wavfile.write(str(mic_path), SAMPLE_RATE, pcm)
            mic_sources.append(mic_path)
        self.tracks.clear()
        if mic_sources:
            mix_wav_files(mic_sources, MIC_WAV_PATH)
            for path in mic_sources:
                path.unlink(missing_ok=True)
        else:
            wavfile.write(str(MIC_WAV_PATH), SAMPLE_RATE, np.zeros(0, dtype=np.int16))

        final_sources = []
        if MIC_WAV_PATH.exists() and MIC_WAV_PATH.stat().st_size > 44:
            final_sources.append(MIC_WAV_PATH)

        if screen_path is not None:
            from media import extract_audio_to_wav

            try:
                extract_audio_to_wav(screen_path, SCREEN_AUDIO_PATH)
                if SCREEN_AUDIO_PATH.stat().st_size > 44:
                    final_sources.append(SCREEN_AUDIO_PATH)
            except subprocess.CalledProcessError:
                print("Could not extract audio from screen capture.", file=sys.stderr)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not final_sources:
            wavfile.write(str(self.output_path), SAMPLE_RATE, np.zeros(0, dtype=np.int16))
        elif len(final_sources) == 1:
            self.output_path.write_bytes(final_sources[0].read_bytes())
        else:
            mix_wav_files(final_sources, self.output_path)

        for path in (MIC_WAV_PATH, SCREEN_AUDIO_PATH, SCREEN_VIDEO_PATH):
            path.unlink(missing_ok=True)

        clear_session_files()
        return self.output_path

    def status_line(self) -> str:
        mic_names = [
            self.mic_labels.get(mid, f"Mic {mid}")[:18]
            for mid in self.config.enabled_mic_ids
        ]
        mic_text = ", ".join(mic_names) if mic_names else "none"
        screen = "on" if self.config.screen_enabled else "off"
        elapsed = int(self.elapsed)
        mins, secs = divmod(elapsed, 60)
        return f"Recording {mins:02d}:{secs:02d} | mics: {mic_text} | screen: {screen}"

    def publish_state(self) -> None:
        write_state(
            enabled_mic_ids=self.config.enabled_mic_ids,
            screen_enabled=self.config.screen_enabled,
            screen_active=self.screen.process is not None,
            elapsed=self.elapsed,
            mic_labels=self.mic_labels,
        )


def run_recording_session(
    output_path: Path = RECORDING_PATH,
    mic_ids: list[int] | None = None,
    screen_enabled: bool = False,
    interactive: bool = False,
) -> None:
    recorder = SessionRecorder(
        output_path=output_path,
        mic_ids=mic_ids,
        screen_enabled=screen_enabled,
    )
    stop_event = threading.Event()

    def _handle_stop(signum, _frame) -> None:
        print(f"\nStopping recording (signal {signum})...", file=sys.stderr)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    if interactive:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    recorder.start()
    print("Recording started.", flush=True)
    if interactive:
        print("Keys: [0-9] toggle mic by device id | s toggle screen | q stop", flush=True)
        print("      Or run toggle-mic / toggle-screen from another terminal.", flush=True)

    try:
        while not stop_event.is_set():
            recorder._apply_control_updates()
            recorder.publish_state()
            print(f"\r{recorder.status_line()}", end="", flush=True)

            if interactive:
                readable, _, _ = select.select([sys.stdin], [], [], 0.25)
                if readable:
                    key = sys.stdin.read(1)
                    if key in {"q", "\x03"}:
                        stop_event.set()
                        break
                    if key == "s":
                        recorder.config.toggle_screen()
                        recorder.config.save()
                        continue
                    if key.isdigit():
                        mic_id = int(key)
                        if mic_id in recorder.mic_labels:
                            recorder.config.toggle_mic(mic_id)
                            recorder.config.save()
            else:
                time.sleep(0.25)
    finally:
        if interactive:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        path = recorder.stop()
        print(f"\nSaved audio to {path}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GapScribe recorder")
    parser.add_argument("--mic", action="append", type=int, dest="mic_ids", default=[])
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    mic_ids = args.mic_ids or None
    run_recording_session(
        mic_ids=mic_ids,
        screen_enabled=args.screen,
        interactive=args.interactive,
    )
