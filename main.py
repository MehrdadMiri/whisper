"""GapScribe CLI – record audio and transcribe with Whisper (Apple Silicon)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def _reexec_with_project_venv() -> None:
    """Use .venv when present so `python main.py` works outside an activated env."""
    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        return
    try:
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except OSError:
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_project_venv()

import typer

from control import (
    SessionConfig,
    clear_session_files,
    conversation_path,
    load_conversation_id,
    resolve_conversation_path,
    start_conversation_session,
)
from devices import find_system_audio_device, get_default_mic_id, list_microphones
from processor import cleanup_recording, transcribe, transcribe_media_file
from recorder import RECORDING_PATH, run_recording_session

app = typer.Typer(
    name="gapscribe",
    help="Record audio/screen and translate mixed speech to English with Whisper.",
    add_completion=False,
)

PID_FILE = Path("/tmp/gapscribe.pid")
LOCK_FILE = Path("/tmp/gapscribe.lock")
STATE_PATH = Path("/tmp/gapscribe.state.json")
RECORDER_SCRIPT = Path(__file__).resolve().parent / "recorder.py"


def _require_apple_silicon() -> None:
    try:
        arm64 = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        arm64 = ""
    if arm64 != "1":
        typer.echo("GapScribe requires an Apple Silicon (M-series) Mac.", err=True)
        raise typer.Exit(code=1)


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except ValueError:
        return None


def _write_pid(pid: int) -> None:
    PID_FILE.write_text(str(pid))


def _clear_pid() -> None:
    PID_FILE.unlink(missing_ok=True)
    LOCK_FILE.unlink(missing_ok=True)


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process(pid: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _is_process_running(pid):
            return True
        time.sleep(0.2)
    return not _is_process_running(pid)


def _ensure_not_recording() -> None:
    existing_pid = _read_pid()
    if existing_pid is not None and _is_process_running(existing_pid):
        typer.echo(f"Recording already in progress (PID {existing_pid}).")
        raise typer.Exit(code=1)


def _recorder_command(
    mic_ids: list[int],
    screen: bool,
    interactive: bool,
) -> list[str]:
    cmd = [sys.executable, str(RECORDER_SCRIPT)]
    for mic_id in mic_ids:
        cmd.extend(["--mic", str(mic_id)])
    if screen:
        cmd.append("--screen")
    if interactive:
        cmd.append("--interactive")
    return cmd


def _init_session(mic_ids: list[int], screen: bool) -> None:
    SessionConfig(enabled_mic_ids=mic_ids, screen_enabled=screen).save()


@app.command("mics")
def list_mics() -> None:
    """List available microphone input devices."""
    for mic in list_microphones():
        default = " (default)" if mic.is_default else ""
        typer.echo(f"[{mic.id}] {mic.name}{default}")


@app.command()
def status() -> None:
    """Show current recording session state."""
    pid = _read_pid()
    if pid is not None and _is_process_running(pid):
        typer.echo(f"Recording in progress (PID {pid}).")
    else:
        typer.echo("No active recording process.")

    if STATE_PATH.exists():
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        typer.echo(f"Elapsed: {data.get('elapsed_seconds', 0)}s")
        typer.echo(f"Enabled mics: {data.get('enabled_mic_ids', [])}")
        typer.echo(f"Screen enabled: {data.get('screen_enabled', False)}")
        typer.echo(f"Screen active: {data.get('screen_active', False)}")

    conversation_id = load_conversation_id()
    if conversation_id:
        typer.echo(f"Conversation id: {conversation_id}")
        typer.echo(f"Transcript path: {conversation_path(conversation_id)}")


@app.command("toggle-mic")
def toggle_mic(
    mic_id: int = typer.Argument(..., help="Microphone device id from 'mics' command."),
) -> None:
    """Enable or disable a microphone during an active recording."""
    known = {mic.id for mic in list_microphones()}
    if mic_id not in known:
        typer.echo(f"Unknown mic id {mic_id}. Run 'python main.py mics'.", err=True)
        raise typer.Exit(code=1)

    config = SessionConfig.load()
    enabled = config.toggle_mic(mic_id)
    config.save()
    state = "enabled" if enabled else "disabled"
    typer.echo(f"Mic {mic_id} {state}.")


@app.command("toggle-screen")
def toggle_screen() -> None:
    """Enable or disable screen capture during an active recording."""
    config = SessionConfig.load()
    enabled = config.toggle_screen()
    config.save()
    state = "enabled" if enabled else "disabled"
    typer.echo(f"Screen capture {state}.")
    if enabled:
        typer.echo("Grant Screen Recording permission if macOS prompts you.")


@app.command()
def start(
    foreground: bool = typer.Option(
        False,
        "--foreground",
        "-f",
        help="Record in this terminal with live status and keyboard toggles.",
    ),
    screen: bool = typer.Option(
        False,
        "--screen",
        "-s",
        help="Also capture screen (and system audio when available).",
    ),
    mic: list[int] = typer.Option(
        [],
        "--mic",
        "-m",
        help="Microphone device id(s). Repeat flag for multiple mics. Defaults to system default.",
    ),
) -> None:
    """Start recording mics (and optionally screen) to /tmp/recording.wav."""
    _ensure_not_recording()
    _clear_pid()
    clear_session_files()

    if RECORDING_PATH.exists():
        RECORDING_PATH.unlink()

    mic_ids = mic or [get_default_mic_id()]
    conversation_id = start_conversation_session()
    _init_session(mic_ids, screen)

    if screen and find_system_audio_device() is None:
        typer.echo(
            "Warning: no system-audio loopback device found "
            "(BlackHole / Loopback / Soundflower). "
            "Screen capture will not include meeting/system sound.",
            err=True,
        )

    if foreground:
        _write_pid(os.getpid())
        LOCK_FILE.write_text("recording")
        typer.echo(f"Conversation id: {conversation_id}")
        typer.echo(f"Transcript path: {conversation_path(conversation_id)}")
        try:
            run_recording_session(
                mic_ids=mic_ids,
                screen_enabled=screen,
                interactive=True,
            )
        finally:
            _clear_pid()
        return

    log_path = Path("/tmp/gapscribe.recorder.log")
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        _recorder_command(mic_ids, screen, interactive=False),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_file.close()
    _write_pid(proc.pid)
    LOCK_FILE.write_text("recording")
    typer.echo(f"Recording started (PID {proc.pid}).")
    typer.echo(f"Conversation id: {conversation_id}")
    typer.echo(f"Transcript path: {conversation_path(conversation_id)}")
    typer.echo(f"Mics: {mic_ids}" + (" | screen: on" if screen else ""))
    typer.echo("Toggle mics:    python main.py toggle-mic <id>")
    typer.echo("Toggle screen:  python main.py toggle-screen")
    typer.echo("Check status:   python main.py status")
    typer.echo("Finish:         python main.py stop")


@app.command()
def stop(
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Optional source ISO language code (e.g. fa). Omit for auto-detection.",
    ),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Parallel minute workers (default: auto, max 4).",
    ),
) -> None:
    """Stop recording, translate minute-by-minute, and write /tmp/conversation_<id>.md."""
    pid = _read_pid()
    if pid is None or not _is_process_running(pid):
        _clear_pid()
        if not RECORDING_PATH.exists():
            typer.echo("No active recording found.")
            raise typer.Exit(code=1)
        typer.echo("No active recorder process; using existing /tmp/recording.wav.")
    else:
        typer.echo(f"Stopping recorder (PID {pid})...")
        os.kill(pid, signal.SIGTERM)
        if not _wait_for_process(pid):
            typer.echo("Recorder did not exit in time; sending SIGKILL.", err=True)
            os.kill(pid, signal.SIGKILL)
            _wait_for_process(pid, timeout=5.0)
        _clear_pid()

    deadline = time.monotonic() + 15.0
    while not RECORDING_PATH.exists() and time.monotonic() < deadline:
        time.sleep(0.2)

    if not RECORDING_PATH.exists():
        mic_fallback = Path("/tmp/gapscribe_mics.wav")
        if mic_fallback.exists() and mic_fallback.stat().st_size > 44:
            typer.echo(
                "Final mix missing; recovering microphone audio from "
                f"{mic_fallback}.",
                err=True,
            )
            RECORDING_PATH.write_bytes(mic_fallback.read_bytes())
        else:
            log_path = Path("/tmp/gapscribe.recorder.log")
            typer.echo("Recording file not found at /tmp/recording.wav.", err=True)
            if log_path.exists():
                typer.echo(f"Recorder log: {log_path}", err=True)
            raise typer.Exit(code=1)

    typer.echo("Translating with Whisper large-v3 (minute-by-minute)...")
    transcript_path = resolve_conversation_path()
    try:
        output = transcribe(
            audio_path=RECORDING_PATH,
            output_path=transcript_path,
            language=language,
            workers=workers or None,
        )
    except Exception as exc:
        typer.echo(f"Translation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    cleanup_recording(RECORDING_PATH)
    clear_session_files()
    typer.echo(f"Translation saved to {output}")


@app.command()
def convert(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Audio or video file."),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output markdown path. Defaults to /tmp/conversation_<id>.md",
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Optional source ISO language code (e.g. fa). Omit for auto-detection.",
    ),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Parallel minute workers (default: auto, max 4).",
    ),
) -> None:
    """Translate an existing audio or video file to English markdown (minute-by-minute)."""
    typer.echo(f"Preparing media: {file}")
    typer.echo("Translating with Whisper large-v3 (minute-by-minute)...")
    try:
        result = transcribe_media_file(
            source_path=file,
            output_path=output,
            language=language,
            workers=workers or None,
        )
    except Exception as exc:
        typer.echo(f"Conversion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Translation saved to {result}")


if __name__ == "__main__":
    _require_apple_silicon()
    app()
