#!/usr/bin/env python3
"""Pre-download Whisper large-v3 from ModelScope (not Hugging Face)."""

from __future__ import annotations

import hashlib
import http.client
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

MODEL_NAME = "large-v3"
REPO_ID = "keepitsimple/faster-whisper-large-v3"
BASE_URL = f"https://www.modelscope.cn/models/{REPO_ID}/resolve/master"
DOWNLOAD_ROOT = Path("./models")
MODEL_DIR = DOWNLOAD_ROOT / "faster-whisper-large-v3"

# Same CTranslate2 conversion as Systran/faster-whisper-large-v3.
FILES: dict[str, dict[str, int | str | None]] = {
    "config.json": {"size": 2394, "sha256": None},
    "preprocessor_config.json": {"size": 340, "sha256": None},
    "tokenizer.json": {"size": 2_480_617, "sha256": None},
    "vocabulary.json": {"size": 1_068_114, "sha256": None},
    "model.bin": {
        "size": 3_087_284_237,
        "sha256": "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1",
    },
}

USER_AGENT = "gapscribe-model-downloader/1.0"
CHUNK_SIZE = 1024 * 1024
MAX_RETRIES = 50
RETRY_BACKOFF_SECONDS = 2.0
RETRY_BACKOFF_MAX_SECONDS = 60.0
# Socket idle timeout while reading the body (connection setup uses the same).
READ_TIMEOUT_SECONDS = 60


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _file_url(filename: str) -> str:
    return f"{BASE_URL}/{filename}"


def _part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _is_complete(path: Path, meta: dict[str, int | str | None]) -> bool:
    if not path.is_file():
        return False
    expected_size = meta.get("size")
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        return False
    expected_sha = meta.get("sha256")
    if isinstance(expected_sha, str):
        return _sha256_file(path) == expected_sha
    return True


def _prepare_part_file(dest: Path, expected_size: int) -> Path:
    """Return a .part path, reclaiming any incomplete final file for resume."""
    part = _part_path(dest)
    if dest.is_file():
        size = dest.stat().st_size
        if expected_size and size == expected_size:
            return part
        # Incomplete/corrupt final file: continue from it instead of restarting.
        if part.is_file():
            if dest.stat().st_size > part.stat().st_size:
                part.unlink()
                dest.replace(part)
            else:
                dest.unlink()
        else:
            dest.replace(part)
    return part


def _open_download(url: str, existing: int):
    headers = {"User-Agent": USER_AGENT}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=READ_TIMEOUT_SECONDS)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, KeyboardInterrupt):
        return False
    if isinstance(
        exc,
        (
            TimeoutError,
            socket.timeout,
            socket.gaierror,
            ConnectionError,
            BrokenPipeError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            urllib.error.URLError,
        ),
    ):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 425, 429, 500, 502, 503, 504}
    return False


def _download_file(
    filename: str,
    dest: Path,
    *,
    console: Console,
    expected_size: int,
) -> None:
    url = _file_url(filename)
    part_path = _prepare_part_file(dest, expected_size)
    attempt = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=36),
        TaskProgressColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task(
            filename,
            total=expected_size or None,
            completed=part_path.stat().st_size if part_path.is_file() else 0,
        )

        while True:
            existing = part_path.stat().st_size if part_path.is_file() else 0
            progress.update(task_id, completed=existing)

            if expected_size and existing >= expected_size:
                break

            try:
                response = _open_download(url, existing)
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and existing > 0:
                    # Server says range is satisfied — treat as complete.
                    break
                if _is_retryable(exc) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = min(
                        RETRY_BACKOFF_MAX_SECONDS,
                        RETRY_BACKOFF_SECONDS * attempt,
                    )
                    console.print(
                        f"  [yellow]Retry {attempt}/{MAX_RETRIES}[/yellow] "
                        f"{filename} after HTTP {exc.code} "
                        f"(resume from {_format_size(existing)}, wait {delay:.0f}s)"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"Failed to download {filename}: HTTP {exc.code}"
                ) from exc
            except Exception as exc:
                if _is_retryable(exc) and attempt < MAX_RETRIES:
                    attempt += 1
                    delay = min(
                        RETRY_BACKOFF_MAX_SECONDS,
                        RETRY_BACKOFF_SECONDS * attempt,
                    )
                    console.print(
                        f"  [yellow]Retry {attempt}/{MAX_RETRIES}[/yellow] "
                        f"{filename} after {type(exc).__name__}: {exc} "
                        f"(resume from {_format_size(existing)}, wait {delay:.0f}s)"
                    )
                    time.sleep(delay)
                    continue
                raise

            stream_failed = False
            try:
                # If the server ignored Range, restart the part file.
                content_range = response.headers.get("Content-Range", "")
                status = getattr(response, "status", None) or response.getcode()
                if existing > 0 and status == 200 and not content_range:
                    console.print(
                        f"  [yellow]Server ignored resume for {filename}; "
                        "restarting file[/yellow]"
                    )
                    existing = 0
                    part_path.unlink(missing_ok=True)
                    progress.update(task_id, completed=0)

                mode = "ab" if existing > 0 else "wb"
                with part_path.open(mode) as handle:
                    while True:
                        try:
                            chunk = response.read(CHUNK_SIZE)
                        except Exception as exc:
                            if _is_retryable(exc) and attempt < MAX_RETRIES:
                                attempt += 1
                                delay = min(
                                    RETRY_BACKOFF_MAX_SECONDS,
                                    RETRY_BACKOFF_SECONDS * attempt,
                                )
                                current = (
                                    part_path.stat().st_size
                                    if part_path.is_file()
                                    else 0
                                )
                                console.print(
                                    f"  [yellow]Retry {attempt}/{MAX_RETRIES}[/yellow] "
                                    f"{filename} after {type(exc).__name__}: {exc} "
                                    f"(resume from {_format_size(current)}, "
                                    f"wait {delay:.0f}s)"
                                )
                                time.sleep(delay)
                                stream_failed = True
                                break
                            raise
                        if not chunk:
                            break
                        handle.write(chunk)
                        handle.flush()
                        progress.update(task_id, advance=len(chunk))
            finally:
                try:
                    response.close()
                except Exception:
                    pass

            if stream_failed:
                continue

            current = part_path.stat().st_size if part_path.is_file() else 0
            if expected_size and current < expected_size:
                if attempt < MAX_RETRIES:
                    attempt += 1
                    delay = min(
                        RETRY_BACKOFF_MAX_SECONDS,
                        RETRY_BACKOFF_SECONDS * attempt,
                    )
                    console.print(
                        f"  [yellow]Retry {attempt}/{MAX_RETRIES}[/yellow] "
                        f"{filename}: connection closed early "
                        f"({_format_size(current)} / {_format_size(expected_size)}), "
                        f"wait {delay:.0f}s"
                    )
                    time.sleep(delay)
                    continue
                raise RuntimeError(
                    f"{filename}: incomplete after download "
                    f"({_format_size(current)} != {_format_size(expected_size)}). "
                    "Partial file kept — re-run python download_model.py to continue."
                )

            attempt = 0
            break

    current = part_path.stat().st_size if part_path.is_file() else 0
    if expected_size and current != expected_size:
        # Keep .part for the next run; do not delete progress.
        raise RuntimeError(
            f"{filename}: incomplete after download "
            f"({_format_size(current)} != {_format_size(expected_size)}). "
            "Re-run python download_model.py to continue."
        )

    part_path.replace(dest)


def main() -> None:
    console = Console()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Model", MODEL_NAME)
    table.add_row("Source", f"ModelScope ({REPO_ID})")
    table.add_row("Destination", str(MODEL_DIR.resolve()))
    console.print(table)
    console.print()
    console.print(
        f"Direct model.bin URL (axel-friendly):\n  {_file_url('model.bin')}"
    )
    console.print()

    cached: dict[str, Path] = {}
    missing: list[str] = []
    for filename, meta in FILES.items():
        path = MODEL_DIR / filename
        if _is_complete(path, meta):
            cached[filename] = path
        else:
            missing.append(filename)

    cached_bytes = sum(path.stat().st_size for path in cached.values())
    if not missing:
        console.print(
            f"[green]Model already downloaded[/green] "
            f"({_format_size(cached_bytes)} on disk)."
        )
        for filename in FILES:
            size = cached[filename].stat().st_size
            console.print(
                f"  [green]✓[/green] {filename} "
                f"(cached, {_format_size(size)})"
            )
        return

    partial_bytes = 0
    for filename in missing:
        part = _part_path(MODEL_DIR / filename)
        dest = MODEL_DIR / filename
        if part.is_file():
            partial_bytes += part.stat().st_size
        elif dest.is_file():
            partial_bytes += dest.stat().st_size

    console.print(
        f"Found {len(cached)}/{len(FILES)} files complete "
        f"({_format_size(cached_bytes)}); "
        f"downloading {len(missing)} remaining"
        + (
            f" (resuming {_format_size(partial_bytes)} already on disk)"
            if partial_bytes
            else ""
        )
        + "…"
    )
    for filename in FILES:
        if filename in cached:
            size = cached[filename].stat().st_size
            console.print(
                f"  [green]✓[/green] {filename} "
                f"(cached, {_format_size(size)})"
            )

    try:
        for filename in missing:
            meta = FILES[filename]
            dest = MODEL_DIR / filename
            expected_size = int(meta["size"] or 0)
            part = _part_path(dest)
            resumed = part.stat().st_size if part.is_file() else (
                dest.stat().st_size if dest.is_file() else 0
            )
            if resumed and not (dest.is_file() and dest.stat().st_size == expected_size):
                console.print(
                    f"  Resuming {filename} from {_format_size(resumed)}…"
                )

            _download_file(
                filename,
                dest,
                console=console,
                expected_size=expected_size,
            )
            expected_sha = meta.get("sha256")
            if isinstance(expected_sha, str):
                console.print(f"  Verifying {filename}…")
                actual = _sha256_file(dest)
                if actual != expected_sha:
                    dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"{filename}: sha256 mismatch "
                        f"(got {actual}, expected {expected_sha}). "
                        "Deleted corrupt file; re-run to download again."
                    )
            console.print(
                f"  [green]✓[/green] {filename} ({_format_size(expected_size)})"
            )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Download interrupted.[/yellow] "
            "Partial files were kept — re-run "
            "[bold]python download_model.py[/bold] to continue."
        )
        raise SystemExit(130) from None

    console.print("\n[green]Model downloaded successfully.[/green]")


if __name__ == "__main__":
    main()
