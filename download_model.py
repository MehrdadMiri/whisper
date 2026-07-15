#!/usr/bin/env python3
"""Pre-download the Whisper model with install-time progress."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, try_to_load_from_cache
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

MODEL_NAME = "large-v3-turbo"
REPO_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
DOWNLOAD_ROOT = Path("./models")
FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
    "model.bin",
)


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.2f} GB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _local_path(filename: str) -> Path | None:
    path = try_to_load_from_cache(REPO_ID, filename, cache_dir=str(DOWNLOAD_ROOT))
    if not isinstance(path, str):
        return None
    resolved = Path(path)
    if not resolved.is_file():
        return None
    # Incomplete Hugging Face downloads leave .incomplete siblings; a real
    # cache entry is a normal file (or symlink to a finished blob).
    return resolved


def _cached_files() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for filename in FILES:
        path = _local_path(filename)
        if path is not None:
            found[filename] = path
    return found


def _remote_sizes() -> dict[str, int]:
    info = HfApi().repo_info(REPO_ID, files_metadata=True)
    return {sibling.rfilename: (sibling.size or 0) for sibling in info.siblings}


def _make_tqdm(progress: Progress, task_id: int):
    class _RichTqdm:
        def __init__(self, *args, **kwargs):
            total = kwargs.get("total")
            initial = kwargs.get("initial", 0)
            if total is not None:
                progress.update(task_id, total=total)
            if initial:
                progress.update(task_id, completed=initial)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def update(self, n: int | float | None = 1) -> None:
            if n:
                progress.update(task_id, advance=n)

        def update_transfer(self, n: int | float | None = 1) -> None:
            self.update(n)

        def close(self) -> None:
            pass

        def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
            pass

        def set_transfer_postfix_str(self, postfix: str, refresh: bool = False) -> None:
            pass

    return _RichTqdm


def main() -> None:
    console = Console()
    cached = _cached_files()
    missing = [name for name in FILES if name not in cached]
    cached_bytes = sum(path.stat().st_size for path in cached.values())

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("Model", MODEL_NAME)
    table.add_row("Repository", REPO_ID)
    table.add_row("Destination", str(DOWNLOAD_ROOT.resolve()))
    console.print(table)
    console.print()

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

    console.print(
        f"Found {len(cached)}/{len(FILES)} files cached "
        f"({_format_size(cached_bytes)}); downloading {len(missing)} remaining…"
    )
    for filename in FILES:
        if filename in cached:
            size = cached[filename].stat().st_size
            console.print(
                f"  [green]✓[/green] {filename} "
                f"(cached, {_format_size(size)})"
            )

    sizes = _remote_sizes()
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

    for filename in missing:
        file_size = sizes.get(filename, 0)
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
            task_id = progress.add_task(filename, total=file_size or None)
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                cache_dir=str(DOWNLOAD_ROOT),
                tqdm_class=_make_tqdm(progress, task_id),
            )
            if file_size:
                progress.update(task_id, completed=file_size)

        console.print(f"  [green]✓[/green] {filename} ({_format_size(file_size)})")

    console.print("\n[green]Model downloaded successfully.[/green]")


if __name__ == "__main__":
    main()
