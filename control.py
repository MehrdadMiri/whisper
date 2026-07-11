"""Shared session state for live recording control."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4

CONTROL_PATH = Path("/tmp/gapscribe.control.json")
STATE_PATH = Path("/tmp/gapscribe.state.json")
SESSION_PATH = Path("/tmp/gapscribe.session.json")


def new_conversation_id() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"


def conversation_path(conversation_id: str) -> Path:
    return Path(f"/tmp/conversation_{conversation_id}.txt")


def save_conversation_id(conversation_id: str) -> None:
    SESSION_PATH.write_text(
        json.dumps({"conversation_id": conversation_id}),
        encoding="utf-8",
    )


def load_conversation_id() -> str | None:
    if not SESSION_PATH.exists():
        return None
    data = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    value = data.get("conversation_id")
    return str(value) if value else None


def start_conversation_session() -> str:
    conversation_id = new_conversation_id()
    save_conversation_id(conversation_id)
    return conversation_id


def resolve_conversation_path() -> Path:
    conversation_id = load_conversation_id() or start_conversation_session()
    return conversation_path(conversation_id)


@dataclass
class SessionConfig:
    enabled_mic_ids: list[int] = field(default_factory=list)
    screen_enabled: bool = False

    @classmethod
    def load(cls, path: Path = CONTROL_PATH) -> SessionConfig:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            enabled_mic_ids=list(data.get("enabled_mic_ids", [])),
            screen_enabled=bool(data.get("screen_enabled", False)),
        )

    def save(self, path: Path = CONTROL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def toggle_mic(self, mic_id: int) -> bool:
        if mic_id in self.enabled_mic_ids:
            self.enabled_mic_ids.remove(mic_id)
            return False
        self.enabled_mic_ids.append(mic_id)
        return True

    def toggle_screen(self) -> bool:
        self.screen_enabled = not self.screen_enabled
        return self.screen_enabled


def write_state(
    *,
    enabled_mic_ids: list[int],
    screen_enabled: bool,
    screen_active: bool,
    elapsed: float,
    mic_labels: dict[int, str],
) -> None:
    STATE_PATH.write_text(
        json.dumps(
            {
                "enabled_mic_ids": enabled_mic_ids,
                "screen_enabled": screen_enabled,
                "screen_active": screen_active,
                "elapsed_seconds": round(elapsed, 1),
                "mic_labels": {str(k): v for k, v in mic_labels.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def clear_session_files() -> None:
    for path in (CONTROL_PATH, STATE_PATH, SESSION_PATH):
        path.unlink(missing_ok=True)
