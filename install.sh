#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Homebrew/bash under Rosetta report x86_64 even on M-series Macs.
if [[ "$(uname -m)" != "arm64" ]]; then
  exec arch -arm64 /bin/bash "$0" "$@"
fi

if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null)" != "1" ]]; then
  echo "Error: GapScribe is built for Apple Silicon (M-series) Macs only." >&2
  exit 1
fi

echo "==> GapScribe installer (Apple Silicon)"

if ! command -v brew &>/dev/null; then
  echo "Error: Homebrew is required. Install from https://brew.sh" >&2
  exit 1
fi

install_if_missing() {
  local pkg="$1"
  if brew list --formula "$pkg" &>/dev/null; then
    echo "    $pkg already installed."
  else
    echo "    Installing $pkg..."
    brew install "$pkg"
  fi
}

echo "==> Checking system dependencies (ffmpeg, portaudio)..."
install_if_missing ffmpeg
install_if_missing portaudio

if [[ -x /opt/homebrew/bin/python3 ]]; then
  PYTHON="/opt/homebrew/bin/python3"
else
  PYTHON="${PYTHON:-python3}"
fi

if ! command -v "$PYTHON" &>/dev/null; then
  echo "Error: arm64 python3 not found. Run: brew install python" >&2
  exit 1
fi

if ! file "$PYTHON" | grep -q "arm64"; then
  echo "Error: $PYTHON is not arm64-native. Use Homebrew Python: brew install python" >&2
  exit 1
fi

echo "==> Using Python: $PYTHON"

venv_has_x86_only_packages() {
  local site_packages so info
  site_packages=$(find .venv/lib -maxdepth 2 -type d -name 'site-packages' 2>/dev/null | head -1)
  [[ -n "$site_packages" ]] || return 1

  while IFS= read -r -d '' so; do
    info=$(file "$so")
    if echo "$info" | grep -q 'x86_64' && ! echo "$info" | grep -q 'arm64'; then
      return 0
    fi
  done < <(find "$site_packages" -name '*.so' -print0 2>/dev/null)

  return 1
}

echo "==> Creating virtual environment (.venv)..."
if [[ -d .venv ]]; then
  venv_python=".venv/bin/python"
  if [[ -x "$venv_python" ]] && file "$venv_python" | grep -q "arm64" && ! venv_has_x86_only_packages; then
    echo "    .venv already exists, reusing it."
  else
    echo "    Removing incompatible .venv and recreating..."
    rm -rf .venv
    "$PYTHON" -m venv .venv
  fi
else
  "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing Python packages..."
pip install faster-whisper sounddevice scipy typer numpy rich

echo "==> Pre-downloading Whisper large-v3 from ModelScope to ./models..."
python download_model.py

echo ""
echo "GapScribe installed successfully."
echo "Activate the environment: source .venv/bin/activate"
echo "Start recording:          python main.py start [--screen] [--mic ID]"
echo "Toggle during recording:  python main.py toggle-mic <id> | toggle-screen"
echo "Stop and translate:       python main.py stop"
echo "Convert a file:           python main.py convert meeting.mp4"
