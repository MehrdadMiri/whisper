#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> GapScribe installer (Apple Silicon)"

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Error: GapScribe is built for Apple Silicon (M-series) Macs only." >&2
  exit 1
fi

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

echo "==> Creating virtual environment (.venv)..."
if [[ -d .venv ]]; then
  venv_python=".venv/bin/python"
  if [[ -x "$venv_python" ]] && file "$venv_python" | grep -q "arm64"; then
    echo "    .venv already exists, reusing it."
  else
    echo "    Removing non-arm64 .venv and recreating..."
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
pip install faster-whisper sounddevice scipy typer numpy

echo "==> Pre-downloading Whisper large-v3-turbo model to ./models..."
mkdir -p models
python - <<'PY'
from faster_whisper import WhisperModel

model = WhisperModel(
    "large-v3-turbo",
    device="cpu",
    compute_type="int8_float16",
    download_root="./models",
    cpu_threads=0,
)
print("Model downloaded successfully.")
PY

echo ""
echo "GapScribe installed successfully."
echo "Activate the environment: source .venv/bin/activate"
echo "Start recording:          python main.py start [--screen] [--mic ID]"
echo "Toggle during recording:  python main.py toggle-mic <id> | toggle-screen"
echo "Stop and transcribe:      python main.py stop"
echo "Convert a file:           python main.py convert meeting.mp4"
