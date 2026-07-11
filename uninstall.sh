#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> GapScribe uninstaller"

if [[ -d .venv ]]; then
  echo "Removing .venv..."
  rm -rf .venv
else
  echo ".venv not found, skipping."
fi

if [[ -d models ]]; then
  echo "Removing ./models cache..."
  rm -rf models
else
  echo "./models not found, skipping."
fi

echo "Removing temporary audio and transcript files from /tmp/..."
rm -f /tmp/recording.wav /tmp/gapscribe.pid /tmp/gapscribe.lock
rm -f /tmp/gapscribe.control.json /tmp/gapscribe.state.json /tmp/gapscribe.session.json
rm -f /tmp/conversation_*.txt
rm -f /tmp/gapscribe_mics.wav /tmp/gapscribe_screen.wav /tmp/gapscribe_screen.mp4
rm -f /tmp/gapscribe_mic_*.wav /tmp/gapscribe_convert_*.wav

echo "GapScribe uninstalled."
