#!/usr/bin/env bash
set -euo pipefail

MARTHA_SOURCE="${MARTHA_SOURCE:-/mnt/c/Users/Bot/Desktop/martha}"
TIMMYBOT_SOURCE="${TIMMYBOT_SOURCE:-/mnt/c/Users/Bot/Desktop/TimmyBot}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"

if [[ ! -d "$MARTHA_SOURCE" ]]; then
  echo "Missing Martha source: $MARTHA_SOURCE" >&2
  exit 1
fi

if [[ ! -d "$TIMMYBOT_SOURCE" ]]; then
  echo "Missing TimmyBot source: $TIMMYBOT_SOURCE" >&2
  exit 1
fi

if [[ ! -w "$(dirname "$WORKSPACE_ROOT")" ]]; then
  echo "Creating $WORKSPACE_ROOT requires sudo/root permissions." >&2
  echo "Run: sudo bash scripts/setup_wsl_workspace.sh" >&2
  exit 1
fi

mkdir -p "$WORKSPACE_ROOT"

find "$WORKSPACE_ROOT" -mindepth 1 -maxdepth 1 \
  ! -name martha \
  ! -name TimmyBot \
  -exec rm -rf {} +

ln -sfn "$MARTHA_SOURCE" "$WORKSPACE_ROOT/martha"
ln -sfn "$TIMMYBOT_SOURCE" "$WORKSPACE_ROOT/TimmyBot"

echo "WSL workspace ready:"
ls -la "$WORKSPACE_ROOT"
