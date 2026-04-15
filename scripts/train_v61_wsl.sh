#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
MARTHA_DIR="${MARTHA_DIR:-$WORKSPACE_ROOT/martha}"
TIMMYBOT_DIR="${TIMMYBOT_DIR:-$WORKSPACE_ROOT/TimmyBot}"
VENV_DIR="${VENV_DIR:-$MARTHA_DIR/.venv_wsl}"
LOCK_FILE="${LOCK_FILE:-$MARTHA_DIR/outputs/training.lock}"

if [[ ! -d "$MARTHA_DIR" ]]; then
  echo "Missing Martha workspace: $MARTHA_DIR" >&2
  echo "Run: sudo bash scripts/setup_wsl_workspace.sh" >&2
  exit 1
fi

if [[ ! -d "$TIMMYBOT_DIR" ]]; then
  echo "Missing TimmyBot workspace: $TIMMYBOT_DIR" >&2
  echo "Run: sudo bash scripts/setup_wsl_workspace.sh" >&2
  exit 1
fi

mkdir -p "$MARTHA_DIR/outputs"

if [[ -e "$LOCK_FILE" ]]; then
  echo "Training lock exists: $LOCK_FILE" >&2
  echo "Another training job may be running. Verify before removing it." >&2
  exit 1
fi

cleanup() {
  rm -f "$LOCK_FILE"
}
trap cleanup EXIT

echo "$$ $(date -Is)" > "$LOCK_FILE"

cd "$MARTHA_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Missing WSL virtualenv: $VENV_DIR" >&2
  echo "Create it with the commands in docs/WSL_V6_TRAINING.md" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"

export PYTHONPATH="$MARTHA_DIR:$TIMMYBOT_DIR:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

# V6.1 consultant training defaults.
export UNSLOTH_BASE_MODEL="${UNSLOTH_BASE_MODEL:-unsloth/gemma-2-2b-it-bnb-4bit}"
export UNSLOTH_TRAIN_DATA="${UNSLOTH_TRAIN_DATA:-outputs/v61/data/v61_consultant_combined_train_chat.jsonl}"
export UNSLOTH_TRAIN_FORMAT="${UNSLOTH_TRAIN_FORMAT:-chat}"
export UNSLOTH_OUTPUT_DIR="${UNSLOTH_OUTPUT_DIR:-outputs/v61/models/gemma_timmy_martha_v61_consultant_lora}"
export UNSLOTH_MAX_SEQ_LENGTH="${UNSLOTH_MAX_SEQ_LENGTH:-1024}"
export UNSLOTH_BATCH_SIZE="${UNSLOTH_BATCH_SIZE:-1}"
export UNSLOTH_GRAD_ACCUM="${UNSLOTH_GRAD_ACCUM:-4}"
export UNSLOTH_MAX_STEPS="${UNSLOTH_MAX_STEPS:-0}"
export UNSLOTH_NUM_EPOCHS="${UNSLOTH_NUM_EPOCHS:-3}"
export UNSLOTH_LEARNING_RATE="${UNSLOTH_LEARNING_RATE:-1e-5}"
export UNSLOTH_LR_SCHEDULER="${UNSLOTH_LR_SCHEDULER:-cosine}"
export UNSLOTH_MIN_LEARNING_RATE="${UNSLOTH_MIN_LEARNING_RATE:-0}"
export UNSLOTH_WARMUP_STEPS="${UNSLOTH_WARMUP_STEPS:-10}"
export UNSLOTH_MAX_GRAD_NORM="${UNSLOTH_MAX_GRAD_NORM:-1.0}"
export UNSLOTH_SEED="${UNSLOTH_SEED:-42}"

python -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.version.cuda); print("cuda available:", torch.cuda.is_available()); print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")'

python train_gemma_unsloth.py
