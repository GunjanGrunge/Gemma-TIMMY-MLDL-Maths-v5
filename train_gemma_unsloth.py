# train_gemma_unsloth.py
from pathlib import Path
import json
import os
import random

# Windows + small VRAM fallback: set before importing torch/unsloth.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import torch


BASE_MODEL = os.getenv("UNSLOTH_BASE_MODEL", "unsloth/gemma-2-2b-it-bnb-4bit")
MAX_SEQ_LENGTH = int(os.getenv("UNSLOTH_MAX_SEQ_LENGTH", "1024"))
LORA_R = int(os.getenv("UNSLOTH_LORA_R", "8"))
LORA_ALPHA = int(os.getenv("UNSLOTH_LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("UNSLOTH_LORA_DROPOUT", "0"))
PER_DEVICE_BATCH_SIZE = int(os.getenv("UNSLOTH_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("UNSLOTH_GRAD_ACCUM", "8"))
MAX_STEPS = int(os.getenv("UNSLOTH_MAX_STEPS", "1"))
NUM_TRAIN_EPOCHS = float(os.getenv("UNSLOTH_NUM_EPOCHS", "1"))
LEARNING_RATE = float(os.getenv("UNSLOTH_LEARNING_RATE", "2e-4"))
LR_SCHEDULER_TYPE = os.getenv("UNSLOTH_LR_SCHEDULER", "cosine")
MIN_LEARNING_RATE = float(os.getenv("UNSLOTH_MIN_LEARNING_RATE", "0"))
WARMUP_STEPS = int(os.getenv("UNSLOTH_WARMUP_STEPS", "50"))
MAX_GRAD_NORM = float(os.getenv("UNSLOTH_MAX_GRAD_NORM", "1.0"))
SEED = int(os.getenv("UNSLOTH_SEED", "3407"))
TRAIN_DATA_PATH = os.getenv("UNSLOTH_TRAIN_DATA")
TRAIN_DATA_FORMAT = os.getenv("UNSLOTH_TRAIN_FORMAT", "auto").lower()
USE_CHAT_TEMPLATE = os.getenv("UNSLOTH_USE_CHAT_TEMPLATE", "1").lower() not in {"0", "false", "no"}
OUTPUT_DIR = Path(os.getenv("UNSLOTH_OUTPUT_DIR", "outputs/v4/models/gemma_math_lora"))

try:
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def require_cuda_torch() -> None:
    if "+cpu" in torch.__version__ or not torch.cuda.is_available():
        raise SystemExit(
            "CUDA PyTorch is not available in this virtual environment.\n"
            f"Detected torch version: {torch.__version__}\n"
            f"torch.version.cuda: {torch.version.cuda}\n"
            f"torch.cuda.is_available(): {torch.cuda.is_available()}\n\n"
            "Install a CUDA-enabled PyTorch build inside .venv, then rerun training. "
            "Installing the CUDA toolkit alone is not enough if PyTorch is still the CPU wheel."
        )


require_cuda_torch()

try:
    import unsloth  # noqa: F401 - import before trl/transformers so Unsloth can patch them
    from unsloth import FastLanguageModel
except NotImplementedError as e:
    raise SystemExit(
        "Unsloth requires an NVIDIA GPU and CUDA runtime. "
        "No supported GPU was detected in this environment. "
        "Run on a CUDA-enabled machine or use an alternative training path."
    ) from e

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# 1. Load base model (4-bit QLoRA)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

# 2. Prepare dataset

def format_alpaca(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"

def format_chat(example):
    messages = example.get("messages", [])
    if USE_CHAT_TEMPLATE and hasattr(tokenizer, "apply_chat_template"):
        system_prompt = ""
        normalized_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if not content:
                continue
            if role == "system":
                system_prompt = content
            elif role in {"user", "assistant"}:
                normalized_messages.append({"role": role, "content": content})

        if system_prompt and normalized_messages:
            first_user_index = next(
                (index for index, msg in enumerate(normalized_messages) if msg["role"] == "user"),
                None,
            )
            if first_user_index is not None:
                normalized_messages[first_user_index]["content"] = (
                    f"{system_prompt}\n\n{normalized_messages[first_user_index]['content']}"
                )

        if normalized_messages:
            try:
                return tokenizer.apply_chat_template(
                    normalized_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as exc:
                print(f"Falling back to plain chat formatting: {exc}")

    user_prompt = ""
    assistant_response = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")
        elif msg.get("role") == "assistant":
            assistant_response = msg.get("content", "")
    return f"### Instruction:\n{user_prompt}\n\n### Input:\n\n### Response:\n{assistant_response}"

def build_alpaca_dataset(chat_path: Path, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with chat_path.open("r", encoding="utf-8") as fin, save_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            example = json.loads(line)
            messages = example.get("messages", [])
            user_prompt = ""
            assistant_response = ""
            for msg in messages:
                if msg.get("role") == "user" and not user_prompt:
                    user_prompt = msg.get("content", "")
                elif msg.get("role") == "assistant" and not assistant_response:
                    assistant_response = msg.get("content", "")

            if not user_prompt or not assistant_response:
                continue

            item = {
                "instruction": user_prompt,
                "input": "",
                "output": assistant_response,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    print(f"Saved {count} Alpaca-format examples to {save_path}")
    return save_path

def get_dataset_path():
    if TRAIN_DATA_PATH:
        path = Path(TRAIN_DATA_PATH)
        if not path.exists():
            raise FileNotFoundError(f"UNSLOTH_TRAIN_DATA does not exist: {path}")
        if TRAIN_DATA_FORMAT == "chat":
            return path, format_chat
        if TRAIN_DATA_FORMAT == "alpaca":
            return path, format_alpaca
        if path.name.endswith("_chat.jsonl"):
            return path, format_chat
        return path, format_alpaca

    candidates = [
        (Path("outputs/v6/data/v6_combined_train_chat.jsonl"), format_chat),
        (Path("outputs/v6/data/v6_curated_train_chat.jsonl"), format_chat),
        (Path("outputs/v52/data/v52_combined_train_chat.jsonl"), format_chat),
        (Path("outputs/v52/data/v52_advanced_train_chat.jsonl"), format_chat),
        (Path("outputs/v51/data/v51_combined_train_chat.jsonl"), format_chat),
        (Path("outputs/v5/data/v5_dl_train_chat.jsonl"), format_chat),
        (Path("outputs/v4/data/v4_train.jsonl"), format_alpaca),
        (Path("outputs/v4/data/v4_train_chat.jsonl"), format_chat),
        (Path("outputs/v3/data/v3_train.jsonl"), format_alpaca),
        (Path("outputs/v3/data/v3_train_chat.jsonl"), format_chat),
        (Path("outputs/v2/data/v2_train.jsonl"), format_alpaca),
        (Path("outputs/v2/data/v2_train_chat.jsonl"), format_chat),
        (Path("outputs/v2/data/v2_quant_chat.jsonl"), format_chat),
        (Path("outputs/v1/data/v1_full_train.jsonl"), format_alpaca),
        (Path("outputs/v1/data/v1_full_train_chat.jsonl"), format_chat),
        (Path("outputs/v1/data/v1_grounded_train.jsonl"), format_alpaca),
        (Path("outputs/v1/data/v1_grounded_chat.jsonl"), format_chat),
        (Path("outputs/v1/data/v1_quant_train.jsonl"), format_alpaca),
        (Path("outputs/v1/data/v1_quant_chat.jsonl"), format_chat),
        (Path("outputs/shared/math_glossary_train.jsonl"), format_alpaca),
        (Path("outputs/shared/math_finetune_chat.jsonl"), format_chat),
        # Legacy flat paths retained for compatibility with older runs.
        (Path("outputs/v1_full_train.jsonl"), format_alpaca),
        (Path("outputs/v1_full_train_chat.jsonl"), format_chat),
        (Path("outputs/v1_grounded_train.jsonl"), format_alpaca),
        (Path("outputs/v1_grounded_chat.jsonl"), format_chat),
        (Path("outputs/v1_quant_train.jsonl"), format_alpaca),
        (Path("outputs/v1_quant_chat.jsonl"), format_chat),
        (Path("outputs/math_glossary_train.jsonl"), format_alpaca),
        (Path("outputs/math_finetune_chat.jsonl"), format_chat),
    ]
    for path, formatter in candidates:
        if path.exists():
            return path, formatter

    raise FileNotFoundError(
        "No training dataset found. Create outputs/v4/data/v4_train_chat.jsonl or set UNSLOTH_TRAIN_DATA."
    )

data_path, formatter = get_dataset_path()
print(f"Dataset: {data_path}")
print(f"Output dir: {OUTPUT_DIR}")


def load_training_texts(path: Path, formatter):
    """Load JSONL as plain training text.

    The generated datasets keep rich nested metadata for eval/reporting, but
    training only needs prompt/response text. Loading only text avoids Arrow
    schema issues when metadata.expected has task-specific shapes.
    """
    rows = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                text = formatter(example)
            except Exception as exc:
                skipped += 1
                print(f"Skipping malformed row {line_number}: {exc}")
                continue
            if text.strip():
                rows.append({"_raw_text": text})
    if not rows:
        raise RuntimeError(f"No usable training rows found in {path}")
    if skipped:
        print(f"Skipped {skipped} malformed rows.")
    return Dataset.from_list(rows)


dataset = load_training_texts(data_path, formatter)

def add_training_text(example):
    text = example["_raw_text"]
    token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    token_count = len(token_ids)
    max_training_tokens = max(1, MAX_SEQ_LENGTH - 2)
    was_truncated = token_count > max_training_tokens
    if was_truncated:
        text = tokenizer.decode(token_ids[:max_training_tokens], skip_special_tokens=True)
    return {
        "text": text,
        "_token_count": min(token_count, max_training_tokens),
        "_was_truncated": was_truncated,
    }

dataset = dataset.map(add_training_text)
original_count = len(dataset)
truncated_count = sum(1 for was_truncated in dataset["_was_truncated"] if was_truncated)
max_training_tokens = max(1, MAX_SEQ_LENGTH - 2)
dataset = dataset.filter(lambda x: x["_token_count"] <= max_training_tokens)
kept_count = len(dataset)
if kept_count == 0:
    raise RuntimeError(
        f"No examples fit UNSLOTH_MAX_SEQ_LENGTH={MAX_SEQ_LENGTH}. "
        "Increase UNSLOTH_MAX_SEQ_LENGTH or shorten the training data."
    )
dataset = dataset.remove_columns([column for column in dataset.column_names if column != "text"])
print(
    f"Training on {kept_count}/{original_count} examples "
    f"with <= {max_training_tokens} tokens before special tokens "
    f"({truncated_count} truncated)."
)

# 3. Train
training_args_kwargs = {
    "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "warmup_steps": WARMUP_STEPS,
    "max_grad_norm": MAX_GRAD_NORM,
    "max_steps": MAX_STEPS if MAX_STEPS > 0 else -1,
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "fp16": not torch.cuda.is_bf16_supported(),
    "bf16": torch.cuda.is_bf16_supported(),
    "logging_steps": 20,
    "output_dir": str(OUTPUT_DIR),
    "optim": "adamw_8bit",
    "lr_scheduler_type": LR_SCHEDULER_TYPE,
    "seed": SEED,
    "data_seed": SEED,
}
if LR_SCHEDULER_TYPE == "cosine_with_min_lr" and MIN_LEARNING_RATE > 0:
    training_args_kwargs["lr_scheduler_kwargs"] = {"min_lr": MIN_LEARNING_RATE}

try:
    training_args = TrainingArguments(**training_args_kwargs)
except TypeError:
    # Older Transformers builds may not support lr_scheduler_kwargs.
    training_args_kwargs.pop("lr_scheduler_kwargs", None)
    training_args = TrainingArguments(**training_args_kwargs)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

trainer.train()
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
