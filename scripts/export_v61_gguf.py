import os
import sys

# Windows + small VRAM fallback / disable eager compilation errors
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

try:
    from unsloth import FastLanguageModel
except ImportError:
    print("Error: Unsloth is not installed in the current environment.")
    sys.exit(1)

LORA_PATH = "archive/gemmafinetunearchive/outputs/v61/models/gemma_timmy_martha_v61_consultant_lora_502_chattemplate"
EXPORT_DIR = "outputs/v61/models/gemma_timmy_martha_v61_consultant_merged"

def export_gguf():
    print(f"Loading LoRA adapter from {LORA_PATH} (base will be auto-detected)")
    
    # Load the model and adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_PATH,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Exporting merged model to GGUF format in {EXPORT_DIR}")
    # This natively handles merging weights and compiling via llama.cpp backend
    # We output purely a gguf q4_k_m model
    model.save_pretrained_gguf(EXPORT_DIR, tokenizer, quantization_method="q4_k_m")
    
    print("Export Complete! GGUF files are located in:", EXPORT_DIR)

if __name__ == "__main__":
    export_gguf()
