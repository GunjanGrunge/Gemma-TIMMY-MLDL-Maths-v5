"""Check whether the current venv has CUDA-enabled PyTorch for Unsloth."""

import torch


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            print(f"device {index}: {props.name}")
            print(f"  capability: {props.major}.{props.minor}")
            print(f"  total memory GB: {props.total_memory / 1024**3:.2f}")
    else:
        print("No CUDA device is visible to PyTorch in this environment.")


if __name__ == "__main__":
    main()
