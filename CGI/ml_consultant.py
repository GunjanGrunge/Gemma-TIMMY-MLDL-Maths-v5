from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


DEFAULT_RUNTIME = Path("runtime")


def resolve_runtime(runtime_path: Path) -> Path:
    if runtime_path.is_file():
        runtime_path = runtime_path.parent

    candidates = [runtime_path]
    if runtime_path.exists():
        candidates.extend(path for path in runtime_path.iterdir() if path.is_dir())

    for candidate in candidates:
        if (candidate / "einstein_v6_hybrid_assistant.py").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find einstein_v6_hybrid_assistant.py under {runtime_path}. "
        "Run download_runtime.py first or pass --runtime-path to a downloaded repo snapshot."
    )


def load_answer_fn(runtime_root: Path):
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    module = importlib.import_module("einstein_v6_hybrid_assistant")
    return module.answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Portable Martha V6 ML consultant CLI.")
    parser.add_argument("--runtime-path", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    runtime_root = resolve_runtime(args.runtime_path)
    answer_question = load_answer_fn(runtime_root)

    rows = []
    for idx, question in enumerate(args.question, 1):
        answer = answer_question(question)
        row = {
            "index": idx,
            "question": question,
            "answer": answer,
            "runtime_root": str(runtime_root),
        }
        rows.append(row)

    if args.as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return

    for row in rows:
        print("=" * 80)
        print(f"Question {row['index']}: {row['question']}")
        print("-" * 80)
        print(row["answer"])
        print()


if __name__ == "__main__":
    main()
