"""CLI entrypoint for the public V6 wrapper."""

from __future__ import annotations

import argparse
import json

from .api import answer_structured


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the scoped Martha V6 hybrid assistant. "
            "Best for ML, DL, stats, forecasting, funnel math, and numeric guardrails."
        )
    )
    parser.add_argument("--question", action="append", required=True, help="Question to answer. Can be passed multiple times.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a compact structured JSON-like record instead of plain text output.",
    )
    args = parser.parse_args()

    for index, question in enumerate(args.question, start=1):
        response = answer_structured(question)
        if args.json:
            print(json.dumps(
                {
                    "question": response.question,
                    "route": response.route,
                    "status": response.status,
                    "in_scope": response.in_scope,
                    "output": response.output,
                },
                ensure_ascii=False,
            ))
        else:
            print("=" * 80)
            print(f"Question {index}: {question}")
            print("-" * 80)
            print(response.output)
            print()


if __name__ == "__main__":
    main()
