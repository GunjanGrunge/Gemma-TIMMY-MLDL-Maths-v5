"""Stable wrapper API around the existing flat-file V6 hybrid runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass

from einstein_v6_hybrid_assistant import (
    answer_question as _answer_question,
    answer_v6_dl,
    answer_v6_guardrail,
    answer_v6_ml,
    answer_v6_stats_da_forecast,
    answer_v6_training_consulting,
)


SOLVERS = (
    answer_v6_guardrail,
    answer_v6_ml,
    answer_v6_dl,
    answer_v6_stats_da_forecast,
    answer_v6_training_consulting,
)


@dataclass(frozen=True)
class HybridAnswer:
    question: str
    route: str
    output: str
    status: str | None = None
    in_scope: bool = True


def route_question(question: str) -> str | None:
    """Return the first matching V6 solver name, or None if the query is out of scope."""
    for solver in SOLVERS:
        if solver(question):
            return solver.__name__
    return None


def answer(question: str) -> str:
    """Return the raw V6 hybrid response text."""
    return _answer_question(question)


def answer_structured(question: str) -> HybridAnswer:
    """Return a stable structured response for downstream callers."""
    route = route_question(question)
    output = _answer_question(question)

    status = None
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        status = payload.get("status")

    return HybridAnswer(
        question=question,
        route=route or "answer_v52_question",
        output=output,
        status=status,
        in_scope=route is not None,
    )
