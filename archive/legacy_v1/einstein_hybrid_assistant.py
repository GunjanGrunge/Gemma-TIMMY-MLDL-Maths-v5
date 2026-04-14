"""
Calculator-backed local Einstein assistant.

This is the reliability layer for exact math. The fine-tuned Gemma adapter is
useful for style and explanation, but deterministic calculators should own
numeric results.

Examples:
python einstein_hybrid_assistant.py --question "A car moves east at 20 m/s and north at 15 m/s. Compute the speed vector magnitude and direction."
python einstein_hybrid_assistant.py --question "Naive Bayes: prior P(spam)=0.4, P(not spam)=0.6, P(offer|spam)=0.5, P(offer|not spam)=0.1. Compute P(spam|offer)."
"""

from __future__ import annotations

import argparse
import re

from math_calculators import naive_bayes_binary, r4, sigmoid_backprop, velocity_vector


def answer_velocity(question: str) -> str | None:
    match = re.search(
        r"east(?:=| at )\s*([-+]?\d+(?:\.\d+)?)\s*m/s.*north(?:=| at )\s*([-+]?\d+(?:\.\d+)?)\s*m/s",
        question,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"v(?:elocity)?\s*=\s*<\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*>",
            question,
            re.IGNORECASE,
        )
    if not match:
        return None

    vx = float(match.group(1))
    vy = float(match.group(2))
    result = velocity_vector(vx, vy)
    squared_sum = vx * vx + vy * vy
    return (
        "Problem: Compute 2D velocity magnitude and direction.\n"
        "Method: Use ||v|| = sqrt(vx^2 + vy^2) and theta = atan2(vy, vx).\n"
        f"Calculation: ||v|| = sqrt({r4(vx)}^2 + {r4(vy)}^2) = sqrt({r4(squared_sum)}) = {r4(result['speed'])} m/s. "
        f"theta = atan2({r4(vy)}, {r4(vx)}) = {r4(result['angle_deg'])} degrees.\n"
        f"Result: Speed = {r4(result['speed'])} m/s; direction = {r4(result['angle_deg'])} degrees north of east.\n"
        "Practical note: Use atan2 so the quadrant is handled correctly."
    )


def answer_naive_bayes(question: str) -> str | None:
    prior_match = re.search(r"P\(spam\)\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.IGNORECASE)
    like_pos_match = re.search(r"P\(([^|()]+)\|spam\)\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.IGNORECASE)
    like_neg_match = re.search(r"P\(([^|()]+)\|not spam\)\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.IGNORECASE)
    if not (prior_match and like_pos_match and like_neg_match):
        return None

    token = like_pos_match.group(1).strip()
    prior = float(prior_match.group(1))
    like_pos = float(like_pos_match.group(2))
    like_neg = float(like_neg_match.group(2))
    result = naive_bayes_binary(prior, like_pos, like_neg)
    prior_neg = 1 - prior
    numerator = like_pos * prior
    return (
        "Problem: Compute a binary Naive Bayes posterior.\n"
        f"Method: P(spam|{token}) = P({token}|spam)P(spam) / P({token}).\n"
        f"Calculation: P({token}) = {r4(like_pos)}*{r4(prior)} + {r4(like_neg)}*{r4(prior_neg)} = {r4(result['evidence'])}. "
        f"Numerator = {r4(numerator)}. Posterior = {r4(numerator)} / {r4(result['evidence'])} = {r4(result['posterior'])}.\n"
        f"Result: P(spam|{token}) = {r4(result['posterior'])}.\n"
        "Practical note: The numerator is likelihood times prior; do not divide evidence by itself."
    )


def answer_sigmoid_backprop(question: str) -> str | None:
    x = re.search(r"x\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.IGNORECASE)
    w = re.search(r"w\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.IGNORECASE)
    b = re.search(r"b\s*=\s*([-+]?\d+(?:\.\d+)?)", question, re.IGNORECASE)
    y = re.search(r"(?:target\s*)?y\s*=\s*([01])", question, re.IGNORECASE)
    if not (x and w and b and y):
        return None

    x_val = float(x.group(1))
    w_val = float(w.group(1))
    b_val = float(b.group(1))
    y_val = int(y.group(1))
    result = sigmoid_backprop(x_val, w_val, b_val, y_val)
    direction = "increase w" if result["dldw"] < 0 else "decrease w"
    return (
        "Problem: Compute one-neuron sigmoid backprop with BCE.\n"
        "Method: z=wx+b, a=sigmoid(z), dL/dz=a-y, dL/dw=(a-y)x.\n"
        f"Calculation: z={r4(w_val)}*{r4(x_val)}+{r4(b_val)}={r4(result['z'])}. "
        f"a=sigmoid({r4(result['z'])})={r4(result['a'])}. "
        f"dL/dz={r4(result['a'])}-{y_val}={r4(result['dldz'])}. "
        f"dL/dw={r4(result['dldz'])}*{r4(x_val)}={r4(result['dldw'])}.\n"
        f"Result: z={r4(result['z'])}, a={r4(result['a'])}, dL/dz={r4(result['dldz'])}, dL/dw={r4(result['dldw'])}; gradient descent should {direction}.\n"
        "Practical note: With sigmoid plus BCE, use dL/dz=a-y."
    )


def answer_seasonal_residual(question: str) -> str | None:
    if not re.search(r"(residual|acf|autocorrelation)", question, re.IGNORECASE):
        return None
    lag_match = re.search(r"lag\s*(\d+)", question, re.IGNORECASE)
    lag = int(lag_match.group(1)) if lag_match else 12
    period = "yearly" if lag == 12 else f"{lag}-period"
    return (
        "Problem: Diagnose seasonal residual autocorrelation.\n"
        "Method: Residual autocorrelation means the forecast errors still contain predictable temporal structure.\n"
        f"Calculation: For monthly data, lag {lag} corresponds to a {period} seasonal pattern. Check residual ACF/PACF and a Ljung-Box test at seasonal lags.\n"
        f"Result: Add seasonal structure: SARIMA seasonal terms with period {lag}, seasonal differencing (1-B^{lag}), Fourier/month indicators, or lag-{lag} features; then recheck residual autocorrelation.\n"
        "Practical note: A good forecast model should leave residuals close to white noise."
    )


def answer_logistic_l2_derivation(question: str) -> str | None:
    if not re.search(r"logistic regression", question, re.IGNORECASE):
        return None
    if not re.search(r"L2|lambda|regulari", question, re.IGNORECASE):
        return None
    return (
        "Problem: Derive the L2-regularized logistic regression gradient.\n"
        "Method: Let p=sigmoid(Xw). Differentiate the mean BCE term and add the L2 derivative.\n"
        "Calculation: For each example, dBCE/dz_i=p_i-y_i and dz_i/dw=x_i. Therefore the mean data-loss gradient is (1/n) sum_i (p_i-y_i)x_i = (1/n)X^T(p-y). The derivative of (lambda/2)||w||^2 is lambda*w.\n"
        "Result: grad J(w) = (1/n)X^T(p-y) + lambda*w.\n"
        "Practical note: Larger lambda shrinks weights toward zero, reducing variance but increasing bias if too large."
    )


def answer_question(question: str) -> str:
    for solver in [
        answer_velocity,
        answer_naive_bayes,
        answer_sigmoid_backprop,
        answer_seasonal_residual,
        answer_logistic_l2_derivation,
    ]:
        response = solver(question)
        if response:
            return response
    return (
        "Problem: No deterministic calculator matched this prompt.\n"
        "Method: Route this to the fine-tuned Gemma explanation model or add a new calculator pattern.\n"
        "Calculation: Not computed.\n"
        "Result: Unsupported by the current hybrid calculator.\n"
        "Practical note: Add a parser/calculator for this task before relying on exact numeric output."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", action="append", required=True)
    args = parser.parse_args()

    for idx, question in enumerate(args.question, start=1):
        print("=" * 80)
        print(f"Question {idx}: {question}")
        print("-" * 80)
        print(answer_question(question))
        print()


if __name__ == "__main__":
    main()
