"""Deterministic calculators for the local Einstein math assistant."""

from __future__ import annotations

import math
from statistics import mean


def r4(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def pct(value: float) -> str:
    return f"{100 * value:.2f}%".rstrip("0").rstrip(".")


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def velocity_vector(vx: float, vy: float) -> dict:
    speed = math.hypot(vx, vy)
    angle = math.degrees(math.atan2(vy, vx))
    return {"speed": speed, "angle_deg": angle}


def naive_bayes_binary(prior_pos: float, likelihood_pos: float, likelihood_neg: float) -> dict:
    prior_neg = 1 - prior_pos
    evidence = likelihood_pos * prior_pos + likelihood_neg * prior_neg
    posterior = (likelihood_pos * prior_pos) / evidence
    return {"evidence": evidence, "posterior": posterior}


def sigmoid_backprop(x: float, w: float, b: float, y: int) -> dict:
    z = w * x + b
    a = sigmoid(z)
    loss = -(y * math.log(a) + (1 - y) * math.log(1 - a))
    dldz = a - y
    dldw = dldz * x
    dldb = dldz
    return {"z": z, "a": a, "loss": loss, "dldz": dldz, "dldw": dldw, "dldb": dldb}


def logistic_l2_gradient(x: list[float], w: list[float], y: int, lam: float) -> dict:
    z = sum(xi * wi for xi, wi in zip(x, w))
    p = sigmoid(z)
    gradient = [(p - y) * xi + lam * wi for xi, wi in zip(x, w)]
    return {"z": z, "p": p, "gradient": gradient}


def classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def mape(actuals: list[float], forecasts: list[float]) -> dict:
    errors = [abs((a - f) / a) for a, f in zip(actuals, forecasts)]
    return {"mape": mean(errors), "errors": errors}


def rmse(actuals: list[float], forecasts: list[float]) -> dict:
    squared_errors = [(a - f) ** 2 for a, f in zip(actuals, forecasts)]
    return {"rmse": math.sqrt(mean(squared_errors)), "squared_errors": squared_errors}


def two_proportion_z_test(n1: int, n2: int, x1: int, x2: int) -> dict:
    p1 = x1 / n1
    p2 = x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se
    return {"p1": p1, "p2": p2, "pooled": pooled, "se": se, "z": z, "significant_005": abs(z) > 1.96}


def sharpe_ratio(returns: list[float], risk_free: float = 0.0) -> dict:
    avg = mean(returns)
    variance = sum((r - avg) ** 2 for r in returns) / (len(returns) - 1)
    stdev = math.sqrt(variance)
    return {"mean": avg, "sample_stdev": stdev, "sharpe": (avg - risk_free) / stdev}
