"""Deterministic deep-learning calculators for the Einstein DL assistant."""

from __future__ import annotations

import math
from statistics import mean


def r4(value: float) -> str:
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(r4(v) for v in values) + "]"


def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(x - max_logit) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_norm(values: list[float]) -> float:
    return math.sqrt(sum(x * x for x in values))


def fmt_matrix(values: list[list[float]]) -> str:
    return "[" + ", ".join(fmt_list(row) for row in values) + "]"


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def broadcast_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> tuple[int, ...]:
    rev_a = list(reversed(shape_a))
    rev_b = list(reversed(shape_b))
    out: list[int] = []
    for idx in range(max(len(rev_a), len(rev_b))):
        dim_a = rev_a[idx] if idx < len(rev_a) else 1
        dim_b = rev_b[idx] if idx < len(rev_b) else 1
        if dim_a == 1:
            out.append(dim_b)
        elif dim_b == 1 or dim_a == dim_b:
            out.append(dim_a)
        else:
            raise ValueError(f"Shapes {shape_a} and {shape_b} cannot broadcast")
    return tuple(reversed(out))


def sigmoid_bce_backprop(x: float, w: float, b: float, y: int) -> dict:
    z = w * x + b
    a = sigmoid(z)
    loss = -(y * math.log(a) + (1 - y) * math.log(1 - a))
    dldz = a - y
    return {
        "z": z,
        "a": a,
        "loss": loss,
        "dldz": dldz,
        "dldw": dldz * x,
        "dldb": dldz,
    }


def softmax_cross_entropy(logits: list[float], true_class: int) -> dict:
    probs = softmax(logits)
    loss = -math.log(probs[true_class])
    gradient = probs[:]
    gradient[true_class] -= 1
    return {"probs": probs, "loss": loss, "gradient": gradient}


def binary_cross_entropy_probability(prob: float, y: int) -> dict:
    loss = -(y * math.log(prob) + (1 - y) * math.log(1 - prob))
    dldp = -(y / prob) + ((1 - y) / (1 - prob))
    return {"loss": loss, "dldp": dldp}


def classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def multiclass_accuracy(labels: list[int], predictions: list[int]) -> dict:
    correct = sum(int(y == yhat) for y, yhat in zip(labels, predictions))
    return {"correct": correct, "accuracy": correct / len(labels)}


def gradient_descent_update(weights: list[float], gradients: list[float], lr: float) -> dict:
    return {"updated_weights": [w - lr * g for w, g in zip(weights, gradients)]}


def weight_decay_sgd_update(weights: list[float], gradients: list[float], lr: float, lam: float) -> dict:
    total_gradient = [g + lam * w for w, g in zip(weights, gradients)]
    updated_weights = [w - lr * tg for w, tg in zip(weights, total_gradient)]
    return {"total_gradient": total_gradient, "updated_weights": updated_weights}


def cosine_similarity(a: list[float], b: list[float]) -> dict:
    numerator = dot(a, b)
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    return {"dot": numerator, "norm_a": norm_a, "norm_b": norm_b, "cosine": numerator / (norm_a * norm_b)}


def semantic_search_rank(query: list[float], documents: list[list[float]]) -> dict:
    scores = [cosine_similarity(query, doc)["cosine"] for doc in documents]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return {"scores": scores, "best": best}


def matmul_shape(a_shape: tuple[int, int], b_shape: tuple[int, int]) -> dict:
    if a_shape[1] != b_shape[0]:
        raise ValueError(f"Invalid matmul shapes: {a_shape} and {b_shape}")
    return {"output_shape": (a_shape[0], b_shape[1])}


def adam_update(
    w: float,
    grad: float,
    m: float,
    v: float,
    t: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> dict:
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1**t)
    v_hat = v_new / (1 - beta2**t)
    step = lr * m_hat / (math.sqrt(v_hat) + eps)
    return {
        "m_new": m_new,
        "v_new": v_new,
        "m_hat": m_hat,
        "v_hat": v_hat,
        "step": step,
        "w_new": w - step,
    }


def momentum_sgd_update(w: float, grad: float, velocity: float, lr: float, momentum: float) -> dict:
    new_velocity = momentum * velocity + grad
    return {"velocity": new_velocity, "w_new": w - lr * new_velocity}


def gradient_clip_by_norm(gradient: list[float], max_norm: float) -> dict:
    norm = math.sqrt(sum(g * g for g in gradient))
    scale = 1.0 if norm <= max_norm else max_norm / norm
    return {"norm": norm, "scale": scale, "clipped": [g * scale for g in gradient]}


def conv2d_output_shape(
    height: int,
    width: int,
    kernel: int,
    stride: int,
    padding: int,
    output_channels: int,
) -> dict:
    out_h = math.floor((height + 2 * padding - kernel) / stride) + 1
    out_w = math.floor((width + 2 * padding - kernel) / stride) + 1
    return {"channels": output_channels, "height": out_h, "width": out_w}


def scaled_dot_product_attention(
    query: list[float],
    keys: list[list[float]],
    values: list[list[float]],
) -> dict:
    dim = len(query)
    scores = [dot(query, key) / math.sqrt(dim) for key in keys]
    weights = softmax(scores)
    output = [
        sum(weights[i] * values[i][j] for i in range(len(values)))
        for j in range(len(values[0]))
    ]
    return {"scores": scores, "weights": weights, "output": output}


def normalize_forward(values: list[float], gamma: float, beta: float, eps: float) -> dict:
    mu = mean(values)
    variance = mean([(x - mu) ** 2 for x in values])
    xhat = [(x - mu) / math.sqrt(variance + eps) for x in values]
    output = [gamma * x + beta for x in xhat]
    return {"mean": mu, "variance": variance, "xhat": xhat, "output": output}


def inverted_dropout(values: list[float], mask: list[int], keep_prob: float) -> dict:
    return {"output": [x * m / keep_prob for x, m in zip(values, mask)]}


def activation_derivative(kind: str, value: float) -> dict:
    kind = kind.lower()
    if kind == "relu":
        derivative = 1.0 if value > 0 else 0.0
    elif kind == "tanh":
        derivative = 1 - math.tanh(value) ** 2
    elif kind == "sigmoid":
        s = sigmoid(value)
        derivative = s * (1 - s)
    else:
        raise ValueError(f"Unsupported activation: {kind}")
    return {"activation": kind, "x": value, "derivative": derivative}


def linear_mse_backprop(x: list[float], w: list[float], b: float, y: float) -> dict:
    yhat = dot(x, w) + b
    err = yhat - y
    return {
        "yhat": yhat,
        "loss": err * err,
        "grad_w": [2 * err * xi for xi in x],
        "grad_b": 2 * err,
    }
