"""
Generate V5 deep-learning focused training data.

V5 is intentionally scoped to DL operations where a useful assistant must be
precise: backprop, softmax CE, optimizers, clipping, normalization, dropout,
attention scores, and tensor shapes.

Outputs:
- outputs/v5/data/v5_dl_train_chat.jsonl
- outputs/v5/data/v5_dl_eval_cases.jsonl
- outputs/v5/reports/v5_dl_dataset_report.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from statistics import mean


RANDOM_SEED = 5505
TRAIN_PATH = Path("outputs/v5/data/v5_dl_train_chat.jsonl")
EVAL_PATH = Path("outputs/v5/data/v5_dl_eval_cases.jsonl")
REPORT_PATH = Path("outputs/v5/reports/v5_dl_dataset_report.md")

SYSTEM_PROMPT = (
    "You are a deep-learning math assistant. Give one correct formula path, "
    "one calculation, one final result, and one concise debugging note. "
    "Do not repeat notes or invent references."
)


@dataclass(frozen=True)
class Example:
    domain: str
    task_type: str
    difficulty: str
    question: str
    answer: str
    expected: dict


def r4(value: float) -> str:
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


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


def fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(r4(v) for v in values) + "]"


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


def answer(problem: str, method: str, calculation: str, result: str, note: str) -> str:
    return (
        f"Problem: {problem}\n"
        f"Method: {method}\n"
        f"Calculation: {calculation}\n"
        f"Result: {result}\n"
        f"Debug note: {note}"
    )


def chat_record(example: Example) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.question},
            {"role": "assistant", "content": example.answer},
        ],
        "metadata": {
            "dataset": "v5_deep_learning_verified",
            "domain": example.domain,
            "task_type": example.task_type,
            "difficulty": example.difficulty,
            "expected": example.expected,
        },
    }


def sigmoid_bce_example(x: float, w: float, b: float, y: int, idx: int) -> Example:
    z = w * x + b
    a = sigmoid(z)
    loss = -(y * math.log(a) + (1 - y) * math.log(1 - a))
    dldz = a - y
    dldw = dldz * x
    dldb = dldz
    direction = "increase w" if dldw < 0 else "decrease w"
    q = f"DL backprop case {idx}: one sigmoid neuron has x={x}, w={w}, b={b}, y={y}. Compute z, a, BCE loss, dL/dz, dL/dw, dL/db, and update direction."
    a_text = answer(
        "Compute sigmoid neuron backprop with binary cross entropy.",
        "Use z=wx+b, a=sigmoid(z), BCE=-[y log(a)+(1-y)log(1-a)], dL/dz=a-y, dL/dw=(a-y)x, dL/db=a-y.",
        f"z={r4(w)}*{r4(x)}+{r4(b)}={r4(z)}. a={r4(a)}. loss={r4(loss)}. dL/dz={r4(dldz)}. dL/dw={r4(dldw)}. dL/db={r4(dldb)}.",
        f"z={r4(z)}, a={r4(a)}, loss={r4(loss)}, dL/dz={r4(dldz)}, dL/dw={r4(dldw)}, dL/db={r4(dldb)}; gradient descent should {direction}.",
        "For sigmoid plus BCE, do not multiply by sigmoid derivative again; dL/dz=a-y already includes it.",
    )
    return Example("Deep Learning", "sigmoid_bce_backprop", "expert", q, a_text, {"z": round(z, 4), "a": round(a, 4), "loss": round(loss, 4), "dldz": round(dldz, 4), "dldw": round(dldw, 4), "dldb": round(dldb, 4), "direction": direction})


def softmax_ce_example(logits: list[float], y: int, idx: int) -> Example:
    probs = softmax(logits)
    loss = -math.log(probs[y])
    grad = probs[:]
    grad[y] -= 1
    q = f"Softmax CE case {idx}: logits={fmt_list(logits)}, true_class={y}. Compute probabilities, loss, and dL/dlogits."
    a_text = answer(
        "Compute softmax cross-entropy gradient.",
        "Use p_i=exp(z_i)/sum_j exp(z_j), loss=-log(p_y), and dL/dz=p-one_hot(y).",
        f"softmax={fmt_list(probs)}. loss=-log({r4(probs[y])})={r4(loss)}. gradient={fmt_list(grad)}.",
        f"probabilities={fmt_list(probs)}, loss={r4(loss)}, dL/dlogits={fmt_list(grad)}.",
        "For softmax plus CE, the stable gradient is p-y_onehot.",
    )
    return Example("Deep Learning", "softmax_cross_entropy", "expert", q, a_text, {"probs": [round(x, 4) for x in probs], "loss": round(loss, 4), "gradient": [round(x, 4) for x in grad]})


def relu_mlp_example(x1: float, x2: float, w1: float, w2: float, b: float, v: float, target: float, idx: int) -> Example:
    h_pre = w1 * x1 + w2 * x2 + b
    h = max(0.0, h_pre)
    yhat = v * h
    loss = 0.5 * (yhat - target) ** 2
    dldyhat = yhat - target
    dldv = dldyhat * h
    dldh = dldyhat * v
    relu_grad = 1.0 if h_pre > 0 else 0.0
    dldhpre = dldh * relu_grad
    dldw1 = dldhpre * x1
    dldw2 = dldhpre * x2
    dldb = dldhpre
    q = f"Two-layer ReLU case {idx}: x=[{x1},{x2}], hidden preactivation h_pre=w1*x1+w2*x2+b with w1={w1}, w2={w2}, b={b}, output weight v={v}, target={target}. Compute forward pass and gradients for 0.5*(yhat-y)^2."
    a_text = answer(
        "Compute a one-hidden-unit ReLU network forward/backward pass.",
        "Forward: h_pre=w1*x1+w2*x2+b, h=max(0,h_pre), yhat=v*h. Backward: dL/dyhat=yhat-y, dL/dv=(dL/dyhat)h, dL/dhpre=(dL/dyhat)v*1[h_pre>0].",
        f"h_pre={r4(h_pre)}, h={r4(h)}, yhat={r4(yhat)}, loss={r4(loss)}. dL/dv={r4(dldv)}, dL/dw1={r4(dldw1)}, dL/dw2={r4(dldw2)}, dL/db={r4(dldb)}.",
        f"loss={r4(loss)}, gradients: dL/dv={r4(dldv)}, dL/dw1={r4(dldw1)}, dL/dw2={r4(dldw2)}, dL/db={r4(dldb)}.",
        "If h_pre<=0, ReLU blocks gradients to w1, w2, and b.",
    )
    return Example("Deep Learning", "relu_mlp_backprop", "nerd", q, a_text, {"h_pre": round(h_pre, 4), "h": round(h, 4), "yhat": round(yhat, 4), "loss": round(loss, 4), "dldv": round(dldv, 4), "dldw1": round(dldw1, 4), "dldw2": round(dldw2, 4), "dldb": round(dldb, 4)})


def mse_linear_example(x: list[float], w: list[float], b: float, y: float, idx: int) -> Example:
    yhat = dot(x, w) + b
    err = yhat - y
    loss = err * err
    grad_w = [2 * err * xi for xi in x]
    grad_b = 2 * err
    q = f"Linear output MSE case {idx}: x={fmt_list(x)}, w={fmt_list(w)}, b={b}, y={y}. Compute yhat, loss, dL/dw, dL/db for L=(yhat-y)^2."
    a_text = answer(
        "Compute linear-layer MSE gradients.",
        "Use yhat=w dot x+b, L=(yhat-y)^2, dL/dw=2(yhat-y)x, dL/db=2(yhat-y).",
        f"yhat={r4(yhat)}, error={r4(err)}, loss={r4(loss)}, dL/dw={fmt_list(grad_w)}, dL/db={r4(grad_b)}.",
        f"yhat={r4(yhat)}, loss={r4(loss)}, dL/dw={fmt_list(grad_w)}, dL/db={r4(grad_b)}.",
        "The factor 2 appears because this loss is not using the 0.5 convention.",
    )
    return Example("Deep Learning", "mse_linear_backprop", "average", q, a_text, {"yhat": round(yhat, 4), "loss": round(loss, 4), "grad_w": [round(x, 4) for x in grad_w], "grad_b": round(grad_b, 4)})


def adam_example(w: float, grad: float, m: float, v: float, t: int, lr: float, beta1: float, beta2: float, eps: float, idx: int) -> Example:
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad * grad
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    update = lr * m_hat / (math.sqrt(v_hat) + eps)
    w_new = w - update
    q = f"Adam case {idx}: w={w}, grad={grad}, m={m}, v={v}, t={t}, lr={lr}, beta1={beta1}, beta2={beta2}, eps={eps}. Compute one Adam update."
    a_text = answer(
        "Compute one Adam optimizer update.",
        "m=beta1*m+(1-beta1)g, v=beta2*v+(1-beta2)g^2, m_hat=m/(1-beta1^t), v_hat=v/(1-beta2^t), w=w-lr*m_hat/(sqrt(v_hat)+eps).",
        f"m_new={r4(m_new)}, v_new={r4(v_new)}, m_hat={r4(m_hat)}, v_hat={r4(v_hat)}, update={r4(update)}.",
        f"w_new={r4(w_new)}.",
        "Bias correction matters most in early Adam steps.",
    )
    return Example("Deep Learning", "adam_update", "advanced", q, a_text, {"w_new": round(w_new, 4), "m_new": round(m_new, 4), "v_new": round(v_new, 4), "m_hat": round(m_hat, 4), "v_hat": round(v_hat, 4)})


def momentum_example(w: float, grad: float, velocity: float, lr: float, momentum: float, idx: int) -> Example:
    v_new = momentum * velocity + grad
    w_new = w - lr * v_new
    q = f"Momentum SGD case {idx}: w={w}, grad={grad}, previous_velocity={velocity}, lr={lr}, momentum={momentum}. Compute new velocity and weight."
    a_text = answer(
        "Compute one momentum SGD update.",
        "Use v_new=momentum*v_old+grad and w_new=w-lr*v_new.",
        f"v_new={momentum}*{velocity}+{grad}={r4(v_new)}. w_new={w}-{lr}*{r4(v_new)}={r4(w_new)}.",
        f"new_velocity={r4(v_new)}, w_new={r4(w_new)}.",
        "Momentum smooths noisy gradients but can overshoot if learning rate is too high.",
    )
    return Example("Deep Learning", "momentum_sgd_update", "average", q, a_text, {"velocity": round(v_new, 4), "w_new": round(w_new, 4)})


def gradient_clipping_example(grads: list[float], threshold: float, idx: int) -> Example:
    norm = l2_norm(grads)
    scale = min(1.0, threshold / norm)
    clipped = [g * scale for g in grads]
    q = f"Gradient clipping case {idx}: gradient={fmt_list(grads)}, max_norm={threshold}. Compute norm, scale, and clipped gradient."
    a_text = answer(
        "Clip a gradient vector by global L2 norm.",
        "If norm>max_norm, use clipped_g=g*(max_norm/norm); otherwise leave g unchanged.",
        f"norm=sqrt(sum(g_i^2))={r4(norm)}. scale=min(1,{threshold}/{r4(norm)})={r4(scale)}. clipped={fmt_list(clipped)}.",
        f"norm={r4(norm)}, scale={r4(scale)}, clipped_gradient={fmt_list(clipped)}.",
        "Gradient clipping changes magnitude, not direction.",
    )
    return Example("Deep Learning", "gradient_clipping", "average", q, a_text, {"norm": round(norm, 4), "scale": round(scale, 4), "clipped": [round(x, 4) for x in clipped]})


def activation_derivative_example(kind: str, value: float, idx: int) -> Example:
    if kind == "relu":
        derivative = 1.0 if value > 0 else 0.0
        formula = "d ReLU(x)/dx = 1 if x>0 else 0"
    elif kind == "tanh":
        derivative = 1 - math.tanh(value) ** 2
        formula = "d tanh(x)/dx = 1 - tanh(x)^2"
    else:
        s = sigmoid(value)
        derivative = s * (1 - s)
        formula = "d sigmoid(x)/dx = sigmoid(x)(1-sigmoid(x))"
    q = f"Activation derivative case {idx}: compute derivative of {kind} at x={value}."
    a_text = answer(
        f"Compute derivative of {kind}.",
        formula + ".",
        f"At x={value}, derivative={r4(derivative)}.",
        f"{kind}'({value})={r4(derivative)}.",
        "Activation derivatives control how much gradient passes backward.",
    )
    return Example("Deep Learning", "activation_derivative", "easy", q, a_text, {"derivative": round(derivative, 4)})


def batchnorm_example(values: list[float], gamma: float, beta: float, eps: float, idx: int) -> Example:
    mu = mean(values)
    var = mean([(x - mu) ** 2 for x in values])
    normalized = [(x - mu) / math.sqrt(var + eps) for x in values]
    output = [gamma * x + beta for x in normalized]
    q = f"BatchNorm case {idx}: activations={fmt_list(values)}, gamma={gamma}, beta={beta}, eps={eps}. Compute mean, variance, normalized values, and output."
    a_text = answer(
        "Compute BatchNorm forward pass for one feature across a mini-batch.",
        "mu=mean(x), var=mean((x-mu)^2), xhat=(x-mu)/sqrt(var+eps), y=gamma*xhat+beta.",
        f"mu={r4(mu)}, var={r4(var)}, xhat={fmt_list(normalized)}, output={fmt_list(output)}.",
        f"BatchNorm output={fmt_list(output)}.",
        "BatchNorm statistics are computed across the batch for each feature.",
    )
    return Example("Deep Learning", "batchnorm_forward", "advanced", q, a_text, {"mean": round(mu, 4), "variance": round(var, 4), "output": [round(x, 4) for x in output]})


def layernorm_example(values: list[float], gamma: float, beta: float, eps: float, idx: int) -> Example:
    mu = mean(values)
    var = mean([(x - mu) ** 2 for x in values])
    normalized = [(x - mu) / math.sqrt(var + eps) for x in values]
    output = [gamma * x + beta for x in normalized]
    q = f"LayerNorm case {idx}: vector={fmt_list(values)}, gamma={gamma}, beta={beta}, eps={eps}. Compute normalized output."
    a_text = answer(
        "Compute LayerNorm for one sample vector.",
        "LayerNorm normalizes across features within one sample: xhat=(x-mean(x))/sqrt(var(x)+eps), y=gamma*xhat+beta.",
        f"mean={r4(mu)}, var={r4(var)}, xhat={fmt_list(normalized)}, output={fmt_list(output)}.",
        f"LayerNorm output={fmt_list(output)}.",
        "LayerNorm does not depend on other batch examples.",
    )
    return Example("Deep Learning", "layernorm_forward", "advanced", q, a_text, {"mean": round(mu, 4), "variance": round(var, 4), "output": [round(x, 4) for x in output]})


def dropout_example(values: list[float], mask: list[int], keep_prob: float, idx: int) -> Example:
    output = [x * m / keep_prob for x, m in zip(values, mask)]
    q = f"Inverted dropout case {idx}: activations={fmt_list(values)}, mask={mask}, keep_prob={keep_prob}. Compute training-time output."
    a_text = answer(
        "Compute inverted dropout output.",
        "Use output=x*mask/keep_prob during training so expected activation scale stays unchanged.",
        f"output={fmt_list(output)}.",
        f"dropout_output={fmt_list(output)}.",
        "At inference, inverted dropout uses no mask and no extra scaling.",
    )
    return Example("Deep Learning", "dropout_forward", "average", q, a_text, {"output": [round(x, 4) for x in output]})


def attention_example(q_vec: list[float], keys: list[list[float]], values: list[list[float]], idx: int) -> Example:
    dim = len(q_vec)
    scores = [dot(q_vec, key) / math.sqrt(dim) for key in keys]
    weights = softmax(scores)
    output = [sum(weights[i] * values[i][j] for i in range(len(values))) for j in range(len(values[0]))]
    q = f"Attention case {idx}: q={fmt_list(q_vec)}, keys={fmt_matrix(keys)}, values={fmt_matrix(values)}. Compute scaled dot-product attention output."
    a_text = answer(
        "Compute scaled dot-product attention for one query.",
        "scores=qK^T/sqrt(d), weights=softmax(scores), output=sum_i weights_i*V_i.",
        f"scores={fmt_list(scores)}, weights={fmt_list(weights)}, output={fmt_list(output)}.",
        f"attention_output={fmt_list(output)}.",
        "Scaling by sqrt(d) prevents dot products from growing too large with dimension.",
    )
    return Example("Deep Learning", "attention_scaled_dot_product", "nerd", q, a_text, {"scores": [round(x, 4) for x in scores], "weights": [round(x, 4) for x in weights], "output": [round(x, 4) for x in output]})


def conv_shape_example(h: int, w: int, kernel: int, stride: int, padding: int, channels_out: int, idx: int) -> Example:
    out_h = math.floor((h + 2 * padding - kernel) / stride) + 1
    out_w = math.floor((w + 2 * padding - kernel) / stride) + 1
    q = f"CNN shape case {idx}: input HxW={h}x{w}, kernel={kernel}, stride={stride}, padding={padding}, output_channels={channels_out}. Compute output shape."
    a_text = answer(
        "Compute 2D convolution output shape.",
        "out=floor((in+2*padding-kernel)/stride)+1 for each spatial dimension.",
        f"out_h=floor(({h}+2*{padding}-{kernel})/{stride})+1={out_h}. out_w=floor(({w}+2*{padding}-{kernel})/{stride})+1={out_w}.",
        f"Output shape is {channels_out}x{out_h}x{out_w} if using channel-first format.",
        "Check divisibility; non-integer spatial sizes indicate a shape configuration problem.",
    )
    return Example("Deep Learning", "cnn_output_shape", "average", q, a_text, {"channels": channels_out, "height": out_h, "width": out_w})


def binary_ce_probability_example(prob: float, y: int, idx: int) -> Example:
    loss = -(y * math.log(prob) + (1 - y) * math.log(1 - prob))
    dlda = -(y / prob) + ((1 - y) / (1 - prob))
    q = f"Binary cross entropy case {idx}: predicted probability p={prob}, label y={y}. Compute BCE loss and dL/dp."
    a_text = answer(
        "Compute binary cross entropy from a predicted probability.",
        "Use BCE=-[y log(p)+(1-y)log(1-p)] and dL/dp=-(y/p)+(1-y)/(1-p).",
        f"loss={r4(loss)}. dL/dp={r4(dlda)}.",
        f"BCE={r4(loss)}, dL/dp={r4(dlda)}.",
        "Use probability p only after sigmoid; for logits, prefer the stable logits form.",
    )
    return Example("Deep Learning", "binary_cross_entropy_probability", "average", q, a_text, {"loss": round(loss, 4), "dldp": round(dlda, 4)})


def classification_metrics_example(tp: int, fp: int, fn: int, tn: int, idx: int) -> Example:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    q = f"Classification metrics case {idx}: TP={tp}, FP={fp}, FN={fn}, TN={tn}. Compute accuracy, precision, recall, specificity, and F1."
    a_text = answer(
        "Compute binary classification metrics from a confusion matrix.",
        "accuracy=(TP+TN)/N, precision=TP/(TP+FP), recall=TP/(TP+FN), specificity=TN/(TN+FP), F1=2PR/(P+R).",
        f"N={total}. accuracy={r4(accuracy)}, precision={r4(precision)}, recall={r4(recall)}, specificity={r4(specificity)}, F1={r4(f1)}.",
        f"accuracy={r4(accuracy)}, precision={r4(precision)}, recall={r4(recall)}, specificity={r4(specificity)}, F1={r4(f1)}.",
        "For imbalanced data, accuracy can look high while recall or precision is poor.",
    )
    return Example("Deep Learning", "classification_metrics", "average", q, a_text, {"accuracy": round(accuracy, 4), "precision": round(precision, 4), "recall": round(recall, 4), "specificity": round(specificity, 4), "f1": round(f1, 4)})


def multiclass_accuracy_example(labels: list[int], predictions: list[int], idx: int) -> Example:
    correct = sum(int(y == yhat) for y, yhat in zip(labels, predictions))
    accuracy = correct / len(labels)
    q = f"Multiclass accuracy case {idx}: labels={labels}, predictions={predictions}. Compute number correct and accuracy."
    a_text = answer(
        "Compute multiclass accuracy.",
        "Accuracy is the count of exactly matched predicted labels divided by total examples.",
        f"correct={correct} out of {len(labels)}. accuracy={correct}/{len(labels)}={r4(accuracy)}.",
        f"correct={correct}, accuracy={r4(accuracy)}.",
        "Accuracy treats all classes equally by count; inspect per-class recall when classes are imbalanced.",
    )
    return Example("Deep Learning", "multiclass_accuracy", "easy", q, a_text, {"correct": correct, "accuracy": round(accuracy, 4)})


def gradient_descent_example(weights: list[float], gradients: list[float], lr: float, idx: int) -> Example:
    new_weights = [w - lr * g for w, g in zip(weights, gradients)]
    q = f"Gradient descent case {idx}: weights={fmt_list(weights)}, gradients={fmt_list(gradients)}, lr={lr}. Compute updated weights."
    a_text = answer(
        "Compute one vanilla gradient descent update.",
        "Use w_new=w-lr*gradient elementwise.",
        f"w_new={fmt_list(new_weights)}.",
        f"updated_weights={fmt_list(new_weights)}.",
        "A positive gradient decreases the weight under gradient descent; a negative gradient increases it.",
    )
    return Example("Deep Learning", "gradient_descent_update", "average", q, a_text, {"updated_weights": [round(x, 4) for x in new_weights]})


def weight_decay_sgd_example(weights: list[float], gradients: list[float], lr: float, lam: float, idx: int) -> Example:
    total_grad = [g + lam * w for w, g in zip(weights, gradients)]
    new_weights = [w - lr * tg for w, tg in zip(weights, total_grad)]
    q = f"Weight decay SGD case {idx}: weights={fmt_list(weights)}, gradients={fmt_list(gradients)}, lr={lr}, lambda={lam}. Compute total gradient and updated weights."
    a_text = answer(
        "Compute SGD with L2 weight decay.",
        "Use total_gradient=gradient+lambda*w, then w_new=w-lr*total_gradient.",
        f"total_gradient={fmt_list(total_grad)}. w_new={fmt_list(new_weights)}.",
        f"total_gradient={fmt_list(total_grad)}, updated_weights={fmt_list(new_weights)}.",
        "Weight decay shrinks weights toward zero while still following the data gradient.",
    )
    return Example("Deep Learning", "weight_decay_sgd_update", "advanced", q, a_text, {"total_gradient": [round(x, 4) for x in total_grad], "updated_weights": [round(x, 4) for x in new_weights]})


def cosine_similarity_example(a: list[float], b: list[float], idx: int) -> Example:
    numerator = dot(a, b)
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    cosine = numerator / (norm_a * norm_b)
    q = f"Cosine similarity case {idx}: vector_a={fmt_list(a)}, vector_b={fmt_list(b)}. Compute cosine similarity."
    a_text = answer(
        "Compute cosine similarity between two embedding vectors.",
        "cosine(a,b)=(a dot b)/(||a|| ||b||).",
        f"dot={r4(numerator)}, ||a||={r4(norm_a)}, ||b||={r4(norm_b)}, cosine={r4(cosine)}.",
        f"cosine_similarity={r4(cosine)}.",
        "Cosine similarity measures direction alignment, not vector magnitude.",
    )
    return Example("Deep Learning", "cosine_similarity", "average", q, a_text, {"cosine": round(cosine, 4)})


def semantic_search_example(query: list[float], docs: list[list[float]], idx: int) -> Example:
    scores = [dot(query, doc) / (l2_norm(query) * l2_norm(doc)) for doc in docs]
    best = max(range(len(scores)), key=lambda i: scores[i])
    q = f"Semantic search case {idx}: query_embedding={fmt_list(query)}, document_embeddings={fmt_matrix(docs)}. Rank documents by cosine similarity."
    a_text = answer(
        "Rank embedding vectors for semantic search.",
        "Compute cosine similarity between the query embedding and each document embedding, then sort by descending score.",
        f"scores={fmt_list(scores)}. best_doc={best}.",
        f"best_document_index={best}, cosine_scores={fmt_list(scores)}.",
        "Normalize or use cosine similarity when embedding magnitude should not dominate semantic ranking.",
    )
    return Example("Deep Learning", "semantic_search_cosine_ranking", "advanced", q, a_text, {"scores": [round(x, 4) for x in scores], "best": best})


def matrix_multiply_example(a: list[list[float]], b: list[list[float]], idx: int) -> Example:
    product = matmul(a, b)
    q = f"Matrix multiplication case {idx}: A={fmt_matrix(a)}, B={fmt_matrix(b)}. Compute A@B."
    a_text = answer(
        "Compute matrix multiplication.",
        "For C=A@B, each C_ij is the dot product of row i of A and column j of B.",
        f"A@B={fmt_matrix(product)}.",
        f"product={fmt_matrix(product)}.",
        "Matrix multiplication requires inner dimensions to match.",
    )
    return Example("Deep Learning", "matrix_multiplication", "average", q, a_text, {"product": [[round(x, 4) for x in row] for row in product]})


def matmul_shape_example(a_shape: tuple[int, int], b_shape: tuple[int, int], idx: int) -> Example:
    out_shape = (a_shape[0], b_shape[1])
    q = f"Tensor matmul shape case {idx}: A shape={a_shape}, B shape={b_shape}. Compute output shape for A@B."
    a_text = answer(
        "Compute matrix multiplication output shape.",
        "For A with shape (m,k) and B with shape (k,n), A@B has shape (m,n).",
        f"m={a_shape[0]}, k={a_shape[1]}, n={b_shape[1]}, output_shape={out_shape}.",
        f"output_shape={out_shape}.",
        "The inner dimensions must match; otherwise matmul is invalid.",
    )
    return Example("Deep Learning", "tensor_matmul_shape", "easy", q, a_text, {"output_shape": out_shape})


def broadcasting_shape_example(shape_a: tuple[int, ...], shape_b: tuple[int, ...], idx: int) -> Example:
    out_shape = broadcast_shape(shape_a, shape_b)
    q = f"Broadcasting shape case {idx}: tensor A shape={shape_a}, tensor B shape={shape_b}. Compute broadcast output shape for elementwise add."
    a_text = answer(
        "Compute tensor broadcasting shape.",
        "Compare dimensions from the right; dimensions are compatible when equal or one of them is 1.",
        f"{shape_a} and {shape_b} broadcast to {out_shape}.",
        f"broadcast_output_shape={out_shape}.",
        "Broadcasting changes how tensors are viewed for elementwise operations; it does not copy values conceptually.",
    )
    return Example("Deep Learning", "tensor_broadcasting_shape", "average", q, a_text, {"output_shape": out_shape})


def build_examples() -> tuple[list[Example], list[Example]]:
    rng = random.Random(RANDOM_SEED)
    examples: list[Example] = []

    for idx in range(260):
        examples.append(sigmoid_bce_example(
            x=rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]),
            w=rng.choice([-1.2, -0.7, -0.3, 0.2, 0.5, 0.9, 1.4]),
            b=rng.choice([-0.6, -0.2, 0.0, 0.3, 0.7]),
            y=rng.choice([0, 1]),
            idx=idx,
        ))

    for idx in range(240):
        logits = [rng.choice([-2, -1, -0.5, 0, 0.5, 1, 2, 3]) for _ in range(3)]
        examples.append(softmax_ce_example(logits, rng.randrange(3), idx))

    for idx in range(220):
        examples.append(relu_mlp_example(
            x1=rng.choice([-2, -1, 1, 2, 3]),
            x2=rng.choice([-3, -1, 1, 2]),
            w1=rng.choice([-1.0, -0.5, 0.4, 0.8, 1.2]),
            w2=rng.choice([-0.7, -0.2, 0.3, 0.9]),
            b=rng.choice([-0.5, 0.0, 0.4]),
            v=rng.choice([-1.1, -0.6, 0.5, 1.0]),
            target=rng.choice([-1, 0, 1, 2]),
            idx=idx,
        ))

    for idx in range(220):
        examples.append(mse_linear_example(
            x=[rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-2, -1, 1, 2, 4])],
            w=[rng.choice([-1.0, -0.5, 0.3, 0.8, 1.2]), rng.choice([-0.8, -0.2, 0.4, 1.0])],
            b=rng.choice([-0.5, 0.0, 0.5]),
            y=rng.choice([-2, -1, 0, 1, 2, 3]),
            idx=idx,
        ))

    for idx in range(180):
        examples.append(adam_example(
            w=rng.choice([-1.0, -0.5, 0.5, 1.0]),
            grad=rng.choice([-0.8, -0.4, -0.1, 0.2, 0.6]),
            m=rng.choice([-0.2, 0.0, 0.1, 0.3]),
            v=rng.choice([0.0, 0.01, 0.04, 0.09]),
            t=rng.choice([1, 2, 3, 5, 10]),
            lr=rng.choice([0.001, 0.003, 0.01]),
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            idx=idx,
        ))

    for idx in range(160):
        examples.append(momentum_example(
            w=rng.choice([-1.0, -0.5, 0.5, 1.0]),
            grad=rng.choice([-0.7, -0.2, 0.3, 0.8]),
            velocity=rng.choice([-0.3, 0.0, 0.2, 0.5]),
            lr=rng.choice([0.01, 0.03, 0.1]),
            momentum=rng.choice([0.8, 0.9, 0.95]),
            idx=idx,
        ))

    for idx in range(180):
        examples.append(gradient_clipping_example(
            grads=[rng.choice([-6, -4, -2, -1, 1, 2, 4, 6]) for _ in range(3)],
            threshold=rng.choice([1.0, 2.0, 5.0]),
            idx=idx,
        ))

    for idx in range(180):
        examples.append(activation_derivative_example(
            rng.choice(["relu", "sigmoid", "tanh"]),
            rng.choice([-3, -1, -0.5, 0, 0.5, 1, 3]),
            idx,
        ))

    for idx in range(130):
        values = [rng.choice([-2, -1, 0, 1, 2, 3, 4]) for _ in range(4)]
        examples.append(batchnorm_example(values, gamma=rng.choice([0.5, 1.0, 1.5]), beta=rng.choice([-0.2, 0.0, 0.3]), eps=1e-5, idx=idx))
        examples.append(layernorm_example(values, gamma=rng.choice([0.5, 1.0, 1.5]), beta=rng.choice([-0.2, 0.0, 0.3]), eps=1e-5, idx=idx))

    for idx in range(140):
        values = [rng.choice([0.5, 1.0, 2.0, 3.0]) for _ in range(4)]
        mask = [rng.choice([0, 1]) for _ in range(4)]
        if sum(mask) == 0:
            mask[0] = 1
        examples.append(dropout_example(values, mask, keep_prob=rng.choice([0.5, 0.8]), idx=idx))

    for idx in range(150):
        q_vec = [rng.choice([-1.0, -0.5, 0.5, 1.0]) for _ in range(2)]
        keys = [[rng.choice([-1.0, 0.0, 1.0]), rng.choice([-1.0, 0.0, 1.0])] for _ in range(2)]
        values = [[rng.choice([-1.0, 0.0, 1.0]), rng.choice([-1.0, 0.0, 1.0])] for _ in range(2)]
        examples.append(attention_example(q_vec, keys, values, idx))

    for idx in range(130):
        examples.append(conv_shape_example(
            h=rng.choice([28, 32, 64, 128]),
            w=rng.choice([28, 32, 64, 128]),
            kernel=rng.choice([1, 3, 5]),
            stride=rng.choice([1, 2]),
            padding=rng.choice([0, 1, 2]),
            channels_out=rng.choice([16, 32, 64, 128]),
            idx=idx,
        ))

    for idx in range(160):
        examples.append(binary_ce_probability_example(
            prob=rng.choice([0.05, 0.1, 0.2, 0.35, 0.65, 0.8, 0.9, 0.95]),
            y=rng.choice([0, 1]),
            idx=idx,
        ))

    for idx in range(140):
        examples.append(classification_metrics_example(
            tp=rng.choice([5, 12, 20, 35, 50]),
            fp=rng.choice([1, 4, 10, 18]),
            fn=rng.choice([1, 5, 12, 20]),
            tn=rng.choice([10, 25, 40, 80]),
            idx=idx,
        ))

    for idx in range(100):
        labels = [rng.randrange(4) for _ in range(8)]
        predictions = [label if rng.random() < 0.65 else rng.randrange(4) for label in labels]
        examples.append(multiclass_accuracy_example(labels, predictions, idx))

    for idx in range(160):
        examples.append(gradient_descent_example(
            weights=[rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]) for _ in range(3)],
            gradients=[rng.choice([-0.8, -0.3, -0.1, 0.2, 0.6, 1.0]) for _ in range(3)],
            lr=rng.choice([0.001, 0.01, 0.03, 0.1]),
            idx=idx,
        ))

    for idx in range(120):
        examples.append(weight_decay_sgd_example(
            weights=[rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]) for _ in range(3)],
            gradients=[rng.choice([-0.8, -0.3, 0.2, 0.6, 1.0]) for _ in range(3)],
            lr=rng.choice([0.001, 0.01, 0.03]),
            lam=rng.choice([0.001, 0.01, 0.05, 0.1]),
            idx=idx,
        ))

    for idx in range(160):
        examples.append(cosine_similarity_example(
            a=[rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]) for _ in range(4)],
            b=[rng.choice([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]) for _ in range(4)],
            idx=idx,
        ))

    for idx in range(100):
        query = [rng.choice([-1.0, -0.5, 0.5, 1.0]) for _ in range(4)]
        docs = [[rng.choice([-1.0, -0.5, 0.5, 1.0]) for _ in range(4)] for _ in range(3)]
        examples.append(semantic_search_example(query, docs, idx))

    for idx in range(120):
        a = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(3)] for _ in range(2)]
        b = [[rng.choice([-2, -1, 0, 1, 2]) for _ in range(2)] for _ in range(3)]
        examples.append(matrix_multiply_example(a, b, idx))

    for idx in range(100):
        m = rng.choice([1, 2, 4, 8, 16])
        k = rng.choice([2, 4, 8, 16, 32])
        n = rng.choice([1, 2, 4, 8, 16])
        examples.append(matmul_shape_example((m, k), (k, n), idx))

    broadcast_pairs = [
        ((32, 1, 128), (1, 10, 128)),
        ((4, 1), (3, 4, 5)),
        ((8, 1, 64), (8, 32, 64)),
        ((1, 16), (32, 16)),
        ((2, 3, 1), (1, 3, 5)),
    ]
    for idx in range(80):
        shape_a, shape_b = rng.choice(broadcast_pairs)
        examples.append(broadcasting_shape_example(shape_a, shape_b, idx))

    examples = unique_by_question(examples)
    rng.shuffle(examples)

    eval_cases = [
        sigmoid_bce_example(2, 0.5, -0.1, 1, 9001),
        softmax_ce_example([2.0, 1.0, 0.1], 0, 9002),
        adam_example(1.0, 0.2, 0.0, 0.0, 1, 0.001, 0.9, 0.999, 1e-8, 9003),
        gradient_clipping_example([3.0, 4.0], 2.0, 9004),
        attention_example([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 2.0], [3.0, 4.0]], 9005),
        conv_shape_example(32, 32, 3, 1, 1, 64, 9006),
        binary_ce_probability_example(0.8, 1, 9007),
        classification_metrics_example(40, 10, 5, 45, 9008),
        multiclass_accuracy_example([0, 1, 2, 2, 1], [0, 2, 2, 2, 1], 9009),
        gradient_descent_example([1.0, -2.0], [0.3, -0.5], 0.1, 9010),
        weight_decay_sgd_example([1.0, -2.0], [0.3, -0.5], 0.1, 0.01, 9011),
        cosine_similarity_example([1.0, 2.0, 0.0], [2.0, 1.0, 0.0], 9012),
        semantic_search_example([1.0, 0.0], [[0.9, 0.1], [0.0, 1.0], [0.7, 0.7]], 9013),
        matrix_multiply_example([[1, 2], [3, 4]], [[2, 0], [1, 2]], 9014),
        matmul_shape_example((32, 128), (128, 10), 9015),
        broadcasting_shape_example((32, 1, 128), (1, 10, 128), 9016),
    ]
    validate(examples)
    validate(eval_cases)
    return examples, eval_cases


def unique_by_question(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    unique: list[Example] = []
    for ex in examples:
        if ex.question in seen:
            continue
        seen.add(ex.question)
        unique.append(ex)
    return unique


def validate(examples: list[Example]) -> None:
    for idx, ex in enumerate(examples, start=1):
        if "Result:" not in ex.answer or "Method:" not in ex.answer:
            raise ValueError(f"Malformed answer at row {idx}")
        if len(ex.answer) > 1700:
            raise ValueError(f"Overlong answer at row {idx}: {len(ex.answer)} chars")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    examples, eval_cases = build_examples()
    write_jsonl(TRAIN_PATH, [chat_record(ex) for ex in examples])
    write_jsonl(EVAL_PATH, [chat_record(ex) for ex in eval_cases])

    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.task_type] = counts.get(ex.task_type, 0) + 1

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as fout:
        fout.write("# V5 Deep Learning Dataset Report\n\n")
        fout.write(f"Training examples: {len(examples)}\n\n")
        fout.write(f"Eval examples: {len(eval_cases)}\n\n")
        fout.write("## Task Counts\n\n")
        for task_type, count in sorted(counts.items()):
            fout.write(f"- `{task_type}`: {count}\n")

    print(f"Wrote {len(examples)} training examples to {TRAIN_PATH}")
    print(f"Wrote {len(eval_cases)} eval examples to {EVAL_PATH}")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
