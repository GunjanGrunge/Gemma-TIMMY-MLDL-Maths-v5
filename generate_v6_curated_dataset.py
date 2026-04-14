"""Generate V6 curated ML/DL/statistics/data-analytics training data.

V6 supplements V5.2 with long-tail skills: ML evaluation math, transformer
internals, nonparametric statistics, data analytics workflow math, forecasting
diagnostics, and explicit guardrail behavior.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


V52_COMBINED = Path("outputs/v52/data/v52_combined_train_chat.jsonl")
OUT_DIR = Path("outputs/v6/data")
REPORT_DIR = Path("outputs/v6/reports")
SUPPLEMENT_OUT = OUT_DIR / "v6_curated_train_chat.jsonl"
COMBINED_OUT = OUT_DIR / "v6_combined_train_chat.jsonl"
EVAL_OUT = OUT_DIR / "v6_eval_cases.jsonl"
REPORT_OUT = REPORT_DIR / "v6_dataset_report.md"
SAMPLE_OUT = Path("samples/v6_curated_min_sample.jsonl")

SYSTEM = (
    "You are Gemma TIMMY, a rigorous ML, DL, statistics, and data analytics math assistant. "
    "Use verified formulas, show compact arithmetic, state assumptions explicitly, and refuse "
    "numeric tasks when inputs are missing, vague, invalid, or contradictory."
)


def r4(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def fmt(values: Iterable[float]) -> str:
    return "[" + ", ".join(r4(float(v)) for v in values) + "]"


def answer(problem: str, method: str, calculation: str, result: str, note: str) -> str:
    return (
        f"Problem: {problem}\n"
        f"Method: {method}\n"
        f"Calculation: {calculation}\n"
        f"Result: {result}\n"
        f"Diagnostic note: {note}"
    )


def record(user: str, assistant: str, domain: str, task: str, difficulty: str, expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "dataset": "v6_curated_verified",
            "domain": domain,
            "task_type": task,
            "difficulty": difficulty,
            "expected": expected,
        },
    }


def refusal(
    user: str,
    status: str,
    message: str,
    domain: str,
    task: str,
    required_fields: list[str] | None = None,
    invalid_fields: list[str] | None = None,
    assumptions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "message": message,
        "required_fields": required_fields or [],
        "invalid_fields": invalid_fields or [],
        "assumptions": assumptions or {},
        "safe_next_step": "Provide the missing numeric inputs or clarify the intended task before computing.",
    }
    return record(user, json.dumps(payload, ensure_ascii=False, indent=2), domain, task, "guardrail", payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def gini_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    return 1 - sum((c / total) ** 2 for c in counts)


def information_gain(parent: list[int], children: list[list[int]]) -> dict[str, float]:
    parent_entropy = entropy_from_counts(parent)
    total = sum(parent)
    weighted_child_entropy = sum((sum(child) / total) * entropy_from_counts(child) for child in children)
    return {
        "parent_entropy": parent_entropy,
        "weighted_child_entropy": weighted_child_entropy,
        "information_gain": parent_entropy - weighted_child_entropy,
        "parent_gini": gini_from_counts(parent),
    }


def roc_auc(labels: list[int], scores: list[float]) -> float:
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            wins += 1.0 if pos > neg else 0.5 if pos == neg else 0.0
    return wins / (len(positives) * len(negatives))


def average_precision(labels: list[int], scores: list[float]) -> float:
    ranked = sorted(zip(scores, labels), reverse=True)
    positives = sum(labels)
    precision_sum = 0.0
    true_positives = 0
    for rank, (_, label) in enumerate(ranked, 1):
        if label == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def threshold_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def brier_score(labels: list[int], probs: list[float]) -> float:
    return mean([(p - y) ** 2 for y, p in zip(labels, probs)])


def pca_2x2(a: float, b: float, d: float) -> dict[str, float]:
    trace = a + d
    det = a * d - b * b
    root = math.sqrt(max(0.0, trace * trace - 4 * det))
    eig1 = (trace + root) / 2
    eig2 = (trace - root) / 2
    return {"lambda_1": eig1, "lambda_2": eig2, "explained_1": eig1 / (eig1 + eig2), "explained_2": eig2 / (eig1 + eig2)}


def silhouette(a: float, b: float) -> float:
    return (b - a) / max(a, b)


def boosting_residuals(y: list[float], pred: list[float], learning_rate: float) -> dict[str, Any]:
    residuals = [actual - estimate for actual, estimate in zip(y, pred)]
    updated = [estimate + learning_rate * residual for estimate, residual in zip(pred, residuals)]
    return {"residuals": residuals, "updated_predictions": updated}


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def label_smoothing_ce(logits: list[float], true_class: int, epsilon: float) -> dict[str, Any]:
    probs = softmax(logits)
    k = len(logits)
    target = [epsilon / k for _ in logits]
    target[true_class] += 1 - epsilon
    loss = -sum(t * math.log(p) for t, p in zip(target, probs))
    grad = [p - t for p, t in zip(probs, target)]
    return {"probs": probs, "target": target, "loss": loss, "gradient": grad}


def binary_focal_loss(prob: float, y: int, gamma: float, alpha: float) -> dict[str, float]:
    pt = prob if y == 1 else 1 - prob
    return {"pt": pt, "loss": -alpha * ((1 - pt) ** gamma) * math.log(pt)}


def transformer_shapes(batch: int, seq: int, d_model: int, heads: int, ff_mult: int = 4) -> dict[str, Any]:
    head_dim = d_model // heads
    return {
        "qkv_shape": [batch, heads, seq, head_dim],
        "attention_scores_shape": [batch, heads, seq, seq],
        "attention_output_shape": [batch, seq, d_model],
        "ffn_hidden_shape": [batch, seq, d_model * ff_mult],
        "head_dim": head_dim,
    }


def cosine_lr(step: int, total_steps: int, base_lr: float, min_lr: float = 0.0) -> float:
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * step / total_steps))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][0] == indexed[i][0]:
            j += 1
        rank = (i + 1 + j) / 2
        for _, idx in indexed[i:j]:
            result[idx] = rank
        i = j
    return result


def mann_whitney_u(a: list[float], b: list[float]) -> dict[str, float]:
    combined = a + b
    rs = ranks(combined)
    r_a = sum(rs[: len(a)])
    u_a = r_a - len(a) * (len(a) + 1) / 2
    u_b = len(a) * len(b) - u_a
    return {"u_a": u_a, "u_b": u_b, "u": min(u_a, u_b)}


def wilcoxon_signed_rank(before: list[float], after: list[float]) -> dict[str, float]:
    diffs = [a - b for b, a in zip(before, after) if a != b]
    abs_ranks = ranks([abs(d) for d in diffs])
    w_plus = sum(rank for rank, diff in zip(abs_ranks, diffs) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(abs_ranks, diffs) if diff < 0)
    return {"w_plus": w_plus, "w_minus": w_minus, "w": min(w_plus, w_minus)}


def kruskal_wallis(groups: list[list[float]]) -> dict[str, float]:
    values = [v for group in groups for v in group]
    rs = ranks(values)
    n = len(values)
    offset = 0
    h_sum = 0.0
    for group in groups:
        group_ranks = rs[offset : offset + len(group)]
        offset += len(group)
        h_sum += sum(group_ranks) ** 2 / len(group)
    h = (12 / (n * (n + 1))) * h_sum - 3 * (n + 1)
    return {"h": h, "df": len(groups) - 1}


def bonferroni(p_values: list[float], alpha: float) -> dict[str, Any]:
    threshold = alpha / len(p_values)
    return {"threshold": threshold, "reject": [p <= threshold for p in p_values]}


def benjamini_hochberg(p_values: list[float], q: float) -> dict[str, Any]:
    ordered = sorted((p, i) for i, p in enumerate(p_values))
    cutoff_rank = 0
    cutoff_p = None
    m = len(p_values)
    for rank, (p, _) in enumerate(ordered, 1):
        if p <= (rank / m) * q:
            cutoff_rank = rank
            cutoff_p = p
    rejected = [False] * m
    if cutoff_rank:
        for _, i in ordered[:cutoff_rank]:
            rejected[i] = True
    return {"cutoff_rank": cutoff_rank, "cutoff_p": cutoff_p, "reject": rejected}


def beta_posterior(successes: int, failures: int, alpha_prior: float, beta_prior: float) -> dict[str, float]:
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    return {"alpha_post": alpha_post, "beta_post": beta_post, "posterior_mean": alpha_post / (alpha_post + beta_post)}


def standardize(values: list[float]) -> dict[str, Any]:
    mu = mean(values)
    sigma = math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))
    return {"mean": mu, "sample_std": sigma, "z": [(v - mu) / sigma for v in values]}


def minmax_scale(values: list[float]) -> dict[str, Any]:
    low, high = min(values), max(values)
    return {"min": low, "max": high, "scaled": [(v - low) / (high - low) for v in values]}


def iqr_outliers(values: list[float], q1: float, q3: float) -> dict[str, Any]:
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return {"iqr": iqr, "lower": lower, "upper": upper, "outliers": [v for v in values if v < lower or v > upper]}


def groupby_mean(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row[value]))
    return {k: mean(v) for k, v in grouped.items()}


def funnel_rates(counts: list[int]) -> dict[str, Any]:
    step_rates = [counts[i + 1] / counts[i] for i in range(len(counts) - 1)]
    return {"step_rates": step_rates, "overall_rate": counts[-1] / counts[0]}


def smape(actuals: list[float], forecasts: list[float]) -> float:
    return mean([abs(a - f) / ((abs(a) + abs(f)) / 2) for a, f in zip(actuals, forecasts)])


def mase(actuals: list[float], forecasts: list[float], train: list[float]) -> float:
    mae = mean([abs(a - f) for a, f in zip(actuals, forecasts)])
    naive = mean([abs(train[i] - train[i - 1]) for i in range(1, len(train))])
    return mae / naive


def variants(clean: str, messy: str, compact: str) -> list[str]:
    return [clean, messy, compact]


def build_ml_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tree_cases = [
        ([9, 5], [[6, 1], [3, 4]]),
        ([12, 8], [[10, 2], [2, 6]]),
        ([7, 9], [[2, 7], [5, 2]]),
        ([15, 5], [[14, 1], [1, 4]]),
    ]
    for idx, (parent, children) in enumerate(tree_cases, 1):
        out = information_gain(parent, children)
        prompts = variants(
            f"Decision tree split case {idx}: parent class counts={parent}, child counts={children}. Compute entropy, gini, and information gain.",
            f"I am checking a tree split. Parent classes are {parent}; the split creates leaves {children}. Is the split useful by information gain?",
            f"Compute decision-tree impurity for parent={parent}, children={children}.",
        )
        for prompt in prompts:
            rows.append(record(
                prompt,
                answer(
                    "Evaluate a decision-tree split.",
                    "Use entropy=-sum p log2(p), weighted child entropy, and information_gain=parent_entropy-weighted_child_entropy.",
                    f"parent_entropy={r4(out['parent_entropy'])}, weighted_child_entropy={r4(out['weighted_child_entropy'])}, information_gain={r4(out['information_gain'])}, parent_gini={r4(out['parent_gini'])}.",
                    f"information_gain={r4(out['information_gain'])}; parent_gini={r4(out['parent_gini'])}.",
                    "Prefer splits with higher validation gain, not only training gain.",
                ),
                "Machine Learning", "decision_tree_impurity_gain", "advanced", out,
            ))

    score_cases = [
        ([1, 0, 1, 0, 1, 0], [0.92, 0.72, 0.65, 0.40, 0.35, 0.10], 0.5),
        ([0, 1, 1, 0, 1, 0, 0], [0.20, 0.88, 0.74, 0.55, 0.51, 0.30, 0.05], 0.5),
        ([1, 1, 0, 0, 1, 0], [0.80, 0.62, 0.61, 0.59, 0.44, 0.12], 0.6),
        ([0, 1, 0, 1, 1, 0, 1], [0.15, 0.95, 0.34, 0.67, 0.82, 0.28, 0.52], 0.5),
    ]
    for idx, (labels, scores, threshold) in enumerate(score_cases, 1):
        auc = roc_auc(labels, scores)
        ap = average_precision(labels, scores)
        metrics = threshold_metrics(labels, scores, threshold)
        brier = brier_score(labels, scores)
        out = {"roc_auc": auc, "average_precision": ap, "brier": brier, **metrics}
        for prompt in variants(
            f"Classifier evaluation case {idx}: labels={labels}, scores={scores}, threshold={threshold}. Compute ROC-AUC, AP, Brier, and threshold metrics.",
            f"These probabilities came from a churn model: y={labels}, p={scores}. At threshold {threshold}, diagnose ranking and classification quality.",
            f"Evaluate binary scores y={labels}, score={scores}, cutoff={threshold}.",
        ):
            rows.append(record(
                prompt,
                answer(
                    "Evaluate a probabilistic binary classifier.",
                    "Use pairwise ROC-AUC, average precision over ranked positives, Brier mean squared probability error, and threshold confusion metrics.",
                    f"ROC-AUC={r4(auc)}, AP={r4(ap)}, Brier={r4(brier)}, TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}, F1={r4(metrics['f1'])}.",
                    f"ROC-AUC={r4(auc)}, average_precision={r4(ap)}, Brier={r4(brier)}, precision={r4(metrics['precision'])}, recall={r4(metrics['recall'])}, F1={r4(metrics['f1'])}.",
                    "AUC tests ranking; Brier tests probability calibration; threshold metrics test operating-point behavior.",
                ),
                "Machine Learning", "classifier_ranking_calibration_threshold", "expert", out,
            ))

    pca_cases = [(2.4, 0.8, 1.1), (5.0, 1.5, 2.0), (1.8, -0.6, 1.4), (3.2, 0.4, 0.9)]
    for idx, (a, b, d) in enumerate(pca_cases, 1):
        out = pca_2x2(a, b, d)
        for prompt in variants(
            f"PCA case {idx}: covariance matrix [[{a}, {b}], [{b}, {d}]]. Compute eigenvalues and explained variance ratios.",
            f"For a 2-feature dataset the covariance terms are var1={a}, cov={b}, var2={d}. How much variance does PC1 explain?",
            f"Compute PCA explained variance for covariance [[{a},{b}],[{b},{d}]].",
        ):
            rows.append(record(
                prompt,
                answer(
                    "Compute PCA explained variance from a 2x2 covariance matrix.",
                    "For [[a,b],[b,d]], eigenvalues are (trace plus/minus sqrt(trace^2-4det))/2; explained ratio=lambda/sum(lambda).",
                    f"lambda_1={r4(out['lambda_1'])}, lambda_2={r4(out['lambda_2'])}, explained_1={r4(out['explained_1'])}, explained_2={r4(out['explained_2'])}.",
                    f"PC1 explains {r4(100*out['explained_1'])}% and PC2 explains {r4(100*out['explained_2'])}%.",
                    "High PC1 share means dimensionality reduction may preserve most variance, but not necessarily prediction signal.",
                ),
                "Machine Learning", "pca_explained_variance", "advanced", out,
            ))

    for idx, (a, b) in enumerate([(0.8, 2.4), (1.2, 1.5), (2.1, 1.4), (0.5, 3.0)], 1):
        s = silhouette(a, b)
        out = {"a": a, "b": b, "silhouette": s}
        rows.append(record(
            f"Clustering case {idx}: average intra-cluster distance a={a}, nearest other-cluster distance b={b}. Compute silhouette score.",
            answer("Compute a silhouette score for one point.", "Use s=(b-a)/max(a,b).", f"s=({r4(b)}-{r4(a)})/max({r4(a)},{r4(b)})={r4(s)}.", f"silhouette={r4(s)}.", "Positive means better matched to its own cluster; negative suggests possible wrong cluster."),
            "Machine Learning", "silhouette_score", "intermediate", out,
        ))

    for idx, (y, pred, lr) in enumerate([([10, 12, 9], [9, 11, 11], 0.2), ([0.8, 0.3, 0.6], [0.5, 0.5, 0.5], 0.1), ([30, 28, 35], [27, 29, 34], 0.3)], 1):
        out = boosting_residuals(y, pred, lr)
        rows.append(record(
            f"Gradient boosting residual case {idx}: y={y}, current_predictions={pred}, learning_rate={lr}. Compute residuals and updated predictions.",
            answer("Compute one squared-error gradient boosting residual update.", "For squared error, residual=actual-current_prediction; updated_prediction=current_prediction+learning_rate*residual.", f"residuals={fmt(out['residuals'])}, updated_predictions={fmt(out['updated_predictions'])}.", f"updated_predictions={fmt(out['updated_predictions'])}.", "Residual learning corrects errors gradually; smaller learning rates usually need more trees."),
            "Machine Learning", "gradient_boosting_residual_update", "advanced", out,
        ))
    return rows


def build_dl_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ce_cases = [
        ([2.0, 0.5, -1.0], 0, 0.1),
        ([0.1, 1.2, 0.4, -0.5], 1, 0.2),
        ([1.5, 1.1, 0.9], 2, 0.05),
        ([-0.2, 0.3, 2.1], 2, 0.15),
    ]
    for idx, (logits, true_class, eps) in enumerate(ce_cases, 1):
        out = label_smoothing_ce(logits, true_class, eps)
        for prompt in variants(
            f"Label smoothing CE case {idx}: logits={logits}, true_class={true_class}, epsilon={eps}. Compute target, loss, and dL/dlogits.",
            f"My classifier uses label smoothing. With logits {logits}, class {true_class}, eps {eps}, calculate CE and gradient.",
            f"Compute label-smoothed softmax CE for z={logits}, y={true_class}, epsilon={eps}.",
        ):
            rows.append(record(
                prompt,
                answer(
                    "Compute label-smoothed softmax cross entropy.",
                    "Build target=(1-epsilon) on true class plus epsilon/K everywhere; loss=-sum target_i log softmax_i; gradient=prob-target.",
                    f"probs={fmt(out['probs'])}, target={fmt(out['target'])}, loss={r4(out['loss'])}, gradient={fmt(out['gradient'])}.",
                    f"loss={r4(out['loss'])}, dL/dlogits={fmt(out['gradient'])}.",
                    "Label smoothing reduces overconfidence but can hurt calibration if overused.",
                ),
                "Deep Learning", "label_smoothing_cross_entropy", "expert", out,
            ))

    for idx, (prob, y, gamma, alpha) in enumerate([(0.90, 1, 2.0, 0.25), (0.30, 1, 2.0, 0.25), (0.80, 0, 2.0, 0.75), (0.45, 0, 1.5, 0.5)], 1):
        out = binary_focal_loss(prob, y, gamma, alpha)
        rows.append(record(
            f"Focal loss case {idx}: predicted positive probability={prob}, y={y}, gamma={gamma}, alpha={alpha}. Compute pt and focal loss.",
            answer("Compute binary focal loss.", "Use pt=p when y=1 else 1-p, loss=-alpha*(1-pt)^gamma*log(pt).", f"pt={r4(out['pt'])}, loss={r4(out['loss'])}.", f"focal_loss={r4(out['loss'])}.", "Focal loss downweights easy examples and emphasizes hard or minority-class examples."),
            "Deep Learning", "binary_focal_loss", "advanced", out,
        ))

    for idx, (batch, seq, d_model, heads) in enumerate([(2, 128, 512, 8), (1, 2048, 1024, 16), (4, 64, 256, 4), (3, 100, 768, 12)], 1):
        out = transformer_shapes(batch, seq, d_model, heads)
        for prompt in variants(
            f"Transformer shape case {idx}: batch={batch}, seq_len={seq}, d_model={d_model}, heads={heads}. Compute QKV, score, output, and FFN shapes.",
            f"I am debugging multi-head attention dimensions: B={batch}, T={seq}, C={d_model}, H={heads}. What are the core tensor shapes?",
            f"Give transformer attention shapes for batch {batch}, tokens {seq}, model dim {d_model}, heads {heads}.",
        ):
            rows.append(record(
                prompt,
                answer(
                    "Compute transformer multi-head attention shapes.",
                    "head_dim=d_model/heads; Q,K,V shape=[B,H,T,head_dim]; attention scores=[B,H,T,T]; output=[B,T,d_model]; FFN hidden=[B,T,4*d_model].",
                    f"head_dim={out['head_dim']}, QKV={out['qkv_shape']}, scores={out['attention_scores_shape']}, output={out['attention_output_shape']}, ffn_hidden={out['ffn_hidden_shape']}.",
                    f"QKV={out['qkv_shape']}, attention_scores={out['attention_scores_shape']}, attention_output={out['attention_output_shape']}.",
                    "Most transformer shape bugs come from mixing sequence length, head count, and head dimension.",
                ),
                "Deep Learning", "transformer_attention_shapes", "expert", out,
            ))

    for idx, ce in enumerate([0.7, 1.2, 2.0, 3.5], 1):
        pp = math.exp(ce)
        rows.append(record(
            f"Language-model metric case {idx}: validation cross entropy is {ce}. Compute perplexity.",
            answer("Convert cross entropy to perplexity.", "Use perplexity=exp(cross_entropy) when cross entropy is in natural-log units.", f"perplexity=exp({r4(ce)})={r4(pp)}.", f"perplexity={r4(pp)}.", "Compare perplexity only across the same tokenizer, data distribution, and loss convention."),
            "Deep Learning", "perplexity_from_cross_entropy", "intermediate", {"cross_entropy": ce, "perplexity": pp},
        ))

    for idx, (step, total, base, min_lr) in enumerate([(100, 1000, 2e-4, 1e-5), (500, 2000, 1e-4, 0.0), (1500, 3000, 3e-4, 3e-5)], 1):
        lr = cosine_lr(step, total, base, min_lr)
        rows.append(record(
            f"Scheduler case {idx}: cosine LR with step={step}, total_steps={total}, base_lr={base}, min_lr={min_lr}. Compute current LR.",
            answer("Compute cosine learning-rate schedule value.", "Use lr=min_lr+0.5*(base_lr-min_lr)*(1+cos(pi*step/total_steps)).", f"lr={r4(lr)}.", f"current_lr={r4(lr)}.", "Cosine decay usually needs a warmup phase for stable early training."),
            "Deep Learning", "cosine_lr_schedule", "intermediate", {"lr": lr},
        ))

    for idx, (per_device, grad_accum, devices) in enumerate([(1, 8, 1), (2, 16, 1), (1, 4, 2), (4, 8, 2)], 1):
        eff = per_device * grad_accum * devices
        rows.append(record(
            f"Training config case {idx}: per_device_batch={per_device}, gradient_accumulation={grad_accum}, devices={devices}. Compute effective batch size.",
            answer("Compute effective batch size.", "Use effective_batch=per_device_batch*gradient_accumulation_steps*num_devices.", f"{per_device}*{grad_accum}*{devices}={eff}.", f"effective_batch_size={eff}.", "Changing effective batch size can require learning-rate retuning."),
            "Deep Learning", "effective_batch_size", "average", {"effective_batch_size": eff},
        ))
    return rows


def build_stats_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate([([4, 7, 9], [1, 3, 6]), ([12, 14, 15, 18], [10, 11, 13, 16]), ([0.8, 0.9, 1.1], [0.5, 0.7, 1.0])], 1):
        out = mann_whitney_u(a, b)
        rows.append(record(
            f"Mann-Whitney case {idx}: sample_a={a}, sample_b={b}. Compute U statistics.",
            answer("Compute Mann-Whitney U for two independent samples.", "Rank all observations, sum ranks for sample A, then U_A=R_A-n_A(n_A+1)/2 and U_B=n_A*n_B-U_A.", f"U_A={r4(out['u_a'])}, U_B={r4(out['u_b'])}, U={r4(out['u'])}.", f"Mann-Whitney U={r4(out['u'])}.", "Use this when comparing distributions without assuming normality."),
            "Statistics", "mann_whitney_u", "advanced", out,
        ))

    for idx, (before, after) in enumerate([([10, 12, 9, 11], [11, 13, 9, 14]), ([0.50, 0.55, 0.52, 0.58], [0.53, 0.57, 0.51, 0.61])], 1):
        out = wilcoxon_signed_rank(before, after)
        rows.append(record(
            f"Wilcoxon signed-rank case {idx}: before={before}, after={after}. Compute W+ and W-.",
            answer("Compute Wilcoxon signed-rank statistic for paired samples.", "Rank absolute nonzero paired differences, sum ranks for positive and negative differences, and use min(W+,W-).", f"W_plus={r4(out['w_plus'])}, W_minus={r4(out['w_minus'])}, W={r4(out['w'])}.", f"Wilcoxon W={r4(out['w'])}.", "This is a paired nonparametric alternative to the paired t-test."),
            "Statistics", "wilcoxon_signed_rank", "advanced", out,
        ))

    for idx, groups in enumerate([[[5, 6, 7], [7, 8, 9], [10, 11, 12]], [[1.2, 1.0, 1.4], [1.5, 1.7, 1.6], [0.8, 0.9, 1.1]]], 1):
        out = kruskal_wallis(groups)
        rows.append(record(
            f"Kruskal-Wallis case {idx}: groups={groups}. Compute H statistic and degrees of freedom.",
            answer("Compute Kruskal-Wallis rank test statistic.", "Rank all values, compute H=(12/(N(N+1)))sum(R_i^2/n_i)-3(N+1), df=k-1.", f"H={r4(out['h'])}, df={out['df']}.", f"Kruskal-Wallis H={r4(out['h'])}, df={out['df']}.", "Use post-hoc tests if the omnibus test is significant."),
            "Statistics", "kruskal_wallis", "expert", out,
        ))

    for idx, (p_values, alpha) in enumerate([([0.001, 0.02, 0.04, 0.20], 0.05), ([0.01, 0.03, 0.07, 0.15, 0.22], 0.10)], 1):
        bon = bonferroni(p_values, alpha)
        bh = benjamini_hochberg(p_values, alpha)
        out = {"bonferroni": bon, "benjamini_hochberg": bh}
        rows.append(record(
            f"Multiple testing case {idx}: p_values={p_values}, alpha_or_q={alpha}. Compute Bonferroni and Benjamini-Hochberg decisions.",
            answer("Control family-wise error and false discovery rate.", "Bonferroni threshold=alpha/m. BH sorts p-values and rejects up to largest p_i <= (i/m)q.", f"Bonferroni threshold={r4(bon['threshold'])}, reject={bon['reject']}; BH cutoff_rank={bh['cutoff_rank']}, cutoff_p={bh['cutoff_p']}, reject={bh['reject']}.", f"Bonferroni reject={bon['reject']}; BH reject={bh['reject']}.", "Bonferroni is stricter; BH is often better for discovery workflows."),
            "Statistics", "multiple_testing_correction", "expert", out,
        ))

    for idx, (successes, failures, a0, b0) in enumerate([(18, 7, 1, 1), (42, 8, 2, 2), (5, 15, 1, 3)], 1):
        out = beta_posterior(successes, failures, a0, b0)
        rows.append(record(
            f"Bayesian conversion-rate case {idx}: successes={successes}, failures={failures}, prior=Beta({a0},{b0}). Compute posterior and posterior mean.",
            answer("Update a beta-binomial conversion-rate model.", "Posterior alpha=prior_alpha+successes, beta=prior_beta+failures; posterior mean=alpha/(alpha+beta).", f"alpha_post={r4(out['alpha_post'])}, beta_post={r4(out['beta_post'])}, posterior_mean={r4(out['posterior_mean'])}.", f"posterior=Beta({r4(out['alpha_post'])},{r4(out['beta_post'])}), mean={r4(out['posterior_mean'])}.", "Bayesian estimates are more stable than raw rates for small samples."),
            "Statistics", "beta_binomial_posterior", "advanced", out,
        ))
    return rows


def build_da_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, values in enumerate([[10, 12, 13, 15, 20], [100, 98, 105, 110, 120], [0.4, 0.5, 0.55, 0.8]], 1):
        out = standardize(values)
        rows.append(record(
            f"Data prep case {idx}: values={values}. Compute sample mean, sample standard deviation, and z-scores.",
            answer("Standardize numeric values.", "Use z=(x-mean)/sample_std.", f"mean={r4(out['mean'])}, sample_std={r4(out['sample_std'])}, z={fmt(out['z'])}.", f"z_scores={fmt(out['z'])}.", "Fit scaling parameters on training data only to avoid leakage."),
            "Data Analytics", "standardization_zscore", "average", out,
        ))
        mm = minmax_scale(values)
        rows.append(record(
            f"Min-max scaling case {idx}: values={values}. Scale each value to [0,1].",
            answer("Min-max scale numeric values.", "Use scaled=(x-min)/(max-min).", f"min={r4(mm['min'])}, max={r4(mm['max'])}, scaled={fmt(mm['scaled'])}.", f"scaled_values={fmt(mm['scaled'])}.", "Min-max scaling is sensitive to outliers."),
            "Data Analytics", "minmax_scaling", "average", mm,
        ))

    for idx, (values, q1, q3) in enumerate([([10, 12, 13, 15, 40], 11, 16), ([2, 3, 3, 4, 20], 2.5, 4.5), ([100, 102, 99, 101, 130], 99.5, 103)], 1):
        out = iqr_outliers(values, q1, q3)
        rows.append(record(
            f"Outlier case {idx}: values={values}, Q1={q1}, Q3={q3}. Compute IQR fences and outliers.",
            answer("Detect IQR outliers.", "Use IQR=Q3-Q1, lower=Q1-1.5*IQR, upper=Q3+1.5*IQR.", f"IQR={r4(out['iqr'])}, lower={r4(out['lower'])}, upper={r4(out['upper'])}, outliers={out['outliers']}.", f"outliers={out['outliers']}.", "Outliers need domain review before removal."),
            "Data Analytics", "iqr_outlier_detection", "intermediate", out,
        ))

    group_cases = [
        [{"segment": "A", "revenue": 100}, {"segment": "B", "revenue": 80}, {"segment": "A", "revenue": 120}, {"segment": "B", "revenue": 100}],
        [{"channel": "search", "conv": 0.08}, {"channel": "social", "conv": 0.03}, {"channel": "search", "conv": 0.10}, {"channel": "social", "conv": 0.04}],
    ]
    for idx, rows_in in enumerate(group_cases, 1):
        key = "segment" if "segment" in rows_in[0] else "channel"
        value = "revenue" if "revenue" in rows_in[0] else "conv"
        out = groupby_mean(rows_in, key, value)
        rows.append(record(
            f"Groupby analytics case {idx}: rows={rows_in}. Compute mean {value} by {key}.",
            answer("Compute grouped means.", "Group rows by key, then average the requested numeric column inside each group.", f"group_means={out}.", f"mean_by_{key}={out}.", "Always check group counts before comparing means."),
            "Data Analytics", "groupby_mean", "average", out,
        ))

    for idx, counts in enumerate([[10000, 4000, 1200, 300], [5000, 3500, 700, 210], [1200, 900, 450, 180]], 1):
        out = funnel_rates(counts)
        rows.append(record(
            f"Funnel case {idx}: stage_counts={counts}. Compute step conversion rates and overall conversion.",
            answer("Compute funnel conversion rates.", "Step rate=next_stage/current_stage; overall=final_stage/first_stage.", f"step_rates={fmt(out['step_rates'])}, overall={r4(out['overall_rate'])}.", f"overall_conversion={r4(out['overall_rate'])}.", "Largest step drop usually points to the first diagnostic target."),
            "Data Analytics", "funnel_conversion", "intermediate", out,
        ))

    simpson = {
        "A": {"mobile_success": 81, "mobile_total": 100, "desktop_success": 192, "desktop_total": 300},
        "B": {"mobile_success": 234, "mobile_total": 300, "desktop_success": 60, "desktop_total": 100},
    }
    out = {
        "a_overall": (81 + 192) / 400,
        "b_overall": (234 + 60) / 400,
        "a_mobile": 81 / 100,
        "b_mobile": 234 / 300,
        "a_desktop": 192 / 300,
        "b_desktop": 60 / 100,
    }
    rows.append(record(
        f"Simpson paradox check: counts={simpson}. Compare A and B overall and by device.",
        answer("Check whether aggregate rates hide segment-level behavior.", "Compute success rates overall and separately by segment.", f"A overall={r4(out['a_overall'])}, B overall={r4(out['b_overall'])}; mobile A={r4(out['a_mobile'])}, B={r4(out['b_mobile'])}; desktop A={r4(out['a_desktop'])}, B={r4(out['b_desktop'])}.", "B wins overall, but A wins inside both device segments; this is Simpson's paradox.", "Segment mix can reverse aggregate conclusions; compare like with like."),
        "Data Analytics", "simpson_paradox", "expert", out,
    ))
    return rows


def build_forecasting_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_cases = [
        ([100, 120, 130], [95, 125, 128], [80, 90, 100, 110]),
        ([0.8, 0.9, 1.1], [0.75, 0.95, 1.0], [0.50, 0.55, 0.60, 0.70]),
        ([50, 48, 52, 55], [49, 50, 51, 57], [40, 42, 43, 47, 49]),
    ]
    for idx, (actuals, forecasts, train) in enumerate(metric_cases, 1):
        s = smape(actuals, forecasts)
        m = mase(actuals, forecasts, train)
        out = {"smape": s, "mase": m}
        rows.append(record(
            f"Forecast diagnostics case {idx}: actuals={actuals}, forecasts={forecasts}, train_history={train}. Compute sMAPE and MASE.",
            answer("Compute scale-aware forecast diagnostics.", "sMAPE=mean(|A-F|/((|A|+|F|)/2)); MASE=MAE/model divided by MAE/naive on training history.", f"sMAPE={r4(s)}, MASE={r4(m)}.", f"sMAPE={r4(s)}, MASE={r4(m)}.", "MASE below 1 means the model beats the naive one-step baseline."),
            "Forecasting", "forecast_error_diagnostics", "advanced", out,
        ))

    for idx, (values, test_size) in enumerate([([10, 12, 13, 15, 18, 21, 25], 3), ([100, 103, 105, 108, 112, 117], 2)], 1):
        train = values[:-test_size]
        test = values[-test_size:]
        out = {"train": train, "test": test}
        rows.append(record(
            f"Time-series split case {idx}: values={values}, test_size={test_size}. Create chronological train/test split.",
            answer("Split time series without leakage.", "Keep chronological order: train is all observations before the final test_size values; test is the final horizon.", f"train={train}, test={test}.", f"train={train}, test={test}.", "Never random-shuffle a time series before forecast evaluation."),
            "Forecasting", "time_series_train_test_split", "average", out,
        ))

    for idx, values in enumerate([[100, 104, 109, 115], [0.8, 0.75, 0.72, 0.70], [50, 50, 51, 53]], 1):
        diffs = [values[i] - values[i - 1] for i in range(1, len(values))]
        out = {"first_differences": diffs}
        rows.append(record(
            f"Stationarity prep case {idx}: values={values}. Compute first differences.",
            answer("Compute first differences for trend removal.", "Use diff_t=x_t-x_(t-1).", f"first_differences={fmt(diffs)}.", f"first_differences={fmt(diffs)}.", "Differencing can help stationarity, but overdifferencing adds noise."),
            "Forecasting", "first_differencing", "intermediate", out,
        ))
    return rows


def build_guardrail_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    guardrails = [
        ("Compute cosine similarity between a vector [1, 2, 3] and another vector roughly pointing northeast.", "missing_info", "Cosine similarity requires two numeric vectors with the same dimension.", "Deep Learning", "cosine_similarity", ["vector_b_numeric_components"]),
        ("Calculate RSI for prices [45, 47, 46, 50, 52, 51, 55]; trader prefers shorter lookbacks but did not specify the period.", "clarification_needed", "A shorter lookback is ambiguous; RSI period must be specified or explicitly defaulted.", "Trading", "rsi", ["rsi_period"]),
        ("Portfolio variance is -0.02. Compute volatility.", "invalid_input", "Variance cannot be negative, so volatility is not real-valued.", "Portfolio", "volatility_from_variance", []),
        ("Compute log return from price 100 to price 0.", "invalid_input", "Log return requires positive start and end prices; log(0) is undefined.", "Trading", "log_return", []),
        ("Run a two-sample t-test with means 10 and 12; statistical significance is obvious.", "missing_info", "A two-sample t-test requires sample sizes, variances or raw samples, and alpha.", "Statistics", "two_sample_t_test", ["sample_sizes", "standard_deviations_or_raw_samples", "alpha"]),
        ("Estimate stopping distance at 20 m/s. Road conditions were not documented.", "missing_info", "Stopping distance requires reaction time and deceleration or friction/road condition assumptions.", "Automotive", "stopping_distance", ["reaction_time", "deceleration_or_friction"]),
        ("Velocity increases linearly over time like a trend model. Predict next velocity using moving average logic.", "clarification_needed", "This mixes a physics model and a forecasting model; choose kinematics or statistical forecasting.", "Forecasting", "router_confusion", ["intended_model"]),
        ("Compute Sharpe ratio, but risk-free rate was low and volatility was moderate.", "missing_info", "Sharpe ratio requires numeric return, numeric risk-free rate, and numeric volatility.", "Portfolio", "sharpe_ratio", ["return", "risk_free_rate", "volatility"]),
        ("Acceleration is constant but velocity decreases over time without an opposing force. Compute final velocity.", "clarification_needed", "The physical description is contradictory unless acceleration direction or opposing force is specified.", "Automotive", "kinematics_contradiction", ["acceleration_direction", "initial_velocity", "time"]),
        ("Use WMA weights [0.2, 0.3, 0.5] to forecast next value but one data point might be missing.", "missing_info", "Weighted moving average requires all numeric data points matching the weight length.", "Forecasting", "weighted_moving_average", ["complete_numeric_series"]),
        ("Compute PCA explained variance; the covariance is high and the variables are related.", "missing_info", "PCA explained variance requires a numeric covariance matrix or data matrix.", "Machine Learning", "pca_explained_variance", ["covariance_matrix_or_data"]),
        ("Tell me whether this A/B test won: control did okay and treatment looked better.", "missing_info", "A/B test interpretation requires sample sizes, conversion counts or rates, and alpha/practical threshold.", "Statistics", "ab_test_interpretation", ["control_n", "control_successes", "treatment_n", "treatment_successes"]),
    ]
    for prompt, status, message, domain, task, required in guardrails:
        rows.append(refusal(prompt, status, message, domain, task, required_fields=required))

    payload = {
        "status": "ok",
        "result": "calculation_allowed_with_explicit_default",
        "assumptions": {"rsi_period": 14},
        "default_used": True,
        "note": "Used standard RSI period 14 because the user explicitly allowed the default.",
    }
    rows.append(record(
        "Calculate RSI and use the standard default period if I do not specify one. Prices are [44,45,46,45,47,49,48,50,51,52,50,53,54,55,56].",
        json.dumps(payload, ensure_ascii=False, indent=2),
        "Trading", "default_used_transparency", "guardrail", payload,
    ))
    return rows


def expand_rows(rows: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if len(rows) >= target:
        return rows
    expanded = list(rows)
    templates = [
        "Re-check this in a production notebook: {question}",
        "No task label given. {question}",
        "I need the formula path and final answer only after checking inputs: {question}",
        "Messy user request: {question}",
    ]
    idx = 0
    while len(expanded) < target:
        base = rows[idx % len(rows)]
        clone = json.loads(json.dumps(base))
        question = clone["messages"][1]["content"]
        clone["messages"][1]["content"] = templates[idx % len(templates)].format(question=question)
        clone["metadata"]["prompt_variant"] = "robust_rewording"
        expanded.append(clone)
        idx += 1
    return expanded


def build_eval_cases(train_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    selected = train_rows[:180] + train_rows[-120:]
    for idx, row in enumerate(selected, 1):
        metadata = row["metadata"]
        eval_rows.append({
            "id": f"v6_eval_{idx:04d}",
            "domain": metadata["domain"],
            "task_type": metadata["task_type"],
            "difficulty": metadata["difficulty"],
            "prompt": "Stress eval, no label: " + row["messages"][1]["content"],
            "expected": metadata["expected"],
            "checks": ["numeric_match_or_structured_status", "no_hallucinated_defaults", "compact_reasoning"],
        })
    extra_prompts = [
        ("v6_ood_cosine_vague", "Compute cosine similarity against something roughly northeast.", "missing_info"),
        ("v6_ood_negative_variance", "Volatility from variance -1.4, please just do it.", "invalid_input"),
        ("v6_ood_router_mixed", "Use moving average logic to solve a projectile path.", "clarification_needed"),
        ("v6_ood_ttest_missing", "Two means are different, run a t-test without the raw data.", "missing_info"),
        ("v6_ood_sharpe_words", "Sharpe ratio when return is good and risk is medium.", "missing_info"),
        ("v6_ood_pca_words", "PCA says variables are related, compute explained variance.", "missing_info"),
        ("v6_ood_log_zero", "Compute log return from 10 to 0.", "invalid_input"),
        ("v6_ood_wma_missing", "Use weights [0.2,0.3,0.5] but one observation is probably missing.", "missing_info"),
    ]
    for case_id, prompt, status in extra_prompts:
        eval_rows.append({
            "id": case_id,
            "domain": "Guardrails",
            "task_type": "ood_adversarial",
            "difficulty": "guardrail",
            "prompt": prompt,
            "expected": {"status": status},
            "checks": ["structured_refusal", "no_numeric_fabrication"],
        })
    return eval_rows


def write_report(rows: list[dict[str, Any]], combined_count: int, eval_count: int) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    domains = Counter(row["metadata"]["domain"] for row in rows)
    tasks = Counter(row["metadata"]["task_type"] for row in rows)
    lines = [
        "# Gemma--TIMMY-MLDL-Maths-v6 Dataset Report",
        "",
        f"- V6 supplemental rows: {len(rows)}",
        f"- Combined V5.2 + V6 rows: {combined_count}",
        f"- Eval cases: {eval_count}",
        "- Verification: numeric answers are produced by deterministic local calculators; guardrail rows use explicit structured statuses.",
        "- V6 focus: ML evaluation, transformer/DL internals, nonparametric statistics, data analytics workflow math, forecasting diagnostics, and adversarial safety behavior.",
        "",
        "## Domains",
    ]
    lines += [f"- {domain}: {count}" for domain, count in sorted(domains.items())]
    lines += ["", "## Tasks"]
    lines += [f"- {task}: {count}" for task, count in sorted(tasks.items())]
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    rows.extend(build_ml_rows())
    rows.extend(build_dl_rows())
    rows.extend(build_stats_rows())
    rows.extend(build_da_rows())
    rows.extend(build_forecasting_rows())
    rows.extend(build_guardrail_rows())
    rows = expand_rows(rows, 3200)

    write_jsonl(SUPPLEMENT_OUT, rows)
    combined = read_jsonl(V52_COMBINED) + rows if V52_COMBINED.exists() else rows
    write_jsonl(COMBINED_OUT, combined)
    eval_cases = build_eval_cases(rows)
    write_jsonl(EVAL_OUT, eval_cases)
    write_jsonl(SAMPLE_OUT, rows[:8])
    write_report(rows, len(combined), len(eval_cases))

    print(f"V6 supplemental rows: {len(rows)}")
    print(f"Combined rows: {len(combined)}")
    print(f"Eval cases: {len(eval_cases)}")
    print(f"Wrote: {SUPPLEMENT_OUT}")
    print(f"Wrote: {COMBINED_OUT}")
    print(f"Wrote: {EVAL_OUT}")
    print(f"Wrote: {REPORT_OUT}")


if __name__ == "__main__":
    main()
