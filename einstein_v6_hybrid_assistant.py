"""V6 hybrid assistant: deterministic V6 calculators + V5.2 fallback.

The raw V6 LoRA adapter is not used for arithmetic. This runtime owns exact
calculations and structured refusals, then falls back to the mature V5.2 router.
"""

from __future__ import annotations

import argparse
import ast
import json
import re

from einstein_dl_hybrid_assistant import parse_int, parse_number, parse_vector_after
from einstein_v52_hybrid_assistant import answer_question as answer_v52_question
from generate_v6_curated_dataset import (
    benjamini_hochberg,
    beta_posterior,
    bonferroni,
    funnel_rates,
    iqr_outliers,
    information_gain,
    label_smoothing_ce,
    mann_whitney_u,
    mase,
    minmax_scale,
    pca_2x2,
    r4,
    smape,
    standardize,
    transformer_shapes,
    wilcoxon_signed_rank,
)


def fmt(values: list[float]) -> str:
    return "[" + ", ".join(r4(float(v)) for v in values) + "]"


def parse_literal_after(pattern: str, question: str):
    match = re.search(pattern, question, re.IGNORECASE)
    if not match:
        return None
    return ast.literal_eval(match.group(1))


def parse_matrix_after(label: str, question: str):
    return parse_literal_after(rf"{re.escape(label)}\s*(?:=|are)?\s*(\[\[.*?\]\])", question)


def parse_any_vector(labels: list[str], question: str) -> list[float] | None:
    for label in labels:
        values = parse_vector_after(label, question)
        if values is not None:
            return values
    return None


def structured_status(status: str, message: str, required: list[str] | None = None, invalid: list[str] | None = None) -> str:
    return json.dumps(
        {
            "status": status,
            "message": message,
            "required_fields": required or [],
            "invalid_fields": invalid or [],
            "safe_next_step": "Provide valid numeric inputs or clarify the intended task before computing.",
        },
        indent=2,
    )


def answer_v6_guardrail(question: str) -> str | None:
    lower = question.lower()
    if "roughly" in lower and ("cosine" in lower or "vector" in lower):
        return structured_status(
            "missing_info",
            "Cosine similarity requires two numeric vectors with the same dimension.",
            ["vector_b_numeric_components"],
        )
    if "variance" in lower:
        match = re.search(
            r"variance\b(?:\s+(?:is|given|as|given\s+as|equals|equal\s+to|of))*\s*[=:]?\s*(-?\d+(?:\.\d+)?)",
            lower,
        )
        if match and float(match.group(1)) < 0:
            return structured_status("invalid_input", "Variance cannot be negative, so volatility is not real-valued.", invalid=["variance"])
    if "log return" in lower and re.search(r"\bto\s+0\b|price\s*0\b", lower):
        return structured_status("invalid_input", "Log return requires positive start and end prices; log(0) is undefined.", invalid=["end_price"])
    if "two-sample t-test" in lower and ("mean" in lower or "means" in lower) and not re.search(r"std|variance|sample|n=", lower):
        return structured_status(
            "missing_info",
            "A two-sample t-test requires sample sizes, variances or raw samples, and alpha.",
            ["sample_sizes", "standard_deviations_or_raw_samples", "alpha"],
        )
    if "stopping distance" in lower and ("road conditions" in lower or "not documented" in lower):
        return structured_status(
            "missing_info",
            "Stopping distance requires reaction time and deceleration or friction/road condition assumptions.",
            ["reaction_time", "deceleration_or_friction"],
        )
    if "moving average logic" in lower and ("projectile" in lower or "velocity" in lower):
        return structured_status(
            "clarification_needed",
            "This mixes a physics model and a forecasting model; choose kinematics or statistical forecasting.",
            ["intended_model"],
        )
    if "sharpe" in lower and re.search(r"\blow\b|\bmoderate\b|\bgood\b", lower):
        return structured_status(
            "missing_info",
            "Sharpe ratio requires numeric return, numeric risk-free rate, and numeric volatility.",
            ["return", "risk_free_rate", "volatility"],
        )
    return None


def answer_v6_ml(question: str) -> str | None:
    lower = question.lower()
    if "decision" in lower and ("tree" in lower or "split" in lower) and "child" in lower:
        parent = parse_literal_after(r"parent(?: class counts)?\s*(?:=|are)?\s*(\[[^\]]+\])", question)
        children = parse_literal_after(r"(?:child counts|children|leaves)\s*(?:=|are|creates leaves)?\s*(\[\[.*?\]\])", question)
        if parent is None or children is None:
            return None
        out = information_gain([int(x) for x in parent], [[int(x) for x in row] for row in children])
        return (
            "Problem: Evaluate a decision-tree split.\n"
            "Method: entropy=-sum p log2(p); information_gain=parent_entropy-weighted_child_entropy; gini=1-sum p^2.\n"
            f"Calculation: parent_entropy={r4(out['parent_entropy'])}, weighted_child_entropy={r4(out['weighted_child_entropy'])}, "
            f"information_gain={r4(out['information_gain'])}, parent_gini={r4(out['parent_gini'])}.\n"
            f"Result: information_gain={r4(out['information_gain'])}; parent_gini={r4(out['parent_gini'])}.\n"
            "Diagnostic note: Higher gain is useful only if it holds on validation data."
        )
    if "pca" in lower and ("covariance" in lower or "[[" in question):
        matrix = parse_matrix_after("covariance matrix", question) or parse_literal_after(r"covariance\s*(\[\[.*?\]\])", question)
        if matrix is None:
            return structured_status("missing_info", "PCA explained variance requires a numeric covariance matrix or data matrix.", ["covariance_matrix_or_data"])
        out = pca_2x2(float(matrix[0][0]), float(matrix[0][1]), float(matrix[1][1]))
        return (
            "Problem: Compute PCA explained variance for a 2x2 covariance matrix.\n"
            "Method: eigenvalues=(trace +/- sqrt(trace^2-4det))/2; explained ratio=lambda/sum(lambda).\n"
            f"Calculation: lambda_1={r4(out['lambda_1'])}, lambda_2={r4(out['lambda_2'])}, explained_1={r4(out['explained_1'])}.\n"
            f"Result: PC1 explains {r4(100*out['explained_1'])}% of variance.\n"
            "Diagnostic note: Variance preservation is not the same as predictive usefulness."
        )
    return None


def answer_v6_dl(question: str) -> str | None:
    lower = question.lower()
    if "label smoothing" in lower or "label-smoothed" in lower:
        logits = parse_vector_after("logits", question) or parse_vector_after("z", question)
        true_class = parse_int("true_class", question) or parse_int("y", question)
        epsilon = parse_number("epsilon", question) or parse_number("eps", question)
        if logits is None or true_class is None or epsilon is None:
            return None
        out = label_smoothing_ce(logits, int(true_class), float(epsilon))
        return (
            "Problem: Compute label-smoothed softmax cross entropy.\n"
            "Method: target=(1-epsilon) on true class plus epsilon/K everywhere; loss=-sum target_i log softmax_i; gradient=prob-target.\n"
            f"Calculation: probs={fmt(out['probs'])}, target={fmt(out['target'])}, loss={r4(out['loss'])}, gradient={fmt(out['gradient'])}.\n"
            f"Result: loss={r4(out['loss'])}, dL/dlogits={fmt(out['gradient'])}.\n"
            "Diagnostic note: Label smoothing reduces overconfidence but can hurt calibration if overused."
        )
    if "transformer" in lower and ("shape" in lower or "attention" in lower):
        batch = parse_int("batch", question) or parse_int("B", question)
        seq = parse_int("seq_len", question) or parse_int("seq", question) or parse_int("T", question)
        d_model = parse_int("d_model", question) or parse_int("C", question)
        heads = parse_int("heads", question) or parse_int("H", question)
        if None in [batch, seq, d_model, heads]:
            return None
        out = transformer_shapes(int(batch), int(seq), int(d_model), int(heads))
        return (
            "Problem: Compute transformer multi-head attention shapes.\n"
            "Method: head_dim=d_model/heads; QKV=[B,H,T,head_dim]; scores=[B,H,T,T]; output=[B,T,d_model].\n"
            f"Calculation: head_dim={out['head_dim']}, QKV={out['qkv_shape']}, scores={out['attention_scores_shape']}.\n"
            f"Result: attention_output={out['attention_output_shape']}, ffn_hidden={out['ffn_hidden_shape']}.\n"
            "Diagnostic note: Most shape bugs mix up heads and head dimension."
        )
    return None


def answer_v6_stats_da_forecast(question: str) -> str | None:
    lower = question.lower()
    values = parse_any_vector(["values", "history"], question)
    if "z-score" in lower or "standard deviation" in lower and values is not None:
        out = standardize(values)
        return (
            "Problem: Standardize numeric values.\n"
            "Method: z=(x-mean)/sample_std.\n"
            f"Calculation: mean={r4(out['mean'])}, sample_std={r4(out['sample_std'])}, z={fmt(out['z'])}.\n"
            f"Result: z_scores={fmt(out['z'])}.\n"
            "Diagnostic note: Fit scaling parameters on training data only to avoid leakage."
        )
    if "min-max" in lower or "minmax" in lower:
        if values is None:
            return None
        out = minmax_scale(values)
        return (
            "Problem: Min-max scale numeric values.\n"
            "Method: scaled=(x-min)/(max-min).\n"
            f"Calculation: min={r4(out['min'])}, max={r4(out['max'])}, scaled={fmt(out['scaled'])}.\n"
            f"Result: scaled_values={fmt(out['scaled'])}.\n"
            "Diagnostic note: Min-max scaling is sensitive to outliers."
        )
    if "iqr" in lower or "outlier" in lower:
        q1 = parse_number("Q1", question)
        q3 = parse_number("Q3", question)
        if values is None or q1 is None or q3 is None:
            return None
        out = iqr_outliers(values, float(q1), float(q3))
        return (
            "Problem: Detect IQR outliers.\n"
            "Method: IQR=Q3-Q1; lower=Q1-1.5*IQR; upper=Q3+1.5*IQR.\n"
            f"Calculation: IQR={r4(out['iqr'])}, lower={r4(out['lower'])}, upper={r4(out['upper'])}, outliers={out['outliers']}.\n"
            f"Result: outliers={out['outliers']}.\n"
            "Diagnostic note: Outliers need domain review before removal."
        )
    if "mann-whitney" in lower:
        a = parse_vector_after("sample_a", question)
        b = parse_vector_after("sample_b", question)
        if a is None or b is None:
            return None
        out = mann_whitney_u(a, b)
        return (
            "Problem: Compute Mann-Whitney U.\n"
            "Method: rank all observations, compute U_A and U_B, then use min(U_A,U_B).\n"
            f"Calculation: U_A={r4(out['u_a'])}, U_B={r4(out['u_b'])}, U={r4(out['u'])}.\n"
            f"Result: Mann-Whitney U={r4(out['u'])}.\n"
            "Diagnostic note: Use this for independent samples without normality assumptions."
        )
    if "wilcoxon" in lower:
        before = parse_vector_after("before", question)
        after = parse_vector_after("after", question)
        if before is None or after is None:
            return None
        out = wilcoxon_signed_rank(before, after)
        return f"Problem: Compute Wilcoxon signed-rank statistic.\nMethod: rank nonzero absolute paired differences and compare W+ and W-.\nCalculation: W_plus={r4(out['w_plus'])}, W_minus={r4(out['w_minus'])}, W={r4(out['w'])}.\nResult: Wilcoxon W={r4(out['w'])}.\nDiagnostic note: This is paired and nonparametric."
    if "multiple testing" in lower or "bonferroni" in lower or "benjamini" in lower:
        p_values = parse_vector_after("p_values", question)
        alpha = parse_number("alpha_or_q", question) or parse_number("alpha", question) or parse_number("q", question)
        if p_values is None or alpha is None:
            return None
        bon = bonferroni(p_values, float(alpha))
        bh = benjamini_hochberg(p_values, float(alpha))
        return (
            "Problem: Correct for multiple testing.\n"
            "Method: Bonferroni threshold=alpha/m; BH rejects up to largest p_i <= (i/m)q.\n"
            f"Calculation: Bonferroni threshold={r4(bon['threshold'])}, reject={bon['reject']}; BH cutoff_rank={bh['cutoff_rank']}, reject={bh['reject']}.\n"
            f"Result: Bonferroni reject={bon['reject']}; BH reject={bh['reject']}.\n"
            "Diagnostic note: BH is less conservative and controls false discovery rate."
        )
    if "beta" in lower and "posterior" in lower:
        successes = parse_int("successes", question)
        failures = parse_int("failures", question)
        if successes is None or failures is None:
            return None
        alpha_prior, beta_prior = (1.0, 1.0)
        prior_match = re.search(
            r"prior\s*=\s*Beta\(\s*([-+]?\d+\.?\d*(?:e[-+]?\d+)?)\s*,\s*([-+]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\)",
            question,
            re.IGNORECASE,
        )
        if prior_match:
            alpha_prior = float(prior_match.group(1))
            beta_prior = float(prior_match.group(2))
        out = beta_posterior(int(successes), int(failures), alpha_prior, beta_prior)
        return (
            "Problem: Update a beta-binomial conversion model.\n"
            "Method: alpha_post=alpha_prior+successes, beta_post=beta_prior+failures, mean=alpha/(alpha+beta).\n"
            f"Calculation: alpha_post={r4(out['alpha_post'])}, beta_post={r4(out['beta_post'])}, posterior_mean={r4(out['posterior_mean'])}.\n"
            f"Result: posterior_mean={r4(out['posterior_mean'])}.\n"
            "Diagnostic note: Bayesian rates are more stable for small samples."
        )
    if "smape" in lower or "mase" in lower:
        actuals = parse_vector_after("actuals", question)
        forecasts = parse_vector_after("forecasts", question)
        train = parse_vector_after("train_history", question)
        if actuals is None or forecasts is None or train is None:
            return None
        s = smape(actuals, forecasts)
        m = mase(actuals, forecasts, train)
        return (
            "Problem: Compute forecast diagnostics.\n"
            "Method: sMAPE=mean(|A-F|/((|A|+|F|)/2)); MASE=MAE divided by naive training MAE.\n"
            f"Calculation: sMAPE={r4(s)}, MASE={r4(m)}.\n"
            f"Result: sMAPE={r4(s)}, MASE={r4(m)}.\n"
            "Diagnostic note: MASE below 1 beats the naive baseline."
        )
    if "funnel" in lower:
        counts = parse_vector_after("stage_counts", question)
        if counts is None:
            return None
        out = funnel_rates([int(x) for x in counts])
        return f"Problem: Compute funnel conversion.\nMethod: step_rate=next/current; overall=final/first.\nCalculation: step_rates={fmt(out['step_rates'])}, overall={r4(out['overall_rate'])}.\nResult: overall_conversion={r4(out['overall_rate'])}.\nDiagnostic note: The largest step drop is the first diagnostic target."
    return None


def answer_v6_training_consulting(question: str) -> str | None:
    lower = question.lower()

    if "effective batch size" in lower or "global batch size" in lower:
        per_device = parse_int("batch size", question) or parse_int("per_device_batch_size", question) or parse_int("micro_batch_size", question)
        grad_accum = parse_int("gradient accumulation", question) or parse_int("grad accumulation", question) or parse_int("grad_accum", question)
        devices = parse_int("gpus", question) or parse_int("gpu", question) or parse_int("devices", question) or parse_int("data parallel gpus", question)
        if devices is None:
            devices = 1
        if per_device is not None and grad_accum is not None:
            effective_batch = int(per_device) * int(grad_accum) * int(devices)
            return (
                "Problem: Compute effective batch size for training.\n"
                "Method: effective_batch_size=per_device_batch_size * gradient_accumulation_steps * number_of_devices.\n"
                f"Calculation: per_device_batch_size={per_device}, gradient_accumulation_steps={grad_accum}, devices={devices}, effective_batch_size={effective_batch}.\n"
                f"Result: effective_batch_size={effective_batch}.\n"
                "Diagnostic note: Larger effective batch sizes usually require retuning learning rate and warmup."
            )
        return (
            "Problem: Explain effective batch size for training.\n"
            "Method: effective_batch_size=per_device_batch_size * gradient_accumulation_steps * number_of_devices.\n"
            "Calculation: Need per-device batch size and gradient accumulation; devices defaults to 1 if unspecified.\n"
            "Result: Provide batch size and gradient accumulation to compute the exact effective batch size.\n"
            "Diagnostic note: Effective batch size controls optimization dynamics more than the micro-batch alone."
        )

    if (
        "overfitting" in lower
        or ("validation loss" in lower and "training loss" in lower)
        or ("after several epochs" in lower and "loss" in lower)
    ):
        return (
            "Problem: Diagnose overfitting during training.\n"
            "Method: Compare training loss, validation loss, and validation metrics across epochs to detect memorization instead of generalization.\n"
            "Calculation: Overfitting is likely when training loss keeps falling while validation loss rises or validation metrics flatten after several epochs.\n"
            "Result: First actions are early stopping, fewer epochs, stronger weight decay, more dropout if supported, data augmentation or more data, and a smaller learning-rate tail near the end of training.\n"
            "Diagnostic note: If the validation set is tiny or noisy, confirm the pattern with multiple eval checkpoints before changing the whole recipe."
        )

    if (
        "oscillating training loss" in lower
        or "loss oscillating" in lower
        or ("training loss" in lower and "oscillat" in lower)
        or ("learning rate" in lower and "fix" in lower)
        or ("lr" in lower and "too high" in lower)
    ):
        return (
            "Problem: Diagnose oscillating training loss and choose a learning-rate fix.\n"
            "Method: Check whether updates are too aggressive relative to batch size, gradient noise, and warmup schedule.\n"
            "Calculation: The highest-probability fix order is lower the learning rate, add or lengthen warmup, increase gradient accumulation if memory is tight, clip gradients, and verify that the scheduler is not spiking LR late in training.\n"
            "Result: Start by reducing learning rate by about 2x to 4x, keep gradient clipping enabled, and rerun a short controlled comparison before changing multiple knobs at once.\n"
            "Diagnostic note: Oscillation can also come from bad labels, unstable mixed precision, or an effective batch size that is too small for the chosen LR."
        )

    return None


def answer_question(question: str) -> str:
    for solver in [answer_v6_guardrail, answer_v6_ml, answer_v6_dl, answer_v6_stats_da_forecast, answer_v6_training_consulting]:
        response = solver(question)
        if response:
            return response
    return answer_v52_question(question)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", action="append", required=True)
    args = parser.parse_args()
    for idx, question in enumerate(args.question, 1):
        print("=" * 80)
        print(f"Question {idx}: {question}")
        print("-" * 80)
        print(answer_question(question))
        print()


if __name__ == "__main__":
    main()
