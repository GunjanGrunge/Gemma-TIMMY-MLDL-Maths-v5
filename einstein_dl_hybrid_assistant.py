"""Calculator-backed deep-learning Einstein assistant.

The V5 LoRA adapter is useful for tone and explanation style. This script owns
exact DL arithmetic so production answers do not depend on model arithmetic.

Examples:
python einstein_dl_hybrid_assistant.py --question "Gradient clipping: gradient=[3.0,4.0], max_norm=2.0. Compute norm, scale, and clipped gradient."
python einstein_dl_hybrid_assistant.py --question "CNN shape: input 32x32, kernel=3, stride=1, padding=1, output_channels=64. Compute output shape."
"""

from __future__ import annotations

import argparse
import ast
import re

from dl_calculators import (
    activation_derivative,
    adam_update,
    binary_cross_entropy_probability,
    broadcast_shape,
    classification_metrics,
    conv2d_output_shape,
    fmt_list,
    fmt_matrix,
    gradient_descent_update,
    gradient_clip_by_norm,
    inverted_dropout,
    linear_mse_backprop,
    matmul,
    matmul_shape,
    momentum_sgd_update,
    multiclass_accuracy,
    normalize_forward,
    r4,
    scaled_dot_product_attention,
    semantic_search_rank,
    sigmoid_bce_backprop,
    softmax_cross_entropy,
    weight_decay_sgd_update,
    cosine_similarity,
)


NUMBER = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"


def parse_number(name: str, question: str, *, default: float | None = None) -> float | None:
    match = re.search(rf"{re.escape(name)}\s*=\s*({NUMBER})", question, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return default


def parse_int(name: str, question: str, *, default: int | None = None) -> int | None:
    value = parse_number(name, question, default=None)
    if value is None:
        return default
    return int(value)


def parse_vector_after(label: str, question: str) -> list[float] | None:
    match = re.search(rf"{re.escape(label)}\s*=\s*(\[[^\]]+\])", question, re.IGNORECASE)
    if not match:
        return None
    value = ast.literal_eval(match.group(1))
    return [float(x) for x in value]


def parse_matrix_after(label: str, question: str) -> list[list[float]] | None:
    match = re.search(rf"{re.escape(label)}\s*=\s*(\[\[.*?\]\])", question, re.IGNORECASE)
    if not match:
        return None
    value = ast.literal_eval(match.group(1))
    return [[float(x) for x in row] for row in value]


def parse_int_vector_after(label: str, question: str) -> list[int] | None:
    values = parse_vector_after(label, question)
    if values is None:
        return None
    return [int(x) for x in values]


def parse_shape_after(label: str, question: str) -> tuple[int, ...] | None:
    match = re.search(rf"{re.escape(label)}\s*=\s*(\([^)]+\))", question, re.IGNORECASE)
    if not match:
        return None
    value = ast.literal_eval(match.group(1))
    return tuple(int(x) for x in value)


def answer_sigmoid_bce(question: str) -> str | None:
    if not re.search(r"sigmoid|BCE|binary cross", question, re.IGNORECASE):
        return None
    x = parse_number("x", question)
    w = parse_number("w", question)
    b = parse_number("b", question)
    y = parse_int("y", question)
    if None in [x, w, b, y]:
        return None

    result = sigmoid_bce_backprop(float(x), float(w), float(b), int(y))
    direction = "increase w" if result["dldw"] < 0 else "decrease w"
    return (
        "Problem: Compute one-neuron sigmoid backprop with BCE.\n"
        "Method: z=wx+b, a=sigmoid(z), BCE=-[y log(a)+(1-y)log(1-a)], dL/dz=a-y, dL/dw=(a-y)x, dL/db=a-y.\n"
        f"Calculation: z={r4(w)}*{r4(x)}+{r4(b)}={r4(result['z'])}. "
        f"a=sigmoid({r4(result['z'])})={r4(result['a'])}. "
        f"loss={r4(result['loss'])}. dL/dz={r4(result['a'])}-{y}={r4(result['dldz'])}. "
        f"dL/dw={r4(result['dldz'])}*{r4(x)}={r4(result['dldw'])}. dL/db={r4(result['dldb'])}.\n"
        f"Result: z={r4(result['z'])}, a={r4(result['a'])}, loss={r4(result['loss'])}, "
        f"dL/dz={r4(result['dldz'])}, dL/dw={r4(result['dldw'])}, dL/db={r4(result['dldb'])}; gradient descent should {direction}.\n"
        "Debug note: For sigmoid plus BCE, do not multiply by sigmoid derivative again."
    )


def answer_softmax_ce(question: str) -> str | None:
    if not re.search(r"softmax|cross.?entropy|CE", question, re.IGNORECASE):
        return None
    logits = parse_vector_after("logits", question)
    true_class = parse_int("true_class", question)
    if true_class is None:
        true_class = parse_int("class", question)
    if logits is None or true_class is None:
        return None

    result = softmax_cross_entropy(logits, true_class)
    return (
        "Problem: Compute softmax cross-entropy and logit gradients.\n"
        "Method: p_i=exp(z_i)/sum_j exp(z_j), loss=-log(p_y), and dL/dz=p-one_hot(y).\n"
        f"Calculation: softmax={fmt_list(result['probs'])}. loss=-log({r4(result['probs'][true_class])})={r4(result['loss'])}. "
        f"gradient={fmt_list(result['gradient'])}.\n"
        f"Result: probabilities={fmt_list(result['probs'])}, loss={r4(result['loss'])}, dL/dlogits={fmt_list(result['gradient'])}.\n"
        "Debug note: For softmax plus CE, use p-y_onehot rather than differentiating softmax manually."
    )


def answer_binary_ce_probability(question: str) -> str | None:
    if not re.search(r"binary cross entropy|BCE", question, re.IGNORECASE):
        return None
    prob = parse_number("p", question)
    y = parse_int("y", question)
    if prob is None or y is None:
        return None
    result = binary_cross_entropy_probability(prob, y)
    return (
        "Problem: Compute binary cross entropy from a probability.\n"
        "Method: BCE=-[y log(p)+(1-y)log(1-p)] and dL/dp=-(y/p)+(1-y)/(1-p).\n"
        f"Calculation: BCE={r4(result['loss'])}, dL/dp={r4(result['dldp'])}.\n"
        f"Result: BCE={r4(result['loss'])}, dL/dp={r4(result['dldp'])}.\n"
        "Debug note: For logits, use a numerically stable BCE-with-logits implementation."
    )


def answer_classification_metrics(question: str) -> str | None:
    if not re.search(r"TP|FP|FN|TN|precision|recall|F1|accuracy", question, re.IGNORECASE):
        return None
    tp = parse_int("TP", question)
    fp = parse_int("FP", question)
    fn = parse_int("FN", question)
    tn = parse_int("TN", question)
    if None in [tp, fp, fn, tn]:
        return None
    result = classification_metrics(int(tp), int(fp), int(fn), int(tn))
    return (
        "Problem: Compute binary classification metrics from a confusion matrix.\n"
        "Method: accuracy=(TP+TN)/N, precision=TP/(TP+FP), recall=TP/(TP+FN), specificity=TN/(TN+FP), F1=2PR/(P+R).\n"
        f"Calculation: accuracy={r4(result['accuracy'])}, precision={r4(result['precision'])}, recall={r4(result['recall'])}, "
        f"specificity={r4(result['specificity'])}, F1={r4(result['f1'])}.\n"
        f"Result: accuracy={r4(result['accuracy'])}, precision={r4(result['precision'])}, recall={r4(result['recall'])}, "
        f"specificity={r4(result['specificity'])}, F1={r4(result['f1'])}.\n"
        "Debug note: On imbalanced data, inspect precision/recall/F1 rather than accuracy alone."
    )


def answer_multiclass_accuracy(question: str) -> str | None:
    if not re.search(r"multiclass accuracy|labels=.*predictions=", question, re.IGNORECASE):
        return None
    labels = parse_int_vector_after("labels", question)
    predictions = parse_int_vector_after("predictions", question)
    if labels is None or predictions is None:
        return None
    result = multiclass_accuracy(labels, predictions)
    return (
        "Problem: Compute multiclass accuracy.\n"
        "Method: Accuracy is exact label matches divided by total examples.\n"
        f"Calculation: correct={result['correct']} out of {len(labels)}, accuracy={r4(result['accuracy'])}.\n"
        f"Result: correct={result['correct']}, accuracy={r4(result['accuracy'])}.\n"
        "Debug note: Use per-class metrics when class frequencies are uneven."
    )


def answer_adam(question: str) -> str | None:
    if not re.search(r"\bAdam\b", question, re.IGNORECASE):
        return None
    w = parse_number("w", question)
    grad = parse_number("grad", question)
    m = parse_number("m", question, default=0.0)
    v = parse_number("v", question, default=0.0)
    t = parse_int("t", question, default=1)
    lr = parse_number("lr", question, default=0.001)
    beta1 = parse_number("beta1", question, default=0.9)
    beta2 = parse_number("beta2", question, default=0.999)
    eps = parse_number("eps", question, default=1e-8)
    if w is None or grad is None:
        return None

    result = adam_update(w, grad, float(m), float(v), int(t), float(lr), float(beta1), float(beta2), float(eps))
    return (
        "Problem: Compute one Adam optimizer update.\n"
        "Method: m=beta1*m+(1-beta1)g, v=beta2*v+(1-beta2)g^2, m_hat=m/(1-beta1^t), v_hat=v/(1-beta2^t), w=w-lr*m_hat/(sqrt(v_hat)+eps).\n"
        f"Calculation: m_new={r4(result['m_new'])}, v_new={r4(result['v_new'])}, "
        f"m_hat={r4(result['m_hat'])}, v_hat={r4(result['v_hat'])}, step={r4(result['step'])}.\n"
        f"Result: w_new={r4(result['w_new'])}.\n"
        "Debug note: Bias correction matters most in the early Adam steps."
    )


def answer_gradient_clipping(question: str) -> str | None:
    if not re.search(r"clipping|clip", question, re.IGNORECASE):
        return None
    gradient = parse_vector_after("gradient", question)
    max_norm = parse_number("max_norm", question)
    if gradient is None or max_norm is None:
        return None

    result = gradient_clip_by_norm(gradient, max_norm)
    return (
        "Problem: Clip a gradient vector by global L2 norm.\n"
        "Method: norm=sqrt(sum(g_i^2)); if norm>max_norm, clipped_g=g*(max_norm/norm).\n"
        f"Calculation: norm={r4(result['norm'])}. scale=min(1,{r4(max_norm)}/{r4(result['norm'])})={r4(result['scale'])}. "
        f"clipped={fmt_list(result['clipped'])}.\n"
        f"Result: norm={r4(result['norm'])}, scale={r4(result['scale'])}, clipped_gradient={fmt_list(result['clipped'])}.\n"
        "Debug note: Clipping changes the magnitude, not the direction."
    )


def answer_weight_decay_sgd(question: str) -> str | None:
    if not re.search(r"weight decay|lambda", question, re.IGNORECASE):
        return None
    weights = parse_vector_after("weights", question)
    gradients = parse_vector_after("gradients", question)
    lr = parse_number("lr", question)
    lam = parse_number("lambda", question)
    if weights is None or gradients is None or lr is None or lam is None:
        return None
    result = weight_decay_sgd_update(weights, gradients, lr, lam)
    return (
        "Problem: Compute SGD with L2 weight decay.\n"
        "Method: total_gradient=gradient+lambda*w, then w_new=w-lr*total_gradient.\n"
        f"Calculation: total_gradient={fmt_list(result['total_gradient'])}, updated_weights={fmt_list(result['updated_weights'])}.\n"
        f"Result: total_gradient={fmt_list(result['total_gradient'])}, updated_weights={fmt_list(result['updated_weights'])}.\n"
        "Debug note: Weight decay shrinks weights while still following the data gradient."
    )


def answer_gradient_descent(question: str) -> str | None:
    if not re.search(r"gradient descent|updated weights", question, re.IGNORECASE):
        return None
    weights = parse_vector_after("weights", question)
    gradients = parse_vector_after("gradients", question)
    lr = parse_number("lr", question)
    if weights is None or gradients is None or lr is None:
        return None
    result = gradient_descent_update(weights, gradients, lr)
    return (
        "Problem: Compute one vanilla gradient descent update.\n"
        "Method: w_new=w-lr*gradient elementwise.\n"
        f"Calculation: updated_weights={fmt_list(result['updated_weights'])}.\n"
        f"Result: updated_weights={fmt_list(result['updated_weights'])}.\n"
        "Debug note: Positive gradients move weights downward under gradient descent."
    )


def answer_attention(question: str) -> str | None:
    if not re.search(r"attention|scaled dot", question, re.IGNORECASE):
        return None
    query = parse_vector_after("q", question) or parse_vector_after("query", question)
    keys = parse_matrix_after("keys", question)
    values = parse_matrix_after("values", question)
    if query is None or keys is None or values is None:
        return None

    result = scaled_dot_product_attention(query, keys, values)
    return (
        "Problem: Compute scaled dot-product attention for one query.\n"
        "Method: scores=qK^T/sqrt(d), weights=softmax(scores), output=sum_i weights_i*V_i.\n"
        f"Calculation: scores={fmt_list(result['scores'])}, weights={fmt_list(result['weights'])}, output={fmt_list(result['output'])}.\n"
        f"Result: attention_output={fmt_list(result['output'])}.\n"
        "Debug note: The output is a weighted sum of value vectors, not a weighted sum of keys."
    )


def answer_cosine_similarity(question: str) -> str | None:
    if not re.search(r"cosine similarity", question, re.IGNORECASE):
        return None
    a = parse_vector_after("vector_a", question) or parse_vector_after("a", question)
    b = parse_vector_after("vector_b", question) or parse_vector_after("b", question)
    if a is None or b is None:
        return None
    result = cosine_similarity(a, b)
    return (
        "Problem: Compute cosine similarity between embedding vectors.\n"
        "Method: cosine(a,b)=(a dot b)/(||a|| ||b||).\n"
        f"Calculation: dot={r4(result['dot'])}, ||a||={r4(result['norm_a'])}, ||b||={r4(result['norm_b'])}, cosine={r4(result['cosine'])}.\n"
        f"Result: cosine_similarity={r4(result['cosine'])}.\n"
        "Debug note: Cosine measures directional alignment, not raw magnitude."
    )


def answer_semantic_search(question: str) -> str | None:
    if not re.search(r"semantic search|document_embeddings|rank documents", question, re.IGNORECASE):
        return None
    query = parse_vector_after("query_embedding", question) or parse_vector_after("query", question)
    documents = parse_matrix_after("document_embeddings", question) or parse_matrix_after("documents", question)
    if query is None or documents is None:
        return None
    result = semantic_search_rank(query, documents)
    return (
        "Problem: Rank documents by embedding similarity.\n"
        "Method: Compute cosine(query, document_i) for each document and choose the highest score.\n"
        f"Calculation: cosine_scores={fmt_list(result['scores'])}, best_document_index={result['best']}.\n"
        f"Result: best_document_index={result['best']}, cosine_scores={fmt_list(result['scores'])}.\n"
        "Debug note: Use cosine or normalized embeddings when magnitude should not dominate semantic ranking."
    )


def answer_cnn_shape(question: str) -> str | None:
    if not re.search(r"CNN|conv|kernel|padding", question, re.IGNORECASE):
        return None
    hw = re.search(r"(?:input\s*)?(?:HxW\s*=\s*)?(\d+)\s*x\s*(\d+)", question, re.IGNORECASE)
    height = int(hw.group(1)) if hw else parse_int("height", question)
    width = int(hw.group(2)) if hw else parse_int("width", question)
    kernel = parse_int("kernel", question)
    stride = parse_int("stride", question)
    padding = parse_int("padding", question)
    output_channels = parse_int("output_channels", question) or parse_int("channels", question)
    if None in [height, width, kernel, stride, padding, output_channels]:
        return None

    result = conv2d_output_shape(int(height), int(width), int(kernel), int(stride), int(padding), int(output_channels))
    return (
        "Problem: Compute 2D convolution output shape.\n"
        "Method: out=floor((in+2*padding-kernel)/stride)+1 for height and width.\n"
        f"Calculation: out_h=floor(({height}+2*{padding}-{kernel})/{stride})+1={result['height']}. "
        f"out_w=floor(({width}+2*{padding}-{kernel})/{stride})+1={result['width']}.\n"
        f"Result: Output shape is {result['channels']}x{result['height']}x{result['width']} in channel-first format.\n"
        "Debug note: Check spatial divisibility when stride is greater than 1."
    )


def answer_matrix_multiply(question: str) -> str | None:
    if not re.search(r"matrix multiplication|A@B|matmul", question, re.IGNORECASE):
        return None
    if re.search(r"shape", question, re.IGNORECASE):
        return None
    a = parse_matrix_after("A", question)
    b = parse_matrix_after("B", question)
    if a is None or b is None:
        return None
    result = matmul(a, b)
    return (
        "Problem: Compute matrix multiplication.\n"
        "Method: Each C_ij is the dot product of row i of A and column j of B.\n"
        f"Calculation: A@B={fmt_matrix(result)}.\n"
        f"Result: product={fmt_matrix(result)}.\n"
        "Debug note: Matrix multiplication requires matching inner dimensions."
    )


def answer_matmul_shape(question: str) -> str | None:
    if not re.search(r"matmul shape|A shape|B shape", question, re.IGNORECASE):
        return None
    a_shape = parse_shape_after("A shape", question)
    b_shape = parse_shape_after("B shape", question)
    if a_shape is None or b_shape is None or len(a_shape) != 2 or len(b_shape) != 2:
        return None
    result = matmul_shape(a_shape, b_shape)
    return (
        "Problem: Compute matrix multiplication output shape.\n"
        "Method: For A with shape (m,k) and B with shape (k,n), A@B has shape (m,n).\n"
        f"Calculation: output_shape={result['output_shape']}.\n"
        f"Result: output_shape={result['output_shape']}.\n"
        "Debug note: The inner dimensions must match."
    )


def answer_broadcasting_shape(question: str) -> str | None:
    if not re.search(r"broadcast", question, re.IGNORECASE):
        return None
    shape_a = parse_shape_after("tensor A shape", question) or parse_shape_after("A shape", question)
    shape_b = parse_shape_after("tensor B shape", question) or parse_shape_after("B shape", question)
    if shape_a is None or shape_b is None:
        return None
    result = broadcast_shape(shape_a, shape_b)
    return (
        "Problem: Compute tensor broadcasting shape.\n"
        "Method: Compare dimensions from the right; dimensions are compatible when equal or one dimension is 1.\n"
        f"Calculation: {shape_a} and {shape_b} broadcast to {result}.\n"
        f"Result: broadcast_output_shape={result}.\n"
        "Debug note: Broadcasting is for elementwise operations and does not mean matrix multiplication."
    )


def answer_normalization(question: str) -> str | None:
    if not re.search(r"BatchNorm|LayerNorm", question, re.IGNORECASE):
        return None
    values = parse_vector_after("activations", question) or parse_vector_after("vector", question)
    gamma = parse_number("gamma", question, default=1.0)
    beta = parse_number("beta", question, default=0.0)
    eps = parse_number("eps", question, default=1e-5)
    if values is None:
        return None

    result = normalize_forward(values, float(gamma), float(beta), float(eps))
    name = "BatchNorm" if re.search(r"BatchNorm", question, re.IGNORECASE) else "LayerNorm"
    scope = "across the mini-batch for one feature" if name == "BatchNorm" else "across features in one sample"
    return (
        f"Problem: Compute {name} forward pass.\n"
        f"Method: Normalize {scope}: xhat=(x-mean)/sqrt(var+eps), output=gamma*xhat+beta.\n"
        f"Calculation: mean={r4(result['mean'])}, variance={r4(result['variance'])}, xhat={fmt_list(result['xhat'])}, output={fmt_list(result['output'])}.\n"
        f"Result: {name} output={fmt_list(result['output'])}.\n"
        "Debug note: BatchNorm and LayerNorm use the same arithmetic here, but their statistic scope is different."
    )


def answer_dropout(question: str) -> str | None:
    if not re.search(r"dropout", question, re.IGNORECASE):
        return None
    values = parse_vector_after("activations", question)
    mask_match = re.search(r"mask\s*=\s*(\[[^\]]+\])", question, re.IGNORECASE)
    keep_prob = parse_number("keep_prob", question)
    if values is None or mask_match is None or keep_prob is None:
        return None
    mask = [int(x) for x in ast.literal_eval(mask_match.group(1))]
    result = inverted_dropout(values, mask, keep_prob)
    return (
        "Problem: Compute inverted dropout training-time output.\n"
        "Method: output=x*mask/keep_prob so the expected activation scale stays unchanged.\n"
        f"Calculation: output={fmt_list(result['output'])}.\n"
        f"Result: dropout_output={fmt_list(result['output'])}.\n"
        "Debug note: At inference, inverted dropout uses no mask and no additional scaling."
    )


def answer_momentum(question: str) -> str | None:
    if not re.search(r"momentum", question, re.IGNORECASE):
        return None
    w = parse_number("w", question)
    grad = parse_number("grad", question)
    velocity = parse_number("previous_velocity", question)
    if velocity is None:
        velocity = parse_number("velocity", question, default=0.0)
    lr = parse_number("lr", question, default=0.01)
    momentum = parse_number("momentum", question, default=0.9)
    if w is None or grad is None:
        return None
    result = momentum_sgd_update(w, grad, float(velocity), float(lr), float(momentum))
    return (
        "Problem: Compute one momentum SGD update.\n"
        "Method: v_new=momentum*v_old+grad and w_new=w-lr*v_new.\n"
        f"Calculation: v_new={r4(momentum)}*{r4(velocity)}+{r4(grad)}={r4(result['velocity'])}. "
        f"w_new={r4(w)}-{r4(lr)}*{r4(result['velocity'])}={r4(result['w_new'])}.\n"
        f"Result: new_velocity={r4(result['velocity'])}, w_new={r4(result['w_new'])}.\n"
        "Debug note: Momentum smooths gradients but can overshoot with a high learning rate."
    )


def answer_activation(question: str) -> str | None:
    if not re.search(r"derivative|activation", question, re.IGNORECASE):
        return None
    match = re.search(r"(relu|sigmoid|tanh)", question, re.IGNORECASE)
    value = parse_number("x", question)
    if match is None or value is None:
        return None
    result = activation_derivative(match.group(1), value)
    return (
        f"Problem: Compute derivative of {result['activation']} at x={r4(value)}.\n"
        "Method: Use the activation derivative formula directly.\n"
        f"Calculation: derivative={r4(result['derivative'])}.\n"
        f"Result: {result['activation']}'({r4(value)})={r4(result['derivative'])}.\n"
        "Debug note: Activation derivatives control how much gradient passes backward."
    )


def answer_linear_mse(question: str) -> str | None:
    if not re.search(r"linear|MSE", question, re.IGNORECASE):
        return None
    x = parse_vector_after("x", question)
    w = parse_vector_after("w", question)
    b = parse_number("b", question)
    y = parse_number("y", question)
    if x is None or w is None or b is None or y is None:
        return None
    result = linear_mse_backprop(x, w, b, y)
    return (
        "Problem: Compute linear-layer MSE gradients.\n"
        "Method: yhat=w dot x+b, L=(yhat-y)^2, dL/dw=2(yhat-y)x, dL/db=2(yhat-y).\n"
        f"Calculation: yhat={r4(result['yhat'])}, loss={r4(result['loss'])}, dL/dw={fmt_list(result['grad_w'])}, dL/db={r4(result['grad_b'])}.\n"
        f"Result: yhat={r4(result['yhat'])}, loss={r4(result['loss'])}, dL/dw={fmt_list(result['grad_w'])}, dL/db={r4(result['grad_b'])}.\n"
        "Debug note: The factor 2 appears because this uses L=(yhat-y)^2."
    )


def answer_question(question: str) -> str:
    for solver in [
        answer_sigmoid_bce,
        answer_softmax_ce,
        answer_binary_ce_probability,
        answer_classification_metrics,
        answer_multiclass_accuracy,
        answer_adam,
        answer_gradient_clipping,
        answer_weight_decay_sgd,
        answer_gradient_descent,
        answer_attention,
        answer_semantic_search,
        answer_cosine_similarity,
        answer_cnn_shape,
        answer_matmul_shape,
        answer_matrix_multiply,
        answer_broadcasting_shape,
        answer_normalization,
        answer_dropout,
        answer_momentum,
        answer_activation,
        answer_linear_mse,
    ]:
        response = solver(question)
        if response:
            return response
    return (
        "Problem: No deterministic DL calculator matched this prompt.\n"
        "Method: Route this to the fine-tuned V5 Gemma model for explanation, or add a parser/calculator for this DL operation.\n"
        "Calculation: Not computed.\n"
        "Result: Unsupported by the current DL hybrid calculator.\n"
        "Debug note: Add exact calculators before trusting numeric answers for new DL task families."
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
