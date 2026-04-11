"""Deterministic statistics, ML, forecasting, and quant calculators for V5.1."""

from __future__ import annotations

import math
from statistics import mean, median


def r4(value: float) -> str:
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(r4(v) for v in values) + "]"


def sample_variance(values: list[float]) -> float:
    avg = mean(values)
    return sum((x - avg) ** 2 for x in values) / (len(values) - 1)


def population_variance(values: list[float]) -> float:
    avg = mean(values)
    return sum((x - avg) ** 2 for x in values) / len(values)


def descriptive_stats(values: list[float]) -> dict:
    sample_var = sample_variance(values)
    pop_var = population_variance(values)
    return {
        "mean": mean(values),
        "median": median(values),
        "sample_variance": sample_var,
        "sample_stddev": math.sqrt(sample_var),
        "population_variance": pop_var,
        "population_stddev": math.sqrt(pop_var),
    }


def z_scores(values: list[float]) -> dict:
    avg = mean(values)
    stdev = math.sqrt(sample_variance(values))
    return {"mean": avg, "sample_stddev": stdev, "z_scores": [(x - avg) / stdev for x in values]}


def covariance_correlation(x: list[float], y: list[float]) -> dict:
    x_mean = mean(x)
    y_mean = mean(y)
    cov = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / (len(x) - 1)
    sx = math.sqrt(sample_variance(x))
    sy = math.sqrt(sample_variance(y))
    return {"covariance": cov, "correlation": cov / (sx * sy), "x_stddev": sx, "y_stddev": sy}


def bayes_binary(prior: float, likelihood_pos: float, likelihood_neg: float) -> dict:
    evidence = likelihood_pos * prior + likelihood_neg * (1 - prior)
    posterior = likelihood_pos * prior / evidence
    return {"evidence": evidence, "posterior": posterior}


def binomial_pmf(n: int, k: int, p: float) -> dict:
    probability = math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))
    return {"probability": probability}


def poisson_pmf(lam: float, k: int) -> dict:
    probability = math.exp(-lam) * (lam**k) / math.factorial(k)
    return {"probability": probability}


def normal_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def normal_probability_between(mu: float, sigma: float, lower: float, upper: float) -> dict:
    z_lower = (lower - mu) / sigma
    z_upper = (upper - mu) / sigma
    probability = normal_cdf(z_upper) - normal_cdf(z_lower)
    return {"z_lower": z_lower, "z_upper": z_upper, "probability": probability}


def one_sample_t_test(values: list[float], mu0: float) -> dict:
    avg = mean(values)
    stdev = math.sqrt(sample_variance(values))
    se = stdev / math.sqrt(len(values))
    return {"mean": avg, "sample_stddev": stdev, "se": se, "t": (avg - mu0) / se, "df": len(values) - 1}


def welch_t_test(a: list[float], b: list[float]) -> dict:
    mean_a = mean(a)
    mean_b = mean(b)
    var_a = sample_variance(a)
    var_b = sample_variance(b)
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    numerator = (var_a / len(a) + var_b / len(b)) ** 2
    denominator = ((var_a / len(a)) ** 2) / (len(a) - 1) + ((var_b / len(b)) ** 2) / (len(b) - 1)
    return {"mean_a": mean_a, "mean_b": mean_b, "se": se, "t": (mean_a - mean_b) / se, "df": numerator / denominator}


def paired_t_test(before: list[float], after: list[float]) -> dict:
    differences = [a - b for a, b in zip(after, before)]
    result = one_sample_t_test(differences, 0.0)
    result["differences"] = differences
    return result


def two_proportion_z_test(n1: int, x1: int, n2: int, x2: int) -> dict:
    p1 = x1 / n1
    p2 = x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return {"p1": p1, "p2": p2, "pooled": pooled, "se": se, "z": (p2 - p1) / se}


def chi_square_2x2(a: int, b: int, c: int, d: int) -> dict:
    rows = [a + b, c + d]
    cols = [a + c, b + d]
    total = sum(rows)
    observed = [[a, b], [c, d]]
    expected = [[rows[i] * cols[j] / total for j in range(2)] for i in range(2)]
    chi2 = sum((observed[i][j] - expected[i][j]) ** 2 / expected[i][j] for i in range(2) for j in range(2))
    return {"expected": expected, "chi2": chi2, "df": 1}


def one_way_anova(groups: list[list[float]]) -> dict:
    all_values = [value for group in groups for value in group]
    grand_mean = mean(all_values)
    ss_between = sum(len(group) * (mean(group) - grand_mean) ** 2 for group in groups)
    ss_within = sum(sum((value - mean(group)) ** 2 for value in group) for group in groups)
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    return {"ss_between": ss_between, "ss_within": ss_within, "f": ms_between / ms_within, "df_between": df_between, "df_within": df_within}


def mean_confidence_interval(values: list[float], critical: float = 1.96) -> dict:
    avg = mean(values)
    se = math.sqrt(sample_variance(values)) / math.sqrt(len(values))
    margin = critical * se
    return {"mean": avg, "se": se, "lower": avg - margin, "upper": avg + margin}


def proportion_confidence_interval(successes: int, n: int, critical: float = 1.96) -> dict:
    p_hat = successes / n
    se = math.sqrt(p_hat * (1 - p_hat) / n)
    margin = critical * se
    return {"p_hat": p_hat, "se": se, "lower": p_hat - margin, "upper": p_hat + margin}


def simple_linear_regression(x: list[float], y: list[float]) -> dict:
    x_bar = mean(x)
    y_bar = mean(y)
    sxx = sum((xi - x_bar) ** 2 for xi in x)
    sxy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    y_hat = [intercept + slope * xi for xi in x]
    ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, y_hat))
    ss_tot = sum((yi - y_bar) ** 2 for yi in y)
    return {"slope": slope, "intercept": intercept, "predictions": y_hat, "r2": 1 - ss_res / ss_tot}


def regression_errors(actuals: list[float], forecasts: list[float]) -> dict:
    errors = [a - f for a, f in zip(actuals, forecasts)]
    abs_errors = [abs(e) for e in errors]
    squared = [e * e for e in errors]
    ape = [abs((a - f) / a) for a, f in zip(actuals, forecasts)]
    return {"mae": mean(abs_errors), "rmse": math.sqrt(mean(squared)), "mape": mean(ape), "errors": errors}


def logistic_l2_gradient(x: list[float], w: list[float], y: int, lam: float) -> dict:
    z = sum(xi * wi for xi, wi in zip(x, w))
    p = 1 / (1 + math.exp(-z))
    gradient = [(p - y) * xi + lam * wi for xi, wi in zip(x, w)]
    return {"z": z, "p": p, "gradient": gradient}


def svm_hinge_loss(y: int, score: float) -> dict:
    margin = y * score
    loss = max(0.0, 1 - margin)
    grad_direction = "no update" if margin >= 1 else "increase y*score"
    return {"margin": margin, "loss": loss, "gradient_direction": grad_direction}


def kmeans_centroid(points: list[list[float]]) -> dict:
    dims = len(points[0])
    centroid = [mean([point[d] for point in points]) for d in range(dims)]
    return {"centroid": centroid}


def moving_average(values: list[float], window: int) -> dict:
    forecast = mean(values[-window:])
    return {"forecast": forecast}


def exponential_smoothing(values: list[float], alpha: float) -> dict:
    level = values[0]
    for value in values[1:]:
        level = alpha * value + (1 - alpha) * level
    return {"forecast": level}


def returns_volatility_sharpe(prices: list[float], risk_free: float = 0.0) -> dict:
    returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
    avg = mean(returns)
    vol = math.sqrt(sample_variance(returns))
    return {"returns": returns, "mean_return": avg, "volatility": vol, "sharpe": (avg - risk_free) / vol}


def max_drawdown(prices: list[float]) -> dict:
    peak = prices[0]
    max_dd = 0.0
    for price in prices:
        peak = max(peak, price)
        drawdown = (price - peak) / peak
        max_dd = min(max_dd, drawdown)
    return {"max_drawdown": max_dd}


def beta(asset_returns: list[float], market_returns: list[float]) -> dict:
    cov_corr = covariance_correlation(asset_returns, market_returns)
    market_var = sample_variance(market_returns)
    return {"beta": cov_corr["covariance"] / market_var, "covariance": cov_corr["covariance"], "market_variance": market_var}


def acceleration(v0: float, v1: float, seconds: float) -> dict:
    return {"acceleration": (v1 - v0) / seconds}


def stopping_distance(speed: float, reaction_time: float, deceleration: float) -> dict:
    reaction_distance = speed * reaction_time
    braking_distance = speed * speed / (2 * deceleration)
    return {"reaction_distance": reaction_distance, "braking_distance": braking_distance, "total_distance": reaction_distance + braking_distance}


def entropy(probabilities: list[float]) -> dict:
    h = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return {"entropy_bits": h}


def kl_divergence(p: list[float], q: list[float]) -> dict:
    kl = sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0)
    return {"kl_bits": kl}
