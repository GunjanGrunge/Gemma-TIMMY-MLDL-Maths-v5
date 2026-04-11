"""Advanced deterministic calculators for V5.2 coverage."""

from __future__ import annotations

import math
from statistics import mean

from stats_calculators import covariance_correlation, r4, sample_variance


def fmt_list(values: list[float]) -> str:
    return "[" + ", ".join(r4(v) for v in values) + "]"


def weighted_moving_average(values: list[float], weights: list[float]) -> dict:
    total_weight = sum(weights)
    normalized = [w / total_weight for w in weights]
    used_values = values[-len(weights) :]
    forecast = sum(v * w for v, w in zip(used_values, normalized))
    return {"used_values": used_values, "normalized_weights": normalized, "forecast": forecast}


def linear_trend_forecast(values: list[float], periods: int = 1) -> dict:
    x = list(range(1, len(values) + 1))
    x_bar = mean(x)
    y_bar = mean(values)
    sxx = sum((xi - x_bar) ** 2 for xi in x)
    sxy = sum((xi - x_bar) * (yi - y_bar) for xi, yi in zip(x, values))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    forecast_x = len(values) + periods
    forecast = intercept + slope * forecast_x
    return {"slope": slope, "intercept": intercept, "forecast_x": forecast_x, "forecast": forecast}


def seasonal_naive_forecast(values: list[float], season_length: int, horizon: int = 1) -> dict:
    index = len(values) - season_length + ((horizon - 1) % season_length)
    forecast = values[index]
    return {"matched_index": index, "forecast": forecast}


def holt_linear_forecast(values: list[float], alpha: float, beta: float, periods: int = 1) -> dict:
    level = values[0]
    trend = values[1] - values[0]
    for value in values[1:]:
        previous_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
    forecast = level + periods * trend
    return {"level": level, "trend": trend, "forecast": forecast}


def ar1_forecast(values: list[float], periods: int = 1) -> dict:
    avg = mean(values)
    y_lag = values[:-1]
    y_now = values[1:]
    denominator = sum((v - avg) ** 2 for v in y_lag)
    phi = sum((yt - avg) * (yl - avg) for yl, yt in zip(y_lag, y_now)) / denominator
    forecast = values[-1]
    for _ in range(periods):
        forecast = avg + phi * (forecast - avg)
    return {"mean": avg, "phi": phi, "forecast": forecast}


def ema(values: list[float], period: int) -> dict:
    alpha = 2 / (period + 1)
    value = values[0]
    for price in values[1:]:
        value = alpha * price + (1 - alpha) * value
    return {"alpha": alpha, "ema": value}


def rsi(prices: list[float], period: int = 14) -> dict:
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    recent = changes[-period:]
    gains = [max(change, 0.0) for change in recent]
    losses = [abs(min(change, 0.0)) for change in recent]
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    rs = math.inf if avg_loss == 0 else avg_gain / avg_loss
    value = 100 if avg_loss == 0 else 100 - (100 / (1 + rs))
    return {"avg_gain": avg_gain, "avg_loss": avg_loss, "rs": rs, "rsi": value}


def bollinger_bands(prices: list[float], window: int = 20, k: float = 2.0) -> dict:
    recent = prices[-window:]
    mid = mean(recent)
    std = math.sqrt(sample_variance(recent))
    return {"middle": mid, "upper": mid + k * std, "lower": mid - k * std, "sample_stddev": std}


def macd(prices: list[float], short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> dict:
    short = ema(prices, short_period)["ema"]
    long = ema(prices, long_period)["ema"]
    line = short - long
    # For a compact deterministic calculator, approximate signal from repeated
    # final MACD line history over the last signal window.
    macd_history = []
    for end in range(max(long_period, len(prices) - signal_period + 1), len(prices) + 1):
        short_i = ema(prices[:end], short_period)["ema"]
        long_i = ema(prices[:end], long_period)["ema"]
        macd_history.append(short_i - long_i)
    signal = ema(macd_history, signal_period)["ema"] if macd_history else line
    return {"ema_short": short, "ema_long": long, "macd": line, "signal": signal, "histogram": line - signal}


def portfolio_return_variance(weights: list[float], expected_returns: list[float], covariance_matrix: list[list[float]]) -> dict:
    expected_return = sum(w * r for w, r in zip(weights, expected_returns))
    variance = 0.0
    for i, wi in enumerate(weights):
        for j, wj in enumerate(weights):
            variance += wi * wj * covariance_matrix[i][j]
    return {"expected_return": expected_return, "variance": variance, "volatility": math.sqrt(variance)}


def two_asset_min_variance_weight(vol_a: float, vol_b: float, corr: float) -> dict:
    var_a = vol_a**2
    var_b = vol_b**2
    cov = corr * vol_a * vol_b
    weight_a = (var_b - cov) / (var_a + var_b - 2 * cov)
    return {"weight_a": weight_a, "weight_b": 1 - weight_a, "covariance": cov}


def normal_var(mean_return: float, volatility: float, z_value: float, portfolio_value: float) -> dict:
    cutoff_return = mean_return - z_value * volatility
    var_amount = max(0.0, -cutoff_return * portfolio_value)
    return {"cutoff_return": cutoff_return, "var_amount": var_amount}


def cohen_d(a: list[float], b: list[float]) -> dict:
    mean_a = mean(a)
    mean_b = mean(b)
    var_a = sample_variance(a)
    var_b = sample_variance(b)
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    return {"mean_a": mean_a, "mean_b": mean_b, "pooled_stddev": pooled, "cohen_d": (mean_a - mean_b) / pooled}


def hypothesis_decision(statistic: float, critical_value: float, p_value: float | None = None, alpha: float = 0.05) -> dict:
    reject = abs(statistic) >= critical_value if p_value is None else p_value < alpha
    evidence = "reject H0" if reject else "fail to reject H0"
    return {"reject": reject, "decision": evidence}


def ab_test_interpretation(control_rate: float, treatment_rate: float, z: float, practical_threshold: float = 0.01) -> dict:
    lift = treatment_rate - control_rate
    relative_lift = lift / control_rate
    statistically_significant = abs(z) >= 1.96
    practically_significant = abs(lift) >= practical_threshold
    return {
        "lift": lift,
        "relative_lift": relative_lift,
        "statistically_significant": statistically_significant,
        "practically_significant": practically_significant,
        "ship": statistically_significant and practically_significant and lift > 0,
    }


def vector_magnitude_angle(x: float, y: float) -> dict:
    magnitude = math.hypot(x, y)
    angle_degrees = math.degrees(math.atan2(y, x))
    return {"magnitude": magnitude, "angle_degrees": angle_degrees}


def vector_components(magnitude: float, angle_degrees: float) -> dict:
    radians = math.radians(angle_degrees)
    return {"x": magnitude * math.cos(radians), "y": magnitude * math.sin(radians)}


def vector_add(a: list[float], b: list[float]) -> dict:
    result = [x + y for x, y in zip(a, b)]
    return {"vector": result, **vector_magnitude_angle(result[0], result[1])}


def relative_velocity(object_velocity: list[float], observer_velocity: list[float]) -> dict:
    relative = [x - y for x, y in zip(object_velocity, observer_velocity)]
    return {"relative_velocity": relative, **vector_magnitude_angle(relative[0], relative[1])}


def projectile_motion(speed: float, angle_degrees: float, gravity: float = 9.81) -> dict:
    components = vector_components(speed, angle_degrees)
    time_of_flight = 2 * components["y"] / gravity
    range_distance = components["x"] * time_of_flight
    max_height = components["y"] ** 2 / (2 * gravity)
    return {
        "vx": components["x"],
        "vy": components["y"],
        "time_of_flight": time_of_flight,
        "range": range_distance,
        "max_height": max_height,
    }


def constant_acceleration_2d(position: list[float], velocity: list[float], acceleration: list[float], time: float) -> dict:
    final_position = [
        position[i] + velocity[i] * time + 0.5 * acceleration[i] * time * time
        for i in range(2)
    ]
    final_velocity = [velocity[i] + acceleration[i] * time for i in range(2)]
    return {"final_position": final_position, "final_velocity": final_velocity}
