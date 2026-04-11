"""Generate dense V5.2 advanced training/eval data.

This extends V5.1 with advanced forecasting, trading indicators, portfolio math,
hypothesis-test interpretation, and vector/kinematics edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from advanced_calculators import (
    ab_test_interpretation,
    ar1_forecast,
    bollinger_bands,
    cohen_d,
    constant_acceleration_2d,
    fmt_list,
    holt_linear_forecast,
    hypothesis_decision,
    linear_trend_forecast,
    macd,
    normal_var,
    portfolio_return_variance,
    projectile_motion,
    relative_velocity,
    rsi,
    seasonal_naive_forecast,
    two_asset_min_variance_weight,
    vector_components,
    vector_magnitude_angle,
    weighted_moving_average,
)
from stats_calculators import r4

SYSTEM = (
    "You are a rigorous ML, DL, statistics, forecasting, trading, and applied math assistant. "
    "Use the right formula, show compact arithmetic, give the final result, and add one diagnostic interpretation."
)

V51_COMBINED = Path("outputs/v51/data/v51_combined_train_chat.jsonl")
OUT_DIR = Path("outputs/v52/data")
REPORT_DIR = Path("outputs/v52/reports")
SUPPLEMENT_OUT = OUT_DIR / "v52_advanced_train_chat.jsonl"
COMBINED_OUT = OUT_DIR / "v52_combined_train_chat.jsonl"
EVAL_OUT = OUT_DIR / "v52_eval_cases.jsonl"
SAMPLE_OUT = Path("samples/v52_advanced_min_sample.jsonl")
REPORT_OUT = REPORT_DIR / "v52_dataset_report.md"


def record(user: str, assistant: str, domain: str, task: str, difficulty: str, expected: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {
            "dataset": "v52_advanced_verified",
            "domain": domain,
            "task_type": task,
            "difficulty": difficulty,
            "expected": expected,
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_forecasting() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    histories = [
        [100, 108, 115, 111, 120],
        [42, 45, 47, 46, 50, 52],
        [200, 212, 218, 230, 235, 240],
        [0.80, 0.83, 0.81, 0.86, 0.88],
        [55, 58, 61, 60, 63, 67],
        [300, 295, 302, 310, 318, 325],
    ]
    weights = [[0.2, 0.3, 0.5], [1, 2, 3], [0.1, 0.2, 0.3, 0.4]]
    for i, values in enumerate(histories, 1):
        w = weights[i % len(weights)]
        out = weighted_moving_average(values, w)
        rows.append(record(
            f"Weighted moving average case {i}: history={values}, weights={w}. Compute the forecast.",
            "Problem: Compute a weighted moving-average forecast.\n"
            "Method: normalize weights and multiply the latest observations by those weights.\n"
            f"Calculation: used_values={fmt_list(out['used_values'])}, normalized_weights={fmt_list(out['normalized_weights'])}, forecast={r4(out['forecast'])}.\n"
            f"Result: weighted moving-average forecast={r4(out['forecast'])}.\n"
            "Diagnostic note: Heavier recent weights react faster but are more noise-sensitive.",
            "Forecasting", "weighted_moving_average", "intermediate", out,
        ))
        trend = linear_trend_forecast(values, 1 + (i % 3))
        rows.append(record(
            f"Linear trend forecast case {i}: history={values}, periods={1 + (i % 3)}. Compute slope, intercept, and forecast.",
            "Problem: Compute a linear trend forecast.\n"
            "Method: regress y on time index, then forecast at the future time index.\n"
            f"Calculation: slope={r4(trend['slope'])}, intercept={r4(trend['intercept'])}, forecast_x={trend['forecast_x']}, forecast={r4(trend['forecast'])}.\n"
            f"Result: trend forecast={r4(trend['forecast'])}.\n"
            "Diagnostic note: Trend forecasts should be checked against seasonality and regime changes.",
            "Forecasting", "linear_trend_forecast", "advanced", trend,
        ))
        holt = holt_linear_forecast(values, 0.35, 0.2, 2)
        rows.append(record(
            f"Holt linear smoothing case {i}: history={values}, alpha=0.35, beta=0.2, periods=2. Compute level, trend, and forecast.",
            "Problem: Compute Holt linear exponential smoothing.\n"
            "Method: recursively update level and trend, then forecast level+horizon*trend.\n"
            f"Calculation: level={r4(holt['level'])}, trend={r4(holt['trend'])}, forecast={r4(holt['forecast'])}.\n"
            f"Result: Holt forecast={r4(holt['forecast'])}.\n"
            "Diagnostic note: Tune alpha and beta on validation error.",
            "Forecasting", "holt_linear_forecast", "expert", holt,
        ))
        ar = ar1_forecast(values, 1)
        rows.append(record(
            f"AR(1) case {i}: values={values}, periods=1. Estimate phi and compute the next forecast.",
            "Problem: Compute a mean-reverting AR(1) forecast.\n"
            "Method: estimate phi from lagged deviations around the mean, then forecast mean+phi(last-mean).\n"
            f"Calculation: mean={r4(ar['mean'])}, phi={r4(ar['phi'])}, forecast={r4(ar['forecast'])}.\n"
            f"Result: AR(1) forecast={r4(ar['forecast'])}.\n"
            "Diagnostic note: Check stationarity before relying on AR forecasts.",
            "Forecasting", "ar1_forecast", "expert", ar,
        ))
    seasonal = [120, 90, 100, 150, 125, 92, 105, 160, 130, 94, 108, 170]
    for horizon in range(1, 5):
        out = seasonal_naive_forecast(seasonal, 4, horizon)
        rows.append(record(
            f"Seasonal naive case {horizon}: values={seasonal}, season_length=4, horizon={horizon}. Compute forecast.",
            "Problem: Compute a seasonal naive forecast.\n"
            "Method: reuse the value from the same seasonal position in the previous cycle.\n"
            f"Calculation: matched_index={out['matched_index']}, forecast={r4(out['forecast'])}.\n"
            f"Result: forecast={r4(out['forecast'])}.\n"
            "Diagnostic note: Seasonal naive is a strong baseline for stable seasonal demand.",
            "Forecasting", "seasonal_naive_forecast", "intermediate", out,
        ))
    return rows


def build_trading_portfolio() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    price_sets = [
        [44, 45, 46, 45, 47, 49, 48, 50, 51, 52, 50, 53, 54, 55, 56],
        [100, 102, 101, 105, 108, 107, 110, 114, 112, 116, 119, 121, 120, 124, 127],
        [210, 208, 211, 215, 214, 218, 220, 219, 225, 228, 226, 231, 235, 237, 240],
        [30, 31, 32, 31, 33, 34, 35, 34, 36, 37, 39, 38, 40, 41, 42],
    ]
    for i, prices in enumerate(price_sets, 1):
        out = rsi(prices, 14)
        rows.append(record(
            f"RSI case {i}: prices={prices}, period=14. Compute RSI.",
            "Problem: Compute RSI.\n"
            "Method: RSI=100-100/(1+RS), where RS=average_gain/average_loss.\n"
            f"Calculation: avg_gain={r4(out['avg_gain'])}, avg_loss={r4(out['avg_loss'])}, RS={r4(out['rs'])}, RSI={r4(out['rsi'])}.\n"
            f"Result: RSI={r4(out['rsi'])}.\n"
            "Diagnostic note: RSI is momentum context, not a standalone trading signal.",
            "Trading", "rsi", "advanced", out,
        ))
        bands = bollinger_bands(prices, 10, 2.0)
        rows.append(record(
            f"Bollinger bands case {i}: prices={prices}, window=10, k=2. Compute lower, middle, and upper bands.",
            "Problem: Compute Bollinger bands.\n"
            "Method: middle=rolling mean, upper=middle+k*std, lower=middle-k*std.\n"
            f"Calculation: middle={r4(bands['middle'])}, std={r4(bands['sample_stddev'])}, lower={r4(bands['lower'])}, upper={r4(bands['upper'])}.\n"
            f"Result: bands=[{r4(bands['lower'])}, {r4(bands['middle'])}, {r4(bands['upper'])}].\n"
            "Diagnostic note: Band touches are volatility context, not automatic reversals.",
            "Trading", "bollinger_bands", "advanced", bands,
        ))
        m = macd(prices, 4, 8, 3)
        rows.append(record(
            f"MACD case {i}: prices={prices}, short=4, long=8, signal=3. Compute MACD, signal, and histogram.",
            "Problem: Compute MACD.\n"
            "Method: MACD=EMA_short-EMA_long, signal=EMA(MACD), histogram=MACD-signal.\n"
            f"Calculation: ema_short={r4(m['ema_short'])}, ema_long={r4(m['ema_long'])}, macd={r4(m['macd'])}, signal={r4(m['signal'])}, histogram={r4(m['histogram'])}.\n"
            f"Result: MACD={r4(m['macd'])}, signal={r4(m['signal'])}, histogram={r4(m['histogram'])}.\n"
            "Diagnostic note: MACD can lag abrupt reversals.",
            "Trading", "macd", "expert", m,
        ))
    portfolios = [
        ([0.6, 0.4], [0.08, 0.12], [[0.04, 0.006], [0.006, 0.09]]),
        ([0.5, 0.3, 0.2], [0.07, 0.10, 0.13], [[0.03, 0.004, 0.002], [0.004, 0.06, 0.005], [0.002, 0.005, 0.10]]),
        ([0.25, 0.35, 0.40], [0.05, 0.09, 0.11], [[0.02, 0.003, 0.001], [0.003, 0.05, 0.004], [0.001, 0.004, 0.08]]),
    ]
    for i, (weights, returns, cov) in enumerate(portfolios, 1):
        out = portfolio_return_variance(weights, returns, cov)
        rows.append(record(
            f"Portfolio case {i}: weights={weights}, expected_returns={returns}, covariance={cov}. Compute expected return, variance, and volatility.",
            "Problem: Compute portfolio return and risk.\n"
            "Method: return=w dot mu, variance=w' Sigma w, volatility=sqrt(variance).\n"
            f"Calculation: expected_return={r4(out['expected_return'])}, variance={r4(out['variance'])}, volatility={r4(out['volatility'])}.\n"
            f"Result: portfolio_return={r4(out['expected_return'])}, volatility={r4(out['volatility'])}.\n"
            "Diagnostic note: Stress-test covariance assumptions.",
            "Portfolio", "portfolio_return_variance", "expert", out,
        ))
    minvar_cases = [(0.18, 0.25, 0.35), (0.12, 0.20, -0.10), (0.30, 0.22, 0.60)]
    for i, (vol_a, vol_b, corr) in enumerate(minvar_cases, 1):
        out = two_asset_min_variance_weight(vol_a, vol_b, corr)
        rows.append(record(
            f"Minimum variance case {i}: vol_a={vol_a}, vol_b={vol_b}, corr={corr}. Compute two-asset min-variance weights.",
            "Problem: Compute unconstrained two-asset minimum-variance weights.\n"
            "Method: w_A=(var_B-cov_AB)/(var_A+var_B-2cov_AB), w_B=1-w_A.\n"
            f"Calculation: covariance={r4(out['covariance'])}, weight_A={r4(out['weight_a'])}, weight_B={r4(out['weight_b'])}.\n"
            f"Result: weights=[{r4(out['weight_a'])}, {r4(out['weight_b'])}].\n"
            "Diagnostic note: Unconstrained solutions can imply short positions.",
            "Portfolio", "two_asset_min_variance", "expert", out,
        ))
    var_cases = [(0.001, 0.02, 1.65, 100000), (-0.0005, 0.015, 2.33, 250000), (0.002, 0.03, 1.96, 50000)]
    for i, (mu, vol, z, value) in enumerate(var_cases, 1):
        out = normal_var(mu, vol, z, value)
        rows.append(record(
            f"Normal VaR case {i}: mean_return={mu}, volatility={vol}, z={z}, portfolio_value={value}. Compute VaR.",
            "Problem: Compute normal value at risk.\n"
            "Method: cutoff_return=mean-z*volatility, VaR=max(0,-cutoff_return*portfolio_value).\n"
            f"Calculation: cutoff_return={r4(out['cutoff_return'])}, VaR={r4(out['var_amount'])}.\n"
            f"Result: VaR={r4(out['var_amount'])}.\n"
            "Diagnostic note: Normal VaR can understate fat-tail losses.",
            "Portfolio", "normal_var", "advanced", out,
        ))
    return rows


def build_interpretation_kinematics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    effect_cases = [
        ([82, 85, 88, 84, 87], [76, 78, 80, 77, 79]),
        ([0.91, 0.88, 0.93, 0.90], [0.84, 0.86, 0.83, 0.85]),
        ([14, 17, 15, 18, 16], [13, 12, 15, 14, 13]),
    ]
    for i, (a, b) in enumerate(effect_cases, 1):
        out = cohen_d(a, b)
        rows.append(record(
            f"Cohen effect size case {i}: group A={a}, group B={b}. Compute Cohen d.",
            "Problem: Compute Cohen's d.\n"
            "Method: d=(mean_A-mean_B)/pooled_stddev.\n"
            f"Calculation: mean_A={r4(out['mean_a'])}, mean_B={r4(out['mean_b'])}, pooled_std={r4(out['pooled_stddev'])}, d={r4(out['cohen_d'])}.\n"
            f"Result: Cohen_d={r4(out['cohen_d'])}.\n"
            "Diagnostic note: Effect size complements p-values by measuring practical magnitude.",
            "Statistics", "cohen_d", "advanced", out,
        ))
    for i, (stat, critical, pval) in enumerate([(2.3, 1.96, None), (1.4, 1.96, 0.16), (-2.8, 2.58, 0.004)], 1):
        out = hypothesis_decision(stat, critical, pval, 0.05)
        rows.append(record(
            f"Hypothesis decision case {i}: statistic={stat}, critical={critical}, p={pval}, alpha=0.05. Interpret reject or fail to reject.",
            "Problem: Interpret a hypothesis-test decision.\n"
            "Method: reject if |statistic|>=critical or p-value<alpha when supplied.\n"
            f"Calculation: statistic={r4(stat)}, critical={r4(critical)}, decision={out['decision']}.\n"
            f"Result: {out['decision']}.\n"
            "Diagnostic note: Failing to reject H0 is not evidence that H0 is true.",
            "Statistics", "hypothesis_decision", "intermediate", out,
        ))
    ab_cases = [(0.14, 0.1769, 1.6128, 0.01), (0.20, 0.226, 2.2, 0.015), (0.08, 0.083, 2.5, 0.01)]
    for i, (control, treatment, z, threshold) in enumerate(ab_cases, 1):
        out = ab_test_interpretation(control, treatment, z, threshold)
        rows.append(record(
            f"A/B test interpretation case {i}: control_rate={control}, treatment_rate={treatment}, z={z}, practical_threshold={threshold}. Should we ship?",
            "Problem: Interpret statistical and practical significance for an A/B test.\n"
            "Method: lift=treatment-control; statistical if |z|>=1.96; practical if lift clears threshold.\n"
            f"Calculation: lift={r4(out['lift'])}, relative_lift={r4(out['relative_lift'])}, statistical={out['statistically_significant']}, practical={out['practically_significant']}.\n"
            f"Result: ship={out['ship']}.\n"
            "Diagnostic note: Do not ship only because p-value is small; check practical impact.",
            "Statistics", "ab_test_interpretation", "advanced", out,
        ))
    vectors = [(20, 15), (-12, 5), (8, -6), (0, 14)]
    for i, (x, y) in enumerate(vectors, 1):
        out = vector_magnitude_angle(x, y)
        rows.append(record(
            f"Vector magnitude case {i}: x={x}, y={y}. Compute magnitude and angle.",
            "Problem: Compute vector magnitude and direction.\n"
            "Method: magnitude=sqrt(x^2+y^2), angle=atan2(y,x).\n"
            f"Calculation: magnitude={r4(out['magnitude'])}, angle={r4(out['angle_degrees'])} degrees.\n"
            f"Result: magnitude={r4(out['magnitude'])}, angle={r4(out['angle_degrees'])} degrees.\n"
            "Diagnostic note: atan2 preserves the quadrant.",
            "Automotive", "vector_magnitude_angle", "basic", out,
        ))
    component_cases = [(30, 40), (25, -20), (12, 90)]
    for i, (mag, angle) in enumerate(component_cases, 1):
        out = vector_components(mag, angle)
        rows.append(record(
            f"Vector components case {i}: magnitude={mag}, angle={angle}. Resolve into x and y components.",
            "Problem: Resolve a vector into components.\n"
            "Method: x=magnitude*cos(theta), y=magnitude*sin(theta).\n"
            f"Calculation: x={r4(out['x'])}, y={r4(out['y'])}.\n"
            f"Result: components=[{r4(out['x'])}, {r4(out['y'])}].\n"
            "Diagnostic note: Convert degrees to radians inside trigonometric functions.",
            "Automotive", "vector_components", "basic", out,
        ))
    rel_cases = [([20, 15], [5, -2]), ([30, 0], [10, 5]), ([12, -4], [-3, 6])]
    for i, (obj, obs) in enumerate(rel_cases, 1):
        out = relative_velocity(obj, obs)
        rows.append(record(
            f"Relative velocity case {i}: object_velocity={obj}, observer_velocity={obs}. Compute relative velocity, magnitude, and angle.",
            "Problem: Compute relative velocity.\n"
            "Method: v_relative=v_object-v_observer.\n"
            f"Calculation: relative={fmt_list(out['relative_velocity'])}, magnitude={r4(out['magnitude'])}, angle={r4(out['angle_degrees'])} degrees.\n"
            f"Result: relative_velocity={fmt_list(out['relative_velocity'])}.\n"
            "Diagnostic note: Define the observer frame before subtracting velocities.",
            "Automotive", "relative_velocity", "intermediate", out,
        ))
    for i, (speed, angle) in enumerate([(30, 40), (22, 35), (18, 55)], 1):
        out = projectile_motion(speed, angle, 9.81)
        rows.append(record(
            f"Projectile motion case {i}: speed={speed}, angle={angle}, gravity=9.81. Compute vx, vy, time of flight, range, and max height.",
            "Problem: Compute ideal projectile motion.\n"
            "Method: vx=vcos(theta), vy=vsin(theta), time=2vy/g, range=vx*time, height=vy^2/(2g).\n"
            f"Calculation: vx={r4(out['vx'])}, vy={r4(out['vy'])}, time={r4(out['time_of_flight'])}, range={r4(out['range'])}, max_height={r4(out['max_height'])}.\n"
            f"Result: range={r4(out['range'])}, max_height={r4(out['max_height'])}.\n"
            "Diagnostic note: This ignores drag and assumes launch and landing at the same height.",
            "Automotive", "projectile_motion", "advanced", out,
        ))
    motion_cases = [([0, 0], [5, 2], [0.5, -0.2], 4), ([10, -3], [2, 6], [-0.1, -9.81], 1.5)]
    for i, (p, v, a, t) in enumerate(motion_cases, 1):
        out = constant_acceleration_2d(p, v, a, t)
        rows.append(record(
            f"Constant acceleration 2D case {i}: position={p}, velocity={v}, acceleration={a}, time={t}. Compute final position and final velocity.",
            "Problem: Compute 2D constant-acceleration motion.\n"
            "Method: p=p0+v0*t+0.5*a*t^2 and v=v0+a*t.\n"
            f"Calculation: final_position={fmt_list(out['final_position'])}, final_velocity={fmt_list(out['final_velocity'])}.\n"
            f"Result: final_position={fmt_list(out['final_position'])}, final_velocity={fmt_list(out['final_velocity'])}.\n"
            "Diagnostic note: Keep coordinate signs consistent.",
            "Automotive", "constant_acceleration_2d", "advanced", out,
        ))
    return rows


def variants(rows: list[dict[str, Any]], copies: int = 12) -> list[dict[str, Any]]:
    prefixes = [
        "",
        "Show the formula and compute carefully: ",
        "I am debugging a model or quantitative workflow. ",
        "Give a compact calculation and one interpretation: ",
        "Act like an ML math tutor and solve: ",
        "Avoid shortcuts and verify the final number: ",
    ]
    out: list[dict[str, Any]] = []
    for repeat in range(copies):
        for row in rows:
            clone = json.loads(json.dumps(row))
            clone["metadata"]["variant"] = repeat
            clone["messages"][1]["content"] = prefixes[repeat % len(prefixes)] + clone["messages"][1]["content"]
            out.append(clone)
    return out


def build_eval_cases(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "weighted_moving_average",
        "linear_trend_forecast",
        "holt_linear_forecast",
        "ar1_forecast",
        "seasonal_naive_forecast",
        "rsi",
        "bollinger_bands",
        "macd",
        "portfolio_return_variance",
        "two_asset_min_variance",
        "normal_var",
        "cohen_d",
        "hypothesis_decision",
        "ab_test_interpretation",
        "vector_magnitude_angle",
        "vector_components",
        "relative_velocity",
        "projectile_motion",
        "constant_acceleration_2d",
    }
    seen: set[str] = set()
    evals = []
    for row in seed_rows:
        task = row["metadata"]["task_type"]
        if task in wanted and task not in seen:
            seen.add(task)
            evals.append({
                "prompt": row["messages"][1]["content"],
                "domain": row["metadata"]["domain"],
                "task_type": task,
                "difficulty": row["metadata"]["difficulty"],
                "expected": row["metadata"]["expected"],
            })
    return evals


def report(supplement: list[dict[str, Any]], combined: list[dict[str, Any]], evals: list[dict[str, Any]]) -> str:
    domains: dict[str, int] = {}
    tasks: dict[str, int] = {}
    for row in supplement:
        domains[row["metadata"]["domain"]] = domains.get(row["metadata"]["domain"], 0) + 1
        tasks[row["metadata"]["task_type"]] = tasks.get(row["metadata"]["task_type"], 0) + 1
    return (
        "# Gemma--TIMMY-MLDL-Maths-v5.2 Dataset Report\n\n"
        f"- V5.2 supplemental rows: {len(supplement)}\n"
        f"- Combined V5/V5.1/V5.2 rows: {len(combined)}\n"
        f"- Eval cases: {len(evals)}\n"
        "- Verification: all numeric answers are produced from deterministic local calculators.\n\n"
        "## Domains\n"
        + "\n".join(f"- {k}: {v}" for k, v in sorted(domains.items()))
        + "\n\n## Tasks\n"
        + "\n".join(f"- {k}: {v}" for k, v in sorted(tasks.items()))
        + "\n"
    )


def main() -> None:
    base = read_jsonl(V51_COMBINED) if V51_COMBINED.exists() else []
    seed_rows = build_forecasting() + build_trading_portfolio() + build_interpretation_kinematics()
    supplement = variants(seed_rows, copies=12)
    combined = base + supplement
    evals = build_eval_cases(seed_rows)

    write_jsonl(SUPPLEMENT_OUT, supplement)
    write_jsonl(COMBINED_OUT, combined)
    write_jsonl(EVAL_OUT, evals)
    write_jsonl(SAMPLE_OUT, supplement[:8])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report(supplement, combined, evals), encoding="utf-8")

    print(f"Wrote supplemental train: {SUPPLEMENT_OUT} ({len(supplement)} rows)")
    print(f"Wrote combined train: {COMBINED_OUT} ({len(combined)} rows)")
    print(f"Wrote eval cases: {EVAL_OUT} ({len(evals)} rows)")
    print(f"Wrote sample: {SAMPLE_OUT} (8 rows)")
    print(f"Wrote report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
