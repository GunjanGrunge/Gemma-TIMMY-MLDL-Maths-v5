"""V5.2 hybrid assistant with advanced forecasting, trading, portfolio, and kinematics routes."""

from __future__ import annotations

import argparse
import ast
import re

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
    vector_add,
    vector_components,
    vector_magnitude_angle,
    weighted_moving_average,
)
from einstein_dl_hybrid_assistant import parse_int, parse_number, parse_vector_after
from einstein_v51_hybrid_assistant import answer_question as answer_v51_question
from stats_calculators import r4


def answer_missing_info(question: str) -> str | None:
    lower = question.lower()
    if "forgot weights" in lower or ("weights" not in lower and "newer points" in lower):
        return (
            "Problem: Required weights are missing for the weighted forecast.\n"
            "Method: Ask for the missing weight vector before computing a numeric answer.\n"
            "Calculation: Not computed because weights are missing.\n"
            "Result: Need weights=[...] to compute the weighted moving average.\n"
            "Diagnostic note: Do not invent missing inputs for production calculations."
        )
    if "no period" in lower or "lookback" in lower and "not" in lower:
        return (
            "Problem: Required lookback period is missing or ambiguous.\n"
            "Method: Ask for period before computing the indicator.\n"
            "Calculation: Not computed because the lookback period is missing.\n"
            "Result: Need period=<integer> before computing this indicator.\n"
            "Diagnostic note: Defaults can hide assumptions, so missing parameters should be explicit."
        )
    if "covariance is" in lower and "spreadsheet" in lower:
        return (
            "Problem: Portfolio risk requires the covariance matrix.\n"
            "Method: Ask for covariance=[...] before computing variance or volatility.\n"
            "Calculation: Not computed because covariance is missing.\n"
            "Result: Need covariance matrix to compute portfolio risk.\n"
            "Diagnostic note: Portfolio volatility is dominated by covariance assumptions."
        )
    if "forgot launch angle" in lower:
        return (
            "Problem: Projectile range requires the launch angle.\n"
            "Method: Ask for angle before resolving velocity components.\n"
            "Calculation: Not computed because angle is missing.\n"
            "Result: Need angle=<degrees> before computing projectile range.\n"
            "Diagnostic note: Speed alone is insufficient for 2D projectile motion."
        )
    if "z is missing" in lower:
        return (
            "Problem: A/B test rollout decision requires statistical evidence such as z or p-value.\n"
            "Method: Ask for z or p-value before deciding statistical significance.\n"
            "Calculation: Not computed because z is missing.\n"
            "Result: Need z=<statistic> or p=<p_value> before deciding rollout.\n"
            "Diagnostic note: Practical lift alone does not establish statistical significance."
        )
    return None


def parse_matrix_after(label: str, question: str) -> list[list[float]] | None:
    match = re.search(rf"{re.escape(label)}\s*=\s*(\[\[.*?\]\])", question, re.IGNORECASE)
    if not match:
        return None
    return [[float(x) for x in row] for row in ast.literal_eval(match.group(1))]


def answer_advanced_forecasting(question: str) -> str | None:
    values = parse_vector_after("values", question) or parse_vector_after("history", question)
    if re.search(r"weighted moving average|WMA|weighted forecast|newer data", question, re.IGNORECASE) or (
        values is not None and parse_vector_after("weights", question) is not None and re.search(r"forecast|predict|next", question, re.IGNORECASE)
    ):
        weights = parse_vector_after("weights", question)
        if values is None or weights is None:
            return None
        out = weighted_moving_average(values, weights)
        return (
            "Problem: Compute a weighted moving-average forecast.\n"
            "Method: normalize weights and multiply them by the latest observations.\n"
            f"Calculation: used_values={fmt_list(out['used_values'])}, normalized_weights={fmt_list(out['normalized_weights'])}, forecast={r4(out['forecast'])}.\n"
            f"Result: weighted moving-average forecast={r4(out['forecast'])}.\n"
            "Diagnostic note: Heavier recent weights react faster but increase noise sensitivity."
        )
    if re.search(r"linear trend|trend forecast|fit a trend", question, re.IGNORECASE):
        periods = parse_int("periods", question, default=1)
        if values is None:
            return None
        out = linear_trend_forecast(values, int(periods))
        return (
            "Problem: Compute a linear trend forecast.\n"
            "Method: regress y on time index and forecast at the future time point.\n"
            f"Calculation: slope={r4(out['slope'])}, intercept={r4(out['intercept'])}, forecast_x={out['forecast_x']}, forecast={r4(out['forecast'])}.\n"
            f"Result: trend forecast={r4(out['forecast'])}.\n"
            "Diagnostic note: Linear trend is fragile when seasonality or regime shifts dominate."
        )
    if re.search(r"seasonal naive|same season|last season", question, re.IGNORECASE):
        season_length = parse_int("season_length", question)
        horizon = parse_int("horizon", question, default=1)
        if values is None or season_length is None:
            return None
        out = seasonal_naive_forecast(values, int(season_length), int(horizon))
        return (
            "Problem: Compute a seasonal naive forecast.\n"
            "Method: reuse the observation from the same seasonal position in the previous cycle.\n"
            f"Calculation: matched_index={out['matched_index']}, forecast={r4(out['forecast'])}.\n"
            f"Result: seasonal naive forecast={r4(out['forecast'])}.\n"
            "Diagnostic note: This is a strong baseline for stable seasonal series."
        )
    if re.search(r"Holt|linear smoothing|level and trend", question, re.IGNORECASE):
        alpha = parse_number("alpha", question)
        beta = parse_number("beta", question)
        periods = parse_int("periods", question, default=1)
        if values is None or alpha is None or beta is None:
            return None
        out = holt_linear_forecast(values, float(alpha), float(beta), int(periods))
        return (
            "Problem: Compute Holt linear exponential smoothing.\n"
            "Method: update level and trend recursively, then forecast level+horizon*trend.\n"
            f"Calculation: level={r4(out['level'])}, trend={r4(out['trend'])}, forecast={r4(out['forecast'])}.\n"
            f"Result: Holt forecast={r4(out['forecast'])}.\n"
            "Diagnostic note: Tune alpha and beta on validation error, not training fit."
        )
    if re.search(r"AR\(1\)|AR1|autoregressive|mean reverting", question, re.IGNORECASE):
        periods = parse_int("periods", question, default=1)
        if values is None:
            return None
        out = ar1_forecast(values, int(periods))
        return (
            "Problem: Compute a mean-reverting AR(1) forecast.\n"
            "Method: estimate phi from lagged deviations around the mean, then iterate forecast=mean+phi(last-mean).\n"
            f"Calculation: mean={r4(out['mean'])}, phi={r4(out['phi'])}, forecast={r4(out['forecast'])}.\n"
            f"Result: AR(1) forecast={r4(out['forecast'])}.\n"
            "Diagnostic note: Check stationarity before using AR-style forecasts."
        )
    return None


def answer_trading_indicators(question: str) -> str | None:
    prices = parse_vector_after("prices", question)
    if prices is None:
        return None
    if re.search(r"RSI|momentum oscillator|overbought|oversold", question, re.IGNORECASE):
        period = parse_int("period", question, default=14)
        out = rsi(prices, int(period))
        return (
            "Problem: Compute RSI.\n"
            "Method: RSI=100-100/(1+RS), where RS=average_gain/average_loss over the lookback period.\n"
            f"Calculation: avg_gain={r4(out['avg_gain'])}, avg_loss={r4(out['avg_loss'])}, RS={r4(out['rs'])}, RSI={r4(out['rsi'])}.\n"
            f"Result: RSI={r4(out['rsi'])}.\n"
            "Diagnostic note: RSI is a momentum indicator, not a standalone buy/sell rule."
        )
    if re.search(r"Bollinger|volatility band|upper.*lower.*band", question, re.IGNORECASE):
        window = parse_int("window", question, default=20)
        k = parse_number("k", question, default=2.0)
        out = bollinger_bands(prices, int(window), float(k))
        return (
            "Problem: Compute Bollinger bands.\n"
            "Method: middle=rolling mean, upper=mean+k*std, lower=mean-k*std.\n"
            f"Calculation: middle={r4(out['middle'])}, std={r4(out['sample_stddev'])}, upper={r4(out['upper'])}, lower={r4(out['lower'])}.\n"
            f"Result: bands=[{r4(out['lower'])}, {r4(out['middle'])}, {r4(out['upper'])}].\n"
            "Diagnostic note: Band touches are volatility context, not proof of reversal."
        )
    if re.search(r"MACD|EMA.*signal|moving average convergence", question, re.IGNORECASE):
        short = parse_int("short", question, default=12)
        long = parse_int("long", question, default=26)
        signal = parse_int("signal", question, default=9)
        out = macd(prices, int(short), int(long), int(signal))
        return (
            "Problem: Compute MACD line, signal, and histogram.\n"
            "Method: MACD=EMA_short-EMA_long; signal=EMA(MACD); histogram=MACD-signal.\n"
            f"Calculation: ema_short={r4(out['ema_short'])}, ema_long={r4(out['ema_long'])}, macd={r4(out['macd'])}, signal={r4(out['signal'])}, histogram={r4(out['histogram'])}.\n"
            f"Result: MACD={r4(out['macd'])}, signal={r4(out['signal'])}, histogram={r4(out['histogram'])}.\n"
            "Diagnostic note: MACD is trend-following and can lag sharp reversals."
        )
    return None


def answer_portfolio(question: str) -> str | None:
    if re.search(r"\bVaR\b|value at risk|worst.*loss|tail loss", question, re.IGNORECASE) or (
        parse_number("mean_return", question) is not None
        and parse_number("volatility", question) is not None
        and parse_number("portfolio_value", question) is not None
    ):
        mean_return = parse_number("mean_return", question)
        volatility = parse_number("volatility", question)
        z = parse_number("z", question, default=1.65)
        portfolio_value = parse_number("portfolio_value", question)
        if None in [mean_return, volatility, portfolio_value]:
            return None
        out = normal_var(float(mean_return), float(volatility), float(z), float(portfolio_value))
        return (
            "Problem: Compute normal VaR.\n"
            "Method: cutoff_return=mean-z*volatility; VaR=max(0,-cutoff_return*portfolio_value).\n"
            f"Calculation: cutoff_return={r4(out['cutoff_return'])}, VaR={r4(out['var_amount'])}.\n"
            f"Result: VaR={r4(out['var_amount'])}.\n"
            "Diagnostic note: Normal VaR can understate tail risk for fat-tailed returns."
        )
    if re.search(r"portfolio", question, re.IGNORECASE) or (
        parse_vector_after("weights", question) is not None
        and parse_vector_after("expected_returns", question) is not None
        and parse_matrix_after("covariance", question) is not None
    ):
        weights = parse_vector_after("weights", question)
        returns = parse_vector_after("expected_returns", question)
        covariance = parse_matrix_after("covariance", question)
        if weights is None or returns is None or covariance is None:
            return None
        out = portfolio_return_variance(weights, returns, covariance)
        return (
            "Problem: Compute portfolio expected return and variance.\n"
            "Method: expected return=w dot mu, variance=w' Sigma w, volatility=sqrt(variance).\n"
            f"Calculation: expected_return={r4(out['expected_return'])}, variance={r4(out['variance'])}, volatility={r4(out['volatility'])}.\n"
            f"Result: portfolio_return={r4(out['expected_return'])}, volatility={r4(out['volatility'])}.\n"
            "Diagnostic note: Covariance estimates can dominate portfolio risk, so stress-test them."
        )
    if re.search(r"minimum variance|min variance|least risky two-asset", question, re.IGNORECASE):
        vol_a = parse_number("vol_a", question)
        vol_b = parse_number("vol_b", question)
        corr = parse_number("corr", question)
        if None in [vol_a, vol_b, corr]:
            return None
        out = two_asset_min_variance_weight(float(vol_a), float(vol_b), float(corr))
        return (
            "Problem: Compute the two-asset minimum-variance weights.\n"
            "Method: w_A=(var_B-cov_AB)/(var_A+var_B-2cov_AB), w_B=1-w_A.\n"
            f"Calculation: covariance={r4(out['covariance'])}, weight_A={r4(out['weight_a'])}, weight_B={r4(out['weight_b'])}.\n"
            f"Result: weights=[{r4(out['weight_a'])}, {r4(out['weight_b'])}].\n"
            "Diagnostic note: Unconstrained minimum-variance weights can become short positions."
        )
    return None


def answer_hypothesis_interpretation(question: str) -> str | None:
    if re.search(r"Cohen|effect size|practical magnitude", question, re.IGNORECASE):
        a = parse_vector_after("group A", question)
        b = parse_vector_after("group B", question)
        if a is None or b is None:
            return None
        out = cohen_d(a, b)
        return (
            "Problem: Compute Cohen's d effect size.\n"
            "Method: d=(mean_A-mean_B)/pooled_stddev.\n"
            f"Calculation: mean_A={r4(out['mean_a'])}, mean_B={r4(out['mean_b'])}, pooled_std={r4(out['pooled_stddev'])}, d={r4(out['cohen_d'])}.\n"
            f"Result: Cohen_d={r4(out['cohen_d'])}.\n"
            "Diagnostic note: Statistical significance and practical effect size answer different questions."
        )
    if re.search(r"hypothesis decision|reject|fail to reject|null", question, re.IGNORECASE):
        statistic = parse_number("statistic", question)
        critical = parse_number("critical", question)
        p_value = parse_number("p", question)
        alpha = parse_number("alpha", question, default=0.05)
        if statistic is None or critical is None:
            return None
        out = hypothesis_decision(float(statistic), float(critical), p_value, float(alpha))
        return (
            "Problem: Interpret a hypothesis-test decision.\n"
            "Method: reject if |statistic| >= critical, or if p-value < alpha when p-value is supplied.\n"
            f"Calculation: statistic={r4(statistic)}, critical={r4(critical)}, decision={out['decision']}.\n"
            f"Result: {out['decision']}.\n"
            "Diagnostic note: Failing to reject H0 is not proof that H0 is true."
        )
    if re.search(r"AB test|A/B test|ship|rollout|experiment", question, re.IGNORECASE) or (
        parse_number("control_rate", question) is not None
        and parse_number("treatment_rate", question) is not None
        and parse_number("z", question) is not None
    ):
        control = parse_number("control_rate", question)
        treatment = parse_number("treatment_rate", question)
        z = parse_number("z", question)
        threshold = parse_number("practical_threshold", question, default=0.01)
        if None in [control, treatment, z]:
            return None
        out = ab_test_interpretation(float(control), float(treatment), float(z), float(threshold))
        return (
            "Problem: Interpret A/B test statistical and practical significance.\n"
            "Method: lift=treatment-control; statistically significant if |z|>=1.96; practically significant if lift clears the threshold.\n"
            f"Calculation: lift={r4(out['lift'])}, relative_lift={r4(out['relative_lift'])}, statistical={out['statistically_significant']}, practical={out['practically_significant']}.\n"
            f"Result: ship={out['ship']}.\n"
            "Diagnostic note: A statistically significant tiny lift may still be a bad product decision."
        )
    return None


def answer_vector_kinematics(question: str) -> str | None:
    if re.search(r"magnitude|angle|speed and direction|speed.*direction", question, re.IGNORECASE):
        x = parse_number("x", question)
        y = parse_number("y", question)
        if x is not None and y is not None:
            out = vector_magnitude_angle(float(x), float(y))
            return f"Problem: Compute 2D vector magnitude and angle.\nMethod: magnitude=sqrt(x^2+y^2), angle=atan2(y,x).\nCalculation: magnitude={r4(out['magnitude'])}, angle={r4(out['angle_degrees'])} degrees.\nResult: magnitude={r4(out['magnitude'])}, angle={r4(out['angle_degrees'])} degrees.\nDiagnostic note: atan2 preserves quadrant information."
    if re.search(r"components|resolve.*vector", question, re.IGNORECASE):
        magnitude = parse_number("magnitude", question)
        angle = parse_number("angle", question)
        if magnitude is None or angle is None:
            return None
        out = vector_components(float(magnitude), float(angle))
        return f"Problem: Resolve a vector into components.\nMethod: x=magnitude*cos(theta), y=magnitude*sin(theta).\nCalculation: x={r4(out['x'])}, y={r4(out['y'])}.\nResult: components=[{r4(out['x'])}, {r4(out['y'])}].\nDiagnostic note: Convert degrees to radians inside trig functions."
    if re.search(r"relative velocity|relative motion|seen by observer", question, re.IGNORECASE) or (
        parse_vector_after("object_velocity", question) is not None and parse_vector_after("observer_velocity", question) is not None
    ):
        obj = parse_vector_after("object_velocity", question)
        obs = parse_vector_after("observer_velocity", question)
        if obj is None or obs is None:
            return None
        out = relative_velocity(obj, obs)
        return f"Problem: Compute relative velocity.\nMethod: v_relative=v_object-v_observer.\nCalculation: relative={fmt_list(out['relative_velocity'])}, magnitude={r4(out['magnitude'])}, angle={r4(out['angle_degrees'])} degrees.\nResult: relative_velocity={fmt_list(out['relative_velocity'])}.\nDiagnostic note: Define the observer frame before subtracting velocities."
    if re.search(r"projectile|launch|trajectory", question, re.IGNORECASE):
        speed = parse_number("speed", question)
        angle = parse_number("angle", question)
        gravity = parse_number("gravity", question, default=9.81)
        if speed is None or angle is None:
            return None
        out = projectile_motion(float(speed), float(angle), float(gravity))
        return f"Problem: Compute ideal projectile motion.\nMethod: vx=vcos(theta), vy=vsin(theta), time=2vy/g, range=vx*time, max_height=vy^2/(2g).\nCalculation: vx={r4(out['vx'])}, vy={r4(out['vy'])}, time={r4(out['time_of_flight'])}, range={r4(out['range'])}, max_height={r4(out['max_height'])}.\nResult: range={r4(out['range'])}, max_height={r4(out['max_height'])}.\nDiagnostic note: This ignores drag and assumes launch/landing at the same height."
    if re.search(r"constant acceleration 2d|where is it and how fast", question, re.IGNORECASE) or (
        parse_vector_after("position", question) is not None
        and parse_vector_after("velocity", question) is not None
        and parse_vector_after("acceleration", question) is not None
        and parse_number("time", question) is not None
    ):
        position = parse_vector_after("position", question)
        velocity = parse_vector_after("velocity", question)
        acceleration = parse_vector_after("acceleration", question)
        time = parse_number("time", question)
        if None in [position, velocity, acceleration, time]:
            return None
        out = constant_acceleration_2d(position, velocity, acceleration, float(time))
        return f"Problem: Compute 2D constant-acceleration motion.\nMethod: p=p0+v0*t+0.5*a*t^2 and v=v0+a*t.\nCalculation: final_position={fmt_list(out['final_position'])}, final_velocity={fmt_list(out['final_velocity'])}.\nResult: final_position={fmt_list(out['final_position'])}, final_velocity={fmt_list(out['final_velocity'])}.\nDiagnostic note: Keep coordinate signs consistent."
    return None


def answer_question(question: str) -> str:
    missing = answer_missing_info(question)
    if missing:
        return missing
    solvers = [
        answer_advanced_forecasting,
        answer_trading_indicators,
        answer_portfolio,
        answer_hypothesis_interpretation,
        answer_vector_kinematics,
    ]
    responses = []
    for solver in solvers:
        response = solver(question)
        if response:
            responses.append(response)
    if len(responses) > 1:
        return "\n\n".join(f"Part {idx}: {response}" for idx, response in enumerate(responses, 1))
    if responses:
        return responses[0]
    return answer_v51_question(question)


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
