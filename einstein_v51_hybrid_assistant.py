"""V5.1 calculator-backed ML/DL/statistics/quant assistant.

The LoRA adapter provides instruction-following style. This hybrid router owns
exact arithmetic for statistics, classical ML, forecasting, trading, kinematics,
semantic-search, and DL calculator families.
"""

from __future__ import annotations

import argparse
import re

from einstein_dl_hybrid_assistant import (
    answer_question as answer_dl_question,
    parse_int,
    parse_number,
    parse_vector_after,
)
from stats_calculators import (
    bayes_binary,
    beta,
    binomial_pmf,
    chi_square_2x2,
    covariance_correlation,
    descriptive_stats,
    entropy,
    exponential_smoothing,
    fmt_list,
    kmeans_centroid,
    kl_divergence,
    logistic_l2_gradient,
    max_drawdown,
    mean_confidence_interval,
    moving_average,
    normal_probability_between,
    one_sample_t_test,
    one_way_anova,
    paired_t_test,
    poisson_pmf,
    proportion_confidence_interval,
    r4,
    regression_errors,
    returns_volatility_sharpe,
    simple_linear_regression,
    stopping_distance,
    svm_hinge_loss,
    two_proportion_z_test,
    welch_t_test,
    z_scores,
)


def parse_values(question: str) -> list[float] | None:
    for label in ["values", "sample", "history", "prices", "actuals"]:
        values = parse_vector_after(label, question)
        if values is not None:
            return values
    return None


def answer_descriptive(question: str) -> str | None:
    if not re.search(r"mean|std|standard deviation|variance|z-score|z score", question, re.IGNORECASE):
        return None
    values = parse_values(question)
    if values is None:
        return None
    stats = descriptive_stats(values)
    z = z_scores(values)
    return (
        "Problem: Compute descriptive statistics and z-scores.\n"
        "Method: mean=sum(x)/n, sample_var=sum((x-mean)^2)/(n-1), pop_var=sum((x-mean)^2)/n, z=(x-mean)/sample_std.\n"
        f"Calculation: mean={r4(stats['mean'])}, median={r4(stats['median'])}, sample_var={r4(stats['sample_variance'])}, "
        f"sample_std={r4(stats['sample_stddev'])}, population_std={r4(stats['population_stddev'])}, z={fmt_list(z['z_scores'])}.\n"
        f"Result: mean={r4(stats['mean'])}; sample_std={r4(stats['sample_stddev'])}; z-scores={fmt_list(z['z_scores'])}.\n"
        "Diagnostic note: Use sample standard deviation when estimating spread from observed data."
    )


def answer_one_sample_t(question: str) -> str | None:
    if not re.search(r"one.?sample|t-test|t test", question, re.IGNORECASE):
        return None
    values = parse_values(question)
    mu0 = parse_number("null mean", question)
    if mu0 is None:
        mu0 = parse_number("mu0", question)
    if values is None or mu0 is None:
        return None
    out = one_sample_t_test(values, mu0)
    return (
        "Problem: Compute a one-sample t statistic.\n"
        "Method: t=(sample_mean-null_mean)/(sample_std/sqrt(n)), df=n-1.\n"
        f"Calculation: mean={r4(out['mean'])}, sample_std={r4(out['sample_stddev'])}, SE={r4(out['se'])}, t={r4(out['t'])}, df={out['df']}.\n"
        f"Result: t={r4(out['t'])} with df={out['df']}.\n"
        "Diagnostic note: Check sampling assumptions before interpreting the test statistic."
    )


def answer_welch(question: str) -> str | None:
    if not re.search(r"Welch|two means|unequal", question, re.IGNORECASE):
        return None
    a = parse_vector_after("group A", question)
    b = parse_vector_after("group B", question)
    if a is None:
        a = parse_vector_after("a", question)
    if b is None:
        b = parse_vector_after("b", question)
    if a is None or b is None:
        return None
    out = welch_t_test(a, b)
    return (
        "Problem: Compare two means with Welch's unequal-variance t-test.\n"
        "Method: t=(mean_A-mean_B)/sqrt(var_A/n_A+var_B/n_B), df by Welch-Satterthwaite.\n"
        f"Calculation: mean_A={r4(out['mean_a'])}, mean_B={r4(out['mean_b'])}, SE={r4(out['se'])}, t={r4(out['t'])}, df={r4(out['df'])}.\n"
        f"Result: difference={r4(out['mean_a']-out['mean_b'])}; t={r4(out['t'])}; df={r4(out['df'])}.\n"
        "Diagnostic note: Welch is the safer default when variances or sample sizes differ."
    )


def answer_paired_t(question: str) -> str | None:
    if not re.search(r"paired", question, re.IGNORECASE):
        return None
    before = parse_vector_after("before", question)
    after = parse_vector_after("after", question)
    if before is None or after is None:
        return None
    out = paired_t_test(before, after)
    return (
        "Problem: Compute a paired t-test from matched observations.\n"
        "Method: form after-before differences and run a one-sample t-test against 0.\n"
        f"Calculation: differences={fmt_list(out['differences'])}, mean_diff={r4(out['mean'])}, SE={r4(out['se'])}, t={r4(out['t'])}, df={out['df']}.\n"
        f"Result: paired t={r4(out['t'])} with df={out['df']}.\n"
        "Diagnostic note: Do not treat matched observations as independent groups."
    )


def answer_two_prop(question: str) -> str | None:
    if not re.search(r"proportion|conversion|rate|z-test|z test", question, re.IGNORECASE):
        return None
    n1 = parse_int("n1", question)
    x1 = parse_int("x1", question)
    n2 = parse_int("n2", question)
    x2 = parse_int("x2", question)
    if None in [n1, x1, n2, x2]:
        return None
    out = two_proportion_z_test(int(n1), int(x1), int(n2), int(x2))
    return (
        "Problem: Compare two proportions with a pooled z-test.\n"
        "Method: pooled=(x1+x2)/(n1+n2), SE=sqrt(pooled(1-pooled)(1/n1+1/n2)), z=(p2-p1)/SE.\n"
        f"Calculation: p1={r4(out['p1'])}, p2={r4(out['p2'])}, pooled={r4(out['pooled'])}, SE={r4(out['se'])}, z={r4(out['z'])}.\n"
        f"Result: lift={r4(out['p2']-out['p1'])}; z={r4(out['z'])}.\n"
        "Diagnostic note: For tiny counts, prefer an exact test."
    )


def answer_probability(question: str) -> str | None:
    if re.search(r"Bayes|posterior", question, re.IGNORECASE):
        prior = parse_number("prior", question)
        like_pos = parse_number("P(signal|A)", question) or parse_number("likelihood_pos", question)
        like_neg = parse_number("P(signal|not A)", question) or parse_number("likelihood_neg", question)
        if None in [prior, like_pos, like_neg]:
            return None
        out = bayes_binary(float(prior), float(like_pos), float(like_neg))
        return (
            "Problem: Compute a Bayesian posterior.\n"
            "Method: evidence=P(signal|A)P(A)+P(signal|not A)P(not A), posterior=P(signal|A)P(A)/evidence.\n"
            f"Calculation: evidence={r4(out['evidence'])}, posterior={r4(out['posterior'])}.\n"
            f"Result: posterior={r4(out['posterior'])}.\n"
            "Diagnostic note: Base rate strongly affects posterior probability."
        )
    if re.search(r"Binomial", question, re.IGNORECASE):
        n = parse_int("n", question)
        k = parse_int("k", question)
        p = parse_number("p", question)
        if None in [n, k, p]:
            return None
        out = binomial_pmf(int(n), int(k), float(p))
        return f"Problem: Compute binomial probability.\nMethod: P(X=k)=C(n,k)p^k(1-p)^(n-k).\nCalculation: P={r4(out['probability'])}.\nResult: P(X={k})={r4(out['probability'])}.\nDiagnostic note: Trials must be independent with fixed p."
    if re.search(r"Poisson", question, re.IGNORECASE):
        lam = parse_number("lambda", question)
        k = parse_int("k", question)
        if lam is None or k is None:
            return None
        out = poisson_pmf(float(lam), int(k))
        return f"Problem: Compute Poisson probability.\nMethod: P(X=k)=exp(-lambda)lambda^k/k!.\nCalculation: P={r4(out['probability'])}.\nResult: P(X={k})={r4(out['probability'])}.\nDiagnostic note: Poisson models counts over fixed exposure."
    if re.search(r"Normal", question, re.IGNORECASE):
        mu = parse_number("mu", question)
        sigma = parse_number("sigma", question)
        lower = parse_number("lower", question)
        upper = parse_number("upper", question)
        if None in [mu, sigma, lower, upper]:
            return None
        out = normal_probability_between(float(mu), float(sigma), float(lower), float(upper))
        return f"Problem: Compute normal interval probability.\nMethod: standardize lower and upper cutoffs, then subtract CDF values.\nCalculation: z_lower={r4(out['z_lower'])}, z_upper={r4(out['z_upper'])}, probability={r4(out['probability'])}.\nResult: probability={r4(out['probability'])}.\nDiagnostic note: Confirm whether sigma is known or estimated."
    return None


def answer_intervals_regression(question: str) -> str | None:
    if re.search(r"confidence interval|CI", question, re.IGNORECASE):
        values = parse_values(question)
        successes = parse_int("successes", question)
        n = parse_int("n", question)
        if values is not None:
            out = mean_confidence_interval(values)
            return f"Problem: Compute a confidence interval for a mean.\nMethod: mean +/- 1.96*SE where SE=sample_std/sqrt(n).\nCalculation: mean={r4(out['mean'])}, SE={r4(out['se'])}, interval=[{r4(out['lower'])}, {r4(out['upper'])}].\nResult: CI=[{r4(out['lower'])}, {r4(out['upper'])}].\nDiagnostic note: Use a t critical value for small samples."
        if successes is not None and n is not None:
            out = proportion_confidence_interval(successes, n)
            return f"Problem: Compute a confidence interval for a proportion.\nMethod: p_hat +/- 1.96*sqrt(p_hat(1-p_hat)/n).\nCalculation: p_hat={r4(out['p_hat'])}, SE={r4(out['se'])}, interval=[{r4(out['lower'])}, {r4(out['upper'])}].\nResult: CI=[{r4(out['lower'])}, {r4(out['upper'])}].\nDiagnostic note: Wilson intervals are better for small samples."
    if re.search(r"linear regression|slope|intercept|R\^?2", question, re.IGNORECASE):
        x = parse_vector_after("x", question)
        y = parse_vector_after("y", question)
        if x is None or y is None:
            return None
        out = simple_linear_regression(x, y)
        return (
            "Problem: Fit simple linear regression.\n"
            "Method: slope=Sxy/Sxx, intercept=y_bar-slope*x_bar, R2=1-SS_res/SS_tot.\n"
            f"Calculation: slope={r4(out['slope'])}, intercept={r4(out['intercept'])}, predictions={fmt_list(out['predictions'])}, R2={r4(out['r2'])}.\n"
            f"Result: y_hat={r4(out['intercept'])}+{r4(out['slope'])}x; R2={r4(out['r2'])}.\n"
            "Diagnostic note: Inspect residuals before trusting the fitted line."
        )
    return None


def answer_ml_quant(question: str) -> str | None:
    if re.search(r"logistic.*L2|L2.*logistic", question, re.IGNORECASE):
        x = parse_vector_after("x", question)
        w = parse_vector_after("w", question)
        y = parse_int("y", question)
        lam = parse_number("lambda", question)
        if x is None or w is None or y is None or lam is None:
            return None
        out = logistic_l2_gradient(x, w, y, lam)
        return f"Problem: Compute logistic regression gradient with L2.\nMethod: z=w dot x, p=sigmoid(z), gradient=(p-y)x+lambda*w.\nCalculation: z={r4(out['z'])}, p={r4(out['p'])}, gradient={fmt_list(out['gradient'])}.\nResult: gradient={fmt_list(out['gradient'])}.\nDiagnostic note: Usually exclude the intercept from L2 regularization."
    if re.search(r"SVM|hinge", question, re.IGNORECASE):
        y = parse_int("y", question)
        score = parse_number("score", question)
        if y is None or score is None:
            return None
        out = svm_hinge_loss(y, score)
        return f"Problem: Compute SVM hinge loss.\nMethod: margin=y*score, loss=max(0,1-margin).\nCalculation: margin={r4(out['margin'])}, loss={r4(out['loss'])}.\nResult: hinge_loss={r4(out['loss'])}; update={out['gradient_direction']}.\nDiagnostic note: Points inside the margin still produce loss."
    if re.search(r"K-means|centroid", question, re.IGNORECASE):
        return None
    return None


def answer_time_series_trading_auto_info(question: str) -> str | None:
    if re.search(r"moving average", question, re.IGNORECASE):
        values = parse_values(question)
        window = parse_int("window", question)
        if values is None or window is None:
            return None
        out = moving_average(values, window)
        return f"Problem: Compute moving-average forecast.\nMethod: forecast=mean(last window observations).\nCalculation: forecast={r4(out['forecast'])}.\nResult: next forecast={r4(out['forecast'])}.\nDiagnostic note: Moving averages lag trend shifts."
    if re.search(r"exponential smoothing", question, re.IGNORECASE):
        values = parse_values(question)
        alpha = parse_number("alpha", question)
        if values is None or alpha is None:
            return None
        out = exponential_smoothing(values, alpha)
        return f"Problem: Compute simple exponential smoothing.\nMethod: level_t=alpha*y_t+(1-alpha)*level_(t-1).\nCalculation: final_level={r4(out['forecast'])}.\nResult: smoothed forecast={r4(out['forecast'])}.\nDiagnostic note: Larger alpha reacts faster but is noisier."
    if re.search(r"MAE|RMSE|MAPE|forecast error", question, re.IGNORECASE):
        actuals = parse_vector_after("actuals", question)
        forecasts = parse_vector_after("forecasts", question)
        if actuals is None or forecasts is None:
            return None
        out = regression_errors(actuals, forecasts)
        return f"Problem: Compute error metrics.\nMethod: MAE=mean(abs(error)), RMSE=sqrt(mean(error^2)), MAPE=mean(abs(error/actual)).\nCalculation: errors={fmt_list(out['errors'])}, MAE={r4(out['mae'])}, RMSE={r4(out['rmse'])}, MAPE={r4(out['mape'])}.\nResult: MAE={r4(out['mae'])}, RMSE={r4(out['rmse'])}, MAPE={r4(out['mape'])}.\nDiagnostic note: RMSE penalizes large misses more than MAE."
    if re.search(r"Sharpe|drawdown|trading risk", question, re.IGNORECASE):
        prices = parse_vector_after("prices", question)
        if prices is None:
            return None
        ret = returns_volatility_sharpe(prices)
        dd = max_drawdown(prices)
        return f"Problem: Compute trading risk metrics.\nMethod: returns are period percentage changes, volatility is sample std, Sharpe=mean/vol, drawdown=(price-peak)/peak.\nCalculation: returns={fmt_list(ret['returns'])}, mean={r4(ret['mean_return'])}, vol={r4(ret['volatility'])}, Sharpe={r4(ret['sharpe'])}, max_drawdown={r4(dd['max_drawdown'])}.\nResult: Sharpe={r4(ret['sharpe'])}; max_drawdown={r4(dd['max_drawdown'])}.\nDiagnostic note: Annualize consistently when comparing strategies."
    if re.search(r"beta|CAPM", question, re.IGNORECASE):
        asset = parse_vector_after("asset returns", question)
        market = parse_vector_after("market returns", question)
        if asset is None or market is None:
            return None
        out = beta(asset, market)
        return f"Problem: Compute market beta.\nMethod: beta=cov(asset,market)/var(market).\nCalculation: covariance={r4(out['covariance'])}, market_variance={r4(out['market_variance'])}, beta={r4(out['beta'])}.\nResult: beta={r4(out['beta'])}.\nDiagnostic note: Estimate beta on enough aligned return observations."
    if re.search(r"stopping distance", question, re.IGNORECASE):
        speed = parse_number("speed", question)
        reaction_time = parse_number("reaction_time", question)
        deceleration = parse_number("deceleration", question)
        if None in [speed, reaction_time, deceleration]:
            return None
        out = stopping_distance(float(speed), float(reaction_time), float(deceleration))
        return f"Problem: Compute vehicle stopping distance.\nMethod: reaction=v*t, braking=v^2/(2a), total=sum.\nCalculation: reaction={r4(out['reaction_distance'])}, braking={r4(out['braking_distance'])}, total={r4(out['total_distance'])} meters.\nResult: total stopping distance={r4(out['total_distance'])} meters.\nDiagnostic note: Braking distance grows with speed squared."
    if re.search(r"entropy", question, re.IGNORECASE):
        values = parse_vector_after("probabilities", question)
        if values is None:
            return None
        out = entropy(values)
        return f"Problem: Compute entropy.\nMethod: H=-sum p_i log2(p_i).\nCalculation: H={r4(out['entropy_bits'])} bits.\nResult: entropy={r4(out['entropy_bits'])} bits.\nDiagnostic note: Entropy is highest for uniform class probabilities."
    if re.search(r"KL|D_KL", question, re.IGNORECASE):
        p = parse_vector_after("p", question)
        q = parse_vector_after("q", question)
        if p is None or q is None:
            return None
        out = kl_divergence(p, q)
        return f"Problem: Compute KL divergence.\nMethod: D_KL(p||q)=sum p_i log2(p_i/q_i).\nCalculation: KL={r4(out['kl_bits'])} bits.\nResult: D_KL(p||q)={r4(out['kl_bits'])} bits.\nDiagnostic note: KL is asymmetric."
    return None


def answer_anova_chi_corr(question: str) -> str | None:
    if re.search(r"chi-square|chi square", question, re.IGNORECASE):
        a = parse_int("a", question)
        b = parse_int("b", question)
        c = parse_int("c", question)
        d = parse_int("d", question)
        if None in [a, b, c, d]:
            return None
        out = chi_square_2x2(int(a), int(b), int(c), int(d))
        expected = [value for row in out["expected"] for value in row]
        return f"Problem: Compute chi-square for a 2x2 table.\nMethod: expected=row_total*col_total/grand_total; chi2=sum((O-E)^2/E).\nCalculation: expected={fmt_list(expected)}, chi2={r4(out['chi2'])}, df={out['df']}.\nResult: chi-square={r4(out['chi2'])} with df={out['df']}.\nDiagnostic note: Expected counts should be large enough for the approximation."
    if re.search(r"ANOVA", question, re.IGNORECASE):
        return None
    if re.search(r"correlation|covariance", question, re.IGNORECASE):
        x = parse_vector_after("x", question)
        y = parse_vector_after("y", question)
        if x is None or y is None:
            return None
        out = covariance_correlation(x, y)
        return f"Problem: Compute sample covariance and Pearson correlation.\nMethod: cov=sum((x-xbar)(y-ybar))/(n-1), corr=cov/(std_x*std_y).\nCalculation: covariance={r4(out['covariance'])}, corr={r4(out['correlation'])}.\nResult: correlation={r4(out['correlation'])}.\nDiagnostic note: Correlation can be distorted by outliers."
    return None


def answer_question(question: str) -> str:
    for solver in [
        answer_welch,
        answer_paired_t,
        answer_one_sample_t,
        answer_two_prop,
        answer_probability,
        answer_intervals_regression,
        answer_ml_quant,
        answer_time_series_trading_auto_info,
        answer_anova_chi_corr,
        answer_descriptive,
    ]:
        response = solver(question)
        if response:
            return response
    dl_response = answer_dl_question(question)
    if "Unsupported by the current DL hybrid calculator" not in dl_response:
        return dl_response
    return (
        "Problem: No deterministic V5.1 calculator matched this prompt.\n"
        "Method: Route this to the fine-tuned adapter for explanation, or add a deterministic calculator for this task family.\n"
        "Calculation: Not computed.\n"
        "Result: Unsupported by the current V5.1 hybrid calculator.\n"
        "Diagnostic note: For production-grade numeric answers, add exact calculators before trusting free-form model arithmetic."
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
