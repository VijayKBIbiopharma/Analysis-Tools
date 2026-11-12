"""Running the test the router chose.

Every function here returns an :class:`Outcome`, and every Outcome carries four
things beyond the p-value, because a p-value on its own has caused more bad
decisions in process development than any other number:

* **the effect** - the difference in averages, the slope, the odds ratio - in
  the units the user supplied, because "significant" says nothing about size;
* **an interval on it**, because a point estimate with no width invites the
  reader to believe the last digit;
* **a verdict in three states, not two.** "Different", "no difference worth
  having", and "inconclusive" are different findings. The last is the one
  routinely misreported as the second, and it is the most common honest answer
  to a small study;
* **where to reproduce it in JMP**, so nobody has to take this tool's word.

The three-state verdict is the part that matters most. When a test does not
reach significance, the question "could this study have detected a difference
worth caring about?" has an answer, and the analysis-path matrix asks for it in
so many words: its last row is *Sample size (only if p > .05)*. A study with no
power to detect a meaningful effect has not shown equivalence; it has shown
nothing, and should say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from analysis_tools.assumptions import ALPHA, check_equal_variance, check_normality, fmt_p
from analysis_tools.routing import (
    BROWN_FORSYTHE, CHI_SQUARE, KRUSKAL_WALLIS, LOGISTIC, ONE_WAY_ANOVA,
    PAIRED_T, REGRESSION, TESTS, TWO_PROPORTIONS, TWO_SAMPLE_T, WELCH_ANOVA,
)

#: Verdicts. Three, not two.
DIFFERENT = "different"
NO_EVIDENCE = "no evidence of a difference"
INCONCLUSIVE = "inconclusive"

#: The power a study is assumed to want when asking what it could have detected.
TARGET_POWER = 0.80


@dataclass
class Outcome:
    """One test, run."""

    test: str
    statistic: Optional[float]
    p_value: Optional[float]
    verdict: str
    plain: str
    estimate: Optional[float] = None
    estimate_label: str = ""
    interval: Optional[tuple] = None
    df: Optional[float] = None
    impact: Optional[str] = None
    detail: dict = field(default_factory=dict)
    cautions: list = field(default_factory=list)
    checks: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return TESTS[self.test][0]

    @property
    def verify_in(self) -> str:
        return TESTS[self.test][1]


def minimum_detectable_difference(sd: float, n1: int, n2: int,
                                  alpha: float = ALPHA,
                                  power: float = TARGET_POWER) -> Optional[float]:
    """The smallest difference in averages this study could have found.

    Exact and t-based rather than the normal approximation, which overstates
    power at the sample sizes development work actually runs at. Solved by
    iteration because the degrees of freedom depend on the sample sizes, not on
    the effect.
    """
    if sd <= 0 or n1 < 2 or n2 < 2:
        return None
    df = n1 + n2 - 2
    se = sd * np.sqrt(1.0 / n1 + 1.0 / n2)
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    t_beta = stats.t.ppf(power, df)
    return float((t_alpha + t_beta) * se)


def _three_state(p: Optional[float], detectable: Optional[float],
                 units: str = "") -> tuple:
    """Turn a p-value into one of three verdicts, with the sentence for it."""
    if p is None:
        return INCONCLUSIVE, "The test could not be computed."
    if p < ALPHA:
        return DIFFERENT, ""
    if detectable is None:
        return NO_EVIDENCE, ""
    return INCONCLUSIVE, (
        "This is not evidence that they are the same. With the numbers of "
        "observations collected, the smallest difference this study could have "
        "detected reliably is about %.4g%s — anything smaller would have gone "
        "unnoticed. To call them equivalent you need an equivalence test against "
        "a margin you set in advance, not a non-significant difference test."
        % (detectable, units)
    )


# ------------------------------------------------------------ two groups, means

def two_sample_t(groups: dict, y_name: str, x_name: str) -> Outcome:
    """Welch's t-test: do two groups have the same average?

    Welch rather than pooled by default. JMP's Fit Y by X reports both, and
    Welch is the one that stays correct when the spreads differ - which is
    common, is not always detected by a test of variances at small n, and costs
    nothing when the spreads are in fact equal.
    """
    (label_a, a), (label_b, b) = list(groups.items())
    a = np.asarray([v for v in a if v == v], dtype=float)
    b = np.asarray([v for v in b if v == v], dtype=float)
    checks = [check_normality(a, "the %s group" % label_a),
              check_normality(b, "the %s group" % label_b),
              check_equal_variance([a, b], [label_a, label_b])]

    statistic, p = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    df = float((a.var(ddof=1) / a.size + b.var(ddof=1) / b.size) ** 2 / (
        (a.var(ddof=1) / a.size) ** 2 / (a.size - 1)
        + (b.var(ddof=1) / b.size) ** 2 / (b.size - 1)
    ))
    crit = stats.t.ppf(1 - ALPHA / 2, df)
    interval = (diff - crit * se, diff + crit * se)

    pooled_sd = float(np.sqrt(((a.size - 1) * a.var(ddof=1)
                               + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)))
    detectable = minimum_detectable_difference(pooled_sd, a.size, b.size)
    verdict, extra = _three_state(float(p), detectable)

    if verdict == DIFFERENT:
        plain = ("**%s differs between the two %s groups.** %s averages %.4g and "
                 "%s averages %.4g, a difference of %.4g (95%% confidence interval "
                 "%.4g to %.4g). Welch's t = %.3f, p = %s."
                 % (y_name, x_name, label_a, a.mean(), label_b, b.mean(), diff,
                    interval[0], interval[1], statistic, fmt_p(p)))
    else:
        plain = ("**No significant difference in %s between the two %s groups.** "
                 "%s averages %.4g and %s averages %.4g, a difference of %.4g "
                 "(95%% confidence interval %.4g to %.4g). Welch's t = %.3f, "
                 "p = %s. %s"
                 % (y_name, x_name, label_a, a.mean(), label_b, b.mean(), diff,
                    interval[0], interval[1], statistic, fmt_p(p), extra))

    return Outcome(
        test=TWO_SAMPLE_T, statistic=float(statistic), p_value=float(p),
        verdict=verdict, plain=plain, estimate=diff,
        estimate_label="difference in averages (%s minus %s)" % (label_a, label_b),
        interval=interval, df=df, checks=checks,
        detail={"n": {label_a: int(a.size), label_b: int(b.size)},
                "mean": {label_a: float(a.mean()), label_b: float(b.mean())},
                "sd": {label_a: float(a.std(ddof=1)), label_b: float(b.std(ddof=1))},
                "pooled_sd": pooled_sd,
                "smallest_detectable_difference": detectable},
    )


def paired_t(before: Sequence[float], after: Sequence[float],
             y_name: str, labels: tuple = ("first", "second")) -> Outcome:
    """Paired t-test: across the same units measured twice, is the average
    difference zero?"""
    a = np.asarray(before, dtype=float)
    b = np.asarray(after, dtype=float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    differences = a - b
    checks = [check_normality(differences, "the paired differences")]

    statistic, p = stats.ttest_rel(a, b)
    mean_diff = float(differences.mean())
    se = float(differences.std(ddof=1) / np.sqrt(differences.size))
    df = differences.size - 1
    crit = stats.t.ppf(1 - ALPHA / 2, df)
    interval = (mean_diff - crit * se, mean_diff + crit * se)
    detectable = (float((stats.t.ppf(1 - ALPHA / 2, df) + stats.t.ppf(TARGET_POWER, df)) * se)
                  if se > 0 else None)
    verdict, extra = _three_state(float(p), detectable)

    plain = ("**%s the average difference in %s is %.4g** (95%% confidence "
             "interval %.4g to %.4g), across %d paired observations. t = %.3f, "
             "p = %s. %s"
             % ("There is a difference:" if verdict == DIFFERENT
                else "No significant difference —", y_name, mean_diff,
                interval[0], interval[1], differences.size, statistic, fmt_p(p),
                extra))
    return Outcome(test=PAIRED_T, statistic=float(statistic), p_value=float(p),
                   verdict=verdict, plain=plain, estimate=mean_diff,
                   estimate_label="average difference (%s minus %s)" % labels,
                   interval=interval, df=float(df), checks=checks,
                   detail={"n_pairs": int(differences.size),
                           "sd_of_differences": float(differences.std(ddof=1)),
                           "smallest_detectable_difference": detectable})


# --------------------------------------------------- three or more groups, means

def compare_groups(groups: dict, y_name: str, x_name: str) -> list:
    """Three or more groups: one-way ANOVA, Welch ANOVA, or Kruskal-Wallis.

    Which one is decided here rather than left to the reader. The chart folds
    two problems into one fallback; they are separated:

    * residuals not normal  -> Kruskal-Wallis, which compares ranks
    * spreads unequal       -> Welch ANOVA, which still compares averages
    * neither               -> one-way ANOVA

    The branch taken and the numbers that decided it travel with the result.
    """
    labels = list(groups)
    arrays = [np.asarray([v for v in groups[k] if v == v], dtype=float) for k in labels]
    usable = [(lab, arr) for lab, arr in zip(labels, arrays) if arr.size >= 1]
    labels = [lab for lab, _ in usable]
    arrays = [arr for _, arr in usable]

    # Normality is a property of the residuals, not of each group separately:
    # testing each group is the same question asked k times with less data each
    # time. But the residuals must be *standardised* within group before they
    # are pooled, and that detail is load-bearing.
    #
    # Pooling raw residuals from groups with different variances gives a
    # mixture of a narrow normal and a wide one, which is heavy-tailed and
    # fails a normality test however normal each group is. An earlier version
    # pooled them raw, so unequal variance masqueraded as non-normality and
    # every heteroscedastic dataset was routed to Kruskal-Wallis - a rank test,
    # answering a different question from the one the user asked. Dividing each
    # group's residuals by its own standard deviation separates the two
    # questions, which is what lets them have two different remedies.
    standardised = []
    for arr in arrays:
        if arr.size >= 2:
            sd = arr.std(ddof=1)
            if sd > 0:
                standardised.append((arr - arr.mean()) / sd)
    residuals = (np.concatenate(standardised) if standardised
                 else np.array([], dtype=float))
    normality = check_normality(residuals, "the residuals")
    variance = check_equal_variance(arrays, labels)
    checks = [normality, variance]

    spread_outcome = Outcome(
        test=BROWN_FORSYTHE, statistic=variance.statistic, p_value=variance.p_value,
        verdict=DIFFERENT if (variance.p_value is not None
                              and variance.p_value < ALPHA) else NO_EVIDENCE,
        plain=variance.plain, checks=[],
    )

    for_test = [arr for arr in arrays if arr.size >= 2]
    if len(for_test) < 2:
        return [Outcome(test=ONE_WAY_ANOVA, statistic=None, p_value=None,
                        verdict=INCONCLUSIVE,
                        plain="At least two groups need two or more observations "
                              "before their averages can be compared.",
                        checks=checks), spread_outcome]

    if not normality.holds:
        statistic, p = stats.kruskal(*for_test)
        chosen, why = KRUSKAL_WALLIS, (
            "The residuals are not normal (%s), so the comparison is of ranks "
            "rather than averages." % normality.plain.split("(")[-1].rstrip(")."))
    elif not variance.holds:
        statistic, p = _welch_anova(for_test)
        chosen, why = WELCH_ANOVA, (
            "The groups do not vary by the same amount (%s), so Welch's version "
            "is used: it compares averages without assuming equal spreads."
            % variance.plain.split("(")[-1].split(")")[0])
    else:
        statistic, p = stats.f_oneway(*for_test)
        chosen, why = ONE_WAY_ANOVA, (
            "The residuals are consistent with normality and the groups vary by "
            "a similar amount, so the ordinary one-way ANOVA applies.")

    means = {lab: float(arr.mean()) for lab, arr in zip(labels, arrays) if arr.size}
    best = max(means, key=means.get)
    worst = min(means, key=means.get)
    impact = _r_squared_adjusted(arrays)
    pooled_sd = _pooled_sd(arrays)
    sizes = [arr.size for arr in arrays if arr.size >= 2]
    detectable = (minimum_detectable_difference(pooled_sd, min(sizes), min(sizes))
                  if pooled_sd and len(sizes) >= 2 else None)
    verdict, extra = _three_state(float(p), detectable)

    if verdict == DIFFERENT:
        plain = ("**%s differs across the %s groups.** Highest is %s at %.4g, "
                 "lowest is %s at %.4g. %s = %.3f, p = %s. %s %s"
                 % (y_name, x_name, best, means[best], worst, means[worst],
                    "H" if chosen == KRUSKAL_WALLIS else "F", statistic, fmt_p(p),
                    why, impact or ""))
    else:
        plain = ("**No significant difference in %s across the %s groups.** "
                 "Highest is %s at %.4g, lowest is %s at %.4g. %s = %.3f, p = %s. "
                 "%s %s"
                 % (y_name, x_name, best, means[best], worst, means[worst],
                    "H" if chosen == KRUSKAL_WALLIS else "F", statistic, fmt_p(p),
                    why, extra))

    centering = Outcome(
        test=chosen, statistic=float(statistic), p_value=float(p), verdict=verdict,
        plain=plain, impact=impact, checks=checks,
        detail={"n": {lab: int(arr.size) for lab, arr in zip(labels, arrays)},
                "mean": means,
                "sd": {lab: (float(arr.std(ddof=1)) if arr.size >= 2 else None)
                       for lab, arr in zip(labels, arrays)},
                "branch_reason": why,
                "smallest_detectable_difference": detectable},
    )
    return [centering, spread_outcome]


def _welch_anova(groups: Sequence[np.ndarray]) -> tuple:
    """Welch's one-way ANOVA. Compares averages without assuming equal spreads."""
    k = len(groups)
    n = np.array([g.size for g in groups], dtype=float)
    means = np.array([g.mean() for g in groups])
    variances = np.array([g.var(ddof=1) for g in groups])
    w = n / variances
    total_w = w.sum()
    grand = (w * means).sum() / total_w
    numerator = (w * (means - grand) ** 2).sum() / (k - 1)
    lam = ((1 - w / total_w) ** 2 / (n - 1)).sum()
    denominator = 1 + (2 * (k - 2) / (k * k - 1)) * lam
    f = numerator / denominator
    df2 = (k * k - 1) / (3 * lam)
    p = stats.f.sf(f, k - 1, df2)
    return float(f), float(p)


def _pooled_sd(arrays: Sequence[np.ndarray]) -> Optional[float]:
    usable = [a for a in arrays if a.size >= 2]
    if len(usable) < 2:
        return None
    ss = sum((a.size - 1) * a.var(ddof=1) for a in usable)
    df = sum(a.size - 1 for a in usable)
    return float(np.sqrt(ss / df)) if df > 0 else None


def _r_squared_adjusted(arrays: Sequence[np.ndarray]) -> Optional[str]:
    """How much of the variation the grouping accounts for."""
    usable = [a for a in arrays if a.size >= 1]
    values = np.concatenate(usable)
    n, k = values.size, len(usable)
    if n <= k or k < 2:
        return None
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total == 0:
        return None
    ss_between = sum(a.size * (a.mean() - grand) ** 2 for a in usable)
    r2 = ss_between / ss_total
    adjusted = 1 - (1 - r2) * (n - 1) / (n - k)
    return ("The grouping accounts for %.1f%% of the variation in the response "
            "(adjusted R² = %.3f)." % (100 * max(adjusted, 0.0), adjusted))


# ------------------------------------------------------------ counts and tables

def chi_square(x_values: Sequence, y_values: Sequence,
               x_name: str, y_name: str) -> list:
    """Are the two categorical columns independent?

    Fisher's exact test replaces the chi-square approximation on a 2x2 table
    with a small expected count, because the approximation is poor exactly
    there and quietly so. The switch is reported.
    """
    table = pd.crosstab(pd.Series(x_values, name=x_name),
                        pd.Series(y_values, name=y_name))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return [Outcome(test=CHI_SQUARE, statistic=None, p_value=None,
                        verdict=INCONCLUSIVE,
                        plain="Both columns need at least two levels present "
                              "before they can be cross-tabulated.")]

    statistic, p, dof, expected = stats.chi2_contingency(table.values)
    small = int((expected < 5).sum())
    cautions = []
    used_fisher = False
    if table.shape == (2, 2) and expected.min() < 5:
        _, p = stats.fisher_exact(table.values)
        used_fisher = True
        cautions.append(
            "An expected count below 5 makes the chi-square approximation "
            "unreliable, so Fisher's exact test was used instead. JMP reports it "
            "on the same Contingency output."
        )
    elif small:
        cautions.append(
            "%d of %d cells have an expected count below 5. The chi-square "
            "approximation is strained; treat the p-value as approximate."
            % (small, expected.size)
        )

    n = int(table.values.sum())
    cramers_v = float(np.sqrt(statistic / (n * (min(table.shape) - 1)))) if n else None
    verdict = DIFFERENT if (p is not None and p < ALPHA) else NO_EVIDENCE
    plain = ("**%s and %s are %s.** %s = %s, p = %s, on %d observations.%s"
             % (x_name, y_name,
                "related" if verdict == DIFFERENT else "not detectably related",
                "Fisher's exact" if used_fisher else "Chi-square",
                "—" if used_fisher else "%.3f" % statistic, fmt_p(p), n,
                "" if cramers_v is None else
                " Cramér's V = %.3f, where 0 is no association and 1 is complete."
                % cramers_v))

    out = [Outcome(test=CHI_SQUARE, statistic=None if used_fisher else float(statistic),
                   p_value=float(p), verdict=verdict, plain=plain,
                   estimate=cramers_v, estimate_label="Cramér's V",
                   df=float(dof), cautions=cautions,
                   detail={"table": table.to_dict(),
                           "expected_min": float(expected.min())})]
    if table.shape == (2, 2):
        out.append(_two_proportions(table, x_name, y_name))
    return out


def _two_proportions(table: pd.DataFrame, x_name: str, y_name: str) -> Outcome:
    """The difference in proportions, with an interval.

    The chi-square says whether; this says how much, which is the number a
    decision actually needs.
    """
    counts = table.values
    outcome_label = str(table.columns[-1])
    successes = counts[:, -1].astype(float)
    totals = counts.sum(axis=1).astype(float)
    p1, p2 = successes[0] / totals[0], successes[1] / totals[1]
    diff = float(p1 - p2)
    se = float(np.sqrt(p1 * (1 - p1) / totals[0] + p2 * (1 - p2) / totals[1]))
    z = stats.norm.ppf(1 - ALPHA / 2)
    interval = (diff - z * se, diff + z * se)
    verdict = DIFFERENT if not (interval[0] <= 0 <= interval[1]) else NO_EVIDENCE
    plain = ("%s is %s in %s (%.1f%%, %d of %d) than in %s (%.1f%%, %d of %d) — a "
             "difference of %.1f percentage points (95%% confidence interval "
             "%.1f to %.1f)."
             % (outcome_label, "more common" if diff > 0 else "less common",
                table.index[0], 100 * p1, successes[0], totals[0],
                table.index[1], 100 * p2, successes[1], totals[1],
                100 * diff, 100 * interval[0], 100 * interval[1]))
    return Outcome(test=TWO_PROPORTIONS, statistic=None, p_value=None,
                   verdict=verdict, plain=plain, estimate=diff,
                   estimate_label="difference in proportion of %s" % outcome_label,
                   interval=interval)


# ---------------------------------------------------------------- relationships

def simple_regression(x: Sequence[float], y: Sequence[float],
                      x_name: str, y_name: str) -> Outcome:
    """Does the response change steadily as this input changes?"""
    import statsmodels.api as sm

    frame = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"),
                          "y": pd.to_numeric(pd.Series(y), errors="coerce")}).dropna()
    if len(frame) < 3:
        return Outcome(test=REGRESSION, statistic=None, p_value=None,
                       verdict=INCONCLUSIVE,
                       plain="At least three complete pairs are needed to fit a line.")
    model = sm.OLS(frame["y"], sm.add_constant(frame["x"])).fit()
    slope = float(model.params["x"])
    p = float(model.pvalues["x"])
    low, high = (float(v) for v in model.conf_int().loc["x"])
    checks = [check_normality(model.resid, "the residuals")]

    verdict = DIFFERENT if p < ALPHA else NO_EVIDENCE
    span = float(frame["x"].max() - frame["x"].min())
    plain = ("**%s %s with %s.** Every 1 unit increase in %s moves %s by %.4g "
             "(95%% confidence interval %.4g to %.4g); across the range observed "
             "that is %.4g in total. p = %s, adjusted R² = %.3f — the input "
             "accounts for %.1f%% of the variation."
             % (y_name, "rises" if slope > 0 else "falls", x_name, x_name, y_name,
                slope, low, high, slope * span, fmt_p(p), model.rsquared_adj,
                100 * max(model.rsquared_adj, 0.0))
             if verdict == DIFFERENT else
             "**No detectable relationship between %s and %s.** The fitted slope "
             "is %.4g (95%% confidence interval %.4g to %.4g), p = %s. The "
             "interval includes zero, so the direction is not established."
             % (y_name, x_name, slope, low, high, fmt_p(p)))
    return Outcome(test=REGRESSION, statistic=float(model.fvalue), p_value=p,
                   verdict=verdict, plain=plain, estimate=slope,
                   estimate_label="slope (%s per unit of %s)" % (y_name, x_name),
                   interval=(low, high), df=float(model.df_resid), checks=checks,
                   impact="Adjusted R² = %.3f." % model.rsquared_adj,
                   detail={"intercept": float(model.params["const"]),
                           "r_squared": float(model.rsquared),
                           "r_squared_adj": float(model.rsquared_adj),
                           "n": int(len(frame))})


def logistic_regression(x: Sequence, y: Sequence, x_name: str, y_name: str) -> Outcome:
    """Does the chance of each outcome change as this input changes?"""
    import statsmodels.api as sm

    frame = pd.DataFrame({"x": pd.Series(x), "y": pd.Series(y)}).dropna()
    outcomes = sorted(frame["y"].astype(str).unique())
    if len(outcomes) != 2:
        return Outcome(test=LOGISTIC, statistic=None, p_value=None,
                       verdict=INCONCLUSIVE,
                       plain="This runs on a two-outcome response. “%s” has %d "
                             "distinct outcomes; a model for more than two is in "
                             "the Relationships module."
                             % (y_name, len(outcomes)))
    target = outcomes[-1]
    frame["binary"] = (frame["y"].astype(str) == target).astype(int)
    frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
    frame = frame.dropna()
    if frame["binary"].nunique() < 2 or len(frame) < 10:
        return Outcome(test=LOGISTIC, statistic=None, p_value=None,
                       verdict=INCONCLUSIVE,
                       plain="Too few observations, or only one outcome present, "
                             "to fit a logistic model.")
    try:
        model = sm.Logit(frame["binary"], sm.add_constant(frame["x"])).fit(disp=False)
    except Exception as exc:                        # noqa: BLE001 - reported
        return Outcome(test=LOGISTIC, statistic=None, p_value=None,
                       verdict=INCONCLUSIVE,
                       plain="The logistic model did not converge (%s). That "
                             "usually means the outcome is perfectly separated by "
                             "the input, which is a finding in itself." % exc)
    coefficient = float(model.params["x"])
    p = float(model.pvalues["x"])
    odds = float(np.exp(coefficient))
    low, high = (float(np.exp(v)) for v in model.conf_int().loc["x"])
    verdict = DIFFERENT if p < ALPHA else NO_EVIDENCE
    plain = ("**The chance of “%s” %s with %s.** Each 1 unit increase in %s "
             "multiplies the odds by %.3f (95%% confidence interval %.3f to "
             "%.3f), p = %s."
             % (target, "rises" if odds > 1 else "falls", x_name, x_name,
                odds, low, high, fmt_p(p))
             if verdict == DIFFERENT else
             "**No detectable effect of %s on the chance of “%s”.** The odds "
             "ratio is %.3f (95%% confidence interval %.3f to %.3f), p = %s. The "
             "interval includes 1, so no direction is established."
             % (x_name, target, odds, low, high, fmt_p(p)))
    return Outcome(test=LOGISTIC, statistic=None, p_value=p, verdict=verdict,
                   plain=plain, estimate=odds,
                   estimate_label="odds ratio per unit of %s" % x_name,
                   interval=(low, high),
                   detail={"outcome_modelled": target, "n": int(len(frame)),
                           "pseudo_r_squared": float(model.prsquared)})
