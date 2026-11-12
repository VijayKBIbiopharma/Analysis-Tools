"""The tests, tested.

Two kinds of check here. The first compares against an independent
implementation or an identity that must hold - scipy, or JMP's documented
equivalence between Welch ANOVA on two groups and the unequal-variance t test.
The second *measures* a claim the tool makes to a user: if it says a study could
detect a difference of d, then simulating that study at that difference should
find it about 80% of the time. A claim nobody measured is a claim nobody should
print.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis_tools.compare import (
    DIFFERENT, INCONCLUSIVE, NO_EVIDENCE, _welch_anova, chi_square,
    compare_groups, logistic_regression, minimum_detectable_difference,
    paired_t, simple_regression, two_sample_t,
)
from analysis_tools.routing import (
    BROWN_FORSYTHE, KRUSKAL_WALLIS, ONE_WAY_ANOVA, WELCH_ANOVA,
)


# ------------------------------------------------------ against scipy and JMP

def test_welch_anova_on_two_groups_equals_the_unequal_variance_t_test():
    """JMP's own documentation states this identity: "the Welch ANOVA is
    equivalent to an unequal variance t test when there are only two levels".
    If it holds, the Welch implementation is right; if it does not, it is
    wrong, and no amount of plausible output would show that."""
    rng = np.random.default_rng(3)
    a, b = rng.normal(10, 1, 30), rng.normal(11, 3, 25)
    f, p_f = _welch_anova([a, b])
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    assert f == pytest.approx(t ** 2, rel=1e-9)
    assert p_f == pytest.approx(p_t, rel=1e-9)


def test_one_way_anova_matches_scipy():
    rng = np.random.default_rng(4)
    groups = {"A": rng.normal(10, 1, 20), "B": rng.normal(10.1, 1, 20),
              "C": rng.normal(10.2, 1, 20)}
    centering, _ = compare_groups(groups, "Y", "X")
    expected = stats.f_oneway(*groups.values())
    assert centering.test == ONE_WAY_ANOVA
    assert centering.statistic == pytest.approx(expected.statistic)
    assert centering.p_value == pytest.approx(expected.pvalue)


def test_the_two_sample_t_matches_scipy_welch():
    rng = np.random.default_rng(5)
    a, b = rng.normal(10, 1, 18), rng.normal(10.9, 2, 22)
    outcome = two_sample_t({"A": a, "B": b}, "Y", "X")
    expected = stats.ttest_ind(a, b, equal_var=False)
    assert outcome.statistic == pytest.approx(expected.statistic)
    assert outcome.p_value == pytest.approx(expected.pvalue)


def test_chi_square_matches_scipy():
    rng = np.random.default_rng(6)
    x = rng.choice(["A", "B", "C"], 200)
    y = rng.choice(["Pass", "Fail"], 200)
    outcome = chi_square(x, y, "X", "Y")[0]
    table = pd.crosstab(pd.Series(x), pd.Series(y))
    expected = stats.chi2_contingency(table.values)
    assert outcome.statistic == pytest.approx(expected[0])


def test_the_regression_slope_is_the_one_least_squares_gives():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([2.1, 4.2, 5.8, 8.1, 10.2, 11.8, 14.1, 16.2])
    outcome = simple_regression(x, y, "X", "Y")
    expected = stats.linregress(x, y)
    assert outcome.estimate == pytest.approx(expected.slope)
    assert outcome.p_value == pytest.approx(expected.pvalue)


# ------------------------------------------------- the branch, and why it moved

def test_normal_and_equal_spread_gives_the_ordinary_anova():
    rng = np.random.default_rng(7)
    groups = {k: rng.normal(10, 1, 25) for k in "ABC"}
    centering, _ = compare_groups(groups, "Y", "X")
    assert centering.test == ONE_WAY_ANOVA
    assert "ordinary one-way ANOVA" in centering.detail["branch_reason"]


def test_unequal_spread_switches_to_welch_not_kruskal_wallis():
    """The chart's single fallback is Kruskal-Wallis for either problem. Unequal
    variance does not call for a rank test: Welch still compares averages, which
    is the question that was asked."""
    rng = np.random.default_rng(8)
    groups = {"A": rng.normal(10, 0.4, 30), "B": rng.normal(10, 0.4, 30),
              "C": rng.normal(10, 4.0, 30)}
    centering, _ = compare_groups(groups, "Y", "X")
    assert centering.test == WELCH_ANOVA
    assert "without assuming equal spreads" in centering.detail["branch_reason"]


def test_non_normal_residuals_switch_to_kruskal_wallis():
    rng = np.random.default_rng(9)
    groups = {"A": rng.exponential(2, 40), "B": rng.exponential(2, 40),
              "C": rng.exponential(2, 40)}
    centering, _ = compare_groups(groups, "Y", "X")
    assert centering.test == KRUSKAL_WALLIS
    assert centering.statistic == pytest.approx(stats.kruskal(*groups.values()).statistic)


def test_the_spread_test_is_returned_alongside_every_group_comparison():
    """The chart gives two tests. Returning only the centering one would lose
    the change that leaves the mean alone and doubles the variance."""
    rng = np.random.default_rng(10)
    groups = {k: rng.normal(10, 1, 20) for k in "ABC"}
    outcomes = compare_groups(groups, "Y", "X")
    assert [o.test for o in outcomes][1] == BROWN_FORSYTHE


def test_a_change_in_spread_alone_is_found_even_though_the_means_match():
    """The case a t-test on its own reports as "no significant difference"."""
    rng = np.random.default_rng(11)
    groups = {"A": rng.normal(10, 0.5, 40), "B": rng.normal(10, 3.0, 40)}
    _, spread = compare_groups(groups, "Y", "X")
    assert spread.verdict == DIFFERENT


# ------------------------------------------------- the claims, measured not told

def test_the_detectable_difference_is_what_it_claims_to_be():
    """The tool tells a user "the smallest difference this study could have
    detected is about d". Simulate exactly that study at exactly that
    difference: it should come out significant about 80% of the time. Measured
    over 2,000 runs, not argued."""
    rng = np.random.default_rng(12)
    sd, n = 1.0, 20
    d = minimum_detectable_difference(sd, n, n)
    hits = 0
    runs = 2000
    for _ in range(runs):
        a = rng.normal(0.0, sd, n)
        b = rng.normal(d, sd, n)
        if stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05:
            hits += 1
    power = hits / runs
    assert 0.76 <= power <= 0.84, "claimed 80%% power, measured %.1f%%" % (100 * power)


def test_the_confidence_interval_covers_the_truth_at_the_rate_it_claims():
    """A 95% interval that covers 80% of the time is a lie told in good faith.
    Measured over 2,000 runs."""
    rng = np.random.default_rng(13)
    truth, covered, runs = 1.5, 0, 2000
    for _ in range(runs):
        a = rng.normal(10.0, 1.0, 15)
        b = rng.normal(10.0 - truth, 1.0, 15)
        out = two_sample_t({"A": a, "B": b}, "Y", "X")
        low, high = out.interval
        if low <= truth <= high:
            covered += 1
    rate = covered / runs
    assert 0.93 <= rate <= 0.97, "claimed 95%% coverage, measured %.1f%%" % (100 * rate)


def test_the_false_alarm_rate_is_the_one_advertised():
    """With no real difference, a 5% test should call it different about 5% of
    the time. Any more and every verdict the tool gives is suspect."""
    rng = np.random.default_rng(14)
    alarms, runs = 0, 2000
    for _ in range(runs):
        groups = {k: rng.normal(10, 1, 12) for k in "ABC"}
        if compare_groups(groups, "Y", "X")[0].p_value < 0.05:
            alarms += 1
    rate = alarms / runs
    assert 0.035 <= rate <= 0.07, "expected about 5%%, measured %.1f%%" % (100 * rate)


# --------------------------------------------------------- three states, not two

def test_a_clear_difference_reads_as_different():
    rng = np.random.default_rng(15)
    out = two_sample_t({"A": rng.normal(10, 1, 30), "B": rng.normal(12, 1, 30)}, "Y", "X")
    assert out.verdict == DIFFERENT


def test_a_small_study_that_finds_nothing_is_inconclusive_not_equivalent():
    """The error this exists to prevent: reading "p > 0.05" as "the same"."""
    rng = np.random.default_rng(16)
    out = two_sample_t({"A": rng.normal(10, 2, 6), "B": rng.normal(10.1, 2, 6)}, "Y", "X")
    assert out.verdict == INCONCLUSIVE
    assert "not evidence that they are the same" in out.plain
    assert "equivalence test" in out.plain


def test_the_inconclusive_wording_names_what_the_study_could_have_seen():
    rng = np.random.default_rng(17)
    out = two_sample_t({"A": rng.normal(10, 1, 8), "B": rng.normal(10.05, 1, 8)}, "Y", "X")
    assert out.detail["smallest_detectable_difference"] > 0
    assert "would have gone unnoticed" in out.plain


# --------------------------------------------------------------- it says enough

def test_every_outcome_carries_an_effect_and_an_interval_where_one_exists():
    rng = np.random.default_rng(18)
    out = two_sample_t({"A": rng.normal(10, 1, 20), "B": rng.normal(11, 1, 20)}, "Y", "X")
    assert out.estimate is not None and out.interval is not None
    assert out.estimate_label and len(out.interval) == 2


def test_paired_data_is_analysed_on_the_differences():
    rng = np.random.default_rng(19)
    before = rng.normal(10, 2, 20)
    after = before + rng.normal(0.8, 0.3, 20)
    out = paired_t(after, before, "Titer")
    assert out.verdict == DIFFERENT
    assert out.estimate == pytest.approx(float(np.mean(after - before)))


def test_fisher_replaces_chi_square_when_an_expected_count_is_small():
    x = ["A"] * 10 + ["B"] * 10
    y = ["Pass"] * 9 + ["Fail"] + ["Pass"] * 8 + ["Fail"] * 2
    outcome = chi_square(x, y, "Line", "Result")[0]
    assert any("Fisher" in c for c in outcome.cautions)


def test_a_two_by_two_table_also_reports_the_difference_in_proportions():
    """The chi-square says whether. The decision needs how much."""
    rng = np.random.default_rng(20)
    x = rng.choice(["A", "B"], 150)
    y = np.where(rng.random(150) < np.where(x == "A", 0.25, 0.6), "Pass", "Fail")
    outcomes = chi_square(x, y, "Line", "Result")
    assert len(outcomes) == 2
    assert "percentage points" in outcomes[1].plain


def test_a_perfectly_separated_logistic_fit_says_so_rather_than_crashing():
    x = list(range(20))
    y = ["Fail"] * 10 + ["Pass"] * 10
    out = logistic_regression(x, y, "X", "Y")
    assert out.verdict in (INCONCLUSIVE, DIFFERENT)
    if out.verdict == INCONCLUSIVE:
        assert "separated" in out.plain or "converge" in out.plain
