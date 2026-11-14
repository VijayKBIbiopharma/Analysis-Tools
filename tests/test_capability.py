"""Stability and capability.

The arithmetic is simple enough to check by hand, so these tests do that rather
than comparing the code against itself. The important tests are the two that
check what the tool *refuses* to do, and the one that re-measures the
false-alarm rate the page quotes to users.
"""

import numpy as np
import pytest
from scipy import stats

from analysis_tools.capability import (
    D2_N2, FALSE_ALARM_BY_N, assess_capability,
    assess_stability, chance_of_a_signal, moving_range_sigma,
)


# ------------------------------------------------------------- the arithmetic

def test_cp_and_cpk_are_the_textbook_ratios():
    """A sample centred at 100 with sd 1 against limits of 94 and 106 has
    Cp = 12/6 = 2 and, being centred, Cpk = Cp."""
    rng = np.random.default_rng(1)
    x = rng.normal(100, 1, 400)
    x = (x - x.mean()) / x.std(ddof=1) + 100       # exactly mean 100, sd 1
    out = assess_capability(x, lsl=94, usl=106)
    assert out.indices["Pp"] == pytest.approx(2.0, rel=1e-6)
    assert out.indices["Ppk"] == pytest.approx(2.0, rel=1e-6)


def test_cpk_falls_when_the_process_is_off_centre():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 400)
    x = (x - x.mean()) / x.std(ddof=1) + 103       # mean 103, sd 1
    out = assess_capability(x, lsl=94, usl=106)
    assert out.indices["Pp"] == pytest.approx(2.0, rel=1e-6)
    assert out.indices["Ppk"] == pytest.approx(1.0, rel=1e-6)


def test_the_short_term_sigma_comes_from_the_average_moving_range():
    x = np.array([10.0, 12.0, 11.0, 13.0, 12.0])
    sigma, mr_bar = moving_range_sigma(x)
    assert mr_bar == pytest.approx(np.mean([2, 1, 2, 1]))
    assert sigma == pytest.approx(mr_bar / D2_N2)


def test_a_drifting_process_has_a_wider_overall_spread_than_short_term():
    """The gap between the two is the drift, and is the finding."""
    rng = np.random.default_rng(3)
    drifting = np.concatenate([rng.normal(100, 0.5, 40), rng.normal(104, 0.5, 40)])
    out = assess_capability(drifting, lsl=94, usl=110)
    assert out.sigma_overall > 2 * out.sigma_within
    assert any("drifts between observations" in c for c in out.cautions)


def test_z_bench_matches_the_normal_tail_it_claims_to_be():
    rng = np.random.default_rng(4)
    x = rng.normal(100, 1, 300)
    out = assess_capability(x, lsl=96, usl=104)
    expected_p = (stats.norm.cdf(96, x.mean(), x.std(ddof=1))
                  + stats.norm.sf(104, x.mean(), x.std(ddof=1)))
    assert out.z_bench == pytest.approx(stats.norm.isf(expected_p))
    assert out.predicted_ppm == pytest.approx(1e6 * expected_p)


# ------------------------------------------------------------ what it refuses

def test_ppm_is_withheld_when_the_data_is_not_normal():
    """A PPM is a claim about the tails, which is where the normal model fails
    first. JMP withholds Z.bench for non-normal data too."""
    rng = np.random.default_rng(5)
    skewed = rng.exponential(2, 120) + 90
    out = assess_capability(skewed, lsl=89, usl=110)
    assert out.predicted_ppm is None and out.z_bench is None
    assert out.withheld and "tails" in out.withheld[0]
    # The indices themselves survive: they are ratios of distances.
    assert "Ppk" in out.indices


def test_ppm_is_given_when_the_data_is_normal():
    rng = np.random.default_rng(6)
    out = assess_capability(rng.normal(100, 1, 120), lsl=96, usl=104)
    assert out.predicted_ppm is not None and not out.withheld


def test_capability_needs_a_specification_limit():
    out = assess_capability([1.0, 2.0, 3.0, 4.0])
    assert "specification limit" in out.plain


def test_a_small_sample_is_told_it_is_small():
    rng = np.random.default_rng(7)
    out = assess_capability(rng.normal(100, 1, 12), lsl=96, usl=104)
    assert any("imprecise" in c for c in out.cautions)


def test_the_interval_on_pp_is_wide_at_small_n_and_is_shown():
    rng = np.random.default_rng(8)
    out = assess_capability(rng.normal(100, 1, 15), lsl=94, usl=106)
    low, high = out.intervals["Pp"]
    assert low < out.indices["Pp"] < high
    assert (high - low) > 0.5, "a 15-point Pp interval should be visibly wide"


# -------------------------------------------------------------- the chart

def test_a_real_shift_is_caught():
    rng = np.random.default_rng(9)
    shifted = np.concatenate([rng.normal(100, 1, 30), rng.normal(106, 1, 30)])
    stability = assess_stability(shifted)
    assert not stability.stable
    rules = {s.rule for s in stability.signals}
    assert any("beyond 3 sigma" in r for r in rules)


def test_a_steady_trend_is_caught_as_a_trend():
    drifting = np.linspace(100, 104, 40) + np.random.default_rng(10).normal(0, 0.05, 40)
    stability = assess_stability(drifting)
    assert any("increasing or decreasing" in s.rule for s in stability.signals)


def test_too_few_points_for_a_chart_returns_nothing_rather_than_guessing():
    assert assess_stability([1.0, 2.0, 3.0]) is None


def test_the_false_alarm_rate_quoted_to_users_is_still_true():
    """The page tells a user "about X% of perfectly stable processes trip a
    signal at this sample size". That number is measured, and if the rules ever
    change it has to be measured again — which is what this test forces."""
    rng = np.random.default_rng(11)
    for n in (30, 60):
        hits, runs = 0, 600
        for _ in range(runs):
            stability = assess_stability(rng.normal(100, 1, n))
            if stability and stability.signals:
                hits += 1
        measured = hits / runs
        claimed = FALSE_ALARM_BY_N[n]
        assert abs(measured - claimed) < 0.06, (
            "n=%d: page claims %.1f%%, measured %.1f%%"
            % (n, 100 * claimed, 100 * measured))


def test_rule_one_alone_matches_its_theoretical_rate():
    """1 - 0.9973^n. If this drifts, the control limits are wrong."""
    rng = np.random.default_rng(12)
    n, runs, hits = 50, 1200, 0
    for _ in range(runs):
        stability = assess_stability(rng.normal(100, 1, n))
        if stability and any(s.rule.startswith("1 point") for s in stability.signals):
            hits += 1
    theoretical = 1 - 0.9973 ** n
    assert abs(hits / runs - theoretical) < 0.05


def test_the_quoted_chance_rises_with_sample_size():
    assert chance_of_a_signal(20) < chance_of_a_signal(60) < chance_of_a_signal(200)


def test_a_clean_chart_says_how_meaningful_it_is():
    rng = np.random.default_rng(13)
    for _ in range(40):
        stability = assess_stability(rng.normal(100, 1, 25))
        if stability and stability.stable:
            assert "would have shown something about" in stability.plain
            return
    pytest.fail("no stable sample found in 40 tries, which is itself suspicious")
