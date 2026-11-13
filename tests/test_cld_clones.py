"""Clone screening and generation limits.

Both modules exist to stop a ranking or a slope being read as evidence when it
is noise, so the tests that matter most are the ones run on data with nothing
in it at all.
"""

import numpy as np
import pandas as pd
import pytest

from analysis_tools.cld_clones import (
    EXTRAPOLATION_CAP, MARGINAL_CORRELATION, _benjamini_hochberg,
    generation_limit,
    screen_early_reads,
)


def screen_frame(n=40, seed=3, signal=0.7, noise=0.35):
    rng = np.random.default_rng(seed)
    late = rng.normal(3.0, 0.5, n)
    return pd.DataFrame({
        "VCD_d3": signal * late + rng.normal(0, noise, n),
        "Via_d3": rng.normal(95, 2, n),
        "Glc_d5": rng.normal(4, 0.6, n),
        "Titer_d14": late,
    })


# ------------------------------------------------------- the screen

def test_a_predictive_read_clears_the_floor():
    result = screen_early_reads(screen_frame(), ["VCD_d3", "Via_d3", "Glc_d5"],
                                "Titer_d14")
    found = {r.name: r for r in result.reads}
    assert found["VCD_d3"].above_noise
    assert found["VCD_d3"].p_adjusted < 0.05


def test_reads_that_carry_nothing_are_named_as_carrying_nothing():
    result = screen_early_reads(screen_frame(), ["VCD_d3", "Via_d3", "Glc_d5"],
                                "Titer_d14")
    found = {r.name: r for r in result.reads}
    assert not found["Via_d3"].above_noise and not found["Glc_d5"].above_noise


def test_pure_noise_nominates_nothing():
    """The failure this module exists to prevent: a ranked list always has a
    top entry, including a list of columns that are pure noise."""
    rng = np.random.default_rng(11)
    n = 40
    frame = pd.DataFrame({c: rng.normal(0, 1, n) for c in "ABCD"})
    frame["Y"] = rng.normal(0, 1, n)
    result = screen_early_reads(frame, list("ABCD"), "Y")
    assert not any(r.above_noise for r in result.reads)
    assert "carries evidence" in result.plain


def test_the_false_nomination_rate_is_what_the_correction_promises():
    """Measured over 40 screens of pure noise, four reads each. With
    Benjamini-Hochberg the share of screens nominating anything should sit near
    5%, not the ~19% that four unadjusted tests at 0.05 would give."""
    hits, runs = 0, 20
    for seed in range(runs):
        rng = np.random.default_rng(1000 + seed)
        n = 30
        frame = pd.DataFrame({c: rng.normal(0, 1, n) for c in "ABCD"})
        frame["Y"] = rng.normal(0, 1, n)
        result = screen_early_reads(frame, list("ABCD"), "Y", seed=seed,
                                    noise_runs=40)
        if any(r.above_noise for r in result.reads):
            hits += 1
    assert hits / runs <= 0.15, "nominated noise in %d of %d screens" % (hits, runs)


def test_the_fit_quality_reported_is_cross_validated_not_in_sample():
    """An in-sample R-squared on a dozen clones is near 1 whatever the data
    says, so a negative value here is the honest signal it should be."""
    rng = np.random.default_rng(12)
    n = 30
    frame = pd.DataFrame({c: rng.normal(0, 1, n) for c in "ABC"})
    frame["Y"] = rng.normal(0, 1, n)
    result = screen_early_reads(frame, list("ABC"), "Y")
    assert result.r_squared_cv < 0.2
    assert any("at or below zero" in c or "provisional" in c
               for c in result.cautions)


def test_reads_that_track_each_other_are_flagged_as_splitting_the_credit():
    rng = np.random.default_rng(13)
    n = 40
    late = rng.normal(3.0, 0.5, n)
    shared = 0.7 * late + rng.normal(0, 0.3, n)
    frame = pd.DataFrame({
        "VCD_d3": shared,
        "VCD_d4": shared + rng.normal(0, 0.05, n),      # near-duplicate
        "Glc_d5": rng.normal(4, 0.6, n),
        "Titer_d14": late,
    })
    result = screen_early_reads(frame, ["VCD_d3", "VCD_d4", "Glc_d5"], "Titer_d14")
    assert any("track each other" in c for c in result.cautions)


def test_too_few_clones_is_refused():
    assert screen_early_reads(screen_frame(n=5), ["VCD_d3"], "Titer_d14").failed


def test_benjamini_hochberg_is_monotone_and_never_shrinks_a_p_value():
    raw = np.array([0.001, 0.01, 0.04, 0.2, 0.9])
    adjusted = _benjamini_hochberg(raw)
    assert np.all(adjusted >= raw - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= -1e-12)


# -------------------------------------------------- the generation limit

def passage_frame(slope=-0.009, noise=0.06, top=60, seed=5):
    """Five passage points evenly spaced up to `top`, four lines at each.

    The spacing scales with `top` rather than being hardcoded: an earlier
    version listed 0, 15, 30, 45, top, so asking for top=30 still produced a
    maximum of 45 and the extrapolation cap was computed from the wrong place.
    """
    rng = np.random.default_rng(seed)
    gens = np.repeat(np.linspace(0, top, 5).round().astype(int), 4)
    return pd.DataFrame({"Generation": gens,
                         "Titer": 3.2 + slope * gens + rng.normal(0, noise, gens.size)})


def test_a_real_drift_produces_a_limit():
    result = generation_limit(passage_frame(), "Generation", "Titer", lsl=2.6)
    assert result.limit is not None and result.limit > 0
    assert "generation limit of" in result.plain


def test_the_limit_comes_off_the_confidence_bound_not_the_fitted_line():
    """The fitted line is the best guess at the average, so half of future
    lines sit below it. A limit read off the line would be crossed by half the
    lines before it."""
    result = generation_limit(passage_frame(), "Generation", "Titer", lsl=2.6)
    from_line = (2.6 - result.intercept) / result.slope
    assert result.limit < from_line, (
        "limit %.1f should be earlier than the line's own crossing %.1f"
        % (result.limit, from_line))


def test_no_detectable_drift_refuses_to_set_a_limit():
    """A limit from data with no measured drift rests on the scatter."""
    result = generation_limit(passage_frame(slope=0.0, noise=0.2),
                              "Generation", "Titer", lsl=2.6)
    assert result.limit is None
    assert "cannot be defended" in result.plain


def test_extrapolation_is_capped_at_twice_the_generations_sampled():
    """A limit of 200 generations from a study that reached 30 is arithmetic.

    The drift has to be real but gentle: a slope too small to detect returns
    "no drift" instead, which is a different and also correct answer.
    """
    result = generation_limit(passage_frame(slope=-0.002, noise=0.005, top=30),
                              "Generation", "Titer", lsl=2.0)
    assert result.slope_p < 0.05, "the fixture needs a detectable drift"
    assert result.limit_capped
    assert result.limit == pytest.approx(30 * EXTRAPOLATION_CAP)
    assert "arithmetic rather than evidence" in result.plain


def test_a_limit_beyond_the_data_says_so():
    result = generation_limit(passage_frame(), "Generation", "Titer", lsl=2.6)
    if result.limit and result.limit > result.max_observed:
        assert any("as far as the study went" in c for c in result.cautions)


def test_a_rising_trend_is_judged_against_the_upper_limit():
    result = generation_limit(passage_frame(slope=+0.012), "Generation", "Titer",
                              usl=3.9)
    assert result.direction == "rising"
    assert result.limit is not None


def test_a_specification_limit_is_required():
    assert generation_limit(passage_frame(), "Generation", "Titer").failed


def test_too_few_passage_points_is_refused():
    frame = pd.DataFrame({"Generation": [0, 10], "Titer": [3.0, 2.9]})
    assert generation_limit(frame, "Generation", "Titer", lsl=2.6).failed


def test_two_distinct_generations_is_cautioned_even_when_it_fits():
    frame = pd.DataFrame({"Generation": [0] * 6 + [60] * 6,
                          "Titer": list(np.linspace(3.2, 3.25, 6))
                                   + list(np.linspace(2.6, 2.65, 6))})
    result = generation_limit(frame, "Generation", "Titer", lsl=2.5)
    assert any("distinct generation numbers" in c for c in result.cautions)


# --------------------------------------------------------------------------
# "No evidence" is a conditional verdict, and has to say so

def test_a_read_that_predicts_alone_is_not_called_worthless():
    """The failure this guards against is somebody stopping a measurement.

    Day 7 tracks the final titre at r = 0.6 but adds nothing once Day 14 is in
    the model, so it falls below the floor. Reported as a flat "no evidence",
    that reads as "Day 7 tells you nothing" — and a programme could drop the
    read on the strength of it.
    """
    rng = np.random.default_rng(404)
    n = 48
    quality = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "Day7": 0.45 + 0.08 * quality + rng.normal(0, 0.09, n),
        "Day14": 0.90 + 0.30 * quality + rng.normal(0, 0.10, n),
        "SeedDensity": rng.normal(0.32, 0.03, n),
        "Final": 3.1 + 0.62 * quality + rng.normal(0, 0.22, n),
    })
    result = screen_early_reads(frame, ["Day7", "Day14", "SeedDensity"], "Final")

    by_name = {r.name: r for r in result.reads}
    assert by_name["Day14"].above_noise
    assert not by_name["Day7"].above_noise
    assert abs(by_name["Day7"].correlation) >= MARGINAL_CORRELATION

    caution = next((c for c in result.cautions if "adds nothing once" in c), None)
    assert caution is not None, "a conditionally-null read was reported as flatly null"
    assert "Day7" in caution
    # and the read that genuinely knows nothing is not swept in with it
    assert "SeedDensity" not in caution


def test_a_genuinely_uninformative_read_gets_no_such_excuse():
    """Pure noise correlates with nothing, so no caution should defend it."""
    rng = np.random.default_rng(9)
    n = 40
    quality = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "Day14": 0.9 + 0.30 * quality + rng.normal(0, 0.08, n),
        "Noise1": rng.normal(0, 1, n), "Noise2": rng.normal(0, 1, n),
        "Final": 3.1 + 0.62 * quality + rng.normal(0, 0.20, n),
    })
    result = screen_early_reads(frame, ["Day14", "Noise1", "Noise2"], "Final")
    assert not any("adds nothing once" in c for c in result.cautions)


def test_the_shipped_clone_example_demonstrates_both_traps():
    frame = pd.read_csv("examples/clone-screen.csv")
    early = ["Day7_Titer_g_L", "Day14_VCD_e6_mL", "Day14_Titer_g_L",
             "Seed_Density_e6_mL"]
    result = screen_early_reads(frame, early, "Day21_Final_Titer_g_L")
    assert result.failed is None
    assert any("track each other" in c for c in result.cautions)
    assert any("adds nothing once" in c for c in result.cautions)


def test_the_shipped_passage_example_finds_a_limit():
    frame = pd.read_csv("examples/passage-stability.csv")
    result = generation_limit(frame, "Generation", "Titer_g_L", lsl=2.5)
    assert result.failed is None
    assert result.slope < 0 and result.slope_p < 0.01
    assert result.limit is not None
    # read off the bound, so earlier than where the fitted line crosses
    fitted_crossing = (2.5 - result.intercept) / result.slope
    assert result.limit < fitted_crossing


def test_a_specification_already_breached_is_not_called_a_generation_limit():
    """LSL above everything measured: the answer is "already out", not "zero
    generations", which reads as a calculation rather than a warning."""
    frame = pd.read_csv("examples/passage-stability.csv")
    result = generation_limit(frame, "Generation", "Titer_g_L", lsl=3.5)
    assert result.failed is None
    assert any("already" in c and "not a generation limit" in c
               for c in result.cautions)


def test_a_limit_inside_the_sampled_range_is_not_warned_about():
    """The strongest case there is - the bound crosses within the data - must
    not pick up the already-breached caution."""
    rng = np.random.default_rng(31)
    gens = np.repeat([0, 10, 20, 30, 40, 50], 4)
    values = 3.0 - 0.012 * gens + rng.normal(0, 0.04, gens.size)
    frame = pd.DataFrame({"Generation": gens, "Titer": values})
    result = generation_limit(frame, "Generation", "Titer", lsl=2.6)
    assert result.failed is None
    assert result.limit is not None
    assert not any("already" in c for c in result.cautions)
