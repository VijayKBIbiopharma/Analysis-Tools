"""Designs, and the analysis of them.

Two things get checked hardest. The defining relation of every tabulated
design, because the generators are a lookup table and a wrong entry would
silently produce a design whose confounding is not what it says. And the
false-positive rate of the effects analysis, because a screening design that
nominates factors which do nothing sends people to the pilot plant for weeks.
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from analysis_tools.doe import (
    GENERATORS, MIN_ERROR_DF, analyse_design, build_design, lenth_margin,
)

FACTORS = [{"name": n, "low": lo, "high": hi} for n, lo, hi in [
    ("Temp", 32, 38), ("pH", 6.8, 7.2), ("Feed", 8, 12),
    ("Seed", 0.3, 0.7), ("DO", 30, 50), ("Agit", 100, 200),
    ("Time", 10, 14), ("Osmo", 280, 320),
]]


def design_for(k, runs=None, **kwargs):
    return build_design(FACTORS[:k], runs=runs, **kwargs)


# ------------------------------------------------------- the design itself

def test_a_full_factorial_confounds_nothing():
    design = design_for(3)
    assert design.n_runs == 8
    assert design.resolution is None and not design.aliases
    assert "nothing is confounded" in design.plain


@pytest.mark.parametrize("k,p,expected_resolution", [
    (3, 1, 3), (4, 1, 4), (5, 1, 5), (5, 2, 3), (6, 1, 6), (6, 2, 4),
    (6, 3, 3), (7, 1, 7), (7, 2, 4), (7, 3, 4), (7, 4, 3), (8, 2, 5),
    (8, 3, 4), (8, 4, 4),
])
def test_every_tabulated_design_has_the_resolution_it_claims(k, p, expected_resolution):
    """The generators are a lookup table; the resolution is computed from the
    defining relation they produce. If a table entry were wrong the two would
    disagree, which is the point of computing it."""
    design = design_for(k, runs=2 ** (k - p))
    assert design.resolution == expected_resolution, (
        "2^(%d-%d): claimed %s, computed %s"
        % (k, p, expected_resolution, design.resolution))


def test_the_defining_relation_has_the_right_number_of_words():
    """A design with p generators has 2^p - 1 words in its defining relation."""
    for (k, p) in GENERATORS:
        if k > 8:
            continue
        design = design_for(k, runs=2 ** (k - p))
        assert len(design.defining_relation) == 2 ** p - 1, (k, p)


def test_resolution_three_confounds_main_effects_with_interactions_and_says_so():
    design = design_for(5, runs=8)
    assert design.resolution == 3
    assert any(len(partner) == 2 for partner in design.aliases["A"])
    assert any("confounded with two-factor interactions" in c
               for c in design.cautions)


def test_resolution_five_keeps_two_factor_interactions_clear_of_each_other():
    design = design_for(5, runs=16)
    assert design.resolution == 5
    for term, partners in design.aliases.items():
        if len(term) == 2:
            assert all(len(p) >= 3 for p in partners), (term, partners)


def test_every_design_column_is_orthogonal_to_every_other():
    """Orthogonality is what makes the effects separable. If it fails the
    design is not the design it claims to be."""
    for k, runs in [(3, None), (4, 8), (5, 16), (6, 16), (7, 16)]:
        design = design_for(k, runs=runs)
        names = [f["name"] for f in FACTORS[:k]]
        coded = design.runs[names].apply(
            lambda col: pd.to_numeric(col) - pd.to_numeric(col).mean())
        for a, b in itertools.combinations(names, 2):
            assert abs(float((coded[a] * coded[b]).sum())) < 1e-9, (k, a, b)


def test_the_run_order_is_randomised_but_the_standard_order_is_kept():
    design = design_for(4)
    assert sorted(design.runs["Run order"]) == list(range(1, design.n_runs + 1))
    assert sorted(design.runs["Std order"]) == list(range(1, design.n_runs + 1))
    assert not design.runs["Run order"].equals(design.runs["Std order"])


def test_the_same_seed_gives_the_same_run_order():
    assert design_for(4, seed=7).runs.equals(design_for(4, seed=7).runs)


def test_centre_points_are_added_and_are_what_makes_curvature_visible():
    design = design_for(3, centre_points=3)
    assert design.n_runs == 11
    assert "curvature detectable" in design.plain


def test_centre_points_are_refused_for_a_categorical_factor_rather_than_faked():
    factors = [{"name": "Temp", "low": 32, "high": 38},
               {"name": "Resin", "low": "A", "high": "B"}]
    design = build_design(factors, centre_points=3)
    assert design.centre_points == 0
    assert any("no middle setting" in c for c in design.cautions)


def test_a_design_with_no_centre_points_says_curvature_is_invisible():
    design = design_for(3)
    assert any("optimum sitting inside the range is invisible" in c
               for c in design.cautions)


# ---------------------------------------------------------- the analysis

def run_with_truth(k=3, centre_points=4, replicates=2, seed=11,
                   temp=1.0, feed=0.5, interaction=0.4, curve=0.0, noise=0.12):
    design = build_design(FACTORS[:k], centre_points=centre_points,
                          replicates=replicates, seed=3)
    runs = design.runs.copy()
    rng = np.random.default_rng(seed)
    t = (pd.to_numeric(runs["Temp"]) - 35) / 3.0
    f = (pd.to_numeric(runs["Feed"]) - 10) / 2.0
    runs["Titer"] = (10 + temp * t + feed * f + interaction * t * f
                     - curve * t ** 2 + rng.normal(0, noise, len(runs)))
    return runs


def test_it_recovers_the_effects_it_was_given():
    """An effect is the change from the low setting to the high one, which is
    twice the coefficient on the coded scale."""
    runs = run_with_truth(temp=1.0, feed=0.5, interaction=0.4)
    analysis = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    found = {e.term: e.effect for e in analysis.effects}
    assert found["Temp"] == pytest.approx(2.0, abs=0.15)
    assert found["Feed"] == pytest.approx(1.0, abs=0.15)
    assert found["Temp x Feed"] == pytest.approx(0.8, abs=0.15)


def test_a_factor_with_no_effect_is_not_nominated():
    runs = run_with_truth()
    analysis = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    ph = next(e for e in analysis.effects if e.term == "pH")
    assert not ph.significant


def test_curvature_is_detected_when_it_is_there():
    runs = run_with_truth(curve=1.2)
    analysis = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    assert analysis.curvature_p is not None and analysis.curvature_p < 0.05
    assert any("response is curved" in c for c in analysis.cautions)


def test_curvature_is_not_claimed_when_the_response_is_straight():
    runs = run_with_truth(curve=0.0, noise=0.1)
    analysis = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    assert analysis.curvature_p is None or analysis.curvature_p > 0.05


def test_lenth_takes_over_when_the_error_estimate_rests_on_almost_nothing():
    runs = run_with_truth(replicates=1, centre_points=0)
    analysis = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    assert analysis.df_error < MIN_ERROR_DF
    assert analysis.used_lenth and analysis.lenth_margin > 0
    assert all(e.p_value is None for e in analysis.effects)


def test_switching_to_lenth_actually_reduces_false_nominations():
    """The reason the switch exists, measured rather than argued. On a design
    with one residual degree of freedom and no real effects at all, ordinary
    p-values nominate factors that do nothing."""
    rng = np.random.default_rng(5)
    design = build_design(FACTORS[:3], seed=3)
    naive_hits = lenth_hits = 0
    runs_count = 150
    for _ in range(runs_count):
        frame = design.runs.copy()
        frame["Titer"] = rng.normal(10, 0.2, len(frame))   # pure noise
        analysis = analyse_design(frame, ["Temp", "pH", "Feed"], "Titer")
        if any(e.significant for e in analysis.effects):
            lenth_hits += 1
        # the same fit, judged the old way
        import statsmodels.api as sm
        coded = pd.DataFrame({
            n: (pd.to_numeric(frame[n]) - pd.to_numeric(frame[n]).mean())
               / (pd.to_numeric(frame[n]).max() - pd.to_numeric(frame[n]).mean())
            for n in ["Temp", "pH", "Feed"]})
        for a, b in itertools.combinations(["Temp", "pH", "Feed"], 2):
            coded["%s x %s" % (a, b)] = coded[a] * coded[b]
        model = sm.OLS(frame["Titer"].to_numpy(float),
                       sm.add_constant(coded.to_numpy(float))).fit()
        if any(p < 0.05 for p in model.pvalues[1:]):
            naive_hits += 1
    naive = naive_hits / runs_count
    lenth = lenth_hits / runs_count
    assert lenth < naive, "Lenth %.0f%% vs naive %.0f%%" % (100 * lenth, 100 * naive)
    assert lenth < 0.25, "Lenth still nominates noise %.0f%% of the time" % (100 * lenth)


def test_lenth_says_it_is_conservative_and_names_the_two_cheap_fixes():
    runs = run_with_truth(replicates=1, centre_points=0)
    analysis = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    advice = " ".join(analysis.cautions)
    assert "conservative" in advice
    assert "untick interactions" in advice and "centre points" in advice


def test_the_recommended_fix_actually_works():
    """The tool tells users that dropping interactions buys error degrees of
    freedom. It should be true."""
    runs = run_with_truth(replicates=1, centre_points=0)
    with_terms = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    without = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer",
                             interactions=False)
    assert without.df_error > with_terms.df_error
    assert not without.used_lenth


def test_a_three_level_factor_is_refused_rather_than_mangled():
    frame = pd.DataFrame({"A": ["lo", "mid", "hi"] * 6,
                          "B": ["lo", "hi"] * 9,
                          "Y": np.random.default_rng(1).normal(10, 1, 18)})
    analysis = analyse_design(frame, ["A", "B"], "Y")
    assert analysis.failed and "two-level factors" in analysis.failed


def test_lenth_margin_is_none_when_there_is_nothing_to_estimate_from():
    assert lenth_margin([1.0, 2.0]) == (None, None)


def test_every_generator_word_uses_only_the_base_design_letters():
    """A (k, p) design builds its base from the first k - p factors and defines
    the remaining p from those. A generator word referring to a factor that is
    itself being generated is circular, and that is exactly the error the first
    version of the table had."""
    from analysis_tools.doe import LETTERS
    for (k, p), words in GENERATORS.items():
        base = set(LETTERS[:k - p])
        assert len(words) == p, "(%d,%d) has %d words for %d generated factors" % (
            k, p, len(words), p)
        for word in words:
            stray = set(word) - base
            assert not stray, (
                "(%d,%d) generator %r uses %s, which is not in the base design %s"
                % (k, p, word, "".join(sorted(stray)), "".join(sorted(base))))


def test_the_run_budget_is_honoured_even_when_no_half_fraction_is_tabulated():
    """Eight factors has no tabulated half-fraction here, so a naive search
    gives up and returns a 256-run full factorial without saying so."""
    design = design_for(8, runs=64)
    assert design.n_runs == 64
    assert design.resolution == 5


def test_an_unconstrained_request_gives_the_full_factorial():
    """Asking for no particular run count must not quietly buy a fraction.
    A fix for the run-budget search once did exactly that: every design became
    a half fraction, with its confounding reported correctly and nobody having
    asked for it."""
    for k in (3, 4, 5):
        design = design_for(k)
        assert design.n_runs == 2 ** k, k
        assert design.resolution is None, k


def test_a_budget_larger_than_the_full_design_still_gives_the_full_design():
    design = design_for(3, runs=64)
    assert design.n_runs == 8 and design.resolution is None


def test_the_curvature_test_finds_curvature_it_should_comfortably_find():
    """With sixteen factorial runs, four centre points and noise far smaller
    than the curvature, this must be detected. An earlier version used the
    variance of the factorial responses as the error term - but those vary
    because the factors were deliberately moved, so the yardstick was mostly
    signal and the test missed exactly the case centre points exist for."""
    runs = run_with_truth(replicates=2, centre_points=4, curve=0.18, noise=0.035)
    analysis = analyse_design(runs, ["Temp", "pH", "Feed"], "Titer")
    assert analysis.curvature_p is not None
    assert analysis.curvature_p < 0.01, (
        "curvature of 0.18 against noise of 0.035 came out at p = %.4f"
        % analysis.curvature_p)


def test_the_curvature_test_does_not_cry_wolf_on_a_flat_response():
    """Measured, because a test made more sensitive has to be checked at the
    other end too."""
    rng = np.random.default_rng(31)
    design = build_design(FACTORS[:3], replicates=2, centre_points=4, seed=3)
    false_alarms, runs_count = 0, 200
    for _ in range(runs_count):
        frame = design.runs.copy()
        t = (pd.to_numeric(frame["Temp"]) - 35) / 3.0
        frame["Titer"] = 10 + 1.0 * t + rng.normal(0, 0.1, len(frame))
        analysis = analyse_design(frame, ["Temp", "pH", "Feed"], "Titer")
        if analysis.curvature_p is not None and analysis.curvature_p < 0.05:
            false_alarms += 1
    rate = false_alarms / runs_count
    assert rate < 0.10, "flags curvature on flat data %.0f%% of the time" % (100 * rate)
