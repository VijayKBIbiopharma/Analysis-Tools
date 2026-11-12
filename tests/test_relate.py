"""Several inputs at once.

The tests that matter here are about the two ways this kind of model misleads:
inputs that moved together, and models chosen by searching. Both produce output
that looks entirely reasonable.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from analysis_tools.relate import (  # noqa: E402
    SELECTION_OPTIMISM, VIF_CONCERN, all_possible_models, fit_model,
    nearest_optimism,
)


def frame_with(n=80, seed=1, collinear=False, categorical=False, noise=0.35):
    rng = np.random.default_rng(seed)
    temp = rng.uniform(30, 40, n)
    ph = rng.uniform(6.6, 7.4, n)
    data = {"Temp_C": temp, "pH": ph}
    signal = 0.18 * temp
    if collinear:
        data["Feed"] = 0.5 * temp + rng.normal(0, 0.3, n)
    if categorical:
        promoter = rng.choice(["CMV", "EF1a", "CAG"], n)
        data["Promoter"] = promoter
        signal = signal + 0.9 * (promoter == "CAG")
    data["Titer"] = 2 + signal + rng.normal(0, noise, n)
    return pd.DataFrame(data)


# --------------------------------------------------------------- the fit

def test_it_recovers_a_coefficient_it_was_given():
    frame = frame_with(n=200, seed=2)
    fit = fit_model(frame, "Titer", ["Temp_C", "pH"])
    temp = next(t for t in fit.terms if t.name == "Temp_C")
    assert temp.coefficient == pytest.approx(0.18, abs=0.03)
    assert temp.interval[0] < 0.18 < temp.interval[1]


def test_a_grouping_input_enters_the_same_model_as_a_continuous_one():
    """Multiple regression and the general linear model are the same fit; a
    grouping input simply arrives as indicator terms."""
    frame = frame_with(n=120, seed=3, categorical=True)
    fit = fit_model(frame, "Titer", ["Temp_C", "Promoter"], categorical=["Promoter"])
    names = [t.name for t in fit.terms]
    assert "Temp_C" in names
    assert any("Promoter" in n for n in names)
    assert fit.anova is not None and "Promoter" in list(fit.anova.index)


def test_the_anova_table_and_r_squared_come_from_statsmodels():
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    frame = frame_with(n=90, seed=4)
    fit = fit_model(frame, "Titer", ["Temp_C", "pH"])
    direct = ols('Q("Titer") ~ Q("Temp_C") + Q("pH")', data=frame).fit()
    assert fit.r_squared_adj == pytest.approx(direct.rsquared_adj)
    assert fit.bic == pytest.approx(direct.bic)
    assert fit.sigma == pytest.approx(np.sqrt(direct.mse_resid))


def test_too_few_rows_for_the_terms_is_refused_with_the_reason():
    frame = frame_with(n=5, seed=5, categorical=True)
    fit = fit_model(frame, "Titer", ["Temp_C", "pH", "Promoter"],
                    categorical=["Promoter"])
    assert fit.failed and "more observations than it has terms" in fit.failed


# ------------------------------------------------ collinearity, which hides

def test_variance_inflation_catches_what_a_correlation_matrix_does_not():
    """An input that is a combination of others can be badly inflated while no
    single pair of columns looks remarkable. This is the case that makes VIF
    worth computing rather than eyeballing a correlation matrix."""
    rng = np.random.default_rng(6)
    n = 120
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    c = a + b + rng.normal(0, 0.25, n)            # a combination, not a pair
    frame = pd.DataFrame({"A": a, "B": b, "C": c,
                          "Y": 1.5 * a + rng.normal(0, 1, n)})

    correlations = frame[["A", "B", "C"]].corr().abs()
    np.fill_diagonal(correlations.values, 0)
    largest_pair = correlations.values.max()

    fit = fit_model(frame, "Y", ["A", "B", "C"])
    worst_vif = max(t.vif for t in fit.terms if t.vif is not None)

    assert largest_pair < 0.8, "no pair looks alarming: %.3f" % largest_pair
    assert worst_vif > 10, "yet inflation is %.1f" % worst_vif


def test_collinearity_is_reported_and_says_more_data_will_not_fix_it():
    frame = frame_with(n=100, seed=7, collinear=True)
    fit = fit_model(frame, "Titer", ["Temp_C", "pH", "Feed"])
    assert any("moved together" in c for c in fit.cautions)
    assert any("needs an experiment" in c for c in fit.cautions)


def test_independent_inputs_raise_no_collinearity_warning():
    frame = frame_with(n=100, seed=8)
    fit = fit_model(frame, "Titer", ["Temp_C", "pH"])
    assert all(t.vif is None or t.vif < VIF_CONCERN for t in fit.terms)


# ------------------------------------------------------- selection, which lies

def test_the_search_recovers_the_model_that_generated_the_data():
    """With no proxy in the candidate set, the search should find the truth."""
    frame = frame_with(n=120, seed=9, categorical=True)
    search = all_possible_models(frame, "Titer", ["Temp_C", "pH", "Promoter"],
                                 categorical=["Promoter"])
    assert search is not None
    chosen = search.best_by_bic[0]
    assert "Temp_C" in chosen and "Promoter" in chosen
    assert "pH" not in chosen


def test_with_a_proxy_present_the_search_may_keep_the_proxy_and_must_say_so():
    """Feed is built as 0.5 x Temp_C plus noise, so the two carry nearly the
    same information and which one a subset search keeps is close to a coin
    toss. That is not a defect in the search - it is a fact about the data -
    but a report that names one as a driver and omits the other, with no
    indication that they are interchangeable, misleads the reader."""
    frame = frame_with(n=120, seed=9, collinear=True, categorical=True)
    search = all_possible_models(frame, "Titer",
                                 ["Temp_C", "pH", "Feed", "Promoter"],
                                 categorical=["Promoter"])
    chosen = search.best_by_bic[0]
    assert ("Temp_C" in chosen) or ("Feed" in chosen)
    assert search.substitutes, "the interchangeable pair must be reported"
    assert "close to arbitrary" in search.substitutes[0]


def test_no_substitute_warning_when_the_inputs_are_independent():
    frame = frame_with(n=120, seed=21, categorical=True)
    search = all_possible_models(frame, "Titer", ["Temp_C", "pH", "Promoter"],
                                 categorical=["Promoter"])
    assert not search.substitutes


def test_mallows_cp_is_near_the_term_count_for_an_adequate_model():
    """Cp far above p means the model is missing something; that is the whole
    use of the statistic, and it is hidden behind a right-click in JMP."""
    frame = frame_with(n=150, seed=10, categorical=True)
    search = all_possible_models(frame, "Titer", ["Temp_C", "pH", "Promoter"],
                                 categorical=["Promoter"])
    best = search.results.iloc[0]
    assert abs(best["Mallows' Cp"] - (best["Count"] + 1)) < 4


def test_the_search_is_ranked_by_bic_and_says_when_aicc_disagrees():
    frame = frame_with(n=60, seed=11, collinear=True, categorical=True)
    search = all_possible_models(frame, "Titer",
                                 ["Temp_C", "pH", "Feed", "Promoter"],
                                 categorical=["Promoter"])
    assert search.results["BIC"].is_monotonic_increasing
    if search.best_by_bic[0] != search.best_by_aicc[0]:
        assert "BIC prefers the smaller model" in search.plain


def test_the_optimism_warning_carries_a_measured_number_not_an_adjective():
    frame = frame_with(n=60, seed=12, collinear=True)
    search = all_possible_models(frame, "Titer", ["Temp_C", "pH", "Feed"])
    assert "pure noise" in search.optimism
    assert "%" in search.optimism and "adjusted R²" in search.optimism


def test_the_quoted_optimism_figures_are_still_true():
    """The page tells users that searching subsets of pure noise yields a
    significant-looking term a measured fraction of the time. If the search or
    the fitting changes, that figure has to be re-measured — which is what this
    forces. Checked on one cell to keep the suite quick."""
    rng = np.random.default_rng(13)
    n, k = 30, 4
    _, (_, claimed_sig) = nearest_optimism(n, k)
    hits, runs = 0, 60
    for _ in range(runs):
        columns = {"X%d" % i: rng.normal(0, 1, n) for i in range(k)}
        frame = pd.DataFrame({**columns, "Y": rng.normal(0, 1, n)})
        search = all_possible_models(frame, "Y", list(columns))
        if search is None:
            continue
        terms = [t.strip() for t in search.best_by_bic[0].split(",")]
        fit = fit_model(frame, "Y", terms)
        if any(t.p_value is not None and t.p_value < 0.05
               for t in fit.terms if t.name != "(intercept)"):
            hits += 1
    measured = hits / runs
    assert abs(measured - claimed_sig) < 0.18, (
        "page claims %.0f%%, measured %.0f%%" % (100 * claimed_sig, 100 * measured))


def test_every_simulated_optimism_figure_is_worse_than_honest():
    """Sanity on the stored table: the honest answer for noise is zero on both
    counts, so every entry should exceed it."""
    for (rows, inputs), (r2, sig) in SELECTION_OPTIMISM.items():
        assert r2 > 0 and sig > 0.05, (rows, inputs)


# ---------------------------------------------------------------- it narrates

def test_an_unsupported_model_says_so_before_listing_its_terms():
    rng = np.random.default_rng(14)
    frame = pd.DataFrame({"A": rng.normal(0, 1, 40), "B": rng.normal(0, 1, 40),
                          "Y": rng.normal(0, 1, 40)})
    fit = fit_model(frame, "Y", ["A", "B"])
    if fit.f_p_value >= 0.05:
        assert "does not explain" in fit.plain
        assert "should not be read as findings" in fit.plain
