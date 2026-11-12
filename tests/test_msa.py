"""Gage R&R.

The ANOVA table is checked against statsmodels, which is an independent
implementation of the same decomposition. The variance components are checked
by simulation: build a measurement system whose repeatability, operator spread
and part spread are known, and see whether the study recovers them.
"""

import numpy as np
import pandas as pd
import pytest

from analysis_tools.msa import MIN_DISTINCT_CATEGORIES, gauge_rr


def study(n_parts=10, n_ops=3, n_reps=3, part_sd=5.0, op_sd=0.5,
          repeat_sd=1.0, seed=1, interaction_sd=0.0):
    rng = np.random.default_rng(seed)
    part_true = rng.normal(100, part_sd, n_parts)
    op_bias = rng.normal(0, op_sd, n_ops) if op_sd else np.zeros(n_ops)
    rows = []
    for i in range(n_parts):
        for j in range(n_ops):
            cell = rng.normal(0, interaction_sd) if interaction_sd else 0.0
            for _ in range(n_reps):
                rows.append({
                    "Part": "P%02d" % i, "Operator": "Op%d" % j,
                    "Reading": part_true[i] + op_bias[j] + cell
                               + rng.normal(0, repeat_sd),
                })
    return pd.DataFrame(rows)


def test_the_anova_table_matches_statsmodels():
    """An independent implementation of the same two-way decomposition."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    frame = study(seed=2)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    model = ols("Reading ~ C(Part) * C(Operator)", data=frame).fit()
    table = sm.stats.anova_lm(model, typ=2)

    assert result.anova["Part"]["SS"] == pytest.approx(table.loc["C(Part)", "sum_sq"])
    assert result.anova["Operator"]["SS"] == pytest.approx(table.loc["C(Operator)", "sum_sq"])
    assert result.anova["Part x Operator"]["SS"] == pytest.approx(
        table.loc["C(Part):C(Operator)", "sum_sq"])
    assert result.anova["Repeatability"]["SS"] == pytest.approx(table.loc["Residual", "sum_sq"])


def test_it_recovers_a_measurement_system_whose_truth_is_known():
    """Repeatability sd 1.0 against part sd 5.0 should come out near 1.0, and
    the Gage R&R percentage near 100*1/sqrt(1+25) = 19.6%."""
    frame = study(n_parts=30, n_reps=4, part_sd=5.0, op_sd=0.0,
                  repeat_sd=1.0, seed=3)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert np.sqrt(result.variances["Repeatability"]) == pytest.approx(1.0, rel=0.12)
    assert result.grr_pct == pytest.approx(100 / np.sqrt(26), rel=0.20)


def test_an_excellent_gauge_reads_as_acceptable():
    frame = study(part_sd=10.0, op_sd=0.05, repeat_sd=0.3, seed=4)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert result.verdict == "acceptable"
    assert result.grr_pct < 10


def test_a_useless_gauge_reads_as_unacceptable_and_says_what_dominates():
    frame = study(part_sd=1.0, op_sd=0.2, repeat_sd=3.0, seed=5)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert result.verdict == "unacceptable"
    assert "instrument" in result.plain


def test_operator_differences_show_up_as_reproducibility():
    frame = study(part_sd=3.0, op_sd=4.0, repeat_sd=0.4, seed=6)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert result.study_var_pct["Reproducibility"] > result.study_var_pct["Repeatability"]
    assert "difference between operators" in result.plain


def test_an_interaction_is_kept_when_it_is_real():
    """An operator who reads differently on different parts."""
    frame = study(part_sd=2.0, op_sd=0.1, repeat_sd=0.3,
                  interaction_sd=2.0, n_reps=4, seed=7)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert not result.interaction_pooled
    assert result.variances["  Operator x Part"] > 0


def test_an_absent_interaction_is_pooled_and_the_pooling_is_reported():
    frame = study(interaction_sd=0.0, seed=8)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert result.interaction_pooled
    assert any("pooled back into repeatability" in c for c in result.cautions)


def test_distinct_categories_are_reported_and_flagged_when_too_few():
    frame = study(part_sd=0.8, repeat_sd=1.2, seed=9)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert result.distinct_categories is not None
    if result.distinct_categories < MIN_DISTINCT_CATEGORIES:
        assert "cannot support a control chart" in result.plain


def test_percent_of_tolerance_is_reported_when_limits_are_given():
    frame = study(seed=10)
    result = gauge_rr(frame, "Part", "Operator", "Reading", lsl=80, usl=120)
    assert result.tolerance_pct is not None
    assert result.tolerance_pct["Total Gage R&R"] > 0


# -------------------------------------------------------------- it refuses

def test_an_unbalanced_study_is_refused_with_the_reason():
    frame = study(seed=11)
    result = gauge_rr(frame.iloc[:-3], "Part", "Operator", "Reading")
    assert "unbalanced" in result.plain or "never measured" in result.plain


def test_a_study_with_no_repeats_is_refused():
    frame = study(n_reps=1, seed=12)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert "needs repeats" in result.plain


def test_one_operator_is_refused():
    frame = study(n_ops=1, seed=13)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert "at least two" in result.plain


def test_a_missing_combination_is_named_rather_than_averaged_over():
    frame = study(seed=14)
    dropped = frame[~((frame["Part"] == "P00") & (frame["Operator"] == "Op1"))]
    result = gauge_rr(dropped, "Part", "Operator", "Reading")
    assert "never measured" in result.plain


def test_too_few_parts_is_a_caution_not_a_refusal():
    frame = study(n_parts=4, seed=15)
    result = gauge_rr(frame, "Part", "Operator", "Reading")
    assert result.grr_pct is not None
    assert any("AIAG asks for ten" in c for c in result.cautions)
