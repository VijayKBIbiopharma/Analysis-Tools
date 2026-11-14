"""The analysis-path matrix.

These tests are deliberately written as the chart, cell by cell, so that
somebody who already knows the chart can check this file against it without
reading the implementation. Each test names the cell it covers.

The source is the Six Sigma project guide's Analyze Path sheets: one X at a
time, and several X's at once, crossed with the response type.
"""

import pandas as pd

from analysis_tools.roles import classify_column
from analysis_tools.routing import (
    BROWN_FORSYTHE, CHI_SQUARE, GLM, LOGISTIC,
    MULTIPLE_REGRESSION, ONE_WAY_ANOVA, PAIRED_T, REGRESSION, TESTS,
    TWO_PROPORTIONS, TWO_SAMPLE_T, plan_analysis,
)

# Columns of each kind, built once and reused so the tests read as the chart.
VARIABLE = classify_column("Titer", pd.Series([2.1, 2.4, 2.9, 3.1, 2.2, 2.8, 3.3, 2.65]))
VARIABLE_2 = classify_column("Feed", pd.Series([8.1, 9.4, 7.9, 10.1, 9.2, 8.8, 7.3, 9.65]))
ATTR2 = classify_column("Line", pd.Series(["A", "B"] * 8))
ATTR2_B = classify_column("Shift", pd.Series(["Day", "Night"] * 8))
ATTR3 = classify_column("Promoter", pd.Series(["CMV", "EF1a", "CAG"] * 5))
ATTR3_B = classify_column("Marker", pd.Series(["Puro", "Hygro", "Blast"] * 5))
Y_ATTR2 = classify_column("Passed", pd.Series(["Y", "N"] * 8))
Y_ATTR3 = classify_column("Grade", pd.Series(["Low", "Mid", "High"] * 5))


def route(xs, y, **kwargs):
    return plan_analysis(xs if isinstance(xs, list) else [xs], y, **kwargs)


def _chosen(plan):
    return [s.test for s in plan.steps]


# ====================================================== one X, response Variables

def test_variables_x_with_variables_y_is_regression():
    plan = route(VARIABLE_2, VARIABLE)
    assert _chosen(plan) == [REGRESSION]
    assert "R-squared" in plan.impact


def test_two_level_attribute_with_variables_y_is_t_test_and_levene():
    """The chart gives two tests here, not one: a t-test for the centre and
    Levene's for the spread."""
    plan = route(ATTR2, VARIABLE)
    assert _chosen(plan) == [TWO_SAMPLE_T, BROWN_FORSYTHE]
    assert plan.centering.test == TWO_SAMPLE_T
    assert plan.spread.test == BROWN_FORSYTHE


def test_paired_data_switches_to_the_paired_t_and_drops_the_spread_test():
    """Paired data has two observations on each unit, so the comparison is of
    the differences and the chart offers no spread test."""
    plan = route(ATTR2, VARIABLE, paired=True)
    assert _chosen(plan) == [PAIRED_T]
    assert plan.spread is None
    assert any("differences" in n for n in plan.notes)


def test_three_level_attribute_with_variables_y_is_anova_and_levene():
    plan = route(ATTR3, VARIABLE)
    assert _chosen(plan) == [ONE_WAY_ANOVA, BROWN_FORSYTHE]


def test_the_anova_cell_separates_the_two_problems_the_chart_folds_together():
    """The chart says "if non-normal OR unequal variances use Kruskal-Wallis".
    Those are two different problems: unequal variances call for Welch ANOVA,
    which still compares averages, and non-normality calls for Kruskal-Wallis,
    which compares ranks. JMP separates them and so does this."""
    plan = route(ATTR3, VARIABLE)
    condition = plan.centering.conditional_on
    assert "Welch" in condition and "Kruskal-Wallis" in condition
    assert "ranks" in condition


# ===================================================== one X, response Attribute

def test_variables_x_with_two_level_y_is_logistic_regression():
    assert _chosen(route(VARIABLE_2, Y_ATTR2)) == [LOGISTIC]


def test_variables_x_with_three_level_y_is_logistic_regression():
    assert _chosen(route(VARIABLE_2, Y_ATTR3)) == [LOGISTIC]


def test_two_by_two_attributes_get_chi_square_and_the_proportions_test():
    """Two levels each is the one cell where the chart offers a second test that
    gives a difference and an interval, not just a p-value."""
    plan = route(ATTR2, Y_ATTR2)
    assert _chosen(plan) == [CHI_SQUARE, TWO_PROPORTIONS]


def test_three_level_attributes_get_chi_square_for_independence():
    plan = route(ATTR3, Y_ATTR3)
    assert _chosen(plan) == [CHI_SQUARE]
    assert "independent" in plan.steps[0].question


def test_no_impact_statistic_is_offered_for_a_categorical_response():
    """The chart says "No Tool Available" here. Offering R-squared anyway would
    be inventing a number people would read as variance explained."""
    plan = route(VARIABLE_2, Y_ATTR2)
    assert plan.impact is None
    assert any("no agreed measure" in n.lower() for n in plan.notes)


# =========================================================== several X's at once

def test_several_continuous_inputs_is_multiple_regression():
    plan = route([VARIABLE_2, classify_column("pH", pd.Series([6.8, 7.1, 6.9, 7.4, 7.0, 6.75, 7.2, 6.95]))],
                 VARIABLE)
    assert _chosen(plan) == [MULTIPLE_REGRESSION]
    assert "subsets" in " ".join(plan.notes).lower()


def test_several_grouping_inputs_is_a_general_linear_model():
    assert _chosen(route([ATTR3, ATTR2], VARIABLE)) == [GLM]


def test_mixing_continuous_and_grouping_inputs_is_a_general_linear_model():
    assert _chosen(route([VARIABLE_2, ATTR3], VARIABLE)) == [GLM]


def test_several_inputs_against_a_categorical_response_is_logistic_regression():
    assert _chosen(route([VARIABLE_2, ATTR3], Y_ATTR2)) == [LOGISTIC]


def test_several_inputs_report_both_r_squared_and_s():
    plan = route([ATTR3, ATTR2], VARIABLE)
    assert "R-squared" in plan.impact and "residuals" in plan.impact


# ================================================================== it refuses

def test_no_inputs_is_refused():
    plan = plan_analysis([], VARIABLE)
    assert not plan.runnable and "at least one input" in plan.blockers[0]


def test_a_column_cannot_explain_itself():
    plan = route(VARIABLE, VARIABLE)
    assert not plan.runnable
    assert "cannot explain itself" in plan.blockers[0]


def test_an_identifier_cannot_take_part():
    ident = classify_column("Run_ID", pd.Series(range(1, 21)))
    plan = route(ident, VARIABLE)
    assert not plan.runnable
    assert "identifier" in plan.blockers[0].lower()


def test_a_constant_cannot_take_part():
    const = classify_column("Site", pd.Series(["Boulder"] * 12))
    plan = route(const, VARIABLE)
    assert not plan.runnable


def test_a_group_of_one_is_excluded_from_the_spread_test_not_blocked():
    """One observation has no spread to contribute, so it drops out of the
    comparison of spreads. JMP 19.1 does exactly this. Blocking the whole
    analysis instead would refuse an ANOVA that is perfectly computable."""
    lopsided = classify_column("Resin", pd.Series(["A"] * 9 + ["B"] * 9 + ["C"]))
    plan = route(lopsided, VARIABLE)
    assert plan.runnable
    assert any("left out of the comparison of spreads" in c for c in plan.cautions)


# ================================================================ it cautions

def test_a_thin_level_is_a_caution_not_a_blocker():
    thin = classify_column("Resin", pd.Series(["A"] * 9 + ["B"] * 9 + ["C"] * 3))
    plan = route(thin, VARIABLE)
    assert plan.runnable
    assert any("rests on 3 observations" in c for c in plan.cautions)


def test_an_ambiguous_column_carries_its_ambiguity_into_the_plan():
    """The Explore page flags that a numeric column at three settings can be
    read two ways. The plan has to carry it, or the user is told once and then
    shown a result that depends on it."""
    temp = classify_column("Temp_C", pd.Series([30, 35, 40] * 4))
    plan = route(temp, VARIABLE)
    assert any("read the other way" in c.lower() for c in plan.cautions)


def test_too_few_rows_for_the_number_of_inputs_is_said_out_loud():
    plan = route([VARIABLE_2, ATTR3], VARIABLE, n_rows=6)
    assert any("little left to learn" in c for c in plan.cautions)


# ============================================================ reproducibility

def test_every_test_carries_the_jmp_path_that_reproduces_it():
    """A result nobody can check independently is a result nobody has to
    believe. Every test names where the same number lives in JMP 19."""
    for key, (name, path) in TESTS.items():
        assert name and path.startswith("Analyze >"), key


def test_the_variance_test_points_at_the_brown_forsythe_row_not_levene():
    """JMP's Unequal Variances report shows both, and they differ: Levene
    centres on the mean, Brown-Forsythe on the median. This tool uses the
    median, so sending a user to the Levene row would show them a different
    number and cost exactly the trust the reference is meant to build."""
    name, path = TESTS[BROWN_FORSYTHE]
    assert "Brown-Forsythe" in name and "Brown-Forsythe row" in path


def test_the_plan_names_the_columns_it_was_built_for():
    plan = route([ATTR3, ATTR2], VARIABLE)
    assert plan.x_names == ["Promoter", "Line"] and plan.y_name == "Titer"
