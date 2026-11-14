"""Column classification.

This is the module everything else turns on: the role a column is given decides
which statistical test the tool will run on it. So these tests are not about
tidy output, they are about whether the right test would be chosen.
"""

import pandas as pd
import pytest

from analysis_tools.roles import (
    ATTRIBUTE_2, ATTRIBUTE_3PLUS, CONSTANT, DATETIME, EMPTY, IDENTIFIER,
    VARIABLES, classify_column, classify_table,
)


def col(values, name="x"):
    return classify_column(name, pd.Series(values))


# ------------------------------------------------------------------ the roles

def test_continuous_measurements_are_variables():
    assert col([2.1, 2.4, 2.9, 3.1, 2.2, 2.8, 3.3, 2.6]).role == VARIABLES


def test_two_text_levels_is_a_two_level_attribute():
    c = col(["pass", "fail", "pass", "pass", "fail"])
    assert c.role == ATTRIBUTE_2
    assert sorted(c.levels) == ["fail", "pass"]
    assert c.counts == {"pass": 3, "fail": 2}


def test_three_text_levels_is_a_three_plus_attribute():
    assert col(["CMV", "EF1a", "CAG", "CMV"]).role == ATTRIBUTE_3PLUS


def test_a_single_value_is_constant_and_says_why_it_is_useless():
    c = col(["Boulder"] * 10)
    assert c.role == CONSTANT
    assert "cannot explain" in " ".join(c.notes)


def test_all_blank_is_empty():
    assert col([None, None, None]).role == EMPTY


@pytest.mark.parametrize("token", ["", " ", "NA", "n/a", "NaN", "NULL", "-", "?", "#N/A", "missing"])
def test_the_tokens_that_mean_nothing_was_recorded(token):
    """A column of five numbers and five of these is a five-row numeric column,
    not a ten-level attribute."""
    c = col([1.5, 2.5, 3.5, 4.5, 5.5, token, token, token, token, token])
    assert c.role == VARIABLES
    assert c.n_missing == 5
    assert c.n_present == 5


def test_zero_is_a_measurement_not_a_missing_value():
    c = col([0, 0, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    assert c.n_missing == 0


# ------------------------------------------------------- identifiers and dates

@pytest.mark.parametrize("name", ["ID", "Run_ID", "RunOrder", "StdOrder", "Sample ID",
                                  "batch_id", "UUID", "index", "Obs", "serial"])
def test_columns_named_like_identifiers_are_identifiers(name):
    assert col(["a", "b", "c", "d", "e", "f"], name=name).role == IDENTIFIER


def test_a_plain_row_counter_is_an_identifier_whatever_it_is_called():
    assert col(list(range(1, 21)), name="Temperature").role == IDENTIFIER


def test_a_value_unique_to_every_row_is_an_identifier():
    assert col(["s%03d" % i for i in range(40)], name="Thing").role == IDENTIFIER


def test_a_precise_measurement_unique_on_every_row_is_still_a_measurement():
    """Uniqueness is what precision looks like, not what an identifier looks
    like. A titer to three decimals is unique on every row of any real dataset,
    and calling it a label would leave the tool unable to analyse the column the
    user actually cares about."""
    values = [round(2.0 + (i * 7919 % 9973) / 1000.0, 3) for i in range(60)]
    assert len(set(values)) == 60
    assert col(values, name="Titer_g_L").role == VARIABLES


def test_a_unique_whole_number_column_can_still_be_an_identifier():
    """A lot number is unique and integral. A measurement is unique and is not."""
    assert col([100000 + (i * 37) for i in range(40)], name="Lot number").role == IDENTIFIER


def test_unique_values_in_a_small_table_are_not_assumed_to_be_identifiers():
    """Five distinct strings across five rows is a five-level factor with one
    observation each, which is a different problem from an identifier."""
    c = col(["a", "b", "c", "d", "e"], name="Thing")
    assert c.role == ATTRIBUTE_3PLUS


def test_dates_are_recognised_and_not_offered_as_inputs():
    c = col(["2026-01-04", "2026-02-11", "2026-03-02", "2026-04-09"], name="Run date")
    assert c.role == DATETIME
    assert not c.analysable


# ------------------------------------------------------------- the hard case

def test_a_numeric_column_at_three_settings_offers_both_readings():
    """Temperature at 30, 35 and 40 is a slope to a regression and three groups
    to an ANOVA. Both are defensible, so neither is chosen silently."""
    c = col([30, 35, 40] * 4, name="Temp_C")
    assert c.role == VARIABLES
    assert c.alternative == ATTRIBUTE_3PLUS
    assert "slope" in c.alternative_reason and "groups" in c.alternative_reason


def test_a_numeric_column_at_two_settings_offers_the_two_level_reading():
    c = col([10, 20] * 6, name="Pressure")
    assert c.alternative == ATTRIBUTE_2


def test_enough_distinct_numbers_and_there_is_no_ambiguity_to_report():
    c = col([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8], name="Yield")
    assert c.role == VARIABLES and c.alternative is None


def test_the_ceiling_is_where_it_claims_to_be():
    """Six distinct numeric values is still ambiguous; seven is not."""
    assert col([1, 2, 3, 4, 5, 6] * 3, name="k").alternative == ATTRIBUTE_3PLUS
    assert col([1, 2, 3, 4, 5, 6, 7] * 3, name="k").alternative is None


# ------------------------------------------------------------------- warnings

def test_a_level_seen_once_is_named_rather_than_left_to_be_discovered():
    c = col(["A", "A", "A", "B", "B", "C"])
    assert any("Only one row at C" in n for n in c.notes)


def test_missing_values_are_counted_out_loud():
    c = col([1.5, 2.5, None, None, 5.5, 6.5, 7.5, 8.5])
    assert any("2 of 8 rows are blank" in n for n in c.notes)


def test_free_text_is_called_out_rather_than_offered_as_a_factor():
    c = col(["comment %d" % i for i in range(60)] + ["comment 0"] * 5, name="Remark")
    assert any("free text" in n for n in c.notes)


# -------------------------------------------------------------- number parsing

def test_thousands_separators_and_percent_signs_parse_as_numbers():
    c = col(["1,200", "2,400", "3,600", "4,800", "6,000", "7,200", "8,400"])
    assert c.role == VARIABLES


def test_a_percent_column_keeps_its_scale():
    """93% reads as 93, not 0.93. Rescaling silently would make two columns
    that were recorded the same way disagree."""
    c = col(["93%", "88%", "71%", "64%", "55%", "97%", "82%"])
    assert c.role == VARIABLES
    assert c.n_distinct == 7


def test_mostly_text_with_a_few_numbers_is_not_numeric():
    c = col(["1", "2", "3", "4", "five", "six", "seven", "eight"], name="k")
    assert not c.numeric


def test_a_numeric_column_with_one_stray_word_is_still_numeric():
    """One typo in forty rows should not change which test gets run.

    The values are deliberately not 0, 1, 2 ... - an earlier version of this test
    used range(40) and was correctly classified as a row counter, which is the
    classifier working and the fixture being wrong.
    """
    values = [round(2.0 + (i * 7919 % 97) / 50.0, 3) for i in range(40)] + ["n/a"]
    c = col(values, name="Titer")
    assert c.role == VARIABLES


# --------------------------------------------------------------- whole tables

def test_a_realistic_sheet_classifies_as_a_practitioner_would_read_it():
    frame = pd.DataFrame({
        "Run_ID": range(1, 13),
        "Titer_g_L": [2.1, 2.4, 2.9, 3.1, 2.2, 2.8, 3.3, 2.6, 2.7, 3.0, 2.5, 2.95],
        "Promoter": ["CMV", "EF1a", "CAG"] * 4,
        "Passed": ["Y", "N"] * 6,
        "Site": ["Boulder"] * 12,
    })
    roles = {c.name: c.role for c in classify_table(frame)}
    assert roles == {
        "Run_ID": IDENTIFIER,
        "Titer_g_L": VARIABLES,
        "Promoter": ATTRIBUTE_3PLUS,
        "Passed": ATTRIBUTE_2,
        "Site": CONSTANT,
    }
