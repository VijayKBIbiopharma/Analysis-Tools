"""What kind of column is this?

Every routing decision in this tool turns on one question, asked of each column:
is it a *variable* (measured on a continuous scale), an *attribute with two
levels*, or an *attribute with three or more levels*? That vocabulary is not
invented here. It is the vocabulary a Six Sigma practitioner already uses, and
it is what the analysis-path matrix keys on: continuous X with continuous Y means
regression; two-level attribute X with continuous Y means a t-test; three-plus
means one-way ANOVA.

Getting this classification right is therefore not a presentation detail. It
decides which test runs.

The hard case, and how it is handled
------------------------------------
A numeric column holding only a handful of distinct values is genuinely
ambiguous. Temperature recorded at 30, 35 and 40 degrees is a *variable* if you
want to fit a slope through it, and an *attribute with three levels* if you want
to compare three group means. Both readings are defensible and they lead to
different tests with different answers.

So this module does not guess. It gives its reading, and where a second reading
is defensible it says so and names what would change. A tool that silently picks
one and runs with it is choosing the user's analysis for them without telling
them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

#: Column roles, in the analysis-path matrix's own words.
VARIABLES = "variables"
ATTRIBUTE_2 = "attribute_2"
ATTRIBUTE_3PLUS = "attribute_3plus"
IDENTIFIER = "identifier"
DATETIME = "datetime"
CONSTANT = "constant"
EMPTY = "empty"

#: How each role reads in the interface and in the matrix.
ROLE_LABELS = {
    VARIABLES: "Variables (continuous)",
    ATTRIBUTE_2: "Attribute, 2 levels",
    ATTRIBUTE_3PLUS: "Attribute, 3 or more levels",
    IDENTIFIER: "Identifier",
    DATETIME: "Date or time",
    CONSTANT: "Constant",
    EMPTY: "Empty",
}

#: Roles that can take part in an analysis as an X or a Y.
ANALYSABLE = (VARIABLES, ATTRIBUTE_2, ATTRIBUTE_3PLUS)

# Values that mean "nothing was recorded". Compared case-insensitively against
# the stripped cell. "0" is deliberately absent: zero is a measurement.
MISSING_TOKENS = {
    "", "na", "n/a", "nan", "null", "none", "nil", "-", "--", "?", "#n/a",
    "#value!", "#div/0!", "missing", "not available", "not applicable",
}

# Column names that are record-keeping rather than data. Matched whole, after
# stripping, case-insensitively, with separators normalised.
_ID_NAME = re.compile(
    r"^(id|row|row ?number|index|no|num|number|#|obs|observation|seq|sequence|"
    r"s/?n|serial|run|run ?order|std ?order|order|sample ?id|record ?id|lot|"
    r"batch ?id|uuid|guid|key|ref|reference)$"
    r"|[ _-]id$|^id[ _-]",
    re.IGNORECASE,
)

# A numeric column with no more distinct values than this is a candidate for
# being read as an attribute instead. Six is deliberate: a DOE factor at five
# levels is still a factor, and a measured response almost never lands on six
# repeated values across a real dataset.
NUMERIC_LEVEL_CEILING = 6

# Below this many rows per level, a group comparison has nothing to say.
MIN_ROWS_PER_LEVEL = 2


@dataclass
class ColumnRole:
    """One column's classification, and what the tool is unsure about."""

    name: str
    role: str
    n: int
    n_missing: int
    n_distinct: int
    levels: Optional[list] = None          # for attribute roles, the levels seen
    counts: Optional[dict] = None          # level -> row count
    numeric: bool = False
    #: A second defensible reading, when one exists. Never applied silently.
    alternative: Optional[str] = None
    #: Why the alternative exists, in a sentence a user can act on.
    alternative_reason: Optional[str] = None
    #: Things the user should know before analysing this column.
    notes: list = field(default_factory=list)

    @property
    def label(self) -> str:
        return ROLE_LABELS[self.role]

    @property
    def analysable(self) -> bool:
        return self.role in ANALYSABLE

    @property
    def n_present(self) -> int:
        return self.n - self.n_missing


def _clean(series: pd.Series) -> pd.Series:
    """Blank out the tokens that mean "nothing was recorded"."""
    if series.dtype == object:
        stripped = series.astype(str).str.strip()
        blank = stripped.str.lower().isin(MISSING_TOKENS)
        return stripped.mask(blank | series.isna())
    return series.mask(series.isna())


def _as_numeric(values: pd.Series) -> Optional[pd.Series]:
    """Return the column as numbers, or None if too few of them parse.

    The 95% threshold is on *present* values, not on rows: a column that is half
    blank and half numbers is a numeric column with gaps, not a text column.
    Thousands separators and a trailing percent sign are stripped; the percent is
    not divided by 100, because a column of "93%" means 93 to everything
    downstream and quietly rescaling it would make two columns disagree.
    """
    present = values.dropna()
    if present.empty:
        return None
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")
    text = present.astype(str).str.strip()
    text = text.str.replace(",", "", regex=False).str.rstrip("%")
    parsed = pd.to_numeric(text, errors="coerce")
    if parsed.notna().sum() / len(present) < 0.95:
        return None
    out = pd.Series(np.nan, index=values.index, dtype="float64")
    out.loc[parsed.index] = parsed
    return out


def _looks_sequential(numeric: pd.Series) -> bool:
    """1, 2, 3, 4 ... is a row counter however the column is named."""
    present = numeric.dropna()
    if len(present) < 6:
        return False
    diffs = present.diff().dropna()
    return bool((diffs == 1).all())


def classify_column(name: str, series: pd.Series) -> ColumnRole:
    """Classify one column, and record any second defensible reading."""
    values = _clean(series)
    n = len(series)
    n_missing = int(values.isna().sum())
    present = values.dropna()
    notes: list = []

    if present.empty:
        return ColumnRole(name=name, role=EMPTY, n=n, n_missing=n_missing, n_distinct=0)

    numeric = _as_numeric(values)
    is_numeric = numeric is not None
    basis = numeric.dropna() if is_numeric else present
    distinct = sorted(basis.unique().tolist()) if is_numeric else sorted(
        basis.astype(str).unique().tolist()
    )
    n_distinct = len(distinct)

    if n_distinct <= 1:
        return ColumnRole(
            name=name, role=CONSTANT, n=n, n_missing=n_missing, n_distinct=n_distinct,
            levels=distinct, numeric=is_numeric,
            notes=["Every row holds the same value, so this column cannot explain "
                   "any difference in anything else."],
        )

    # Dates, before the identifier test: a date column is often unique per row.
    if not is_numeric:
        as_dates = pd.to_datetime(present, errors="coerce", format="mixed")
        if as_dates.notna().sum() / len(present) >= 0.95:
            return ColumnRole(
                name=name, role=DATETIME, n=n, n_missing=n_missing,
                n_distinct=n_distinct,
                notes=["Read as a date. Use it to order the rows, not as an input."],
            )

    # Identifiers: named like one, or unique across enough rows to be one, or a
    # plain row counter.
    #
    # The uniqueness test applies to text only, and that restriction is
    # load-bearing rather than fussy. A titer measured to three decimals is
    # unique on every row of a real dataset - uniqueness is what precision looks
    # like, not what an identifier looks like. An earlier version of this applied
    # the test to numbers too and classified every continuous response as a
    # label, which would have left the tool unable to analyse the one column the
    # user cared about. A numeric column is only a candidate when every value is
    # a whole number, which a lot or sample number is and a measurement is not.
    unique_per_row = n_distinct == len(present) and len(present) > 20
    if unique_per_row and is_numeric:
        unique_per_row = bool(np.all(np.mod(basis.to_numpy(dtype=float), 1) == 0))
    if _ID_NAME.search(str(name).strip()) or unique_per_row or (
        is_numeric and _looks_sequential(numeric)
    ):
        return ColumnRole(
            name=name, role=IDENTIFIER, n=n, n_missing=n_missing, n_distinct=n_distinct,
            numeric=is_numeric,
            notes=["Read as a label for the row rather than a measurement."],
        )

    counts = basis.value_counts().to_dict()
    smallest = min(counts.values())

    if is_numeric and n_distinct > NUMERIC_LEVEL_CEILING:
        role, alternative, reason = VARIABLES, None, None
    elif is_numeric:
        # The ambiguous case. A numeric column at a handful of settings is both.
        role = VARIABLES
        alternative = ATTRIBUTE_2 if n_distinct == 2 else ATTRIBUTE_3PLUS
        reason = (
            "This column holds numbers but only %d distinct values (%s). Read as "
            "Variables it fits a slope and asks whether the response changes "
            "steadily. Read as an attribute it compares the %d groups against each "
            "other and makes no assumption that the steps are evenly spaced."
            % (n_distinct, ", ".join(_short(v) for v in distinct), n_distinct)
        )
    elif n_distinct == 2:
        role, alternative, reason = ATTRIBUTE_2, None, None
    else:
        role, alternative, reason = ATTRIBUTE_3PLUS, None, None
        if n_distinct > 50 and n_distinct > 0.5 * len(present):
            notes.append(
                "%d distinct values across %d rows. That is close to one per row, "
                "which usually means free text rather than a factor."
                % (n_distinct, len(present))
            )

    if role in (ATTRIBUTE_2, ATTRIBUTE_3PLUS) or alternative:
        if smallest < MIN_ROWS_PER_LEVEL:
            thin = [str(k) for k, v in counts.items() if v < MIN_ROWS_PER_LEVEL]
            notes.append(
                "Only one row at %s. A level seen once contributes nothing to a "
                "comparison of spread and very little to a comparison of means."
                % ", ".join(thin[:4])
            )

    if n_missing:
        notes.append(
            "%d of %d rows are blank (%.0f%%)." % (n_missing, n, 100.0 * n_missing / n)
        )

    return ColumnRole(
        name=name, role=role, n=n, n_missing=n_missing, n_distinct=n_distinct,
        levels=distinct if role in (ATTRIBUTE_2, ATTRIBUTE_3PLUS) or alternative else None,
        counts={str(k): int(v) for k, v in counts.items()} if role != VARIABLES or alternative else None,
        numeric=is_numeric, alternative=alternative, alternative_reason=reason,
        notes=notes,
    )


def _short(value) -> str:
    """A number as a person would write it: 5 not 5.0, 0.25 not 0.25000000001."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return ("%.6g" % value)
    return str(value)


def classify_table(frame: pd.DataFrame) -> list:
    """Classify every column in a table."""
    return [classify_column(str(c), frame[c]) for c in frame.columns]
