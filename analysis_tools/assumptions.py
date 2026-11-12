"""The checks that decide which test gets run.

Several cells of the analysis-path matrix carry a condition rather than a single
answer. On a wall chart that condition is an instruction: the practitioner is
expected to test normality themselves, test equal variances themselves, and
remember to switch. In practice the switch is often missed, because the chart is
read at the start and the data is looked at afterwards.

Here the checks run every time, the branch is taken on the result, and the
numbers that decided it are reported beside the answer.

Reproducing these in JMP
------------------------
Every check names where the same number lives in JMP 19, because a result nobody
can check independently is a result nobody has to believe. Two details matter
for that to work:

*Normality* is Shapiro-Wilk. JMP reports both Shapiro-Wilk and Anderson-Darling
for a fitted Normal, but its Anderson-Darling p-value is simulated by parametric
bootstrap, so it will not match any closed-form approximation exactly.
Shapiro-Wilk is the standard algorithm in both, so the numbers agree.

*Equal variance* is Levene's test centred on the **median**, which JMP reports
on the row labelled **Brown-Forsythe** - its "Levene" row is centred on the
mean. Pointing a user at the wrong row of the right report would be worse than
giving no reference at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats

#: The package these results are quoted against, so a user can reproduce them.
REFERENCE_PACKAGE = "JMP 19"

#: Below this many observations a normality test has almost no power: it will
#: fail to reject nearly anything, so "passed" carries no information.
MIN_FOR_NORMALITY = 8

#: JMP warns when a level holds fewer than five observations. Same threshold.
THIN_LEVEL = 5

ALPHA = 0.05


@dataclass
class Check:
    """One assumption, tested."""

    name: str
    statistic: float
    p_value: Optional[float]
    holds: bool
    plain: str
    verify_in: str = ""
    #: True when the data is too thin for the check to mean anything, in which
    #: case `holds` is a default rather than a finding.
    underpowered: bool = False
    #: Levels left out of the calculation, and why.
    excluded: tuple = ()


def check_normality(values: Sequence[float], what: str = "the data") -> Check:
    """Is this sample consistent with a normal distribution?

    A failure is not a verdict on the data. Non-normal data is ordinary; it
    changes which test applies and nothing more. The wording matters, because a
    user told their data "failed" a test will go looking for a fault that is not
    there.
    """
    x = np.asarray([v for v in values if v == v], dtype=float)
    n = x.size
    where = "Analyze > Distribution > Continuous Fit > Fit Normal > Goodness of Fit"

    if n < MIN_FOR_NORMALITY:
        return Check(
            name="Shapiro-Wilk normality", statistic=float("nan"), p_value=None,
            holds=True, underpowered=True, verify_in=where,
            plain="Only %d values in %s — too few for a normality test to carry "
                  "any weight. Proceeding on the usual assumption of normality, "
                  "which has not been checked." % (n, what),
        )
    if np.allclose(x, x[0]):
        return Check(
            name="Shapiro-Wilk normality", statistic=float("nan"), p_value=None,
            holds=True, underpowered=True, verify_in=where,
            plain="Every value in %s is identical, so there is no distribution "
                  "to test." % what,
        )

    statistic, p = stats.shapiro(x)
    holds = p >= ALPHA
    if holds:
        plain = ("%s is consistent with a normal distribution (Shapiro-Wilk "
                 "W = %.4f, p = %s)." % (what.capitalize(), statistic, fmt_p(p)))
    else:
        plain = ("%s departs from a normal distribution (Shapiro-Wilk W = %.4f, "
                 "p = %s). That is ordinary and is not a fault in the data — it "
                 "changes which test applies."
                 % (what.capitalize(), statistic, fmt_p(p)))
    return Check(name="Shapiro-Wilk normality", statistic=float(statistic),
                 p_value=float(p), holds=holds, plain=plain, verify_in=where)


def check_equal_variance(groups: Sequence[Sequence[float]],
                         labels: Optional[Sequence[str]] = None) -> Check:
    """Do the groups vary by the same amount?

    Levene's test centred on the median, which is the Brown-Forsythe form. The
    median-centred version stays honest when the data is skewed, which is
    exactly when the question gets asked, and the analysis-path matrix says the
    same in its own words: "good test even if data are not normal".

    Levels holding a single observation are left out rather than blocking the
    test, which is what JMP 19.1 does. One observation has no spread to
    contribute; it is excluded here and named in the result so nobody has to
    wonder why the counts do not add up.
    """
    where = "Analyze > Fit Y by X > Oneway > Unequal Variances (Brown-Forsythe row)"
    names = list(labels) if labels is not None else ["group %d" % (i + 1) for i in range(len(groups))]
    cleaned = [np.asarray([v for v in g if v == v], dtype=float) for g in groups]

    excluded = tuple(names[i] for i, g in enumerate(cleaned) if g.size < 2)
    keep = [(names[i], g) for i, g in enumerate(cleaned) if g.size >= 2]

    if len(keep) < 2:
        return Check(
            name="Brown-Forsythe test for equal variances", statistic=float("nan"),
            p_value=None, holds=True, underpowered=True, verify_in=where,
            excluded=excluded,
            plain="At least two groups need two or more observations each before "
                  "their spreads can be compared.",
        )
    if any(np.allclose(g, g[0]) for _, g in keep):
        flat = ", ".join(n for n, g in keep if np.allclose(g, g[0]))
        return Check(
            name="Brown-Forsythe test for equal variances", statistic=float("nan"),
            p_value=None, holds=True, underpowered=True, verify_in=where,
            excluded=excluded,
            plain="Every observation in %s is identical, so the spreads cannot be "
                  "compared." % flat,
        )

    statistic, p = stats.levene(*[g for _, g in keep], center="median")
    holds = p >= ALPHA
    spreads = ", ".join("%s %.4g" % (n, g.std(ddof=1)) for n, g in keep)
    note = ""
    if excluded:
        note = (" %s left out of this test, having only one observation."
                % ", ".join(excluded))
    if holds:
        plain = ("The groups vary by a similar amount (Brown-Forsythe, p = %s). "
                 "Standard deviations: %s.%s" % (fmt_p(p), spreads, note))
    else:
        plain = ("The groups do not vary by the same amount (Brown-Forsythe, "
                 "p = %s). Standard deviations: %s.%s A difference in spread is a "
                 "finding in its own right, not merely an inconvenience for the "
                 "comparison of averages." % (fmt_p(p), spreads, note))
    return Check(name="Brown-Forsythe test for equal variances",
                 statistic=float(statistic), p_value=float(p), holds=holds,
                 plain=plain, verify_in=where, excluded=excluded)


def fmt_p(value: Optional[float]) -> str:
    """A p-value as a reader expects to see it."""
    if value is None or value != value:
        return "—"
    if value < 0.0001:
        return "< 0.0001"
    return "%.4f" % value
