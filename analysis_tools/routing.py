"""Which analysis applies to these columns?

The decision is made on three things, and nothing else:

1. how many inputs (X) were chosen,
2. what kind each input is - Variables, Attribute with 2 levels, Attribute with
   3 or more,
3. what kind the response (Y) is.

That is the analysis-path matrix a Six Sigma practitioner already works from,
usually printed on a wall or followed by hand through a menu. Implementing it
has one obvious benefit and one less obvious one. The obvious benefit is that
the right test gets run without anyone having to remember the chart. The less
obvious one is that several cells of the chart carry a condition - "one-way
ANOVA, or Kruskal-Wallis if non-normal or unequal variances" - which a person
has to evaluate for themselves, test for themselves, and remember to act on.
Here it is evaluated, and the branch taken is reported with the numbers that
decided it.

Two paths, not one
------------------
Each cell routes to a *centering* test and, where one exists, a *spread* test.
Asking only whether the average moved misses the case that matters most often in
practice: a change that leaves the mean alone and doubles the variance. A t-test
on its own reports that as "no significant difference".

What this module does not do
----------------------------
It does not run anything. It decides what should be run, says why, and says what
would stop it. Executing the chosen test is the job of the module that owns that
test, which keeps the routing readable and separately testable - the routing is
the part that has to be right, and it is the part a reviewer will want to check
against the chart they already know.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from analysis_tools.roles import ATTRIBUTE_2, ATTRIBUTE_3PLUS, VARIABLES

# Test identifiers. Strings rather than an enum so that a plan can be compared,
# logged and written to an audit trail without translation.
REGRESSION = "regression"
MULTIPLE_REGRESSION = "multiple_regression"
GLM = "general_linear_model"
LOGISTIC = "logistic_regression"
TWO_SAMPLE_T = "two_sample_t"
PAIRED_T = "paired_t"
ONE_WAY_ANOVA = "one_way_anova"
KRUSKAL_WALLIS = "kruskal_wallis"
LEVENE = "levene"
CHI_SQUARE = "chi_square"
TWO_PROPORTIONS = "two_proportions"

#: How each test reads, and where the same thing lives in Minitab. The Minitab
#: path is carried so a result can be reproduced independently by someone who
#: does not trust this tool yet, which is the fastest way to earn the trust.
TESTS = {
    REGRESSION: ("Regression", "Stat > Regression > Fitted Line Plot"),
    MULTIPLE_REGRESSION: ("Multiple regression", "Stat > Regression > Regression > Fit Regression Model"),
    GLM: ("General Linear Model", "Stat > ANOVA > General Linear Model"),
    LOGISTIC: ("Logistic regression", "Stat > Regression > Binary Logistic Regression"),
    TWO_SAMPLE_T: ("2-sample t-test", "Stat > Basic Statistics > 2-Sample t"),
    PAIRED_T: ("Paired t-test", "Stat > Basic Statistics > Paired t"),
    ONE_WAY_ANOVA: ("One-way ANOVA", "Stat > ANOVA > One-Way"),
    KRUSKAL_WALLIS: ("Kruskal-Wallis", "Stat > Nonparametrics > Kruskal-Wallis"),
    LEVENE: ("Levene's test for equal variances", "Stat > ANOVA > Test for Equal Variances"),
    CHI_SQUARE: ("Chi-square test", "Stat > Tables > Cross Tabulation and Chi-Square"),
    TWO_PROPORTIONS: ("2-proportions test", "Stat > Basic Statistics > 2 Proportions"),
}


@dataclass
class Step:
    """One test the plan intends to run."""

    test: str
    purpose: str                      # "centering" or "spread"
    question: str                     # what it answers, in a sentence
    #: Set when the cell offers a choice and the data decided it. Holds the
    #: check that was run and what it found.
    conditional_on: Optional[str] = None

    @property
    def name(self) -> str:
        return TESTS[self.test][0]

    @property
    def minitab(self) -> str:
        return TESTS[self.test][1]


@dataclass
class Plan:
    """What applies to the columns chosen, and what stands in the way."""

    x_names: list
    y_name: str
    x_roles: list                     # the role of each X, in order
    y_role: str
    steps: list = field(default_factory=list)
    impact: Optional[str] = None      # the statistic that sizes the effect
    #: Reasons the plan cannot run at all. Non-empty means refuse.
    blockers: list = field(default_factory=list)
    #: Things that do not stop the analysis but change how far to trust it.
    cautions: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def runnable(self) -> bool:
        return not self.blockers and bool(self.steps)

    @property
    def centering(self) -> Optional[Step]:
        return next((s for s in self.steps if s.purpose == "centering"), None)

    @property
    def spread(self) -> Optional[Step]:
        return next((s for s in self.steps if s.purpose == "spread"), None)


def _single_x(x_role: str, y_role: str) -> tuple:
    """The one-X-at-a-time cells. Returns (steps, impact, notes)."""
    if y_role == VARIABLES:
        if x_role == VARIABLES:
            return (
                [Step(REGRESSION, "centering",
                      "Does the response change steadily as this input changes?")],
                "R-squared (adjusted): the share of the variation in the response "
                "this input accounts for.",
                ["There is no separate test for spread here. Whether the scatter "
                 "widens or narrows across the range is read off the residuals, "
                 "which the result shows."],
            )
        if x_role == ATTRIBUTE_2:
            return (
                [Step(TWO_SAMPLE_T, "centering", "Do the two groups have the same average?"),
                 Step(LEVENE, "spread", "Do the two groups vary by the same amount?")],
                "R-squared (adjusted), from the equivalent one-way analysis.",
                [],
            )
        # three or more levels
        return (
            [Step(ONE_WAY_ANOVA, "centering", "Do the groups have the same average?",
                  conditional_on="Kruskal-Wallis is used instead when the residuals "
                                 "are not normal or the groups do not vary by the "
                                 "same amount. Both checks are run and reported."),
             Step(LEVENE, "spread", "Do the groups vary by the same amount?")],
            "R-squared (adjusted): the share of the variation in the response the "
            "grouping accounts for.",
            [],
        )

    # Y is an attribute
    if x_role == VARIABLES:
        return (
            [Step(LOGISTIC, "centering",
                  "Does the chance of each outcome change as this input changes?")],
            None,
            ["There is no agreed measure of how much of the response this explains "
             "when the response is a category. The chart says so too, rather than "
             "offering a number that would not mean what people assume."],
        )
    both_two = x_role == ATTRIBUTE_2 and y_role == ATTRIBUTE_2
    steps = [Step(CHI_SQUARE, "centering",
                  "Are the proportions the same across the groups?"
                  if both_two else "Are the two columns independent of each other?")]
    if both_two:
        steps.append(Step(TWO_PROPORTIONS, "centering",
                          "What is the difference in proportion, with an interval?"))
    return (steps, None, [])


def _several_x(x_roles: list, y_role: str) -> tuple:
    """The several-X-at-once cells."""
    if y_role != VARIABLES:
        return (
            [Step(LOGISTIC, "centering",
                  "Which inputs change the chance of each outcome?")],
            None,
            [],
        )
    kinds = set(x_roles)
    if kinds == {VARIABLES}:
        step, note = MULTIPLE_REGRESSION, (
            "With several continuous inputs, which to keep in the model is a "
            "question in its own right. The model-reduction step searches the "
            "subsets rather than leaving every input in."
        )
    else:
        step, note = GLM, (
            "A General Linear Model carries continuous and grouping inputs "
            "together, which is what mixing them requires."
            if kinds != {ATTRIBUTE_2, ATTRIBUTE_3PLUS} and VARIABLES in kinds
            else "A General Linear Model compares groups while holding the other "
                 "groupings constant."
        )
    return (
        [Step(step, "centering", "Which inputs change the response, with the others "
                                 "accounted for?")],
        "R-squared (adjusted) and s, the standard deviation of the residuals: the "
        "variation in the response the model does not explain.",
        [note],
    )


def plan_analysis(x_roles: list, y_role, *, paired: bool = False,
                  n_rows: Optional[int] = None) -> Plan:
    """Decide what applies.

    ``x_roles`` and ``y_role`` are ColumnRole objects from ``roles.py``. Passing
    the roles rather than the raw columns keeps this module free of any opinion
    about the data itself: it routes, and it can be read against the chart.
    """
    x_list = list(x_roles)
    names = [r.name for r in x_list]
    kinds = [r.role for r in x_list]
    plan = Plan(x_names=names, y_name=y_role.name, x_roles=kinds, y_role=y_role.role)

    if not x_list:
        plan.blockers.append("Choose at least one input to analyse.")
        return plan
    if y_role.name in names:
        plan.blockers.append(
            "“%s” is selected as both an input and the response. A column cannot "
            "explain itself." % y_role.name
        )
        return plan
    for role in x_list + [y_role]:
        if not role.analysable:
            plan.blockers.append(
                "“%s” is %s, so it cannot take part in an analysis."
                % (role.name, role.label.lower())
            )
    if plan.blockers:
        return plan

    if len(x_list) == 1:
        steps, impact, notes = _single_x(kinds[0], y_role.role)
        if paired and steps and steps[0].test == TWO_SAMPLE_T:
            steps[0] = Step(PAIRED_T, "centering",
                            "Across the same units measured twice, is the average "
                            "difference zero?")
            steps = [s for s in steps if s.test != LEVENE]
            notes = notes + [
                "Paired data has two observations on each unit, so the comparison "
                "is of the differences. There is no separate spread test for paired "
                "data; the chart says so too."
            ]
    else:
        steps, impact, notes = _several_x(kinds, y_role.role)

    plan.steps, plan.impact = steps, impact
    plan.notes.extend(notes)
    _add_cautions(plan, x_list, y_role, n_rows)
    return plan


def _add_cautions(plan: Plan, x_list: list, y_role, n_rows: Optional[int]) -> None:
    """What will not stop the analysis but changes how far to trust it."""
    for role in x_list:
        if role.counts:
            thin = sorted((v, k) for k, v in role.counts.items())
            fewest, level = thin[0]
            if fewest < 2:
                plan.blockers.append(
                    "“%s” has only one row at %s. A group of one has no spread to "
                    "compare and contributes almost nothing to a comparison of "
                    "averages." % (role.name, level)
                )
            elif fewest < 5:
                plan.cautions.append(
                    "“%s” has only %d rows at %s. Anything this analysis says about "
                    "that level rests on %d observations."
                    % (role.name, fewest, level, fewest)
                )
        if role.alternative:
            plan.cautions.append(
                "“%s” is being read as %s. Read the other way the analysis would be "
                "different — see the Explore page."
                % (role.name, role.label.lower())
            )

    if n_rows is not None:
        wanted = 3 * (len(plan.x_names) + 1)
        if n_rows < wanted:
            plan.cautions.append(
                "%d rows for %d input%s. A model has little left to learn from once "
                "it has fitted the terms; expect wide intervals and treat the "
                "ranking as provisional."
                % (n_rows, len(plan.x_names), "" if len(plan.x_names) == 1 else "s")
            )
