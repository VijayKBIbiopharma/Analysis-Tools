"""Compare groups: pick a response and some inputs, and it chooses the test.

The page deliberately shows the *plan* before the result. A user who sees only
an answer has to take the choice of test on trust; a user who sees "two levels
against a continuous response, so a t-test for the centre and Brown-Forsythe
for the spread" can disagree before any arithmetic happens.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis_tools import charts

from analysis_tools.compare import (
    DIFFERENT, INCONCLUSIVE, NO_EVIDENCE, chi_square, compare_groups,
    logistic_regression, paired_t, simple_regression, two_sample_t,
)
from analysis_tools.roles import ATTRIBUTE_2, ATTRIBUTE_3PLUS, VARIABLES
from analysis_tools.routing import (
    CHI_SQUARE, LOGISTIC, PAIRED_T, REGRESSION, TWO_SAMPLE_T, plan_analysis,
)

VERDICT_STYLE = {
    DIFFERENT: ("🔴", "error"),
    NO_EVIDENCE: ("🟢", "success"),
    INCONCLUSIVE: ("🟡", "warning"),
}


def render(table) -> None:
    frame: pd.DataFrame = table.frame
    roles = st.session_state.get("roles")
    if not roles:
        st.info("Open **Explore** first so the columns can be classified.", icon="👈")
        return

    by_name = {r.name: r for r in roles}
    usable = [r.name for r in roles if r.analysable]
    if len(usable) < 2:
        st.error(
            "At least two columns have to be usable before anything can be "
            "compared. See **Explore** for why the others were set aside.",
            icon="🚫",
        )
        return

    st.subheader("Compare groups")
    st.caption(
        "Choose what you are measuring and what you think moves it. The test is "
        "chosen from the column types — the same routing a Six Sigma analysis "
        "path gives, run rather than printed."
    )

    left, right = st.columns(2)
    y_name = left.selectbox("The response — what you are measuring", usable,
                            index=len(usable) - 1)
    x_choices = [n for n in usable if n != y_name]
    x_names = right.multiselect("The input(s) — what you think moves it",
                                x_choices, default=x_choices[:1])
    if not x_names:
        st.info("Choose at least one input.", icon="👈")
        return

    y_role = by_name[y_name]
    x_roles = [by_name[n] for n in x_names]

    paired = False
    if len(x_roles) == 1 and x_roles[0].role == ATTRIBUTE_2 and y_role.role == VARIABLES:
        paired = st.checkbox(
            "These are paired — the same units measured under both conditions",
            help="Paired data has two observations on each unit, such as the same "
                 "sample measured by two methods. It is compared on the "
                 "differences, which is a more sensitive test when it applies "
                 "and the wrong test when it does not.",
        )

    plan = plan_analysis(x_roles, y_role, paired=paired, n_rows=len(frame))

    if plan.blockers:
        for blocker in plan.blockers:
            st.error(blocker, icon="🚫")
        return

    with st.container(border=True):
        st.markdown("**What applies here, and why**")
        st.caption(
            "%s against %s — %s."
            % (", ".join("%s (%s)" % (r.name, r.label.lower()) for r in x_roles),
               "%s (%s)" % (y_role.name, y_role.label.lower()),
               "one input" if len(x_roles) == 1 else "%d inputs" % len(x_roles))
        )
        for step in plan.steps:
            st.markdown("- **%s** — %s" % (step.name, step.question))
            if step.conditional_on:
                st.caption("   " + step.conditional_on)
        if plan.impact:
            st.caption("Effect size: " + plan.impact)
        for note in plan.notes:
            st.caption("· " + note)

    for caution in plan.cautions:
        st.warning(caution, icon="⚠️")

    if not st.button("Run it", type="primary"):
        return

    outcomes = _run(plan, frame, x_roles, y_role, paired)
    if not outcomes:
        st.info(
            "That combination is routed to the Relationships module, which is "
            "not built yet. Everything with a single input is available here.",
            icon="🚧",
        )
        return

    st.divider()
    for outcome in outcomes:
        _show(outcome)

    # The analysis path routes to two graphical tools in every cell, not one.
    # The statistics say whether; the pictures say what it looks like, which is
    # where an outlier carrying the whole result becomes visible.
    _pictures(frame, x_roles[0], y_role)


def _pictures(frame, x_role, y_role) -> None:
    from analysis_tools.roles import ATTRIBUTE_2, ATTRIBUTE_3PLUS, VARIABLES

    x_name, y_name = x_role.name, y_role.name
    st.markdown("#### The pictures")
    st.caption(
        "The analysis path asks for two of these in every cell. A p-value says "
        "whether the groups differ; these say whether one observation is "
        "carrying the result, whether a spread is lopsided, and whether the "
        "shape is what the test assumed.")

    if y_role.role == VARIABLES and x_role.role in (ATTRIBUTE_2, ATTRIBUTE_3PLUS):
        st.altair_chart(charts.box_by_group(frame, x_name, y_name),
                        use_container_width=True)
        st.altair_chart(charts.main_effects(frame, x_name, y_name),
                        use_container_width=True)
        if x_role.role == ATTRIBUTE_3PLUS:
            picture = charts.anom(frame, x_name, y_name)
            if picture is not None:
                st.altair_chart(picture, use_container_width=True)
                st.caption(
                    "ANOM answers what an ANOVA does not: an ANOVA says the "
                    "groups are not all alike, this says which ones stand out. "
                    "The limits use a Bonferroni approximation to Nelson's h "
                    "factors, which is slightly conservative — JMP uses exact "
                    "values, so its limits sit a touch tighter.")
    elif y_role.role == VARIABLES and x_role.role == VARIABLES:
        st.altair_chart(charts.scatter_with_fit(frame, x_name, y_name),
                        use_container_width=True)
        pair = frame[[x_name, y_name]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 3:
            import numpy as np
            slope, intercept = np.polyfit(pair[x_name], pair[y_name], 1)
            fitted = intercept + slope * pair[x_name]
            st.altair_chart(
                charts.residuals_vs_fitted(fitted, pair[y_name] - fitted),
                use_container_width=True)


def _run(plan, frame, x_roles, y_role, paired):
    """Execute the plan. Only the single-input cells are wired up so far."""
    if len(x_roles) != 1:
        return []
    x_role, x_name, y_name = x_roles[0], x_roles[0].name, y_role.name
    data = frame[[x_name, y_name]].dropna()

    if y_role.role == VARIABLES and x_role.role in (ATTRIBUTE_2, ATTRIBUTE_3PLUS):
        values = pd.to_numeric(data[y_name], errors="coerce")
        groups = {str(level): values[data[x_name].astype(str) == str(level)].tolist()
                  for level in sorted(data[x_name].astype(str).unique())}
        if x_role.role == ATTRIBUTE_2:
            if paired:
                a, b = list(groups.values())
                size = min(len(a), len(b))
                if size < 2:
                    st.error("Paired comparison needs the same units in both "
                             "groups, with at least two pairs.", icon="🚫")
                    return []
                if len(a) != len(b):
                    st.warning(
                        "The two groups hold %d and %d rows. Paired comparison "
                        "uses the first %d of each, in file order. If the rows "
                        "are not aligned that way, this is the wrong test."
                        % (len(a), len(b), size), icon="⚠️")
                return [paired_t(a[:size], b[:size], y_name,
                                 tuple(list(groups)[:2]))]
            return [two_sample_t(groups, y_name, x_name)]
        return compare_groups(groups, y_name, x_name)

    if y_role.role == VARIABLES and x_role.role == VARIABLES:
        return [simple_regression(data[x_name], data[y_name], x_name, y_name)]

    if y_role.role in (ATTRIBUTE_2, ATTRIBUTE_3PLUS) and x_role.role == VARIABLES:
        return [logistic_regression(data[x_name], data[y_name], x_name, y_name)]

    return chi_square(data[x_name], data[y_name], x_name, y_name)


def _show(outcome) -> None:
    icon, _ = VERDICT_STYLE.get(outcome.verdict, ("•", "info"))
    with st.container(border=True):
        st.markdown("#### %s %s" % (icon, outcome.name))
        st.markdown(outcome.plain)

        if outcome.estimate is not None:
            cols = st.columns(3)
            cols[0].metric(outcome.estimate_label.capitalize(),
                           "%.4g" % outcome.estimate)
            if outcome.interval:
                cols[1].metric("95% confidence interval",
                               "%.4g to %.4g" % outcome.interval)
            if outcome.p_value is not None:
                cols[2].metric("p-value", "%.4g" % outcome.p_value)

        if outcome.impact:
            st.caption(outcome.impact)
        for caution in outcome.cautions:
            st.warning(caution, icon="⚠️")

        if outcome.checks:
            with st.expander("What was checked before this test was chosen"):
                for check in outcome.checks:
                    st.markdown("- " + check.plain)
                    if check.verify_in:
                        st.caption("   In JMP: %s" % check.verify_in)

        if outcome.detail:
            with st.expander("The numbers"):
                st.json(outcome.detail, expanded=False)

        st.caption("**Reproduce it in JMP:** %s" % outcome.verify_in)
