"""Relationships: several inputs at once."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis_tools import charts
from analysis_tools.relate import (
    MAX_SUBSET_TERMS, VIF_CONCERN, VIF_SERIOUS, all_possible_models, fit_model,
)
from analysis_tools.app.fitness import explain
from analysis_tools.roles import ATTRIBUTE_2, ATTRIBUTE_3PLUS, VARIABLES


def render(table) -> None:
    frame: pd.DataFrame = table.frame
    roles = st.session_state.get("roles")
    if not roles:
        st.info("Open **Explore** first so the columns can be classified.", icon="👈")
        return

    by_name = {r.name: r for r in roles}
    responses = [r.name for r in roles if r.role == VARIABLES]
    inputs_available = [r.name for r in roles if r.analysable]

    if not responses or len(inputs_available) < 2:
        st.info(explain("Relationships", roles, table.source_name), icon="🧭")
        return

    st.subheader("Relationships")
    st.caption(
        "Which inputs move the response, with the others accounted for. "
        "Continuous and grouping inputs go into the same model — a grouping "
        "input simply enters as a set of indicator terms."
    )

    left, right = st.columns(2)
    y_name = left.selectbox("The response", responses, index=len(responses) - 1)
    choices = [n for n in inputs_available if n != y_name]
    x_names = right.multiselect("The inputs", choices, default=choices[:3])
    if len(x_names) < 1:
        st.info("Choose at least one input.", icon="👈")
        return

    categorical = [n for n in x_names
                   if by_name[n].role in (ATTRIBUTE_2, ATTRIBUTE_3PLUS)]
    options = st.columns(2)
    interactions = options[0].checkbox(
        "Include two-way interactions",
        help="An interaction means the effect of one input depends on the level "
             "of another. Worth including when you suspect it; it costs degrees "
             "of freedom, so not by default.")
    search = options[1].checkbox(
        "Search every subset and rank them",
        value=len(x_names) <= 6,
        help="Fits every combination of the inputs and ranks by BIC, which is "
             "what JMP's All Possible Models does. Read the warning that comes "
             "with it.")

    if not st.button("Fit it", type="primary"):
        return

    fit = fit_model(frame, y_name, x_names, categorical=categorical,
                    interactions=interactions)
    if fit.failed:
        st.error(fit.failed, icon="🚫")
        return

    with st.container(border=True):
        st.markdown("#### The model")
        st.markdown(fit.plain)

        metrics = st.columns(4)
        metrics[0].metric("Adjusted R²", "%.3f" % fit.r_squared_adj)
        metrics[1].metric("s (residual sd)", "%.4g" % fit.sigma)
        metrics[2].metric("BIC", "%.1f" % fit.bic)
        metrics[3].metric("Rows used", fit.n)

        rows = []
        for term in fit.terms:
            rows.append({
                "Term": term.name,
                "Coefficient": round(term.coefficient, 5),
                "95% interval": ("%.4g to %.4g" % term.interval) if term.interval else "—",
                "p": "%.4g" % term.p_value if term.p_value is not None else "—",
                "VIF": round(term.vif, 1) if term.vif is not None else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "VIF is variance inflation: how much wider a coefficient's interval "
            "is than it would be if that input were independent of the others. "
            "Above %.0f is worth attention, above %.0f is serious."
            % (VIF_CONCERN, VIF_SERIOUS))

        if fit.anova is not None:
            with st.expander("The ANOVA table"):
                st.dataframe(fit.anova, use_container_width=True)

        if fit.fitted is not None and fit.residuals is not None:
            st.altair_chart(
                charts.residuals_vs_fitted(fit.fitted, fit.residuals),
                use_container_width=True)
            st.caption(
                "A funnel means the spread changes with the level; a curve "
                "means a straight line was the wrong shape. Neither shows up "
                "in adjusted R².")

        for caution in fit.cautions:
            st.warning(caution, icon="⚠️")
        if fit.checks:
            with st.expander("What was checked"):
                for check in fit.checks:
                    st.markdown("- " + check.plain)
        st.caption("**Reproduce it in JMP:** %s" % fit.verify_in)

    if not search:
        return
    if len(x_names) > MAX_SUBSET_TERMS:
        st.info("Too many inputs to search every subset.", icon="ℹ️")
        return

    result = all_possible_models(frame, y_name, x_names, categorical=categorical)
    if result is None:
        st.info("The subset search could not run on this data.", icon="ℹ️")
        return

    with st.container(border=True):
        st.markdown("#### Every subset, ranked")
        st.markdown(result.plain)
        st.dataframe(result.results, hide_index=True, use_container_width=True)
        st.caption(
            "Mallows' Cp should land near the term count plus one; far above "
            "means the model is missing something. JMP computes it too, hidden "
            "behind a right-click on the table.")
        st.warning(result.optimism, icon="⚠️")
        for substitute in result.substitutes:
            st.warning(substitute, icon="🔀")
        st.caption("**Reproduce it in JMP:** %s" % result.verify_in)
