"""Measurement systems: can the method see what you are chasing?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis_tools import charts
from analysis_tools.msa import ACCEPTABLE, MARGINAL, gauge_rr
from analysis_tools.app.fitness import explain
from analysis_tools.roles import ATTRIBUTE_2, ATTRIBUTE_3PLUS, VARIABLES

BADGE = {"acceptable": "🟢", "marginal": "🟡", "unacceptable": "🔴", "unknown": "⚪"}


def render(table) -> None:
    frame: pd.DataFrame = table.frame
    roles = st.session_state.get("roles")
    if not roles:
        st.info("Open **Explore** first so the columns can be classified.", icon="👈")
        return

    measurements = [r.name for r in roles if r.role == VARIABLES]
    groupings = [r.name for r in roles
                 if r.role in (ATTRIBUTE_2, ATTRIBUTE_3PLUS)] + \
                [r.name for r in roles if r.role == "identifier"]

    if not measurements or len(groupings) < 2:
        st.info(explain("Measurement systems", roles, table.source_name),
                icon="🧭")
        return

    st.subheader("Measurement systems")
    st.caption(
        "Every number downstream inherits the measurement. A capability index, "
        "a clone ranking and a comparability study all take the measured value "
        "at face value — so if a third of what you see is the assay, a third of "
        "each of those conclusions is about the assay."
    )

    if not measurements or len(groupings) < 2:
        st.error(
            "A Gage R&R needs a continuous measurement, a column identifying the "
            "part or sample, and a column identifying the operator. This file "
            "does not have all three — see **Explore** for how each column was "
            "read.", icon="🚫")
        return

    cols = st.columns(3)
    part = cols[0].selectbox("Part or sample", groupings,
                             help="What was measured repeatedly. Ten parts "
                                  "spanning the range of the process is the "
                                  "AIAG recommendation.")
    operator = cols[1].selectbox(
        "Operator, analyst or instrument",
        [g for g in groupings if g != part],
        help="Who or what did the measuring. Reproducibility is the difference "
             "between these.")
    measurement = cols[2].selectbox("The measurement", measurements)

    spec = st.columns(3)
    use_spec = spec[0].checkbox("I have specification limits",
                                help="With limits, the study also reports the "
                                     "measurement error as a share of the "
                                     "tolerance, which is often the number that "
                                     "decides whether the method is fit for use.")
    lsl = spec[1].number_input("Lower spec limit", value=None, placeholder="none",
                               format="%.6g") if use_spec else None
    usl = spec[2].number_input("Upper spec limit", value=None, placeholder="none",
                               format="%.6g") if use_spec else None

    if not st.button("Run the study", type="primary"):
        return

    result = gauge_rr(frame, part, operator, measurement, lsl=lsl, usl=usl)

    if not result.study_var_pct:
        st.error(result.plain, icon="🚫")
        return

    with st.container(border=True):
        st.markdown("#### %s %s" % (BADGE.get(result.verdict, "•"),
                                    result.verdict.capitalize()))
        st.markdown(result.plain)

        st.altair_chart(
            charts.variability_chart(frame, part, operator, measurement),
            use_container_width=True)

        metrics = st.columns(4)
        metrics[0].metric("Total Gage R&R", "%.1f%%" % result.grr_pct)
        metrics[1].metric("Repeatability", "%.1f%%" % result.study_var_pct["Repeatability"])
        metrics[2].metric("Reproducibility", "%.1f%%" % result.study_var_pct["Reproducibility"])
        if result.distinct_categories is not None:
            metrics[3].metric("Distinct categories", result.distinct_categories)

        rows = []
        for name, variance in result.variances.items():
            row = {
                "Source": name,
                "Variance": round(variance, 6),
                "Std dev": round(variance ** 0.5, 5),
                "% of study variation": round(result.study_var_pct[name], 2),
            }
            if result.tolerance_pct:
                row["% of tolerance"] = round(result.tolerance_pct[name], 2)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        st.caption("AIAG: under %.0f%% acceptable, %.0f–%.0f%% conditional, "
                   "over %.0f%% not acceptable."
                   % (ACCEPTABLE, ACCEPTABLE, MARGINAL, MARGINAL))

        for caution in result.cautions:
            st.info(caution, icon="ℹ️")

        with st.expander("The ANOVA behind it"):
            st.dataframe(
                pd.DataFrame([
                    {"Source": k, "SS": round(v["SS"], 5), "df": v["df"],
                     "MS": round(v["MS"], 5)}
                    for k, v in result.anova.items()
                ]), hide_index=True, use_container_width=True)
            if result.interaction_p is not None:
                st.caption("Operator x Part interaction p = %.4f."
                           % result.interaction_p)

        st.caption("**Reproduce it in JMP:** %s" % result.verify_in)
        st.caption("JMP needs the measurements stacked in a single column, which "
                   "is the layout this page expects too.")
