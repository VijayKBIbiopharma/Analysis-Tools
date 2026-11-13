"""Design of experiments: build one, or analyse the one you ran."""

from __future__ import annotations

import itertools

import pandas as pd
import streamlit as st

from analysis_tools import charts
from analysis_tools.doe import analyse_design, build_design
from analysis_tools.roles import ATTRIBUTE_2, VARIABLES


def render(table) -> None:
    build, analyse = st.tabs(["Build a design", "Analyse a design you ran"])
    with build:
        _build()
    with analyse:
        _analyse(table)


def _build() -> None:
    st.caption(
        "A screening design finds **which** factors matter: two levels each, "
        "as wide as the process allows. Keep the levels wide — a factor that "
        "does nothing over a narrow range may do plenty over a real one."
    )

    count = st.number_input("How many factors?", min_value=2, max_value=8,
                            value=3, step=1)
    rows = []
    for i in range(int(count)):
        cols = st.columns([2, 1, 1])
        name = cols[0].text_input("Factor %d" % (i + 1), value="Factor %d" % (i + 1),
                                  key="doe_name_%d" % i)
        low = cols[1].text_input("Low", value="", key="doe_low_%d" % i,
                                 placeholder="e.g. 32 or Resin A")
        high = cols[2].text_input("High", value="", key="doe_high_%d" % i,
                                  placeholder="e.g. 38 or Resin B")
        if name and low and high:
            rows.append({"name": name, "low": low, "high": high})

    if len(rows) < 2:
        st.info("Fill in at least two factors with both settings.", icon="👈")
        return

    settings = st.columns(4)
    budget = settings[0].selectbox(
        "Runs you can afford", [8, 16, 32, 64, 128],
        index=1,
        help="A full factorial on %d factors is %d runs. Asking for fewer buys "
             "a fraction, which confounds effects — the design will say exactly "
             "which." % (len(rows), 2 ** len(rows)))
    centre = settings[1].number_input("Centre points", 0, 8, 3,
                                      help="The only way a two-level design can "
                                           "detect curvature, and they buy error "
                                           "degrees of freedom cheaply.")
    replicates = settings[2].number_input("Replicates", 1, 4, 1,
                                          help="Replication is the only thing "
                                               "that measures pure error.")
    seed = settings[3].number_input("Run-order seed", 1, 9999, 29)

    design = build_design(rows, runs=int(budget), centre_points=int(centre),
                          replicates=int(replicates), seed=int(seed))
    if design.n_runs == 0:
        st.error(design.plain, icon="🚫")
        return

    st.markdown(design.plain)
    for caution in design.cautions:
        st.warning(caution, icon="⚠️")

    if design.aliases:
        with st.expander("What is confounded with what", expanded=design.resolution == 3):
            st.caption(
                "Each term on the left cannot be told apart from the terms on "
                "the right by any analysis of this design. This is the price of "
                "the run count, and it is worth reading before the experiment "
                "rather than after.")
            st.dataframe(
                pd.DataFrame([{"Term": term, "Cannot be separated from": ", ".join(partners)}
                              for term, partners in design.aliases.items()]),
                hide_index=True, use_container_width=True)

    st.markdown("#### The run sheet")
    st.caption("Run them in the **Run order** column, not the standard order. "
               "Randomisation is what stops a drift in the room becoming a "
               "factor effect.")
    st.dataframe(design.runs, hide_index=True, use_container_width=True)
    st.download_button("Download the run sheet",
                       design.runs.to_csv(index=False).encode("utf-8"),
                       file_name="run-sheet.csv", mime="text/csv")
    st.caption("**In JMP:** %s" % design.verify_in)


def _analyse(table) -> None:
    if table is None:
        st.info("Load your completed run sheet in the sidebar — the factor "
                "columns plus the result.", icon="👈")
        return

    frame: pd.DataFrame = table.frame
    roles = st.session_state.get("roles")
    if not roles:
        st.info("Open **Explore** first.", icon="👈")
        return

    responses = [r.name for r in roles if r.role == VARIABLES]
    candidates = [r.name for r in roles if r.analysable]
    if not responses:
        st.error("No continuous result column in this file.", icon="🚫")
        return

    left, right = st.columns(2)
    response = left.selectbox("The result", responses, index=len(responses) - 1)
    factor_names = right.multiselect(
        "The factors you varied", [c for c in candidates if c != response],
        default=[c for c in candidates if c != response][:3])
    if len(factor_names) < 2:
        st.info("Choose at least two factors.", icon="👈")
        return

    interactions = st.checkbox(
        "Fit two-factor interactions", value=True,
        help="Untick if you do not expect them. It frees degrees of freedom "
             "for estimating error, which is often worth more than the "
             "interactions themselves.")

    if not st.button("Analyse it", type="primary"):
        return

    analysis = analyse_design(frame, factor_names, response,
                              interactions=interactions)
    if analysis.failed:
        st.error(analysis.failed, icon="🚫")
        return

    st.markdown(analysis.plain)
    metrics = st.columns(3)
    if analysis.r_squared_adj is not None:
        metrics[0].metric("Adjusted R²", "%.3f" % analysis.r_squared_adj)
    if analysis.sigma is not None:
        metrics[1].metric("s (residual sd)", "%.4g" % analysis.sigma)
    if analysis.df_error is not None:
        metrics[2].metric("Error degrees of freedom", "%g" % analysis.df_error)

    st.altair_chart(charts.effects_pareto(analysis.effects,
                                          analysis.lenth_margin, response),
                    use_container_width=True)
    st.altair_chart(charts.effects_normal_plot(analysis.effects),
                    use_container_width=True)

    st.dataframe(
        pd.DataFrame([{
            "Term": e.term,
            "Effect (low to high)": round(e.effect, 5),
            "Coefficient": round(e.coefficient, 5),
            "p": "%.4g" % e.p_value if e.p_value is not None else "Lenth",
            "Keep": "yes" if e.significant else "",
        } for e in analysis.effects]),
        hide_index=True, use_container_width=True)

    for caution in analysis.cautions:
        st.warning(caution, icon="⚠️")

    real_pairs = [e.term for e in analysis.effects if e.significant and " x " in e.term]
    if real_pairs:
        st.markdown("#### The interactions that matter")
        for term in real_pairs[:3]:
            first, second = [part.strip() for part in term.split(" x ")]
            st.altair_chart(
                charts.interaction_plot(frame, first, second, response),
                use_container_width=True)

    st.caption("**In JMP:** %s" % analysis.verify_in)
