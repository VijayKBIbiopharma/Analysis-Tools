"""Capability: is the process stable, and is it capable — in that order."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis_tools import charts
from analysis_tools.capability import MINIMUM_CAPABLE, assess_capability
from analysis_tools.app.fitness import explain
from analysis_tools.roles import VARIABLES


def render(table) -> None:
    frame: pd.DataFrame = table.frame
    roles = st.session_state.get("roles")
    if not roles:
        st.info("Open **Explore** first so the columns can be classified.", icon="👈")
        return

    measurements = [r.name for r in roles if r.role == VARIABLES]
    if not measurements:
        st.info(explain("Capability", roles, table.source_name), icon="🧭")
        return

    st.subheader("Capability")
    st.caption(
        "Two questions, and the order matters. A capability index computed on a "
        "process that is not in control describes the sample you happen to have, "
        "not a process."
    )

    cols = st.columns([2, 1, 1, 1])
    column = cols[0].selectbox("The measurement", measurements)
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    lsl = cols[1].number_input("Lower spec limit", value=None, placeholder="none",
                               format="%.6g")
    usl = cols[2].number_input("Upper spec limit", value=None, placeholder="none",
                               format="%.6g")
    target = cols[3].number_input("Target", value=None, placeholder="none",
                                  format="%.6g")

    if lsl is None and usl is None:
        st.info("Enter at least one specification limit. Capability is a "
                "comparison against a requirement — without one there is "
                "nothing to compare to.", icon="👈")
        st.caption("For reference: %d values, mean %.6g, standard deviation %.6g, "
                   "range %.6g to %.6g."
                   % (len(series), series.mean(), series.std(ddof=1),
                      series.min(), series.max()))
        return
    if lsl is not None and usl is not None and usl <= lsl:
        st.error("The upper limit has to be above the lower one.", icon="🚫")
        return

    result = assess_capability(series, lsl=lsl, usl=usl, target=target, name=column)

    if result.stability is not None:
        with st.container(border=True):
            st.markdown("#### 1. Is it stable?")
            st.markdown(result.stability.plain)
            flagged = sorted({i for signal in result.stability.signals
                              for i in signal.points})
            st.altair_chart(
                charts.control_chart(series.values, result.stability.centre,
                                     result.stability.upper,
                                     result.stability.lower, flagged, column),
                use_container_width=True)
            st.caption("**Reproduce it in JMP:** %s" % result.stability.verify_in)
    else:
        st.info("Fewer than eight measurements, so no control chart is drawn — "
                "a chart on that many points cannot distinguish a signal from "
                "the noise.", icon="ℹ️")

    with st.container(border=True):
        st.markdown("#### 2. Is it capable?")
        st.markdown(result.plain)
        st.altair_chart(
            charts.capability_histogram(series.values, lsl=lsl, usl=usl,
                                        target=target, name=column),
            use_container_width=True)

        if result.indices:
            overall = {k: v for k, v in result.indices.items() if k.startswith("P")}
            within = {k: v for k, v in result.indices.items() if k.startswith("C")}
            left, right = st.columns(2)
            with left:
                st.markdown("**Overall — what the customer receives**")
                st.caption("Uses the long-term standard deviation: everything, "
                           "including drift between runs.")
                st.dataframe(_index_frame(overall, result.intervals),
                             hide_index=True, use_container_width=True)
            with right:
                st.markdown("**Within — what the process could do**")
                st.caption("Uses the short-term standard deviation from the "
                           "moving range: the noise between consecutive units.")
                st.dataframe(_index_frame(within, result.intervals),
                             hide_index=True, use_container_width=True)

        metrics = st.columns(3)
        metrics[0].metric("Observed outside spec",
                          "%d of %d" % (round(result.observed_out * result.n), result.n))
        if result.predicted_ppm is not None:
            metrics[1].metric("Predicted PPM", "%,.0f".replace(",", "") % result.predicted_ppm)
            metrics[2].metric("Z.bench", "%.2f" % result.z_bench)

        for text in result.withheld:
            st.warning(text, icon="⚠️")
        for caution in result.cautions:
            st.info(caution, icon="ℹ️")

        st.caption("A capability index of %.2f is the usual minimum." % MINIMUM_CAPABLE)
        st.caption("**Reproduce it in JMP:** %s" % result.verify_in)


def _index_frame(indices: dict, intervals: dict) -> pd.DataFrame:
    rows = []
    for name, value in indices.items():
        interval = intervals.get(name)
        rows.append({
            "Index": name,
            "Value": round(value, 3),
            "95% interval": ("%.3f to %.3f" % interval) if interval else "—",
        })
    return pd.DataFrame(rows)
