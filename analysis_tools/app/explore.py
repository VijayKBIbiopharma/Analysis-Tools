"""What have I actually got?

The first page, and the one that earns its place quietly. Before anyone asks a
question of a dataset it is worth knowing what is in it: which columns are
measurements, which are groups, which are labels that should take no part in
anything, what is missing, and where the tool's own reading is uncertain.

Catching "this column is 60% blank" or "this could be read two ways" here saves
running an analysis whose answer was never going to be trustworthy.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis_tools.roles import (
    ATTRIBUTE_2, ATTRIBUTE_3PLUS, CONSTANT, DATETIME, EMPTY, IDENTIFIER,
    ROLE_LABELS, VARIABLES, classify_table,
)

ROLE_ICONS = {
    VARIABLES: "📈",
    ATTRIBUTE_2: "⚖️",
    ATTRIBUTE_3PLUS: "🗂️",
    IDENTIFIER: "🏷️",
    DATETIME: "📅",
    CONSTANT: "➖",
    EMPTY: "∅",
}


def render(table) -> None:
    frame: pd.DataFrame = table.frame
    # Already classified at load; recompute only if something cleared it.
    roles = st.session_state.get("roles")
    if not roles:
        roles = classify_table(frame)
        st.session_state["roles"] = roles

    st.subheader("What is in this file")
    read_as = table.source_name + (" — sheet “%s”" % table.sheet if table.sheet else "")
    st.caption("%s · %s" % (read_as, table.shape_text))
    for note in table.notes:
        st.info(note, icon="ℹ️")

    usable = [r for r in roles if r.analysable]
    uncertain = [r for r in roles if r.alternative]
    blocked = [r for r in roles if not r.analysable]

    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(frame):,}")
    cols[1].metric("Columns", len(frame.columns))
    cols[2].metric("Usable for analysis", len(usable))
    cols[3].metric("Needs a decision", len(uncertain))

    if not usable:
        st.error(
            "No column in this file can take part in an analysis. Every one is a "
            "label, a date, constant or empty. Check that the header row is the "
            "first row of the sheet.",
            icon="🚫",
        )
        return

    if uncertain:
        st.warning(
            "**%d column%s can be read two ways, and the reading changes which "
            "test applies.** They are marked below. Nothing has been chosen for "
            "you."
            % (len(uncertain), "" if len(uncertain) == 1 else "s"),
            icon="⚠️",
        )

    st.markdown("#### Columns")
    for role in roles:
        _column_card(role, frame)

    if blocked:
        st.markdown("#### Taking no part in any analysis")
        st.caption(
            "These are labels, dates, constants or empty columns. They are kept "
            "in the file and ignored by the analyses."
        )
        st.dataframe(
            pd.DataFrame([
                {"Column": r.name, "Read as": r.label,
                 "Why": r.notes[0] if r.notes else ""}
                for r in blocked
            ]),
            hide_index=True, use_container_width=True,
        )

    with st.expander("First rows of the file"):
        st.dataframe(frame.head(25), use_container_width=True)


def _column_card(role, frame: pd.DataFrame) -> None:
    icon = ROLE_ICONS.get(role.role, "•")
    flag = " · **needs a decision**" if role.alternative else ""
    with st.expander("%s  **%s** — %s%s" % (icon, role.name, role.label, flag),
                     expanded=bool(role.alternative)):
        left, right = st.columns([2, 3])
        with left:
            st.markdown(
                "- **Distinct values:** %d\n- **Present:** %d of %d rows"
                % (role.n_distinct, role.n_present, role.n)
            )
            if role.role == VARIABLES and role.numeric:
                series = pd.to_numeric(frame[role.name], errors="coerce").dropna()
                if not series.empty:
                    st.markdown(
                        "- **Range:** %.6g to %.6g\n- **Mean:** %.6g\n- **Median:** %.6g"
                        % (series.min(), series.max(), series.mean(), series.median())
                    )
            if role.counts:
                top = sorted(role.counts.items(), key=lambda kv: -kv[1])[:8]
                st.markdown("**Levels**")
                st.dataframe(
                    pd.DataFrame(top, columns=["Level", "Rows"]),
                    hide_index=True, use_container_width=True,
                )
        with right:
            if role.alternative:
                st.markdown("**This column can be read two ways**")
                st.caption(role.alternative_reason)
                st.radio(
                    "Read %s as:" % role.name,
                    options=[role.role, role.alternative],
                    format_func=lambda r: ROLE_LABELS[r],
                    key="role_choice_%s" % role.name,
                    horizontal=True,
                )
            for note in role.notes:
                st.caption("· " + note)
            if role.role == VARIABLES and role.numeric and not role.alternative:
                series = pd.to_numeric(frame[role.name], errors="coerce").dropna()
                if len(series) > 1:
                    st.bar_chart(
                        pd.cut(series, bins=min(20, max(5, role.n_distinct)))
                        .value_counts().sort_index().reset_index(drop=True),
                        height=120,
                    )
