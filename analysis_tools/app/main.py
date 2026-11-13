"""Analysis Tools — the application entry point.

Run locally:      streamlit run app.py      (from the repository root)
In a container:   the Dockerfile's CMD does the same thing.

Imports here are absolute rather than relative on purpose. Streamlit executes
its entry point as a script rather than importing it as part of a package, so a
relative import raises "attempted relative import with no known parent package"
the moment the app starts. The thin app.py at the repository root is the entry
point; it puts the root on the path and calls main().

Nothing here writes to disk. An uploaded file is held in memory for the length
of the session and is gone when the browser disconnects.
"""

from __future__ import annotations

import streamlit as st

from analysis_tools import __version__
from analysis_tools.app import (capability_page, cld_page, compare_page,
                                doe_page, explore, msa_page, relate_page)
from analysis_tools.io import excel_sheet_names, read_delimited, read_excel
from analysis_tools.logging_setup import configure, get

configure()
log = get("app")

DELIMITED = ("csv", "tsv", "txt", "tab")
WORKBOOK = ("xlsx", "xlsm", "xls")

#: The modules, in the order the work has to happen. Everything not yet built
#: is listed rather than hidden, so the shape of the tool is visible and nobody
#: goes looking for something that is not there.
MODULES = [
    ("Explore", "What is in this file", True),
    ("Compare groups", "Which test applies, and run it", True),
    ("Capability", "Is the process capable, and is it stable", True),
    ("Measurement systems", "Gage R&R, and whether the method can see it", True),
    ("Relationships", "Regression, model reduction", True),
    ("Design of experiments", "Design, analyse, reduce, optimise", True),
    ("Cell line development",
     "Vector selection, clone screening, generation limits", True),
    ("Learn", "How each method works, and when it does not", False),
]


def main() -> None:
    st.set_page_config(
        page_title="Analysis Tools",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _sidebar_header()
    table = _file_section()
    choice = _module_picker(table is not None)

    st.title("Analysis Tools")
    st.caption(
        "Statistics, design of experiments and machine learning for process and "
        "product development. Names the right technique, explains the result in "
        "plain language, and says when the data cannot support an answer."
    )

    if table is None:
        if choice == "Design of experiments":
            doe_page.render(None)
            return
        _welcome()
        return

    pages = {"Explore": explore.render, "Compare groups": compare_page.render,
             "Capability": capability_page.render,
             "Measurement systems": msa_page.render,
             "Relationships": relate_page.render,
             "Design of experiments": doe_page.render,
             "Cell line development": cld_page.render}
    renderer = pages.get(choice)
    if renderer is not None:
        renderer(table)
    else:
        _not_built_yet(choice)


def _sidebar_header() -> None:
    st.sidebar.markdown("### Analysis Tools")
    st.sidebar.caption("version %s" % __version__)
    st.sidebar.divider()


#: The bundled examples, with the module each one is for. Offered because a
#: tool that cannot be opened without first finding a file is a tool most
#: people never open - on a phone or a locked-down desktop, "upload a CSV" is
#: where the trial ends. These are read-only and synthetic; no real process
#: data ships with the application.
EXAMPLES = {
    "— none, I will upload my own —": None,
    "Construct history (vector selection)": "history.csv",
    "Clone screen (early reads vs final)": "clone-screen.csv",
    "Passage stability (generation limits)": "passage-stability.csv",
    "Completed DOE (effects, curvature)": "doe-completed.csv",
    "Capability run (Cp, Cpk, stability)": "capability-run.csv",
    "Gage study (measurement systems)": "gage-study.csv",
    "Process runs (compare groups)": "process-runs.csv",
}


def _examples_directory():
    """Where the bundled examples live, wherever the app was started from."""
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent / "examples"


def _file_section():
    """Upload, held in memory only — or one of the bundled examples."""
    st.sidebar.markdown("#### Your data")

    # The example picker goes first, above the uploader, because of what the
    # uploader looks like on a phone: a large "drag and drop file here" box,
    # an instruction a phone cannot follow, filling the screen and pushing
    # everything else below the fold. Ordered this way there is always
    # something to tap.
    chosen = st.sidebar.selectbox(
        "Start with an example", list(EXAMPLES),
        help="Synthetic data, shipped with the tool. Each one is built to "
             "show a particular module — including the cases where it "
             "refuses to give an answer.")

    uploaded = st.sidebar.file_uploader(
        "or load your own file",
        type=list(DELIMITED + WORKBOOK),
        help="Held in memory for this session only. Never written to disk. "
             "A file you load here is used instead of the example above.",
    )

    if uploaded is None:
        filename = EXAMPLES[chosen]
        if filename is None:
            st.session_state.pop("table", None)
            return None
        path = _examples_directory() / filename
        try:
            data, name = path.read_bytes(), filename
        except OSError as exc:                    # noqa: BLE001 - shown
            st.sidebar.error("That example is missing from this copy.")
            log.warning("example %s unreadable: %s", filename, exc)
            return None
    else:
        data, name = uploaded.getvalue(), uploaded.name
    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""

    try:
        if suffix in WORKBOOK:
            sheets = excel_sheet_names(data)
            sheet = sheets[0] if len(sheets) == 1 else st.sidebar.selectbox(
                "Sheet", sheets, help="%d sheets in this workbook" % len(sheets)
            )
            table = read_excel(data, sheet=sheet, source_name=name)
        else:
            table = read_delimited(data, source_name=name)
    except Exception as exc:                      # noqa: BLE001 - shown to the user
        st.sidebar.error("Could not read that file.")
        st.error(
            "**%s could not be read.**\n\n%s\n\nIf it is a spreadsheet, check the "
            "header row is the first row of the sheet. If it is a text file, check "
            "it is comma, tab, semicolon or pipe separated." % (name, exc),
            icon="🚫",
        )
        log.warning("failed to read %s: %s", name, exc)
        return None

    log.info("read %s: %s", name, table.shape_text)
    st.sidebar.success("%s%s" % (table.shape_text,
                                 "" if uploaded is not None else "  ·  example"))
    st.session_state["table"] = table
    return table


def _module_picker(have_data: bool) -> str:
    st.sidebar.divider()
    st.sidebar.markdown("#### Analysis")
    labels, lookup = [], {}
    for name, blurb, built in MODULES:
        label = name if built else "%s  ·  not built yet" % name
        labels.append(label)
        lookup[label] = name
    # The design builder needs no data - you use it before the experiment -
    # so the picker stays live even with nothing loaded.
    chosen = st.sidebar.radio("Module", labels, label_visibility="collapsed")
    blurb = next(b for n, b, _ in MODULES if n == lookup[chosen])
    st.sidebar.caption(blurb)
    return lookup[chosen]


def _welcome() -> None:
    st.info("Load a file in the sidebar to begin.", icon="👈")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "#### What this does\n"
            "Reads your file, works out what each column is — a measurement, a "
            "group, a label — and then names the analysis that fits and runs it.\n\n"
            "It is built to say *no* as readily as *yes*: where the data cannot "
            "support an answer it says so, rather than printing a number that "
            "looks defensible and is not."
        )
    with right:
        st.markdown(
            "#### Your data stays here\n"
            "The file you load is held in memory for this session and is never "
            "written to disk. Nothing is sent anywhere.\n\n"
            "Run it on your own machine and the data never leaves it at all."
        )
    st.markdown("#### What it will cover")
    st.dataframe(
        [{"Module": n, "What it answers": b, "Status": "built" if ok else "not built yet"}
         for n, b, ok in MODULES],
        hide_index=True, use_container_width=True,
    )


def _not_built_yet(name: str) -> None:
    blurb = next(b for n, b, _ in MODULES if n == name)
    st.warning(
        "**%s is not built yet.**\n\n%s\n\nThe file you loaded is still here — "
        "switch to **Explore** to look at it." % (name, blurb),
        icon="🚧",
    )


if __name__ == "__main__":
    main()
