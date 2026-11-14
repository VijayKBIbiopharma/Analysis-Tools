"""Cell line development: which vector to build, which clones to keep, how far
they can be taken.

Three questions, in the order a programme meets them. The first is the one
where the arithmetic is most willing to mislead, so the page puts what the
history can and cannot support *above* the number rather than in a footnote
under it.
"""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from analysis_tools import charts
from analysis_tools.cld_clones import (generation_limit, screen_early_reads)
from analysis_tools.cld_vector import (plan_vector, read_catalogue,
                                       replicates_for)
from analysis_tools.app.fitness import explain
from analysis_tools.io import read_delimited, read_excel
from analysis_tools.roles import (ATTRIBUTE_2, ATTRIBUTE_3PLUS, DATETIME,
                                  IDENTIFIER, VARIABLES)

GROUPING = (ATTRIBUTE_2, ATTRIBUTE_3PLUS)

#: Words that mark a column as a property of the molecule rather than a part
#: you choose. Getting this wrong is the one mistake this page must not make
#: quietly: a protein class left in the "parts" box is enumerated as though a
#: programme could pick its own modality, and the candidate count silently
#: multiplies by the number of modalities in the history. The hints set the
#: defaults and raise a question; they do not decide anything on their own.
MOLECULE_HINTS = ("protein", "class", "modality", "molecule", "format",
                  "mw", "molecular", "weight", "kda", "pi", "isoelectric",
                  "charge", "sequence", "target", "antigen", "species",
                  "subclass", "isotype", "glyco")

#: And words that mark a column as the thing a build is judged on.
RESPONSE_HINTS = ("titer", "titre", "yield", "productivity", "qp", "expression",
                  "g_l", "gl", "concentration", "potency")

#: Words that mark a column as the *late* read in a screen - the one the early
#: ones are meant to predict. Checked before RESPONSE_HINTS, since a screen
#: usually holds several titre columns and only one of them is the endpoint.
LATE_HINTS = ("final", "harvest", "end", "endpoint", "fedbatch", "fb",
              "day21", "d21", "production", "late")


def _looks_like(name: str, hints) -> bool:
    """Does this column name carry one of the hint words?

    camelCase is split as well as underscores, because the column that caused
    this function to exist was "ProteinClass" - underscore splitting alone
    left it unmatched and it defaulted into the parts box.

    Short hints are matched as whole words only. "pi" as a substring matches
    "Copies", which would mark copy number - a part a programme genuinely
    chooses - as a property of the molecule.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name))
    tokens = {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}
    # The undivided name too, so "pI" - which camel-splits to p and I -
    # still matches the two-letter hint.
    tokens.add(re.sub(r"[^a-z0-9]+", "", str(name).lower()))
    whole = "".join(tokens)
    return any(h in tokens if len(h) <= 2 else (h in tokens or h in whole)
               for h in hints)


def render(table) -> None:
    vector, clones, limits = st.tabs(
        ["Vector selection", "Clone screening", "Generation limits"])
    with vector:
        _vector(table)
    with clones:
        _clones(table)
    with limits:
        _limits(table)


def _roles_or_stop():
    roles = st.session_state.get("roles")
    if not roles:
        st.info("Open **Explore** first so the columns can be classified.",
                icon="👈")
        return None
    return {r.name: r for r in roles}


# --------------------------------------------------------------- tab 1

def _vector(table) -> None:
    st.caption(
        "A new programme arrives. The site has a catalogue of parts it can "
        "build with and a history of what previous constructs did. This picks "
        "the build and fills a plate to confirm it."
    )
    by_name = _roles_or_stop()
    if by_name is None:
        return
    frame: pd.DataFrame = table.frame

    responses = [n for n, r in by_name.items() if r.role == VARIABLES]
    categorical = [n for n, r in by_name.items() if r.role in GROUPING]
    usable = [n for n, r in by_name.items()
              if r.role not in (IDENTIFIER, DATETIME)]
    if not responses or not categorical:
        st.info(explain("Cell line development", list(by_name.values()),
                        table.source_name),
                icon="🧭")
        st.caption(
            "Vector selection in particular needs a column naming each "
            "construct part — promoter, marker, helper plasmid.")
        return

    # Defaults from the column names, because the two boxes below are not
    # interchangeable: a molecule attribute left among the parts gets
    # enumerated as though a programme could choose its own modality.
    likely_response = next((n for n in responses
                            if _looks_like(n, RESPONSE_HINTS)), responses[-1])
    molecule_like = [n for n in usable if _looks_like(n, MOLECULE_HINTS)]

    top = st.columns(2)
    response = top[0].selectbox(
        "What the build is judged on", responses,
        index=responses.index(likely_response))
    parts_default = [n for n in categorical
                     if n != response and n not in molecule_like][:4]
    factors = top[1].multiselect(
        "The construct parts you choose between", 
        [n for n in categorical if n != response],
        default=parts_default,
        help="Promoter, selection marker, helper plasmid, copy number — "
             "whatever the catalogue offers. Only things a programme can "
             "actually decide.")
    protein = st.multiselect(
        "What describes the molecule itself",
        [n for n in usable if n != response and n not in factors],
        default=[n for n in molecule_like
                 if n != response and n not in factors],
        help="Class, molecular weight, pI. These are *not* things you choose; "
             "they are what arrives. Naming them is what lets the tool tell "
             "you when your molecule is unlike anything in the history.")
    if not factors:
        st.info("Choose at least one construct part.", icon="👈")
        return

    # The guardrail. The tool cannot know which is which, but it can ask
    # rather than quietly scoring a modality as a buildable option.
    misplaced = [n for n in factors if _looks_like(n, MOLECULE_HINTS)]
    if misplaced:
        st.warning(
            "**%s %s like %s of the molecule rather than %s you choose.** If "
            "so, move %s to the box below: left among the parts, %s enumerated "
            "as buildable options, so the tool will offer you a plate that "
            "varies something a programme cannot pick — and the candidate "
            "count multiplies by the number of %s in the history."
            % (", ".join(misplaced),
               "looks" if len(misplaced) == 1 else "look",
               "a property" if len(misplaced) == 1 else "properties",
               "something" if len(misplaced) == 1 else "things",
               "it" if len(misplaced) == 1 else "them",
               "it is" if len(misplaced) == 1 else "they are",
               "levels" if len(misplaced) > 1 else "levels"),
            icon="⚠️")

    catalogue = _catalogue_section(frame, factors)
    programme = _programme_section(frame, protein)

    settings = st.columns(3)
    wells = settings[0].number_input("Wells on the confirmation plate",
                                     min_value=4, max_value=96, value=12, step=4)
    confidence = settings[1].select_slider(
        "Interval", options=[0.80, 0.90, 0.95], value=0.90,
        format_func=lambda v: "%.0f%%" % (100 * v))
    settings[2].caption(
        "The interval is built from errors the model made on runs it had never "
        "seen, so it does not narrow just because the model memorised its own "
        "data.")

    if not st.button("Plan the build", type="primary", key="cld_plan"):
        return

    with st.spinner("Fitting the history and scoring the catalogue…"):
        plan = plan_vector(frame, catalogue, response, protein,
                           programme=programme or None, wells=int(wells),
                           confidence=float(confidence))
    if plan.failed:
        st.error(plan.failed, icon="🚫")
        return

    _transferability_panel(plan)

    with st.container(border=True):
        st.markdown("#### The build")
        st.markdown(plan.plain)
        metrics = st.columns(4)
        metrics[0].metric("Predicted %s" % response,
                          "%.4g" % plan.best["Predicted"])
        metrics[1].metric("Interval half-width", "±%.4g" % plan.interval_width)
        metrics[2].metric("Cross-validated R²", "%.3f" % plan.r_squared_cv)
        metrics[3].metric("Candidates scored",
                          "%d" % plan.n_candidates,
                          delta=None if not plan.sampled
                          else "of %d possible" % plan.n_possible,
                          delta_color="off")
        st.caption(
            "Cross-validated R² is measured on constructs the model never saw, "
            "which is the only version of it worth reading. It also sets how "
            "hard the plate concentrates: %d replicates here."
            % replicates_for(plan.r_squared_cv))

        for caution in plan.cautions:
            st.warning(caution, icon="⚠️")

        best, safest = st.columns(2)
        with best:
            st.markdown("**Highest predicted**")
            st.dataframe(_as_rows(plan.best, plan.candidates, protein),
                         hide_index=True, use_container_width=True)
        with safest:
            st.markdown("**Highest lower bound** — least likely to disappoint")
            st.dataframe(_as_rows(plan.safest, plan.candidates, protein),
                         hide_index=True, use_container_width=True)

        st.markdown("#### The confirmation plate")
        st.caption(
            "Run them in well order. A single observation of a vector is "
            "mostly noise, so the plate builds fewer candidates more times "
            "— measured over 4,000 simulated sheets, building everything once "
            "was the worst of the five strategies tried.")
        st.dataframe(plan.sheet, hide_index=True, use_container_width=True)
        st.download_button("Download the plate layout",
                           plan.sheet.to_csv(index=False).encode("utf-8"),
                           file_name="confirmation-plate.csv", mime="text/csv")

        with st.expander("Every candidate, ranked"):
            st.dataframe(plan.candidates, hide_index=True,
                         use_container_width=True)
        st.caption("**Reproduce it in JMP:** %s" % plan.verify_in)


def _transferability_panel(plan) -> None:
    transfer = plan.transferability
    if transfer is None:
        return
    standing = (transfer.per_programme[0]["standing"]
                if transfer.per_programme else None)
    if not transfer.can_transfer:
        box, icon = st.error, "🚫"
    elif standing == "untested":
        box, icon = st.warning, "⚠️"
    elif standing == "thin":
        box, icon = st.warning, "⚠️"
    else:
        box, icon = st.success, "✅"
    detail = []
    if transfer.varying:
        detail.append("varies across the history: %s" % ", ".join(transfer.varying))
    if transfer.constant:
        detail.append("held constant: %s" % ", ".join(transfer.constant))
    box("**What this history can say about your molecule** — %s%s"
        % (transfer.why, ("  \n" + "; ".join(detail)) if detail else ""),
        icon=icon)


def _catalogue_section(frame: pd.DataFrame, factors) -> dict:
    """What can be built, which is not the same as what has been built."""
    with st.expander("The catalogue — what the site can build", expanded=False):
        st.caption(
            "By default the catalogue is whatever appears in the history. If "
            "the site can build parts it never has, add them: they will be "
            "listed as unbuildable-by-evidence rather than silently scored, "
            "because no model can place a level it has never seen.")
        uploaded = st.file_uploader(
            "A catalogue sheet (optional)", type=["csv", "tsv", "txt", "xlsx", "xlsm"],
            key="cld_catalogue",
            help="Columns: Factor, Levels (semicolon or pipe separated), or "
                 "Factor with Min / Max / Step for a numeric part.")
        if uploaded is not None:
            data = uploaded.getvalue()
            try:
                if uploaded.name.lower().endswith((".xlsx", ".xlsm")):
                    sheet = read_excel(data, source_name=uploaded.name)
                else:
                    sheet = read_delimited(data, source_name=uploaded.name)
                parsed = read_catalogue(sheet.frame)
                if parsed:
                    st.success("Read %d factors: %s"
                               % (len(parsed), ", ".join(parsed)))
                    return {k: v for k, v in parsed.items() if k in factors} or parsed
                st.warning("No factors found in that sheet — falling back to "
                           "the levels in the history.", icon="⚠️")
            except Exception as exc:                  # noqa: BLE001 - shown
                st.error("Could not read that catalogue: %s" % exc, icon="🚫")

        catalogue = {}
        for name in factors:
            observed = sorted(frame[name].dropna().astype(str).unique())
            extra = st.text_input(
                "%s — parts you can build but have not" % name,
                key="cld_extra_%s" % name, value="",
                placeholder="separate with semicolons")
            added = [p.strip() for p in extra.split(";") if p.strip()]
            catalogue[name] = observed + [a for a in added if a not in observed]
        return catalogue


def _programme_section(frame: pd.DataFrame, protein) -> dict:
    """The incoming molecule."""
    if not protein:
        return {}
    with st.expander("The incoming programme", expanded=True):
        st.caption(
            "Describe the molecule you are about to express. Leaving this "
            "blank predicts for whatever the history holds first, which is "
            "rarely what you want.")
        programme = {}
        # The categorical attributes first, so the numeric defaults can be
        # taken from runs of that kind of molecule. Defaulting each column to
        # its own overall median produced a chimera - an Fc-fusion at the
        # median weight of the whole history, which is a bispecific's - and
        # the first thing the page showed was a prediction for a molecule
        # that does not exist.
        ordered = ([n for n in protein
                    if not pd.api.types.is_numeric_dtype(frame[n].dropna())
                    or frame[n].dropna().nunique() <= 6]
                   + [n for n in protein
                      if pd.api.types.is_numeric_dtype(frame[n].dropna())
                      and frame[n].dropna().nunique() > 6])
        columns = st.columns(min(3, len(protein)))
        for i, name in enumerate(ordered):
            column = columns[i % len(columns)]
            series = frame[name].dropna()
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > 6:
                matching = frame
                for chosen, value in programme.items():
                    narrowed = matching[matching[chosen].astype(str) == str(value)]
                    if not narrowed.empty:
                        matching = narrowed
                like_it = matching[name].dropna()
                default = float((like_it if not like_it.empty else series).median())
                programme[name] = column.number_input(
                    name, value=default, key="cld_prog_%s" % name,
                    help="History runs from %.4g to %.4g over everything; %.4g "
                         "is typical of the molecule chosen here"
                         % (series.min(), series.max(), default))
            else:
                levels = sorted(series.astype(str).unique())
                picked = column.selectbox(
                    name, levels + ["something else…"], key="cld_prog_%s" % name)
                if picked == "something else…":
                    picked = column.text_input(
                        "%s — what is it?" % name, key="cld_prog_other_%s" % name,
                        placeholder="a modality the history does not hold")
                if picked:
                    programme[name] = picked
        return {n: programme[n] for n in protein if n in programme}


def _as_rows(choice: dict, candidates: pd.DataFrame, protein) -> pd.DataFrame:
    hide = set(protein)
    return pd.DataFrame([
        {"Part": key, "Chosen": choice[key]}
        for key in candidates.columns
        if key not in hide and key not in ("Predicted", "Lower bound",
                                           "Upper bound")
    ] + [
        {"Part": "Predicted", "Chosen": "%.4g" % choice["Predicted"]},
        {"Part": "Interval", "Chosen": "%.4g to %.4g"
         % (choice["Lower bound"], choice["Upper bound"])},
    ])


# --------------------------------------------------------------- tab 2

def _clones(table) -> None:
    st.caption(
        "A screen measures hundreds of clones early and keeps a handful. The "
        "question is whether the early measurement predicts the one that "
        "matters — and the answer is often no, which is worth knowing before "
        "the handful is chosen."
    )
    by_name = _roles_or_stop()
    if by_name is None:
        return
    frame: pd.DataFrame = table.frame
    numeric = [n for n, r in by_name.items() if r.role == VARIABLES]
    if len(numeric) < 2:
        st.info(explain("Cell line development", list(by_name.values()),
                        table.source_name),
                icon="🧭")
        st.caption("Clone screening needs one or more early reads and the "
                   "late measurement they are meant to predict.")
        return

    # The endpoint, not whichever column happens to come last. Defaulting to
    # the last numeric column meant a file whose final column was viability
    # screened the early reads against viability.
    likely_late = next((n for n in numeric if _looks_like(n, LATE_HINTS)), None)
    if likely_late is None:
        likely_late = next((n for n in reversed(numeric)
                            if _looks_like(n, RESPONSE_HINTS)), numeric[-1])

    left, right = st.columns(2)
    late = left.selectbox("The measurement that matters", numeric,
                          index=numeric.index(likely_late), key="cld_late")
    early = right.multiselect("The early reads", [n for n in numeric if n != late],
                              default=[n for n in numeric if n != late],
                              key="cld_early")
    if not early:
        st.info("Choose at least one early read.", icon="👈")
        return

    if not st.button("Screen them", type="primary", key="cld_screen"):
        return
    with st.spinner("Shuffling the response to establish the noise floor…"):
        result = screen_early_reads(frame, early, late)
    if result.failed:
        st.error(result.failed, icon="🚫")
        return

    with st.container(border=True):
        st.markdown("#### What the early reads carry")
        st.markdown(result.plain)
        metrics = st.columns(3)
        metrics[0].metric("Cross-validated R²", "%.3f" % result.r_squared_cv
                          if result.r_squared_cv is not None else "—")
        metrics[1].metric("Clones used", result.n)
        metrics[2].metric(
            "Noise floor, all reads", "%.4f" % result.noise_ceiling
            if result.noise_ceiling is not None else "—",
            help="The highest score any read reached when the response was "
                 "shuffled. Each read also has its own floor, drawn as a tick "
                 "on its bar below — that is the one its verdict is based on.")

        st.altair_chart(
            charts.importance_bars(result.reads, result.noise_ceiling, late),
            use_container_width=True)
        st.caption(
            "The tick on each bar is where that read scores when the response "
            "is shuffled — pure chance. A bar that edges past its own tick and "
            "is still grey did not survive the adjustment for how many reads "
            "were tested.")

        st.dataframe(pd.DataFrame([
            {"Read": r.name,
             "Importance": round(r.importance, 4),
             "Correlation with %s" % late:
                 round(r.correlation, 3) if r.correlation is not None else "—",
             "p (adjusted)": "%.4g" % r.p_adjusted
                 if r.p_adjusted is not None else "—",
             "Verdict": "carries information" if r.above_noise else "no evidence"}
            for r in result.reads]), hide_index=True, use_container_width=True)
        st.caption(
            "The p-values are adjusted for the number of reads tested — "
            "test six and one clears 0.05 by chance about a quarter of the "
            "time. Importance is a share, so two reads measuring the same "
            "thing split the credit between them.")

        for caution in result.cautions:
            st.warning(caution, icon="⚠️")
        st.caption("**Reproduce it in JMP:** %s" % result.verify_in)


# --------------------------------------------------------------- tab 3

def _limits(table) -> None:
    st.caption(
        "A cell line is taken out to some generation and the question is how "
        "far it stays inside specification. The limit is read off the "
        "confidence bound rather than the fitted line, and is not projected "
        "beyond twice the generations actually run."
    )
    by_name = _roles_or_stop()
    if by_name is None:
        return
    frame: pd.DataFrame = table.frame
    numeric = [n for n, r in by_name.items() if r.role == VARIABLES]
    if len(numeric) < 2:
        st.info(explain("Cell line development", list(by_name.values()),
                        table.source_name),
                icon="🧭")
        st.caption("Generation limits need a passage or generation column and "
                   "a measurement to track against it.")
        return

    guess = next((i for i, n in enumerate(numeric)
                  if any(w in n.lower() for w in
                         ("gener", "passage", "pdl", "day", "age"))), 0)
    left, right = st.columns(2)
    generation = left.selectbox("Generation or passage", numeric, index=guess,
                                key="cld_gen")
    tracked = [n for n in numeric if n != generation]
    # Again the hints rather than the last column: the specification limit
    # below belongs to a particular measurement, and tracking the wrong one
    # against it produces a confident answer to a question nobody asked.
    likely = next((n for n in tracked if _looks_like(n, RESPONSE_HINTS)),
                  tracked[-1])
    response = right.selectbox("What is tracked", tracked,
                               index=tracked.index(likely), key="cld_resp")

    st.markdown("**The specification**")
    st.caption("Give the limit the measurement would drift towards. Both, if "
               "the drift could go either way.")
    spec = st.columns(2)
    use_lsl = spec[0].checkbox("Lower limit", value=True, key="cld_use_lsl")
    lsl = spec[0].number_input("LSL", value=0.0, key="cld_lsl",
                               disabled=not use_lsl, format="%.6g")
    use_usl = spec[1].checkbox("Upper limit", key="cld_use_usl")
    usl = spec[1].number_input("USL", value=0.0, key="cld_usl",
                               disabled=not use_usl, format="%.6g")

    if not st.button("Find the limit", type="primary", key="cld_limit"):
        return
    result = generation_limit(frame, generation, response,
                              lsl=float(lsl) if use_lsl else None,
                              usl=float(usl) if use_usl else None)
    if result.failed:
        st.error(result.failed, icon="🚫")
        return

    with st.container(border=True):
        st.markdown("#### How far it can be taken")
        st.markdown(result.plain)
        metrics = st.columns(4)
        metrics[0].metric(
            "Generation limit",
            "—" if result.limit is None else
            ("≥ %.0f" % result.limit if result.limit_capped
             else "%.0f" % result.limit))
        metrics[1].metric("Drift per generation",
                          "—" if result.slope is None else "%.5g" % result.slope)
        metrics[2].metric("Furthest actually run",
                          "—" if result.max_observed is None
                          else "%.0f" % result.max_observed)
        metrics[3].metric("Passage points", result.n)

        if result.curve is not None:
            target = (float(lsl) if use_lsl and (result.slope or 0) < 0
                      else float(usl) if use_usl else None)
            st.altair_chart(
                charts.drift_with_bound(
                    frame[[generation, response]].apply(
                        pd.to_numeric, errors="coerce").dropna(),
                    result.curve, generation, response, spec=target,
                    limit=result.limit, max_observed=result.max_observed),
                use_container_width=True)

        for caution in result.cautions:
            st.warning(caution, icon="⚠️")
        st.caption("**Reproduce it in JMP:** %s" % result.verify_in)
