"""Which construct should we build for an incoming molecule?

A new programme arrives with a protein that has not been expressed here before.
The site has a catalogue of parts it can build with - promoters, markers,
helper plasmids, copy numbers - and a history of what previous constructs did.
The question is which combination to build, and what to put on the plate to
confirm it.

The measurement everything turns on
-----------------------------------
Predicting for a new *vector* is not the same problem as predicting for a new
*protein*, and treating them alike is the most damaging thing this module could
do.

A vector factor can be extrapolated a little: the history holds four promoters,
you ask about a fifth, the model maps it to the nearest and says so. A protein
attribute cannot. If every historical run was a monoclonal antibody, the
history contains no information whatsoever about how a bispecific responds to a
promoter choice. The arithmetic will not complain - a model fitted on one
protein class returns a confident number for another, and the interval beside
it is *narrow*, because nothing in the residuals knows the question changed.

So the history is measured before anything is predicted, and a programme whose
protein is unlike anything run before is told so in those words rather than
handed a number with a tight band on it.

Four hundred runs of one molecule are four hundred runs of one molecule. The
effective sample size for a cross-protein question is one.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

VERIFY = "Analyze > Fit Model > Standard Least Squares, then Prediction Profiler"

#: Full enumeration is preferred wherever it is affordable: sampling means the
#: best build may simply not be among the ones scored, which is invisible in
#: the output.
CANDIDATE_CAP = 20000

#: Below this many distinct proteins, a protein effect is a line through two
#: points.
THIN_PROTEIN_COUNT = 2

#: How to spend the wells. Measured over 4,000 simulated 48-well sheets drawn
#: from 36 candidates, scoring how often the sheet identifies the truly best
#: vector:
#:
#:     strategy            poor prediction   good prediction
#:     36 distinct x 1          33.1%             36.4%
#:     top 24 x 2               39.1%             42.3%
#:     top 16 x 3               41.9%             48.4%
#:     top 12 x 4               41.4%             55.9%
#:     top 8  x 6               37.2%             61.4%
#:
#: Running everything once is the worst option in both columns: a single
#: observation of a vector is mostly noise, so the sheet ends up ranking noise.
#: How far to concentrate depends on how much the ranking can be trusted, which
#: cross-validation measures rather than taste deciding.
def replicates_for(r_squared: Optional[float]) -> int:
    """How many times to build each candidate, from the model's own R-squared."""
    value = 0.0 if r_squared is None else max(0.0, min(1.0, r_squared))
    r = np.sqrt(value)                     # correlation: the ranking is what matters
    return 3 if r < 0.45 else (4 if r < 0.70 else 6)


@dataclass
class Transferability:
    n_proteins: int
    varying: list = field(default_factory=list)
    constant: list = field(default_factory=list)
    can_transfer: bool = False
    why: str = ""
    per_programme: list = field(default_factory=list)


@dataclass
class VectorPlan:
    transferability: Optional[Transferability] = None
    candidates: Optional[pd.DataFrame] = None
    n_candidates: int = 0
    n_possible: int = 0
    sampled: bool = False
    coverage: list = field(default_factory=list)
    best: Optional[dict] = None
    safest: Optional[dict] = None
    sheet: Optional[pd.DataFrame] = None
    r_squared_cv: Optional[float] = None
    interval_width: Optional[float] = None
    plain: str = ""
    cautions: list = field(default_factory=list)
    verify_in: str = VERIFY
    failed: Optional[str] = None


def assess_transferability(history: pd.DataFrame, protein_columns: Sequence[str],
                           programme: Optional[dict] = None) -> Transferability:
    """What the history can say about a protein it has never seen."""
    varying, constant, detail = [], [], {}
    for name in protein_columns:
        values = history[name].dropna()
        levels = sorted(values.astype(str).unique())
        numeric = pd.to_numeric(values, errors="coerce")
        is_numeric = numeric.notna().all()
        detail[name] = {
            "levels": levels, "numeric": is_numeric,
            "low": float(numeric.min()) if is_numeric else None,
            "high": float(numeric.max()) if is_numeric else None,
        }
        (varying if len(levels) > 1 else constant).append(name)

    # Distinct protein identities: the unique combinations, which is the sample
    # size that matters for a question about a different molecule. Four hundred
    # runs of one molecule are one protein, not four hundred.
    if protein_columns:
        signatures = history[list(protein_columns)].astype(str).agg("\u0001".join, axis=1)
        n_proteins = int(signatures.nunique())
    else:
        n_proteins = 0

    if n_proteins < 2:
        why = ("Every historical run is the same protein, so nothing here can "
               "say how a different one behaves.")
    elif not varying:
        why = ("No protein attribute varies across the historical runs, so the "
               "data cannot separate a protein effect from anything else.")
    elif n_proteins <= THIN_PROTEIN_COUNT:
        why = ("Only %d distinct proteins appear in the history. A protein "
               "effect fitted on two molecules is a line through two points."
               % n_proteins)
    else:
        why = "%d distinct proteins appear in the history." % n_proteins

    out = Transferability(
        n_proteins=n_proteins, varying=varying, constant=constant,
        can_transfer=n_proteins >= 2 and bool(varying), why=why)

    if programme:
        novel, unseen, no_evidence = [], [], []
        for name, value in programme.items():
            info = detail.get(name)
            if info is None:
                novel.append("%s = %s is not an attribute the history records at "
                             "all" % (name, value))
                continue
            if name in constant:
                no_evidence.append(
                    "%s is held at %s in every historical run, so the data "
                    "cannot say what changing it does"
                    % (name, info["levels"][0] if info["levels"] else "one value"))
                continue
            if info["numeric"]:
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(number) and not (info["low"] <= number <= info["high"]):
                    unseen.append("%s = %s is outside the historical range %.4g "
                                  "to %.4g" % (name, value, info["low"], info["high"]))
            elif str(value) not in info["levels"]:
                unseen.append("%s = %s is a value the historical runs do not "
                              "contain (they hold %s)"
                              % (name, value, ", ".join(info["levels"][:5])))
        if novel or unseen or no_evidence:
            standing = "untested"
        elif n_proteins <= THIN_PROTEIN_COUNT:
            standing = "thin"
        else:
            standing = "supported"
        out.per_programme = [{"standing": standing, "novel": novel,
                              "unseen": unseen, "no_evidence": no_evidence}]
    return out


def enumerate_candidates(catalogue: dict, cap: int = CANDIDATE_CAP,
                         seed: int = 13) -> tuple:
    """Every buildable vector, or a sample spanning the catalogue if too many."""
    names = list(catalogue)
    sizes = [len(catalogue[n]) for n in names]
    total = int(np.prod(sizes)) if sizes else 0
    if total == 0:
        return pd.DataFrame(), 0, False
    if total <= cap:
        rows = list(itertools.product(*[catalogue[n] for n in names]))
        return pd.DataFrame(rows, columns=names), total, False
    rng = np.random.default_rng(seed)
    seen, rows = set(), []
    while len(rows) < cap and len(seen) < cap * 4:
        pick = tuple(catalogue[n][rng.integers(len(catalogue[n]))] for n in names)
        if pick in seen:
            continue
        seen.add(pick)
        rows.append(pick)
    return pd.DataFrame(rows, columns=names), total, True



def read_catalogue(frame: pd.DataFrame) -> dict:
    """A catalogue sheet into {factor: [levels]}.

    Two shapes, because both turn up in practice. A categorical factor lists
    its levels in one cell - separated by semicolons, pipes or commas, since
    whoever typed it had no reason to know which - and a numeric factor gives
    Min, Max and Step instead. Copy number as 1 to 4 step 1 is the same
    statement as "1; 2; 3; 4" and should not need rewriting.
    """
    columns = {str(c).strip().lower(): c for c in frame.columns}
    factor_col = columns.get("factor") or columns.get("name") or frame.columns[0]
    level_col = columns.get("levels") or columns.get("level")
    out = {}
    for _, row in frame.iterrows():
        name = str(row[factor_col]).strip()
        if not name or name.lower() == "nan":
            continue
        levels = []
        raw = row[level_col] if level_col is not None else None
        if raw is not None and pd.notna(raw) and str(raw).strip():
            parts = [p.strip() for p in re.split(r"[;|,]", str(raw))]
            levels = [p for p in parts if p]
        else:
            low, high, step = (row.get(columns.get(k)) for k in ("min", "max", "step"))
            low, high = pd.to_numeric(pd.Series([low, high]), errors="coerce")
            step = pd.to_numeric(pd.Series([step]), errors="coerce").iloc[0]
            if pd.notna(low) and pd.notna(high):
                if pd.isna(step) or step <= 0:
                    step = 1.0
                count = int(np.floor((high - low) / step)) + 1
                values = low + step * np.arange(max(1, count))
                whole = np.all(np.mod(values, 1) == 0)
                levels = [int(v) if whole else float(v) for v in values]
        if levels:
            out[name] = levels
    return out

def level_coverage(history: pd.DataFrame, catalogue: dict) -> list:
    """Which catalogue levels the history has ever actually built.

    A level nobody has run cannot be placed by any model, so every candidate
    containing it is a guess. On a catalogue offering six helper plasmids where
    only three have been used, that is half the catalogue - and the only
    visible sign would be a smaller number of scored candidates than the
    arithmetic implied.
    """
    out = []
    for name, levels in catalogue.items():
        if name not in history.columns:
            out.append({"factor": name, "built": [], "never_built": list(levels),
                        "thin": []})
            continue
        counts = history[name].astype(str).value_counts()
        built = [l for l in levels if str(l) in counts.index]
        never = [l for l in levels if str(l) not in counts.index]
        thin = [{"level": l, "runs": int(counts[str(l)])}
                for l in built if counts[str(l)] < 3]
        if never or thin:
            out.append({"factor": name, "built": built, "never_built": never,
                        "thin": thin})
    return out


def plan_vector(history: pd.DataFrame, catalogue: dict, response: str,
                protein_columns: Sequence[str],
                programme: Optional[dict] = None,
                wells: int = 12, confidence: float = 0.9,
                seed: int = 29) -> VectorPlan:
    """History and catalogue in, a scored run sheet out."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    plan = VectorPlan()
    factor_names = [n for n in catalogue if n in history.columns]
    if not factor_names:
        plan.failed = ("None of the catalogue factors appear in the historical "
                       "runs, so nothing can be learned about them.")
        return plan

    features = factor_names + [c for c in protein_columns if c in history.columns]
    data = history[features + [response]].dropna().copy()
    data[response] = pd.to_numeric(data[response], errors="coerce")
    data = data.dropna()
    if len(data) < 12:
        plan.failed = ("%d complete historical runs. A model for choosing "
                       "between vectors needs more than that to be worth "
                       "trusting." % len(data))
        return plan

    plan.transferability = assess_transferability(data, list(protein_columns),
                                                  programme)
    plan.coverage = level_coverage(data, catalogue)

    categorical = [c for c in features
                   if not pd.api.types.is_numeric_dtype(data[c])]
    numeric = [c for c in features if c not in categorical]
    model = Pipeline([
        ("encode", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric)])),
        ("forest", RandomForestRegressor(n_estimators=300, random_state=seed,
                                         min_samples_leaf=2, n_jobs=-1)),
    ])

    X, y = data[features], data[response].to_numpy(dtype=float)
    folds = KFold(n_splits=min(5, len(data)), shuffle=True, random_state=seed)

    # A conformal interval built from out-of-fold errors. Every historical run
    # is predicted by a model that never saw it, and the chosen quantile of
    # those errors is the half-width. It assumes nothing about the shape of
    # the errors - which matters, because titre residuals are rarely
    # symmetric - and unlike an interval taken from the training residuals it
    # does not narrow just because the model memorised its own data.
    #
    # Out-of-fold rather than a single hold-out split, for a reason that was
    # visible in testing: a split-conformal band set aside a quarter of the
    # history, so its errors came from a model fitted on 52 runs while the
    # prediction came from one fitted on all 70. The band it produced covered
    # 100% of held-out runs where it claimed 90% - safe, but wide enough to be
    # useless. Out-of-fold errors come from models fitted on four fifths of
    # the data and there are four times as many of them, so the quantile is
    # both closer to the truth and better estimated. The guarantee is
    # approximate rather than exact; the coverage is measured in the tests
    # instead of being taken on trust.
    out_of_fold = cross_val_predict(model, X, y, cv=folds)
    plan.r_squared_cv = float(1.0 - np.sum((y - out_of_fold) ** 2)
                              / np.sum((y - y.mean()) ** 2))
    residuals = np.sort(np.abs(y - out_of_fold))
    k = int(np.ceil((len(residuals) + 1) * confidence)) - 1
    half_width = float(residuals[min(k, len(residuals) - 1)])
    plan.interval_width = half_width

    model.fit(X, y)

    buildable = {n: [l for l in catalogue[n]
                     if str(l) in data[n].astype(str).unique()]
                 for n in factor_names}
    missing = [n for n, levels in buildable.items() if not levels]
    if missing:
        plan.failed = ("No level of %s has ever been built here, so no "
                       "candidate using it can be scored."
                       % ", ".join(missing))
        return plan

    candidates, total, sampled = enumerate_candidates(buildable, seed=seed)
    plan.n_candidates, plan.n_possible, plan.sampled = len(candidates), total, sampled

    scored = candidates.copy()
    for name in protein_columns:
        if name in features:
            value = (programme or {}).get(name)
            scored[name] = value if value is not None else data[name].iloc[0]
    scored["Predicted"] = model.predict(scored[features])
    scored["Lower bound"] = scored["Predicted"] - half_width
    scored["Upper bound"] = scored["Predicted"] + half_width
    plan.candidates = scored.sort_values("Predicted", ascending=False).reset_index(drop=True)

    plan.best = plan.candidates.iloc[0].to_dict()
    plan.safest = plan.candidates.sort_values(
        "Lower bound", ascending=False).iloc[0].to_dict()

    reps = replicates_for(plan.r_squared_cv)
    distinct = max(2, min(len(plan.candidates), wells // reps))
    picked = plan.candidates.head(distinct)
    sheet = pd.concat([picked.assign(Replicate=i + 1) for i in range(reps)],
                      ignore_index=True)
    while len(sheet) < wells:
        extra = picked.iloc[[len(sheet) % distinct]].assign(Replicate=reps + 1)
        sheet = pd.concat([sheet, extra], ignore_index=True)
    rng = np.random.default_rng(seed)
    sheet = sheet.head(wells).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "Well", np.arange(1, len(sheet) + 1))
    plan.sheet = sheet

    plan.plain, plan.cautions = _narrate(plan, response, reps, distinct, wells,
                                         confidence)
    return plan


def _narrate(plan: VectorPlan, response: str, reps: int, distinct: int,
             wells: int, confidence: float) -> tuple:
    transfer = plan.transferability
    cautions = []

    standing = (transfer.per_programme[0]["standing"]
                if transfer and transfer.per_programme else None)
    if transfer and not transfer.can_transfer:
        cautions.append(
            "**No prediction for a different protein can be defended from this "
            "history.** %s The numbers below are what the model returns; they "
            "are the shape of the fit rather than evidence about your molecule."
            % transfer.why)
    elif standing == "untested":
        reasons = (transfer.per_programme[0]["novel"]
                   + transfer.per_programme[0]["unseen"]
                   + transfer.per_programme[0]["no_evidence"])
        cautions.append(
            "**This programme sits outside what the history covers.** %s. A "
            "prediction here is extrapolation, and the interval beside it does "
            "not widen to account for that — nothing in the residuals knows the "
            "question changed." % "; ".join(reasons[:3]))
    elif standing == "thin":
        cautions.append(
            "Only %d distinct proteins in the history, so the protein effect "
            "rests on very little. Treat the ranking as a starting point."
            % transfer.n_proteins)

    if plan.r_squared_cv is not None and plan.r_squared_cv <= 0:
        cautions.append(
            "**Cross-validated R² is %.3f — at or below zero.** On constructs "
            "it has not seen, this model does worse than predicting the "
            "overall average every time. Nothing in the catalogue has been "
            "shown to matter for this response; the most likely reasons are "
            "that the parts genuinely do not move it, or that whatever does "
            "move it is not in the history." % plan.r_squared_cv)
    elif plan.r_squared_cv is not None and plan.r_squared_cv < 0.2:
        cautions.append(
            "Cross-validated R² is %.3f. On constructs it has not seen, this "
            "model barely predicts — which is why the sheet spreads across %d "
            "candidates rather than concentrating on the top few."
            % (plan.r_squared_cv, distinct))
    if plan.sampled:
        cautions.append(
            "The catalogue affords %s combinations, too many to score them "
            "all, so %s were sampled across it. The best build may not be "
            "among them." % (f"{plan.n_possible:,}", f"{plan.n_candidates:,}"))
    for row in plan.coverage:
        if row["never_built"]:
            cautions.append(
                "%s: %s %s never been built here, so %s excluded from the "
                "candidates — no model can place a level it has never seen."
                % (row["factor"], ", ".join(str(l) for l in row["never_built"]),
                   "has" if len(row["never_built"]) == 1 else "have",
                   "it is" if len(row["never_built"]) == 1 else "they are"))

    if plan.r_squared_cv is not None and plan.r_squared_cv <= 0:
        # Worse than predicting the overall average. The ranking is then an
        # ordering of noise, and leading with "build this one" would be the
        # single most misleading thing this page could say - the table below
        # looks identical whether the model knows anything or nothing.
        plain = ("**This history cannot choose between these builds.** On "
                 "constructs it has not seen the model predicts *worse than "
                 "the overall average* (cross-validated R² = %.3f), so the "
                 "ranking below is ordering noise and its top entry is "
                 "arbitrary. The plate is still worth running — it just cannot "
                 "be aimed yet. %d wells across %d candidates, %d replicates "
                 "each, will measure them properly; what it learns is what "
                 "makes the next one aimable."
                 % (plan.r_squared_cv, wells, distinct, reps))
        return plain, cautions

    best_name = " + ".join(
        str(plan.best[k]) for k in plan.candidates.columns
        if k not in ("Predicted", "Lower bound", "Upper bound")
        and k not in (transfer.varying + transfer.constant if transfer else []))
    agree = plan.best is plan.safest or all(
        plan.best.get(k) == plan.safest.get(k) for k in plan.candidates.columns
        if k not in ("Predicted", "Lower bound", "Upper bound"))

    plain = ("**Build %s.** Predicted %s %.4g, with a %.0f%% interval of %.4g "
             "to %.4g. %s The sheet puts %d wells across %d candidate%s, %d "
             "replicate%s each — concentrating rather than spreading, because "
             "a single observation of a vector is mostly noise."
             % (best_name, response, plan.best["Predicted"], 100 * confidence,
                plan.best["Lower bound"], plan.best["Upper bound"],
                "The highest prediction is also the safest choice."
                if agree else
                "A different build has the highest *lower bound* — the choice "
                "least likely to disappoint, which is often what a programme "
                "actually wants.",
                wells, distinct, "" if distinct == 1 else "s",
                reps, "" if reps == 1 else "s"))
    return plain, cautions
