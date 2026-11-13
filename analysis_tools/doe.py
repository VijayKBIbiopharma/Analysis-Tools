"""Designing an experiment, and analysing the one you ran.

Two jobs, and the project guide separates them for a reason. In the Analyze
phase a design is run to find out *which* inputs matter - a screen, two levels
each, as wide as the process allows. In the Improve phase a design is run to
find *where to set* the ones that do - narrower levels, and a shape that can
follow a curve.

The thing that goes wrong
------------------------
Fractional designs buy their small run count by confounding effects with each
other. In a sixteen-run design for five factors, every main effect is clear but
the two-factor interactions are tangled in pairs; in an eight-run design for
the same five, main effects are themselves tangled with two-factor
interactions. The arithmetic gives a number for each either way, and nothing in
the output says that the number attributed to temperature might be an
interaction between pH and feed wearing temperature's name.

So the alias structure is computed from the design that was actually generated,
not looked up from the table the design was requested from, and it is printed
with the design. Anyone reading an effect from a fractional design needs to
know what else it could be.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

VERIFY_DESIGN = "DOE > Classical > Screening Design (or Response Surface Design)"
VERIFY_ANALYSE = "Analyze > Fit Model > Standard Least Squares (effects screening)"

#: Standard generators for two-level fractional designs, as tabulated in every
#: design text. Each word defines one added factor in terms of the base
#: design's factors, so a (k, p) entry has p words drawn only from the first
#: k - p letters.
#:
#: That last constraint is not decoration. The first version of this table had
#: 2^(7-2) as ABCD and ABEF - and with seven factors and two generators the
#: base design is only A to E, so ABEF referred to F, the factor that generator
#: was itself defining. There is a test that checks every word against the
#: base alphabet, and another that computes each design's resolution from its
#: defining relation and compares it with the published value, so a wrong entry
#: cannot survive quietly.
GENERATORS = {
    (3, 1): ["AB"],
    (4, 1): ["ABC"],
    (5, 1): ["ABCD"],
    (5, 2): ["AB", "AC"],
    (6, 1): ["ABCDE"],
    (6, 2): ["ABC", "BCD"],
    (6, 3): ["AB", "AC", "BC"],
    (7, 1): ["ABCDEF"],
    (7, 2): ["ABCD", "ABDE"],
    (7, 3): ["ABC", "BCD", "ACD"],
    (7, 4): ["AB", "AC", "BC", "ABC"],
    (8, 2): ["ABCD", "ABEF"],
    (8, 3): ["ABC", "ABD", "BCDE"],
    (8, 4): ["BCD", "ACD", "ABC", "ABD"],
}

LETTERS = "ABCDEFGHJKLMNP"      # I is skipped: it names the identity column


@dataclass
class Design:
    """A generated design, and what it cannot tell you."""

    runs: pd.DataFrame
    factors: list
    kind: str
    n_runs: int
    resolution: Optional[int] = None
    defining_relation: list = field(default_factory=list)
    aliases: dict = field(default_factory=dict)
    centre_points: int = 0
    replicates: int = 1
    blocks: int = 1
    plain: str = ""
    cautions: list = field(default_factory=list)
    verify_in: str = VERIFY_DESIGN


def _coded_full(k: int) -> np.ndarray:
    """The full 2^k design in -1/+1, standard order."""
    rows = list(itertools.product([-1, 1], repeat=k))
    return np.array([list(reversed(r)) for r in rows], dtype=int)


def _word_column(design: np.ndarray, word: str, letters: str) -> np.ndarray:
    """The product column for a word such as "ABC"."""
    column = np.ones(design.shape[0], dtype=int)
    for letter in word:
        column = column * design[:, letters.index(letter)]
    return column


def _defining_relation(generators: dict, letters: str) -> list:
    """Every word in the defining relation, from the generator words.

    A design with p generators has 2^p - 1 words in its defining relation: each
    generator, and every product of them. The shortest word is the resolution,
    and that is what decides what is tangled with what.
    """
    base = []
    for new_letter, word in generators.items():
        base.append("".join(sorted(set(word + new_letter))))
    words = set()
    for size in range(1, len(base) + 1):
        for combination in itertools.combinations(base, size):
            merged: dict = {}
            for word in combination:
                for letter in word:
                    merged[letter] = merged.get(letter, 0) + 1
            # Letters appearing twice cancel: A x A = I.
            remaining = "".join(sorted(l for l, c in merged.items() if c % 2))
            if remaining:
                words.add(remaining)
    return sorted(words, key=lambda w: (len(w), w))


def _aliases_for(terms: Sequence[str], relation: Sequence[str],
                 letters: str) -> dict:
    """What each term is confounded with, up to two-factor interactions."""
    out = {}
    for term in terms:
        tangled = []
        for word in relation:
            merged: dict = {}
            for letter in term + word:
                merged[letter] = merged.get(letter, 0) + 1
            partner = "".join(sorted(l for l, c in merged.items() if c % 2))
            if partner and partner != term and len(partner) <= 3:
                tangled.append(partner)
        if tangled:
            out[term] = sorted(set(tangled), key=lambda w: (len(w), w))
    return out


def build_design(factors: Sequence[dict], runs: Optional[int] = None,
                 centre_points: int = 0, replicates: int = 1,
                 blocks: int = 1, seed: int = 29) -> Design:
    """Generate a two-level screening design.

    ``factors`` is a list of ``{"name": ..., "low": ..., "high": ...}``. A
    numeric low and high give a continuous factor; strings give a categorical
    one at two levels.
    """
    k = len(factors)
    if k < 2:
        return Design(runs=pd.DataFrame(), factors=list(factors), kind="none",
                      n_runs=0,
                      plain="A design needs at least two factors to be worth running.")
    if k > len(LETTERS):
        return Design(runs=pd.DataFrame(), factors=list(factors), kind="none",
                      n_runs=0,
                      plain="More factors than this design builder handles (%d)." % len(LETTERS))

    letters = LETTERS[:k]
    wanted = runs or 2 ** k
    # The smallest fraction that fits the run budget and that we have
    # generators for. Searching p upward rather than stepping it one at a time
    # matters: an eight-factor design has no tabulated half-fraction here, so a
    # one-step search would give up at p = 0 and quietly return 256 runs.
    # Only fractionate if the full factorial will not fit. Searching for a
    # fraction unconditionally turns every unconstrained request into a half
    # fraction, which is both wrong and silent.
    p = 0
    if 2 ** k > wanted:
        for candidate in range(1, k):
            if (k, candidate) in GENERATORS and 2 ** (k - candidate) <= wanted:
                p = candidate
                break

    cautions = []
    if p == 0:
        base = _coded_full(k)
        matrix = base
        kind = "full factorial"
        resolution, relation, aliases = None, [], {}
    else:
        base = _coded_full(k - p)
        generator_words = GENERATORS[(k, p)]
        assigned = {}
        columns = [base[:, i] for i in range(k - p)]
        for index, word in enumerate(generator_words):
            new_letter = letters[k - p + index]
            assigned[new_letter] = word
            columns.append(_word_column(base, word, letters[:k - p]))
        matrix = np.column_stack(columns)
        kind = "fractional factorial 2^(%d-%d)" % (k, p)
        relation = _defining_relation(assigned, letters)
        resolution = min(len(w) for w in relation) if relation else None
        main_and_two = list(letters) + [a + b for a, b in
                                        itertools.combinations(letters, 2)]
        aliases = _aliases_for(main_and_two, relation, letters)

    frame = pd.DataFrame(matrix, columns=[f["name"] for f in factors])
    frame = pd.concat([frame] * max(1, replicates), ignore_index=True)

    if centre_points:
        numeric = all(_is_numeric(f) for f in factors)
        if numeric:
            centre = pd.DataFrame(
                [[0] * k] * centre_points, columns=[f["name"] for f in factors])
            frame = pd.concat([frame, centre], ignore_index=True)
        else:
            cautions.append(
                "Centre points were asked for but at least one factor is "
                "categorical, which has no middle setting. They were not added.")
            centre_points = 0

    coded = frame.copy()
    for factor in factors:
        frame[factor["name"]] = _decode(coded[factor["name"]], factor)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    frame.insert(0, "Std order", np.arange(1, len(frame) + 1))
    frame.insert(1, "Run order", np.argsort(order) + 1)
    if blocks > 1:
        frame.insert(2, "Block", (np.arange(len(frame)) % blocks) + 1)
    frame = frame.sort_values("Run order").reset_index(drop=True)

    design = Design(runs=frame, factors=list(factors), kind=kind,
                    n_runs=len(frame), resolution=resolution,
                    defining_relation=relation, aliases=aliases,
                    centre_points=centre_points, replicates=replicates,
                    blocks=blocks, cautions=cautions)
    design.plain, extra = _describe(design, k)
    design.cautions.extend(extra)
    return design


def _is_numeric(factor: dict) -> bool:
    try:
        float(factor["low"]), float(factor["high"])
        return True
    except (TypeError, ValueError):
        return False


def _decode(column: pd.Series, factor: dict) -> pd.Series:
    if _is_numeric(factor):
        low, high = float(factor["low"]), float(factor["high"])
        mid = (low + high) / 2.0
        half = (high - low) / 2.0
        return column.map(lambda c: mid + c * half)
    return column.map({-1: str(factor["low"]), 1: str(factor["high"]), 0: "centre"})


def _describe(design: Design, k: int) -> tuple:
    cautions = []
    if design.resolution is None:
        plain = ("**Full factorial, %d factors in %d runs.** Every main effect "
                 "and every interaction is estimated separately — nothing is "
                 "confounded with anything." % (k, design.n_runs))
    else:
        plain = ("**%s, resolution %s, %d runs.** The run count is bought by "
                 "confounding: the defining relation is I = %s."
                 % (design.kind.capitalize(), _roman(design.resolution),
                    design.n_runs, " = ".join(design.defining_relation)))
        if design.resolution <= 3:
            cautions.append(
                "**Resolution III: main effects are confounded with two-factor "
                "interactions.** A large reading for one factor may be an "
                "interaction between two others wearing its name, and no "
                "analysis of this data can separate them. Worth it only to "
                "sift many factors quickly, expecting to follow up."
            )
        elif design.resolution == 4:
            cautions.append(
                "Resolution IV: main effects are clear of two-factor "
                "interactions, but the two-factor interactions are confounded "
                "with each other. An interaction that shows up cannot be "
                "attributed to a particular pair without a follow-up."
            )
    if design.centre_points:
        plain += (" %d centre point%s included, which is what makes curvature "
                  "detectable — without them a two-level design cannot tell a "
                  "straight response from a curved one."
                  % (design.centre_points,
                     "" if design.centre_points == 1 else "s"))
    elif design.resolution is None or design.resolution >= 4:
        cautions.append(
            "No centre points. A two-level design assumes the response moves "
            "in a straight line between the settings; with no middle setting "
            "run, that assumption cannot be checked and an optimum sitting "
            "inside the range is invisible."
        )
    if design.replicates < 2 and not design.centre_points:
        cautions.append(
            "No replication and no centre points, so there is no estimate of "
            "pure error. Lack of fit cannot be separated from ordinary "
            "run-to-run scatter."
        )
    return plain, cautions


def _roman(n: Optional[int]) -> str:
    return {3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII"}.get(n, str(n))


# --------------------------------------------------------------- the analysis

@dataclass
class Effect:
    """One term's estimated effect."""

    term: str
    effect: float
    coefficient: float
    p_value: Optional[float] = None
    significant: bool = False
    aliased_with: list = field(default_factory=list)


#: Below this many residual degrees of freedom, p-values rest on too little to
#: be trusted and Lenth's method takes over.
MIN_ERROR_DF = 3
#: Between MIN_ERROR_DF and this, p-values are real but thin, and said to be.
COMFORTABLE_ERROR_DF = 8


@dataclass
class DesignAnalysis:
    effects: list = field(default_factory=list)
    df_error: Optional[float] = None
    r_squared_adj: Optional[float] = None
    sigma: Optional[float] = None
    curvature_p: Optional[float] = None
    lack_of_fit_p: Optional[float] = None
    lenth_margin: Optional[float] = None
    used_lenth: bool = False
    plain: str = ""
    cautions: list = field(default_factory=list)
    verify_in: str = VERIFY_ANALYSE
    failed: Optional[str] = None


def lenth_margin(effects: Sequence[float], alpha: float = 0.05) -> tuple:
    """Lenth's margin of error, for a design with no degrees of freedom left.

    A saturated design spends every run on estimating terms, leaving nothing to
    estimate error with, so there is no p-value to be had. Lenth's method gets
    one anyway by assuming most effects are nothing and using their spread as
    the yardstick. Returns (margin of error, simultaneous margin).
    """
    values = np.abs(np.asarray(effects, dtype=float))
    m = values.size
    if m < 3:
        return None, None
    s0 = 1.5 * np.median(values)
    kept = values[values < 2.5 * s0]
    pse = 1.5 * np.median(kept) if kept.size else s0
    if pse <= 0:
        return None, None
    d = max(1, m // 3)
    me = float(stats.t.ppf(1 - alpha / 2, d) * pse)
    gamma = (1 + 0.95 ** (1 / m)) / 2
    sme = float(stats.t.ppf(gamma, d) * pse)
    return me, sme


def analyse_design(frame: pd.DataFrame, factor_names: Sequence[str],
                   response: str, interactions: bool = True) -> DesignAnalysis:
    """Analyse a run design: effects, significance, curvature, lack of fit."""
    import statsmodels.api as sm

    data = frame[list(factor_names) + [response]].dropna().copy()
    data[response] = pd.to_numeric(data[response], errors="coerce")
    data = data.dropna()
    if len(data) < len(factor_names) + 2:
        return DesignAnalysis(failed="Too few complete runs to analyse.")

    coded = pd.DataFrame(index=data.index)
    centre_mask = pd.Series(False, index=data.index)
    for name in factor_names:
        column = data[name]
        numeric = pd.to_numeric(column, errors="coerce")
        if numeric.notna().all():
            low, high = numeric.min(), numeric.max()
            mid = (low + high) / 2.0
            half = (high - low) / 2.0
            coded[name] = 0.0 if half == 0 else (numeric - mid) / half
        else:
            levels = sorted(column.astype(str).unique())
            if len(levels) != 2:
                return DesignAnalysis(
                    failed="“%s” has %d levels. This analysis handles two-level "
                           "factors; use Relationships for more."
                           % (name, len(levels)))
            coded[name] = column.astype(str).map({levels[0]: -1.0, levels[1]: 1.0})
        centre_mask = centre_mask | (coded[name].abs() < 1e-9)

    factorial = ~centre_mask
    n_centre = int(centre_mask.sum())

    terms = {name: coded.loc[factorial, name].to_numpy(dtype=float)
             for name in factor_names}
    if interactions:
        for a, b in itertools.combinations(factor_names, 2):
            terms["%s x %s" % (a, b)] = terms[a] * terms[b]

    y = data.loc[factorial, response].to_numpy(dtype=float)
    design_matrix = np.column_stack([np.ones(y.size)] + list(terms.values()))
    labels = ["(intercept)"] + list(terms)
    if design_matrix.shape[1] > y.size:
        return DesignAnalysis(
            failed="%d terms from %d factorial runs. Drop the interactions or "
                   "run more points." % (design_matrix.shape[1] - 1, y.size))

    model = sm.OLS(y, design_matrix).fit()
    analysis = DesignAnalysis(
        r_squared_adj=float(model.rsquared_adj) if y.size > design_matrix.shape[1] else None,
        sigma=float(np.sqrt(model.mse_resid)) if model.df_resid > 0 else None,
    )

    # When almost every run is spent estimating terms, the error estimate
    # rests on a handful of residuals - or on one - and the p-values that come
    # out of it are arithmetic rather than evidence. Measured on a 2^3 with all
    # interactions fitted (one residual degree of freedom), terms whose true
    # effect is exactly zero came out "significant" at p < 0.05.
    #
    # So Lenth's method takes over below three residual degrees of freedom, not
    # only at zero. It assumes most effects are nothing and uses their spread
    # as the yardstick, which is a far better assumption than trusting one
    # residual.
    coefficients = model.params[1:]
    df_error = float(model.df_resid)
    me, _ = lenth_margin(2 * coefficients) if df_error < MIN_ERROR_DF else (None, None)
    analysis.used_lenth = me is not None
    analysis.lenth_margin = me
    analysis.df_error = df_error

    for i, label in enumerate(labels[1:], start=1):
        coefficient = float(model.params[i])
        effect = 2.0 * coefficient
        if analysis.used_lenth:
            p, significant = None, abs(effect) > me
        else:
            p = float(model.pvalues[i])
            significant = p < 0.05
        analysis.effects.append(Effect(term=label, effect=effect,
                                       coefficient=coefficient, p_value=p,
                                       significant=significant))
    analysis.effects.sort(key=lambda e: -abs(e.effect))

    # Curvature: the centre points sit at the middle of every factor, so the
    # factorial model predicts the overall average for them. If they come out
    # systematically off it, the response is curved and no two-level design can
    # follow it.
    if n_centre >= 2:
        centre_values = data.loc[centre_mask, response].to_numpy(dtype=float)
        n_f, n_c = y.size, centre_values.size
        factorial_mean, centre_mean = float(y.mean()), float(centre_values.mean())

        # The curvature sum of squares, on one degree of freedom, tested
        # against the model's residual mean square.
        #
        # The error term matters enormously and the obvious choice is wrong.
        # An earlier version used the variance of the factorial responses as
        # the yardstick - but those responses vary because the factors were
        # deliberately moved, so that variance is mostly signal. Dividing by it
        # made the test so conservative it missed a curvature of -0.18 in data
        # whose noise was 0.035, which is the exact situation centre points are
        # run to detect. The residual mean square is the error with the factor
        # effects taken out, which is what this comparison needs.
        ss_curvature = (n_f * n_c * (factorial_mean - centre_mean) ** 2) / (n_f + n_c)
        residual_ss = float(((centre_values - centre_mean) ** 2).sum())
        residual_df = n_c - 1
        if model.df_resid > 0:
            residual_ss += float(model.ssr)
            residual_df += int(model.df_resid)
        mse = residual_ss / residual_df if residual_df > 0 else 0.0
        if mse > 0:
            f_stat = ss_curvature / mse
            analysis.curvature_p = float(stats.f.sf(f_stat, 1, residual_df))
            if analysis.curvature_p < 0.05:
                analysis.cautions.append(
                    "**The response is curved** (centre points differ from the "
                    "factorial average, p = %.4f). A two-level design fits "
                    "straight lines, so it can tell you a factor matters but "
                    "not where the optimum is — and the optimum may well sit "
                    "inside the range you ran. A response surface design is "
                    "the follow-up." % analysis.curvature_p)

    if analysis.used_lenth:
        analysis.cautions.append(
            "Almost every run is spent estimating terms — %g degree%s of "
            "freedom left for error — so an ordinary p-value would rest on "
            "nearly nothing and would call genuinely null terms significant. "
            "Lenth's method is used instead: it assumes most effects are "
            "nothing and uses their spread as the yardstick, giving a margin "
            "of error of %.4g. Effects larger than that are called real. It is "
            "a sound method and it is not the same thing as a test."
            % (df_error, "" if df_error == 1 else "s", me))
        if len(analysis.effects) < 7:
            analysis.cautions.append(
                "Lenth's margin is estimated from only %d effects, which makes "
                "it conservative — it will miss a moderate real effect rather "
                "than risk a false one. **The design, not the analysis, is the "
                "constraint here.** Two cheap fixes: untick interactions if you "
                "do not expect them, which frees %d degrees of freedom for "
                "error and restores ordinary p-values; or add centre points, "
                "which buy error degrees of freedom without changing the "
                "factorial at all."
                % (len(analysis.effects),
                   sum(1 for e in analysis.effects if " x " in e.term)))
    elif df_error < COMFORTABLE_ERROR_DF:
        analysis.cautions.append(
            "Only %g residual degrees of freedom. The p-values below are real "
            "tests but the error they are measured against rests on %g "
            "observation%s, so a borderline result should not be leaned on. "
            "Dropping the interactions you do not expect, or adding centre "
            "points, buys error degrees of freedom cheaply."
            % (df_error, df_error, "" if df_error == 1 else "s"))

    real = [e for e in analysis.effects if e.significant]
    if real:
        analysis.plain = ("**%s.** The largest is %s, changing %s by %.4g "
                          "between its two settings.%s"
                          % (", ".join(e.term for e in real[:5]),
                             real[0].term, response, real[0].effect,
                             "" if analysis.r_squared_adj is None else
                             " Adjusted R² = %.3f." % analysis.r_squared_adj))
    else:
        analysis.plain = ("**No factor stands out.** Either nothing here moves "
                          "%s over the ranges you ran, or the ranges were too "
                          "narrow to show it. Widening the settings is usually "
                          "more informative than adding runs." % response)
    return analysis
