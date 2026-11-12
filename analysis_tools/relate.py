"""Several inputs at once: which of them actually move the response?

This is the right-hand half of the analysis path - all continuous inputs give
multiple regression, all grouping inputs or a mixture give a general linear
model - and it is where the two expensive mistakes in applied statistics live.

**Collinearity.** When two inputs moved together in the data you happen to have,
no amount of arithmetic can say which of them matters. The model does not fail;
it splits the credit arbitrarily, the coefficients become unstable, and a
pairwise correlation matrix often shows nothing wrong. Variance inflation is
reported for every term because it catches what a correlation does not.

**Selection.** Searching every subset and reporting the best one's p-values as
though that model had been chosen in advance is the most common way a report
ends up confidently wrong. Best-subsets searching is offered here because the
chart asks for it, with the size of the optimism measured and printed rather
than mentioned in a footnote.

Everything is fitted with statsmodels, which is the reference a reviewer can
check against, and quoted against JMP's Fit Model.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from analysis_tools.assumptions import ALPHA, check_normality, fmt_p

VERIFY_FIT = "Analyze > Fit Model > Standard Least Squares"
VERIFY_SUBSETS = ("Analyze > Fit Model > Stepwise > red triangle > All Possible "
                  "Models (choose the lowest BIC or AICc)")
VERIFY_VIF = ("Analyze > Fit Model > Standard Least Squares > right-click "
              "Parameter Estimates > Columns > VIF")

#: Variance inflation above this is usually taken as a problem. 5 is the
#: stricter convention, 10 the looser one; both are reported against.
VIF_CONCERN = 5.0
VIF_SERIOUS = 10.0

#: Searching more subsets than this gets slow and, more to the point, the
#: optimism from selecting among them stops being something a warning can fix.
MAX_SUBSET_TERMS = 12

#: How optimistic the winner of an all-subsets search is, measured on data where
#: the response is pure noise and no input has any effect at all. Keyed by
#: (rows, candidate inputs); the values are the best model's adjusted R-squared
#: and how often that winning model contains a term significant at 0.05.
#:
#: The honest value for both is zero. Measured over 120 simulated datasets per
#: cell:
#:
#:     rows  inputs  models   best adj-R2   a "significant" term
#:       30       4      15         0.047                   18%
#:       30       6      63         0.088                   29%
#:       60       6      63         0.032                   18%
#:       60       8     255         0.053                   41%
#:      100       8     255         0.029                   35%
#:
#: This is why the search is offered with the number attached. A practitioner
#: who searches 255 models of noise and reports the winner will find something
#: to report two times in five.
SELECTION_OPTIMISM = {
    (30, 4): (0.047, 0.18), (30, 6): (0.088, 0.29), (60, 6): (0.032, 0.18),
    (60, 8): (0.053, 0.41), (100, 8): (0.029, 0.35),
}


def nearest_optimism(n_rows: int, n_inputs: int) -> tuple:
    """The measured optimism for the closest case that was simulated."""
    key = min(SELECTION_OPTIMISM,
              key=lambda k: abs(k[0] - n_rows) / 100.0 + abs(k[1] - n_inputs) / 10.0)
    return key, SELECTION_OPTIMISM[key]


@dataclass
class Term:
    """One term in a fitted model."""

    name: str
    coefficient: Optional[float]
    p_value: Optional[float]
    interval: Optional[tuple] = None
    vif: Optional[float] = None


@dataclass
class ModelFit:
    """A fitted model, and what to distrust about it."""

    formula: str
    n: int
    terms: list = field(default_factory=list)
    anova: Optional[pd.DataFrame] = None
    r_squared: Optional[float] = None
    r_squared_adj: Optional[float] = None
    sigma: Optional[float] = None
    aicc: Optional[float] = None
    bic: Optional[float] = None
    f_p_value: Optional[float] = None
    plain: str = ""
    cautions: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    verify_in: str = VERIFY_FIT
    failed: Optional[str] = None


def _quote(name: str) -> str:
    """Column names in a formula. Q() handles spaces, units and punctuation."""
    return 'Q("%s")' % name.replace('"', '\\"')


def _term_label(raw: str, names: Sequence[str]) -> str:
    """Turn statsmodels' formula names back into the user's column names."""
    label = raw
    for name in names:
        label = label.replace('C(Q("%s"))' % name, name).replace(_quote(name), name)
    return label.replace("Intercept", "(intercept)")


def fit_model(frame: pd.DataFrame, y_name: str, x_names: Sequence[str],
              categorical: Sequence[str] = (), interactions: bool = False) -> ModelFit:
    """Fit the response against the inputs.

    One code path for multiple regression and the general linear model, because
    they are the same fit: a grouping input simply enters as a set of indicator
    terms. Separating them would be a presentation choice dressed as a
    statistical one.
    """
    columns = [y_name] + list(x_names)
    data = frame[columns].dropna().copy()
    data[y_name] = pd.to_numeric(data[y_name], errors="coerce")
    data = data.dropna()
    n = len(data)

    pieces = []
    for name in x_names:
        if name in categorical:
            data[name] = data[name].astype(str)
            pieces.append("C(%s)" % _quote(name))
        else:
            data[name] = pd.to_numeric(data[name], errors="coerce")
            pieces.append(_quote(name))
    data = data.dropna()
    n = len(data)

    joiner = " * " if interactions and len(pieces) > 1 else " + "
    formula = "%s ~ %s" % (_quote(y_name), joiner.join(pieces))

    terms_needed = len(pieces) + 1
    if n < terms_needed + 2:
        return ModelFit(formula=formula, n=n,
                        failed="Only %d complete rows for %d terms. A model needs "
                               "more observations than it has terms to estimate, "
                               "with room to spare." % (n, terms_needed))
    try:
        model = ols(formula, data=data).fit()
    except Exception as exc:                        # noqa: BLE001 - reported
        return ModelFit(formula=formula, n=n,
                        failed="The model could not be fitted: %s" % exc)

    fit = ModelFit(
        formula=formula, n=n,
        r_squared=float(model.rsquared), r_squared_adj=float(model.rsquared_adj),
        sigma=float(np.sqrt(model.mse_resid)), bic=float(model.bic),
        aicc=_aicc(model), f_p_value=float(model.f_pvalue),
    )

    intervals = model.conf_int()
    vifs = _vif(model)
    for raw in model.params.index:
        fit.terms.append(Term(
            name=_term_label(raw, x_names),
            coefficient=float(model.params[raw]),
            p_value=float(model.pvalues[raw]),
            interval=(float(intervals.loc[raw, 0]), float(intervals.loc[raw, 1])),
            vif=vifs.get(raw),
        ))

    try:
        table = sm.stats.anova_lm(model, typ=2)
        table.index = [_term_label(str(i), x_names) for i in table.index]
        fit.anova = table
    except Exception:                               # noqa: BLE001 - optional
        fit.anova = None

    fit.checks = [check_normality(model.resid, "the residuals")]
    fit.cautions = _cautions(fit, n, len(pieces))
    fit.plain = _summarise(fit, y_name)
    return fit


def _aicc(model) -> Optional[float]:
    """AIC with the small-sample correction. JMP reports AICc, not AIC, and at
    development sample sizes they differ enough to change which model wins."""
    k = int(model.df_model) + 1
    n = int(model.nobs)
    if n - k - 1 <= 0:
        return None
    return float(model.aic + 2 * k * (k + 1) / (n - k - 1))


def _vif(model) -> dict:
    """Variance inflation for each term.

    How much wider a coefficient's confidence interval is than it would be if
    this input were uncorrelated with the others. VIF 9 means three times
    wider. It catches what a pairwise correlation cannot: an input that is a
    combination of two others can be badly inflated while no pair of columns
    looks remarkable.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        design = np.asarray(model.model.exog, dtype=float)
        names = list(model.model.exog_names)
        if design.shape[1] < 3:
            return {}
        return {names[i]: float(variance_inflation_factor(design, i))
                for i in range(len(names)) if names[i] != "Intercept"}
    except Exception:                               # noqa: BLE001 - optional
        return {}


def _cautions(fit: ModelFit, n: int, n_inputs: int) -> list:
    out = []
    inflated = [t for t in fit.terms
                if t.vif is not None and t.vif >= VIF_CONCERN and t.name != "(intercept)"]
    if inflated:
        worst = max(inflated, key=lambda t: t.vif)
        level = "seriously" if worst.vif >= VIF_SERIOUS else "appreciably"
        out.append(
            "**These inputs moved together in this data.** %s has a variance "
            "inflation of %.1f, which means its confidence interval is %.1f times "
            "wider than it would be if the inputs were independent — the model is "
            "%s unable to separate their effects. Terms affected: %s. This is not "
            "fixed by more rows of the same data; it needs an experiment that "
            "varies them independently."
            % (worst.name, worst.vif, math.sqrt(worst.vif), level,
               ", ".join("%s (%.1f)" % (t.name, t.vif) for t in inflated))
        )
    if n < 10 * n_inputs:
        out.append(
            "%d rows for %d inputs. Below about ten observations per input the "
            "coefficients move a lot from sample to sample; treat the ranking as "
            "provisional." % (n, n_inputs)
        )
    return out


def _summarise(fit: ModelFit, y_name: str) -> str:
    real = [t for t in fit.terms
            if t.name != "(intercept)" and t.p_value is not None and t.p_value < ALPHA]
    if fit.f_p_value is not None and fit.f_p_value >= ALPHA:
        return ("**The model as a whole does not explain %s** (F-test p = %s). "
                "Adjusted R² = %.3f. Individual terms are listed below, but with "
                "the overall model unsupported they should not be read as "
                "findings." % (y_name, fmt_p(fit.f_p_value), fit.r_squared_adj))
    if not real:
        return ("**The model explains %s overall** (F-test p = %s, adjusted R² = "
                "%.3f) **but no single input stands out.** That pattern usually "
                "means the inputs are telling the same story — check the variance "
                "inflation column."
                % (y_name, fmt_p(fit.f_p_value), fit.r_squared_adj))
    ranked = sorted(real, key=lambda t: t.p_value)
    return ("**%s %s %s.** Adjusted R² = %.3f — the model accounts for %.0f%% of "
            "the variation in %s, with a residual standard deviation of %.4g."
            % (", ".join(t.name for t in ranked[:4]),
               "moves" if len(ranked) == 1 else "move", y_name,
               fit.r_squared_adj, 100 * max(fit.r_squared_adj, 0.0), y_name,
               fit.sigma))


# ------------------------------------------------------------- model reduction

#: Two inputs correlated above this carry substantially the same information,
#: and which of them a subset search keeps is close to arbitrary.
PROXY_CORRELATION = 0.85


@dataclass
class SubsetSearch:
    """Every model that was tried, ranked, and how optimistic the winner is."""

    results: pd.DataFrame
    best_by_bic: tuple
    best_by_aicc: tuple
    n_models: int
    plain: str = ""
    optimism: str = ""
    substitutes: list = field(default_factory=list)
    verify_in: str = VERIFY_SUBSETS


def all_possible_models(frame: pd.DataFrame, y_name: str, x_names: Sequence[str],
                        categorical: Sequence[str] = (),
                        max_terms: Optional[int] = None) -> Optional[SubsetSearch]:
    """Fit every subset and rank them, as JMP's All Possible Models does.

    Ranked by BIC and AICc, which is what JMP recommends choosing on, with
    adjusted R² and Mallows' Cp beside them. Cp is in JMP too, hidden behind a
    right-click; it is shown here because a model whose Cp is far above its term
    count is missing something.
    """
    inputs = list(x_names)
    if not inputs or len(inputs) > MAX_SUBSET_TERMS:
        return None
    ceiling = max_terms or len(inputs)

    full = fit_model(frame, y_name, inputs, categorical=categorical)
    if full.failed or full.sigma is None:
        return None
    mse_full = full.sigma ** 2

    rows = []
    for size in range(1, ceiling + 1):
        for combination in itertools.combinations(inputs, size):
            candidate = fit_model(frame, y_name, list(combination),
                                  categorical=categorical)
            if candidate.failed or candidate.r_squared_adj is None:
                continue
            n, p = candidate.n, size + 1
            sse = (candidate.sigma ** 2) * (n - p)
            cp = sse / mse_full - (n - 2 * p) if mse_full > 0 else float("nan")
            rows.append({
                "Terms": ", ".join(combination),
                "Count": size,
                "Adjusted R²": round(candidate.r_squared_adj, 4),
                "BIC": round(candidate.bic, 2),
                "AICc": round(candidate.aicc, 2) if candidate.aicc is not None else None,
                "Mallows' Cp": round(cp, 2),
                "s": round(candidate.sigma, 5),
            })
    if not rows:
        return None

    results = pd.DataFrame(rows).sort_values("BIC").reset_index(drop=True)
    best_bic = (results.iloc[0]["Terms"], results.iloc[0]["BIC"])
    by_aicc = results.dropna(subset=["AICc"]).sort_values("AICc")
    best_aicc = ((by_aicc.iloc[0]["Terms"], by_aicc.iloc[0]["AICc"])
                 if len(by_aicc) else best_bic)

    agree = best_bic[0] == best_aicc[0]
    plain = ("%d models fitted. The lowest BIC is **%s**%s."
             % (len(rows), best_bic[0],
                " and AICc agrees" if agree else
                ", while the lowest AICc picks **%s** — when they disagree, BIC "
                "prefers the smaller model and AICc the better-predicting one"
                % best_aicc[0]))

    (ref_rows, ref_inputs), (ref_r2, ref_sig) = nearest_optimism(full.n, len(inputs))
    optimism = (
        "**The winner's statistics are optimistic, and not slightly.** Searching "
        "%d models and reporting the best one's adjusted R² and p-values is not "
        "the same as testing a model chosen in advance: the search finds whatever "
        "fits this sample, including the part of it that is noise.\n\n"
        "Measured on data where the response is **pure noise and no input has any "
        "effect at all** — %d rows and %d candidate inputs, the closest case "
        "simulated to yours — the best subset still came out with an adjusted R² "
        "of %.3f, and **%.0f%% of the time the winning model contained a term "
        "significant at p < 0.05.** The honest value for both is zero.\n\n"
        "So treat the selected model as a hypothesis to confirm on data it has "
        "not seen, not as a result. The p-values below are for the model as if "
        "you had chosen it in advance, which you did not."
        % (len(rows), ref_rows, ref_inputs, ref_r2, 100 * ref_sig)
    )
    substitutes = _substitutes_for(frame, best_bic[0], inputs, categorical)
    return SubsetSearch(results=results, best_by_bic=best_bic,
                        best_by_aicc=best_aicc, n_models=len(rows),
                        plain=plain, optimism=optimism, substitutes=substitutes)


def _substitutes_for(frame: pd.DataFrame, chosen: str, inputs: Sequence[str],
                     categorical: Sequence[str]) -> list:
    """Inputs the search dropped that say nearly the same thing as one it kept.

    This matters more than it first appears. When two inputs carry
    substantially the same information, which one a subset search keeps is
    close to arbitrary - a different sample would have picked the other, and
    the output gives no sign of that. A reader sees the retained input named as
    a driver and the dropped one absent, and concludes something the data
    cannot support.

    The case that bites hardest is a proxy: a sensor reading that tracks the
    parameter actually being controlled. The search keeps whichever happened to
    fit a fraction better, and the report then credits the thermometer rather
    than the heater.
    """
    kept = {t.strip() for t in chosen.split(",")}
    dropped = [name for name in inputs if name not in kept]
    numeric = [name for name in inputs
               if name not in categorical
               and pd.api.types.is_numeric_dtype(
                   pd.to_numeric(frame[name], errors="coerce"))]
    pairs = []
    for out_name in dropped:
        if out_name not in numeric:
            continue
        for in_name in kept:
            if in_name not in numeric:
                continue
            both = frame[[out_name, in_name]].apply(
                pd.to_numeric, errors="coerce").dropna()
            if len(both) < 3:
                continue
            r = abs(float(both[out_name].corr(both[in_name])))
            if r >= PROXY_CORRELATION:
                pairs.append(
                    "**%s was dropped and %s was kept, but they correlate at "
                    "%.2f** — they carry substantially the same information, and "
                    "which one the search keeps is close to arbitrary. A "
                    "different sample would plausibly have kept %s instead. Do "
                    "not read this as evidence that %s matters and %s does not."
                    % (out_name, in_name, r, out_name, in_name, out_name))
    return pairs
