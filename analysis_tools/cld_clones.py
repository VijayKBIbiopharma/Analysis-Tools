"""Two questions a cell line development group arrives with.

**Which early reads predict the late ones?** A screen measures a lot in week
one and the thing that matters at day fourteen. If a week-one read predicts the
day-fourteen titer, the screen can be cut short and more clones put through it.
If it does not, measuring it is costing bench time for nothing.

**Where should the generation limit sit?** Titer and product quality drift as a
line is passaged. The limit is often set by convention - sixty generations,
because that is what the last programme did. It can be set from the data
instead, at the generation where the confidence limit on the trend meets the
specification.

Both carry the same risk, which is why they are in one module: a result that
looks like evidence and is not. A ranked list of early reads always has
something at the top, and a regression through passage number always has a
slope. The work here is in saying when those are noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from analysis_tools.assumptions import ALPHA, fmt_p

VERIFY_SCREEN = "Analyze > Fit Model > Standard Least Squares (or Predictor Screening)"
VERIFY_TREND = "Analyze > Fit Y by X > Bivariate > Fit Line > Confid Curves Fit"

#: How many times the response is shuffled to establish what importance looks
#: like when there is nothing to find.
#:
#: The permutation test costs one model fit per shuffle, so this number and the
#: forest size together set how long a screen takes. Measured on 40 clones and
#: four reads: 200 shuffles of a 200-tree forest took 44 seconds, which is not
#: a wait anyone should have at a screen. 100 shuffles of a 60-tree forest
#: takes about six. Both numbers were chosen by timing rather than by taste -
#: 60 trees is ample to rank a handful of features, and 100 shuffles resolves a
#: p-value to about 0.01, which is finer than any decision here needs.
NOISE_RUNS = 100
FOREST_TREES = 60

#: An extrapolation cap, in the spirit of ICH Q1E's rule for shelf life: never
#: project further beyond the data than the data itself spans.
EXTRAPOLATION_CAP = 2.0

#: Above this, a read that failed the floor still tracks the response on
#: its own well enough that "no evidence" would be read as more than it
#: means. r = 0.5 is a quarter of the variation explained single-handed.
MARGINAL_CORRELATION = 0.5


# ---------------------------------------------------- which reads predict

@dataclass
class ReadImportance:
    name: str
    importance: float
    above_noise: bool
    noise_ceiling: float
    p_value: Optional[float] = None
    p_adjusted: Optional[float] = None
    correlation: Optional[float] = None


@dataclass
class ScreenResult:
    reads: list = field(default_factory=list)
    r_squared_cv: Optional[float] = None
    n: int = 0
    noise_ceiling: Optional[float] = None
    plain: str = ""
    cautions: list = field(default_factory=list)
    verify_in: str = VERIFY_SCREEN
    failed: Optional[str] = None


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Adjust for having asked the question once per read.

    Testing six reads at 0.05 each gives roughly a one-in-four chance that at
    least one clears by luck. Benjamini-Hochberg controls the share of the
    reads called real that are not, which is the quantity that matters when the
    output is a shortlist to go and measure again.
    """
    m = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def screen_early_reads(frame: pd.DataFrame, early: Sequence[str], late: str,
                       seed: int = 7,
                       noise_runs: int = NOISE_RUNS) -> ScreenResult:
    """Which early measurements carry information about the late one.

    The ranked list is the easy part. Every ranking has a top entry, including
    a ranking of columns that are pure noise, and a list presented without a
    floor invites the reader to act on whatever came first.

    So the response is shuffled many times over and the whole procedure
    repeated. That
    establishes what the best importance score looks like when there is
    certainly nothing to find, and any read that fails to clear it is reported
    as carrying no evidence rather than as merely ranking low.

    The fit quality is cross-validated, not in-sample. An in-sample R-squared
    on a dozen clones and six reads is close to 1 whatever the data says.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_score

    columns = list(early) + [late]
    data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(data)
    if n < 8 or len(early) < 1:
        return ScreenResult(failed="At least eight clones with complete readings "
                                   "are needed before a screen can be assessed.")

    X = data[list(early)].to_numpy(dtype=float)
    y = data[late].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    def importances(target: np.ndarray) -> np.ndarray:
        forest = RandomForestRegressor(
            n_estimators=FOREST_TREES, random_state=seed, min_samples_leaf=2,
            n_jobs=-1)
        forest.fit(X, target)
        return forest.feature_importances_

    observed = importances(y)

    # The null, per read rather than across all of them.
    #
    # The first version compared every read against the *largest* importance
    # any read reached under shuffling. That is a family-wise bar, and forest
    # importances sum to one, so two genuinely predictive reads split the
    # credit between them and the bar sits above both: a read built to be
    # predictive was reported as carrying no evidence. Each read is now judged
    # against its own shuffled distribution, which is the comparison that
    # answers "could this read have scored this highly by chance".
    null = np.empty((noise_runs, X.shape[1]))
    for i in range(noise_runs):
        null[i] = importances(rng.permutation(y))
    # A one-sided permutation p-value, with the observed value included in the
    # count so it can never read as exactly zero.
    p_values = (1.0 + (null >= observed).sum(axis=0)) / (noise_runs + 1.0)
    p_adjusted = _benjamini_hochberg(p_values)
    ceiling = float(np.quantile(null.max(axis=1), 0.95))

    folds = KFold(n_splits=min(5, n), shuffle=True, random_state=seed)
    scores = cross_val_score(
        RandomForestRegressor(n_estimators=FOREST_TREES, random_state=seed,
                              min_samples_leaf=2, n_jobs=-1),
        X, y, cv=folds, scoring="r2")
    r2_cv = float(np.mean(scores))

    reads = []
    for i, name in enumerate(early):
        correlation = float(np.corrcoef(X[:, i], y)[0, 1]) if n > 2 else None
        reads.append(ReadImportance(
            name=name, importance=float(observed[i]),
            above_noise=bool(p_adjusted[i] < ALPHA),
            noise_ceiling=float(np.quantile(null[:, i], 0.95)),
            p_value=float(p_values[i]), p_adjusted=float(p_adjusted[i]),
            correlation=correlation))
    reads.sort(key=lambda r: -r.importance)

    real = [r for r in reads if r.above_noise]
    cautions = []
    if r2_cv <= 0:
        cautions.append(
            "**Cross-validated R² is %.3f — at or below zero.** On clones it "
            "has not seen, the model predicts worse than simply guessing the "
            "average. Whatever the ranking below says, this set of early reads "
            "does not predict %s." % (r2_cv, late))
    # Reads that track each other split the credit between them, so a pair of
    # genuine predictors can each fall short while either alone would clear.
    if len(early) > 1:
        correlations = data[list(early)].corr().abs()
        np.fill_diagonal(correlations.values, 0.0)
        pairs = [(a, b, correlations.loc[a, b])
                 for i, a in enumerate(early) for b in list(early)[i + 1:]
                 if correlations.loc[a, b] >= 0.7]
        if pairs:
            cautions.append(
                "**%s track each other** (correlation %.2f). An importance "
                "score is the credit a read gets once the others are in the "
                "model, so two reads carrying the same information split it "
                "and may each fall short of the floor when either alone would "
                "clear it. Dropping one and re-running says which."
                % (" and ".join("%s/%s" % (a, b) for a, b, _ in pairs[:2]),
                   pairs[0][2]))

    # "No evidence" here means no evidence *once the other reads are in the
    # model*, which is not what a reader hears. A read that tracks the
    # response perfectly well on its own, but adds nothing the others have not
    # already said, comes out below the floor - and someone acting on that
    # could stop making a measurement that would predict perfectly well if it
    # were the only one taken.
    standalone = [r for r in reads
                  if not r.above_noise and r.correlation is not None
                  and abs(r.correlation) >= MARGINAL_CORRELATION]
    if standalone and real:
        cautions.append(
            "**%s %s no evidence here, yet %s the response on %s own** "
            "(correlation %s). The verdict is *adds nothing once the other "
            "reads are in the model* — not *knows nothing*. If you are "
            "deciding which measurements to keep taking, screen each one "
            "against %s by itself before dropping it."
            % (", ".join(r.name for r in standalone[:3]),
               "carries" if len(standalone) == 1 else "carry",
               "it tracks" if len(standalone) == 1 else "they track",
               "its" if len(standalone) == 1 else "their",
               ", ".join("%.2f" % abs(r.correlation) for r in standalone[:3]),
               late))

    if n < 20:
        cautions.append(
            "%d clones. Importance rankings from fewer than about twenty move "
            "a great deal between samples; treat the order as provisional even "
            "for the reads that clear the floor." % n)

    if not real:
        plain = ("**None of these early reads carries evidence about %s.** "
                 "Every one scores below what a shuffled response achieves by "
                 "chance, so the ranking below is ordering noise. Cross-"
                 "validated R² is %.3f." % (late, r2_cv))
    else:
        plain = ("**%s %s %s.** %s clear the floor set by shuffling the "
                 "response %d times; the rest score no better than chance and "
                 "are carrying no evidence. Cross-validated R² is %.3f — that "
                 "is the fit on clones the model has not seen, not on the ones "
                 "it was built from."
                 % (", ".join(r.name for r in real[:4]),
                    "predicts" if len(real) == 1 else "predict", late,
                    "It does" if len(real) == 1 else "They do",
                    noise_runs, r2_cv))

    return ScreenResult(reads=reads, r_squared_cv=r2_cv, n=n,
                        noise_ceiling=ceiling, plain=plain, cautions=cautions)


# ------------------------------------------------------- generation limits

@dataclass
class GenerationLimit:
    slope: Optional[float] = None
    slope_p: Optional[float] = None
    intercept: Optional[float] = None
    limit: Optional[float] = None
    limit_capped: bool = False
    max_observed: Optional[float] = None
    direction: str = ""
    n: int = 0
    r_squared: Optional[float] = None
    curve: Optional[pd.DataFrame] = None
    plain: str = ""
    cautions: list = field(default_factory=list)
    verify_in: str = VERIFY_TREND
    failed: Optional[str] = None


def generation_limit(frame: pd.DataFrame, generation: str, response: str,
                     lsl: Optional[float] = None,
                     usl: Optional[float] = None) -> GenerationLimit:
    """Where the confidence limit on the drift meets the specification.

    The limit is read off the *confidence bound*, not off the fitted line. The
    fitted line is the best guess at the average; half of future lines would
    sit below it. Setting a generation limit where the average meets the
    specification means half the lines cross it earlier, which is not a limit
    anyone would defend in a filing. Reading it off the one-sided 95% bound
    makes the claim "we are confident the average is still inside spec here",
    which is the claim people think they are making.

    Extrapolation is capped at twice the generations actually sampled, in the
    spirit of ICH Q1E's rule for shelf life. A limit of 90 generations from a
    study that reached 30 is arithmetic, not evidence.
    """
    import statsmodels.api as sm

    data = frame[[generation, response]].apply(
        pd.to_numeric, errors="coerce").dropna()
    n = len(data)
    if n < 4:
        return GenerationLimit(failed="At least four passage points are needed "
                                      "to fit a trend.")
    if lsl is None and usl is None:
        return GenerationLimit(failed="A specification limit is needed — the "
                                      "generation limit is where the trend "
                                      "meets it.")

    x = data[generation].to_numpy(dtype=float)
    y = data[response].to_numpy(dtype=float)
    distinct = len(np.unique(x))
    model = sm.OLS(y, sm.add_constant(x)).fit()
    slope = float(model.params[1])
    slope_p = float(model.pvalues[1])
    intercept = float(model.params[0])
    max_observed = float(x.max())

    result = GenerationLimit(
        slope=slope, slope_p=slope_p, intercept=intercept, n=n,
        r_squared=float(model.rsquared), max_observed=max_observed,
        direction="falling" if slope < 0 else "rising")

    if distinct < 3:
        result.cautions.append(
            "Only %d distinct generation numbers. A line through %d points "
            "cannot distinguish a drift from scatter — three points is the "
            "bare minimum and more is much better." % (distinct, distinct))

    if slope_p >= ALPHA:
        result.plain = (
            "**No drift was detected** (slope %.5g per generation, p = %s). "
            "That is not the same as there being none: with %d points this "
            "study could only have detected a steeper trend. A generation "
            "limit cannot be defended from data that shows no measured drift — "
            "it would rest on the scatter, not on the trend."
            % (slope, fmt_p(slope_p), n))
        return result

    # The one-sided 95% bound on the fitted mean, projected forward.
    reach = max_observed * EXTRAPOLATION_CAP
    grid = np.linspace(float(x.min()), reach, 400)
    design = sm.add_constant(grid)
    prediction = model.get_prediction(design)
    frame_out = prediction.summary_frame(alpha=2 * (1 - 0.95))
    curve = pd.DataFrame({
        "Generation": grid,
        "Fitted": frame_out["mean"].to_numpy(),
        "Lower bound": frame_out["mean_ci_lower"].to_numpy(),
        "Upper bound": frame_out["mean_ci_upper"].to_numpy(),
    })
    result.curve = curve

    bound = curve["Lower bound"] if slope < 0 else curve["Upper bound"]
    target = lsl if slope < 0 else usl
    if target is None:
        result.plain = (
            "The trend is %s (%.5g per generation, p = %s) but the "
            "specification limit it would eventually meet was not given."
            % (result.direction, slope, fmt_p(slope_p)))
        return result

    crossed = np.where((bound <= target) if slope < 0 else (bound >= target))[0]
    if crossed.size:
        result.limit = float(curve["Generation"].iloc[crossed[0]])
        result.limit_capped = False
    else:
        result.limit = reach
        result.limit_capped = True

    # A limit at or before the first passage point means the bound was already
    # the wrong side of the specification before any drift happened, which is
    # not a generation limit at all - it is a line that is out of
    # specification now. Reported as "a limit of 0 generations" that reads as
    # a calculation rather than as the thing it is. The usual cause is the
    # wrong column being tracked against the limit, or a limit typed for a
    # different measurement.
    if not result.limit_capped and result.limit <= float(x.min()):
        observed_side = ("below" if slope < 0 else "above")
        result.cautions.append(
            "**The confidence bound is already %s %.4g at generation %.0f, "
            "before any drift.** This is not a generation limit — %s is out "
            "of specification at the earliest passage measured. Check that "
            "%.4g is the limit for %s and not for another measurement."
            % (observed_side, target, x.min(), response, target, response))

    if result.limit_capped:
        result.plain = (
            "**%s is %s by %.5g per generation** (p = %s), but the 95%% "
            "confidence bound does not reach %.4g within twice the generations "
            "sampled. The limit is reported as **at least %.0f generations**, "
            "which is as far as this data supports — the study reached "
            "generation %.0f, and projecting further than twice that is "
            "arithmetic rather than evidence."
            % (response, result.direction, abs(slope), fmt_p(slope_p),
               target, reach, max_observed))
    else:
        result.plain = (
            "**A generation limit of %.0f.** %s is %s by %.5g per generation "
            "(p = %s), and the 95%% confidence bound on the trend meets %.4g "
            "at generation %.0f. The limit is read off the confidence bound "
            "rather than the fitted line: the line is the best guess at the "
            "average, and half of future lines would sit below it."
            % (result.limit, response, result.direction, abs(slope),
               fmt_p(slope_p), target, result.limit))
        if result.limit > max_observed:
            result.cautions.append(
                "The limit of %.0f generations sits beyond generation %.0f, "
                "which is as far as the study went. Everything past that point "
                "assumes the drift stays straight — a line that bends down "
                "later would not show here."
                % (result.limit, max_observed))

    return result
