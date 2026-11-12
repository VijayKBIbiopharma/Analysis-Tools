"""Can the measurement system see what you are chasing?

This is the first question in the project guide's own order, and the reason is
worth stating plainly: every number downstream inherits the measurement. A
capability index, a clone ranking, a comparability study and a trend report all
take the measured value at face value, so if a third of the observed variation
is the assay then a third of each of those conclusions is about the assay.

Gage R&R splits the total variation three ways:

* **Repeatability** - the same operator measuring the same part twice. The
  instrument's own noise.
* **Reproducibility** - the difference between operators measuring the same
  part. People, setup, technique.
* **Part-to-part** - genuine differences between the things being measured.
  This is the only part you actually want to see.

The ANOVA method is used rather than the older average-and-range method,
because it is the only one that can estimate the operator-by-part interaction -
an operator who reads high on small parts and low on large ones - and because
it is what JMP computes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

VERIFY_GAUGE = ("Analyze > Quality and Process > Variability/Attribute Gauge Chart "
                "> Gauge Studies > Gauge R&R (Crossed)")

#: AIAG's bands for the measurement system's share of total variation.
ACCEPTABLE = 10.0
MARGINAL = 30.0

#: AIAG wants at least five distinct categories: fewer and the method cannot
#: separate the parts into enough groups to support a control chart on them.
MIN_DISTINCT_CATEGORIES = 5

#: The interaction is pooled into error above this p-value. 0.25 rather than
#: 0.05 is the standard convention here, and deliberately generous: dropping a
#: real interaction is more costly than carrying a spurious one.
POOL_INTERACTION_ABOVE = 0.25


@dataclass
class GaugeRR:
    """A crossed Gage R&R study, by the ANOVA method."""

    n_parts: int
    n_operators: int
    n_replicates: int
    variances: dict = field(default_factory=dict)
    study_var_pct: dict = field(default_factory=dict)
    tolerance_pct: Optional[dict] = None
    distinct_categories: Optional[int] = None
    interaction_pooled: bool = False
    interaction_p: Optional[float] = None
    anova: dict = field(default_factory=dict)
    verdict: str = ""
    plain: str = ""
    cautions: list = field(default_factory=list)
    verify_in: str = VERIFY_GAUGE

    @property
    def grr_pct(self) -> Optional[float]:
        return self.study_var_pct.get("Total Gage R&R")


def gauge_rr(frame: pd.DataFrame, part: str, operator: str, measurement: str,
             lsl: Optional[float] = None, usl: Optional[float] = None) -> GaugeRR:
    """A crossed Gage R&R on a balanced parts x operators x replicates study."""
    data = frame[[part, operator, measurement]].dropna().copy()
    data[measurement] = pd.to_numeric(data[measurement], errors="coerce")
    data = data.dropna()
    data[part] = data[part].astype(str)
    data[operator] = data[operator].astype(str)

    parts = sorted(data[part].unique())
    operators = sorted(data[operator].unique())
    p, o = len(parts), len(operators)
    counts = data.groupby([part, operator]).size()

    result = GaugeRR(n_parts=p, n_operators=o,
                     n_replicates=int(counts.iloc[0]) if len(counts) else 0)

    if p < 2 or o < 2:
        result.plain = ("A crossed study needs at least two parts and at least "
                        "two operators. This data has %d part(s) and %d "
                        "operator(s)." % (p, o))
        return result
    if len(counts) != p * o:
        missing = p * o - len(counts)
        result.plain = ("%d of the %d part-by-operator combinations were never "
                        "measured. A crossed Gage R&R needs every operator to "
                        "measure every part." % (missing, p * o))
        return result
    if counts.nunique() != 1:
        result.plain = ("The study is unbalanced — between %d and %d repeats per "
                        "part and operator. The ANOVA method here needs the same "
                        "number of repeats throughout."
                        % (counts.min(), counts.max()))
        return result

    n = int(counts.iloc[0])
    if n < 2:
        result.plain = ("Only one measurement per part and operator, so "
                        "repeatability cannot be separated from everything else. "
                        "A Gage R&R needs repeats.")
        return result

    # Sums of squares for a balanced two-way crossed design, written out. The
    # arithmetic is simple enough to read, and reading it is how a reviewer
    # checks it.
    grand = data[measurement].mean()
    part_means = data.groupby(part)[measurement].mean()
    op_means = data.groupby(operator)[measurement].mean()
    cell_means = data.groupby([part, operator])[measurement].mean()

    ss_part = o * n * float(((part_means - grand) ** 2).sum())
    ss_op = p * n * float(((op_means - grand) ** 2).sum())
    ss_int = 0.0
    for (pi, oi), cell in cell_means.items():
        ss_int += n * (cell - part_means[pi] - op_means[oi] + grand) ** 2
    ss_err = float(sum(
        ((group[measurement] - cell_means[(pi, oi)]) ** 2).sum()
        for (pi, oi), group in data.groupby([part, operator])
    ))

    df_part, df_op = p - 1, o - 1
    df_int, df_err = (p - 1) * (o - 1), p * o * (n - 1)
    ms_part, ms_op = ss_part / df_part, ss_op / df_op
    ms_int = ss_int / df_int if df_int else 0.0
    ms_err = ss_err / df_err if df_err else 0.0

    interaction_p = (float(stats.f.sf(ms_int / ms_err, df_int, df_err))
                     if ms_err > 0 and df_int else None)
    pooled = interaction_p is None or interaction_p > POOL_INTERACTION_ABOVE

    if pooled:
        # The interaction goes back into error, which is standard practice when
        # it is not supported. It is reported, not done quietly.
        ms_repeat = (ss_int + ss_err) / (df_int + df_err)
        v_interaction = 0.0
        v_operator = max(0.0, (ms_op - ms_repeat) / (p * n))
        v_part = max(0.0, (ms_part - ms_repeat) / (o * n))
    else:
        ms_repeat = ms_err
        v_interaction = max(0.0, (ms_int - ms_err) / n)
        v_operator = max(0.0, (ms_op - ms_int) / (p * n))
        v_part = max(0.0, (ms_part - ms_int) / (o * n))

    v_repeat = ms_repeat
    v_reproduce = v_operator + v_interaction
    v_grr = v_repeat + v_reproduce
    v_total = v_grr + v_part

    variances = {
        "Repeatability": v_repeat,
        "Reproducibility": v_reproduce,
        "  Operator": v_operator,
        "  Operator x Part": v_interaction,
        "Total Gage R&R": v_grr,
        "Part-to-part": v_part,
        "Total variation": v_total,
    }
    sd_total = float(np.sqrt(v_total)) if v_total > 0 else 0.0
    study = {k: (100 * float(np.sqrt(max(v, 0.0))) / sd_total if sd_total else None)
             for k, v in variances.items()}

    tolerance = None
    if lsl is not None and usl is not None and usl > lsl:
        width = usl - lsl
        tolerance = {k: 100 * 6 * float(np.sqrt(max(v, 0.0))) / width
                     for k, v in variances.items()}

    sd_part, sd_grr = float(np.sqrt(v_part)), float(np.sqrt(v_grr))
    ndc = int(1.41 * sd_part / sd_grr) if sd_grr > 0 else None

    cautions = []
    if any(v == 0.0 for k, v in variances.items() if k in ("  Operator", "Part-to-part")):
        zeroed = [k.strip() for k, v in variances.items()
                  if v == 0.0 and k in ("  Operator", "  Operator x Part")]
        if zeroed:
            cautions.append(
                "The variance estimate for %s came out negative and is reported "
                "as zero. That happens when the component is genuinely small "
                "relative to the noise; it is said out loud rather than hidden."
                % " and ".join(zeroed))
    if p < 10:
        cautions.append(
            "%d parts. AIAG asks for ten, chosen to span the range of the "
            "process — with fewer, the part-to-part variation is estimated from "
            "a thin sample and the Gage R&R percentage inherits that." % p)
    if pooled and interaction_p is not None:
        cautions.append(
            "The operator-by-part interaction was not supported (p = %.3f, above "
            "the 0.25 convention) so it was pooled back into repeatability. JMP "
            "does the same." % interaction_p)

    grr_pct = study["Total Gage R&R"]
    if grr_pct is None:
        verdict, headline = "unknown", "The study could not be summarised."
    elif grr_pct < ACCEPTABLE:
        verdict = "acceptable"
        headline = ("**The measurement system is acceptable.** It accounts for "
                    "%.1f%% of the total variation, below the 10%% AIAG "
                    "threshold." % grr_pct)
    elif grr_pct < MARGINAL:
        verdict = "marginal"
        headline = ("**The measurement system is marginal.** It accounts for "
                    "%.1f%% of the total variation — AIAG calls 10–30%% "
                    "conditionally acceptable, depending on what the measurement "
                    "is used for and what improving it would cost." % grr_pct)
    else:
        verdict = "unacceptable"
        headline = ("**The measurement system is not acceptable.** It accounts "
                    "for %.1f%% of the total variation. More than %.0f%% of what "
                    "you are seeing is the method rather than the product, so "
                    "every number downstream carries that." % (grr_pct, grr_pct))

    detail = (" Repeatability (the instrument) is %.1f%% and reproducibility "
              "(between operators) is %.1f%%, so the %s is the larger problem."
              % (study["Repeatability"], study["Reproducibility"],
                 "instrument" if study["Repeatability"] > study["Reproducibility"]
                 else "difference between operators"))
    categories = ""
    if ndc is not None:
        categories = (" It can distinguish %d distinct categories of part; AIAG "
                      "asks for at least %d%s."
                      % (ndc, MIN_DISTINCT_CATEGORIES,
                         "" if ndc >= MIN_DISTINCT_CATEGORIES else
                         ", so it cannot support a control chart on individual "
                         "measurements"))

    result.n_replicates = n
    result.variances = variances
    result.study_var_pct = study
    result.tolerance_pct = tolerance
    result.distinct_categories = ndc
    result.interaction_pooled = pooled
    result.interaction_p = interaction_p
    result.anova = {
        "Part": {"SS": ss_part, "df": df_part, "MS": ms_part},
        "Operator": {"SS": ss_op, "df": df_op, "MS": ms_op},
        "Part x Operator": {"SS": ss_int, "df": df_int, "MS": ms_int},
        "Repeatability": {"SS": ss_err, "df": df_err, "MS": ms_err},
    }
    result.verdict = verdict
    result.plain = headline + detail + categories
    result.cautions = cautions
    return result
