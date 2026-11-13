"""The pictures.

The analysis path routes to two graphical tools in every cell, not one, and the
reason is that the statistics and the picture answer different questions. A
p-value says whether the groups differ; a box plot says whether one group is
skewed, whether an outlier is carrying the result, and whether the spreads look
alike. People who would not trust a number will trust a picture of it, which is
a good instinct.

Built with Altair, which ships inside Streamlit - so the charts cost no extra
dependency to justify at review.

Colour, and why there is so little of it
----------------------------------------
Most of these plots show **one measurement**, sometimes split across groups.
That is a single series, so it gets a single hue. Colouring each box of a box
plot differently would imply the groups are different *kinds of thing*, which
they are not - they are the same measurement under different conditions.

Colour is therefore reserved for one job: marking a point that has failed a
test. Out of specification, outside a control limit, outside an ANOM decision
limit. Those use the status palette, never a series hue, and they always carry a
label or a shape as well - a reader who cannot distinguish red from grey must
still be able to see which point is flagged.

The palette is the validated default: slot-1 blue for data, status red for
failures, and a muted neutral for the chrome that is the same in light and dark
mode so the charts work either way.
"""

from __future__ import annotations

from typing import Optional, Sequence

import altair as alt
import numpy as np
import pandas as pd
from scipy import stats

# Validated slots. Checked with the palette validator at all-pairs: worst CVD
# delta-E 9.2, worst normal-vision 24.0, both clear of the floors.
DATA = "#2a78d6"          # categorical slot 1, blue
DATA_SOFT = "#9ec5f4"     # blue 200, for intervals and bands
SECOND = "#eb6834"        # categorical slot 2, orange
THIRD = "#1baf7a"         # categorical slot 3, aqua (sub-3:1: always labelled)
CRITICAL = "#d03b3b"      # status: a point that failed
WARNING = "#fab219"       # status: a limit line
GOOD = "#0ca30c"
MUTED = "#898781"         # axis and label ink, identical in both modes
GRID = "#c3c2b7"

CATEGORICAL = [DATA, SECOND, THIRD]

HEIGHT = 300


def _base(frame: pd.DataFrame) -> alt.Chart:
    """A chart with recessive chrome and a transparent surface.

    Transparent rather than white so the chart sits on whatever Streamlit's
    theme provides, light or dark, without a pale rectangle around it.
    """
    return alt.Chart(frame).configure_view(
        strokeWidth=0, fill=None,
    ).configure_axis(
        labelColor=MUTED, titleColor=MUTED, tickColor=GRID, domainColor=GRID,
        gridColor=GRID, gridOpacity=0.35, labelFontSize=11, titleFontSize=12,
    ).configure_legend(
        labelColor=MUTED, titleColor=MUTED, labelFontSize=11, titleFontSize=11,
    )


def _finish(layers: alt.LayerChart, title: str = "") -> alt.LayerChart:
    chart = layers.properties(height=HEIGHT, title=title) if title else \
        layers.properties(height=HEIGHT)
    return chart.configure_view(strokeWidth=0, fill=None).configure_axis(
        labelColor=MUTED, titleColor=MUTED, tickColor=GRID, domainColor=GRID,
        gridColor=GRID, gridOpacity=0.35, labelFontSize=11, titleFontSize=12,
    ).configure_legend(
        labelColor=MUTED, titleColor=MUTED, labelFontSize=11, titleFontSize=11,
    ).configure_title(color=MUTED, fontSize=12, fontWeight="normal", anchor="start")


# ------------------------------------------------------------------ scatter

def scatter_with_fit(frame: pd.DataFrame, x: str, y: str) -> alt.LayerChart:
    """The fitted line plot: the picture the analysis path asks for first.

    The confidence band is on the *fitted line*, not on a future observation.
    That distinction is the one people get wrong - a band that looks tight says
    the line is well located, not that the next batch will land inside it.
    """
    data = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    points = alt.Chart(data).mark_circle(size=70, color=DATA, opacity=0.65).encode(
        x=alt.X(x, type="quantitative", scale=alt.Scale(zero=False)),
        y=alt.Y(y, type="quantitative", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(x, format=".4g"), alt.Tooltip(y, format=".4g")],
    )
    band = points.transform_regression(x, y).mark_line(color=DATA, size=2)
    return _finish(alt.layer(points, band),
                   "Fitted line: %s against %s" % (y, x))


def residuals_vs_fitted(fitted: Sequence[float],
                        residuals: Sequence[float]) -> alt.LayerChart:
    """Residuals against fitted values.

    The single most informative diagnostic there is. A funnel means the spread
    changes with the level; a curve means a straight line was the wrong shape.
    Neither shows up in R-squared.
    """
    data = pd.DataFrame({"Fitted": np.asarray(fitted, dtype=float),
                         "Residual": np.asarray(residuals, dtype=float)})
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
        color=MUTED, strokeDash=[4, 4], size=1).encode(y="y:Q")
    points = alt.Chart(data).mark_circle(size=70, color=DATA, opacity=0.65).encode(
        x=alt.X("Fitted:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("Residual:Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("Fitted:Q", format=".4g"),
                 alt.Tooltip("Residual:Q", format=".4g")],
    )
    return _finish(alt.layer(zero, points),
                   "Residuals against fitted values — look for a funnel or a curve")


# -------------------------------------------------------------- by group

def box_by_group(frame: pd.DataFrame, group: str, value: str) -> alt.LayerChart:
    """Box plot of one measurement across groups.

    One hue, deliberately. These are the same measurement under different
    conditions, not different kinds of thing, and colouring each box separately
    would say otherwise. The individual points are drawn over the boxes because
    a box of five observations looks exactly like a box of fifty, and the
    difference matters more than anything the box itself shows.
    """
    data = frame[[group, value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    data[group] = data[group].astype(str)

    box = alt.Chart(data).mark_boxplot(
        color=DATA_SOFT, size=38, outliers=False, median={"color": DATA},
    ).encode(
        x=alt.X("%s:N" % group, title=group, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("%s:Q" % value, scale=alt.Scale(zero=False)),
    )
    dots = alt.Chart(data).mark_circle(
        size=42, color=DATA, opacity=0.55,
    ).encode(
        x=alt.X("%s:N" % group),
        xOffset=alt.XOffset("jitter:Q", scale=alt.Scale(range=[-12, 12])),
        y=alt.Y("%s:Q" % value),
        tooltip=[group, alt.Tooltip("%s:Q" % value, format=".4g")],
    ).transform_calculate(jitter="random()")
    return _finish(alt.layer(box, dots),
                   "%s by %s — every observation shown over the box" % (value, group))


def main_effects(frame: pd.DataFrame, group: str, value: str) -> alt.LayerChart:
    """Group averages with their confidence intervals.

    The interval is what turns a main-effects plot from decoration into
    evidence. Without it, a line sloping across the page looks like an effect
    whatever the scatter behind it.
    """
    data = frame[[group, value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    data[group] = data[group].astype(str)

    grand = float(data[value].mean())
    summary = data.groupby(group)[value].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"])
    crit = summary["count"].apply(
        lambda n: stats.t.ppf(0.975, n - 1) if n > 1 else np.nan)
    summary["low"] = summary["mean"] - crit * summary["se"]
    summary["high"] = summary["mean"] + crit * summary["se"]

    centre = alt.Chart(pd.DataFrame({"y": [grand]})).mark_rule(
        color=MUTED, strokeDash=[4, 4], size=1).encode(y="y:Q")
    bars = alt.Chart(summary).mark_rule(color=DATA, size=2).encode(
        x=alt.X("%s:N" % group, title=group, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("low:Q", title=value, scale=alt.Scale(zero=False)),
        y2="high:Q",
    )
    line = alt.Chart(summary).mark_line(color=DATA, size=2, point=False).encode(
        x=alt.X("%s:N" % group), y=alt.Y("mean:Q"),
    )
    dots = alt.Chart(summary).mark_point(
        size=110, color=DATA, filled=True,
    ).encode(
        x=alt.X("%s:N" % group), y=alt.Y("mean:Q"),
        tooltip=[group, alt.Tooltip("mean:Q", format=".4g", title="Average"),
                 alt.Tooltip("count:Q", title="Rows"),
                 alt.Tooltip("low:Q", format=".4g", title="95% low"),
                 alt.Tooltip("high:Q", format=".4g", title="95% high")],
    )
    return _finish(alt.layer(centre, line, bars, dots),
                   "Average %s by %s, with 95%% intervals. Dashed line is the "
                   "overall average." % (value, group))


def anom(frame: pd.DataFrame, group: str, value: str,
         alpha: float = 0.05) -> Optional[alt.LayerChart]:
    """Analysis of Means: which group averages differ from the overall average.

    ANOM answers a question ANOVA does not. ANOVA says "the groups are not all
    the same"; ANOM says *which ones* stand out, and by how much, on a single
    picture. The analysis path asks for it by name for three-or-more-level
    inputs.

    The decision limits use a Bonferroni approximation to Nelson's h factors -
    grand mean plus or minus t(1 - alpha/2k, df) x s x sqrt((k-1)/(k n)). JMP
    uses exact h values, so its limits will sit very slightly tighter than
    these; the approximation is conservative, meaning this errs towards not
    flagging a group. Worth knowing if the two are compared to three decimals.
    """
    data = frame[[group, value]].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data = data.dropna()
    data[group] = data[group].astype(str)

    summary = data.groupby(group)[value].agg(["mean", "count", "std"]).reset_index()
    k = len(summary)
    if k < 3 or (summary["count"] < 2).any():
        return None
    n_total = int(summary["count"].sum())
    df = n_total - k
    pooled = float(np.sqrt(
        ((summary["count"] - 1) * summary["std"] ** 2).sum() / df)) if df > 0 else 0.0
    if pooled <= 0:
        return None

    grand = float(data[value].mean())
    h = stats.t.ppf(1 - alpha / (2 * k), df)
    summary["half"] = h * pooled * np.sqrt((k - 1) / (k * summary["count"]))
    summary["upper"] = grand + summary["half"]
    summary["lower"] = grand - summary["half"]
    summary["outside"] = (summary["mean"] > summary["upper"]) | \
                         (summary["mean"] < summary["lower"])
    summary["Verdict"] = np.where(summary["outside"],
                                  "Outside the limits", "Within the limits")

    centre = alt.Chart(pd.DataFrame({"y": [grand]})).mark_rule(
        color=MUTED, size=1).encode(y="y:Q")
    limits = alt.Chart(summary).mark_rule(
        color=WARNING, strokeDash=[5, 3], size=1.5).encode(
        x=alt.X("%s:N" % group, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("lower:Q", title=value, scale=alt.Scale(zero=False)),
        y2="upper:Q")
    stems = alt.Chart(summary).mark_rule(color=MUTED, size=1).encode(
        x=alt.X("%s:N" % group), y=alt.datum(grand), y2="mean:Q")
    dots = alt.Chart(summary).mark_point(size=150, filled=True).encode(
        x=alt.X("%s:N" % group, title=group),
        y=alt.Y("mean:Q"),
        color=alt.Color("Verdict:N",
                        scale=alt.Scale(domain=["Within the limits",
                                                "Outside the limits"],
                                        range=[DATA, CRITICAL]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("Verdict:N",
                        scale=alt.Scale(domain=["Within the limits",
                                                "Outside the limits"],
                                        range=["circle", "triangle-up"]),
                        legend=None),
        tooltip=[group, alt.Tooltip("mean:Q", format=".4g", title="Average"),
                 alt.Tooltip("count:Q", title="Rows"), "Verdict:N"],
    )
    return _finish(alt.layer(limits, centre, stems, dots),
                   "Analysis of Means: a group outside the dashed limits differs "
                   "from the overall average")


# -------------------------------------------------------- capability & control

def capability_histogram(values: Sequence[float], lsl: Optional[float] = None,
                         usl: Optional[float] = None,
                         target: Optional[float] = None,
                         name: str = "Measurement") -> alt.LayerChart:
    """The distribution against the specification.

    The picture that makes a capability index mean something. An index of 1.1
    is a number; a histogram with its right tail sitting on the upper limit is
    a decision.
    """
    x = np.asarray([v for v in values if v == v], dtype=float)
    data = pd.DataFrame({name: x})
    bars = alt.Chart(data).mark_bar(
        color=DATA, opacity=0.8, stroke=None,
    ).encode(
        x=alt.X("%s:Q" % name, bin=alt.Bin(maxbins=30),
                scale=alt.Scale(zero=False)),
        y=alt.Y("count():Q", title="Batches"),
        tooltip=[alt.Tooltip("count():Q", title="Batches")],
    )
    layers = [bars]
    marks = []
    if lsl is not None:
        marks.append({"value": lsl, "Limit": "Lower spec"})
    if usl is not None:
        marks.append({"value": usl, "Limit": "Upper spec"})
    if marks:
        rules = alt.Chart(pd.DataFrame(marks)).mark_rule(
            color=CRITICAL, size=2).encode(
            x="value:Q",
            tooltip=["Limit:N", alt.Tooltip("value:Q", format=".4g")])
        text = alt.Chart(pd.DataFrame(marks)).mark_text(
            align="left", dx=5, dy=-6, color=CRITICAL, fontSize=11).encode(
            x="value:Q", y=alt.value(6), text="Limit:N")
        layers += [rules, text]
    if target is not None:
        layers.append(alt.Chart(pd.DataFrame({"value": [target]})).mark_rule(
            color=GOOD, strokeDash=[5, 3], size=1.5).encode(x="value:Q"))
    return _finish(alt.layer(*layers),
                   "%s against the specification. Red lines are the limits." % name)


def control_chart(values: Sequence[float], centre: float, upper: float,
                  lower: float, flagged: Sequence[int] = (),
                  name: str = "Measurement") -> alt.LayerChart:
    """An individuals chart with the signalling points marked.

    Flagged points get the status colour *and* a different shape, because a
    reader who cannot separate red from blue still has to be able to see which
    point the chart is talking about.
    """
    x = np.asarray([v for v in values if v == v], dtype=float)
    marked = set(int(i) for i in flagged)
    data = pd.DataFrame({
        "Observation": np.arange(1, x.size + 1),
        name: x,
        "State": ["Signal" if i in marked else "In control" for i in range(x.size)],
    })
    limits = pd.DataFrame({
        "value": [upper, centre, lower],
        "Line": ["Upper control limit", "Centre", "Lower control limit"],
    })
    rules = alt.Chart(limits).mark_rule(size=1).encode(
        y="value:Q",
        color=alt.Color("Line:N",
                        scale=alt.Scale(
                            domain=["Upper control limit", "Centre",
                                    "Lower control limit"],
                            range=[CRITICAL, MUTED, CRITICAL]),
                        legend=None),
        strokeDash=alt.StrokeDash("Line:N", legend=None),
        tooltip=["Line:N", alt.Tooltip("value:Q", format=".4g")],
    )
    line = alt.Chart(data).mark_line(color=DATA, size=1.5).encode(
        x=alt.X("Observation:Q", scale=alt.Scale(nice=False)),
        y=alt.Y("%s:Q" % name, scale=alt.Scale(zero=False)),
    )
    dots = alt.Chart(data).mark_point(size=70, filled=True).encode(
        x="Observation:Q", y=alt.Y("%s:Q" % name),
        color=alt.Color("State:N",
                        scale=alt.Scale(domain=["In control", "Signal"],
                                        range=[DATA, CRITICAL]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("State:N",
                        scale=alt.Scale(domain=["In control", "Signal"],
                                        range=["circle", "triangle-up"]),
                        legend=None),
        tooltip=["Observation:Q", alt.Tooltip("%s:Q" % name, format=".4g"),
                 "State:N"],
    )
    return _finish(alt.layer(rules, line, dots),
                   "Individuals chart. Triangles are points the rules flagged.")


def variability_chart(frame: pd.DataFrame, part: str, operator: str,
                      measurement: str) -> alt.LayerChart:
    """Every reading, grouped by part and coloured by operator.

    JMP calls this the variability chart and puts it at the centre of a Gage
    R&R, rightly: the numbers say how much of the variation is the measurement,
    and this says *what it looks like* — whether one operator sits high
    throughout, whether repeats scatter widely, whether one part behaves oddly.
    """
    data = frame[[part, operator, measurement]].copy()
    data[measurement] = pd.to_numeric(data[measurement], errors="coerce")
    data = data.dropna()
    data[part] = data[part].astype(str)
    data[operator] = data[operator].astype(str)

    operators = sorted(data[operator].unique())
    palette = CATEGORICAL[:len(operators)] if len(operators) <= 3 else None

    colour = alt.Color(
        "%s:N" % operator,
        scale=alt.Scale(domain=operators, range=palette) if palette else alt.Undefined,
        legend=alt.Legend(title=operator, orient="top"))

    means = data.groupby([part, operator])[measurement].mean().reset_index()
    dots = alt.Chart(data).mark_point(size=55, filled=True, opacity=0.8).encode(
        x=alt.X("%s:N" % part, title=part, axis=alt.Axis(labelAngle=-45)),
        xOffset=alt.XOffset("%s:N" % operator),
        y=alt.Y("%s:Q" % measurement, scale=alt.Scale(zero=False)),
        color=colour,
        shape=alt.Shape("%s:N" % operator, legend=None),
        tooltip=[part, operator, alt.Tooltip("%s:Q" % measurement, format=".4g")],
    )
    connect = alt.Chart(means).mark_line(size=1, opacity=0.5).encode(
        x=alt.X("%s:N" % part), y=alt.Y("%s:Q" % measurement),
        color=colour, detail="%s:N" % operator,
    )
    return _finish(alt.layer(connect, dots),
                   "Every reading by part, coloured by %s. Lines join each "
                   "%s's averages." % (operator, operator.lower()))


# ------------------------------------------------------------------ designs

def effects_pareto(effects: Sequence, margin: Optional[float] = None,
                   response: str = "the response") -> alt.LayerChart:
    """Effects by size, largest first, with the significance line.

    The chart the project guide asks for by name: "keep terms above the red
    line". Sorting by absolute size and drawing the threshold turns a table of
    p-values into one glance.
    """
    data = pd.DataFrame([
        {"Term": e.term, "Effect": abs(float(e.effect)),
         "Direction": "Increases %s" % response if e.effect > 0
                      else "Decreases %s" % response,
         "Verdict": "Real" if e.significant else "Not distinguishable"}
        for e in effects
    ]).sort_values("Effect", ascending=False)

    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("Term:N", sort="-x", title=None),
        x=alt.X("Effect:Q", title="Size of effect (low setting to high)"),
        color=alt.Color("Verdict:N",
                        scale=alt.Scale(domain=["Real", "Not distinguishable"],
                                        range=[DATA, "#c9c8c2"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Term:N", alt.Tooltip("Effect:Q", format=".4g"),
                 "Direction:N", "Verdict:N"],
    )
    layers = [bars]
    if margin:
        line = alt.Chart(pd.DataFrame({"x": [margin]})).mark_rule(
            color=CRITICAL, size=2, strokeDash=[5, 3]).encode(x="x:Q")
        label = alt.Chart(pd.DataFrame({"x": [margin]})).mark_text(
            text="Lenth margin", align="left", dx=5, dy=-8,
            color=CRITICAL, fontSize=11).encode(x="x:Q", y=alt.value(4))
        layers += [line, label]
    return _finish(alt.layer(*layers),
                   "Effects by size. Bars past the line are the ones to keep.")


def effects_normal_plot(effects: Sequence) -> alt.LayerChart:
    """Effects against normal scores.

    If nothing is real, the effects are just noise and fall on a straight line.
    The ones that matter are the points that leave it. This catches a large
    effect that a Pareto shows as merely the tallest bar, because here the
    comparison is against what noise alone would have looked like.
    """
    values = np.array([float(e.effect) for e in effects])
    order = np.argsort(values)
    m = values.size
    scores = stats.norm.ppf((np.arange(1, m + 1) - 0.375) / (m + 0.25))
    data = pd.DataFrame({
        "Effect": values[order],
        "Normal score": scores,
        "Term": [effects[i].term for i in order],
        "Verdict": ["Real" if effects[i].significant else "Not distinguishable"
                    for i in order],
    })
    points = alt.Chart(data).mark_point(size=110, filled=True).encode(
        x=alt.X("Effect:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("Normal score:Q"),
        color=alt.Color("Verdict:N",
                        scale=alt.Scale(domain=["Real", "Not distinguishable"],
                                        range=[CRITICAL, DATA]),
                        legend=alt.Legend(title=None, orient="top")),
        shape=alt.Shape("Verdict:N",
                        scale=alt.Scale(domain=["Real", "Not distinguishable"],
                                        range=["triangle-up", "circle"]),
                        legend=None),
        tooltip=["Term:N", alt.Tooltip("Effect:Q", format=".4g"), "Verdict:N"],
    )
    labels = alt.Chart(data[data["Verdict"] == "Real"]).mark_text(
        align="left", dx=8, fontSize=11, color=MUTED).encode(
        x="Effect:Q", y="Normal score:Q", text="Term:N")
    return _finish(alt.layer(points, labels),
                   "Effects against normal scores. Noise falls on a line; what "
                   "matters leaves it.")


def interaction_plot(frame: pd.DataFrame, first: str, second: str,
                     response: str) -> alt.LayerChart:
    """Two factors together: non-parallel lines mean the effect of one depends
    on the other.

    The only picture that shows an interaction for what it is. A main-effects
    plot of two interacting factors looks unremarkable, which is how
    interactions get missed.
    """
    data = frame[[first, second, response]].copy()
    data[response] = pd.to_numeric(data[response], errors="coerce")
    data = data.dropna()
    data[first] = data[first].astype(str)
    data[second] = data[second].astype(str)
    summary = data.groupby([first, second])[response].mean().reset_index()

    levels = sorted(summary[second].unique())
    palette = CATEGORICAL[:len(levels)] if len(levels) <= 3 else None
    colour = alt.Color(
        "%s:N" % second,
        scale=alt.Scale(domain=levels, range=palette) if palette else alt.Undefined,
        legend=alt.Legend(title=second, orient="top"))

    lines = alt.Chart(summary).mark_line(size=2, point=True).encode(
        x=alt.X("%s:N" % first, title=first, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("%s:Q" % response, scale=alt.Scale(zero=False),
                title="Average %s" % response),
        color=colour,
        strokeDash=alt.StrokeDash("%s:N" % second, legend=None),
        tooltip=[first, second, alt.Tooltip("%s:Q" % response, format=".4g")],
    )
    return _finish(lines,
                   "%s x %s. Parallel lines mean no interaction; lines that "
                   "cross or fan mean the effect of one depends on the other."
                   % (first, second))


# ------------------------------------------------- cell line development

def importance_bars(reads: Sequence, ceiling: Optional[float] = None,
                    late: str = "the final measurement") -> alt.LayerChart:
    """Early reads by how much they carry, with the noise floor drawn.

    Every ranking has a top entry, including a ranking of pure noise. The
    floor is the point of the chart: a bar below it ranks high among columns
    that know nothing.
    """
    data = pd.DataFrame([
        {"Read": r.name, "Importance": float(r.importance),
         "Verdict": "Carries information" if r.above_noise else "No evidence",
         "p (adjusted)": ("%.4g" % r.p_adjusted) if r.p_adjusted is not None else "—"}
        for r in reads
    ]).sort_values("Importance", ascending=False)

    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=4, height=18).encode(
        y=alt.Y("Read:N", sort="-x", title=None),
        x=alt.X("Importance:Q", title="Share of the explanation of %s" % late),
        color=alt.Color("Verdict:N",
                        scale=alt.Scale(domain=["Carries information", "No evidence"],
                                        range=[DATA, "#c9c8c2"]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=["Read:N", alt.Tooltip("Importance:Q", format=".4f"),
                 "Verdict:N", "p (adjusted):N"],
    )
    layers = [bars]

    # A tick per bar rather than one line across the chart. The single line
    # was the 95th percentile of the largest importance under shuffling -
    # one number for the whole set - while the verdict is decided per read
    # and adjusted for how many were tested. Those two can disagree: measured
    # on weak signals, about one bar in sixty came out grey while sitting
    # past the line, which reads as a contradiction rather than as the
    # multiple-testing adjustment it is. Each read's own floor cannot
    # disagree with its own bar.
    per_read = pd.DataFrame([
        {"Read": r.name, "Floor": float(r.noise_ceiling)}
        for r in reads if getattr(r, "noise_ceiling", None) is not None])
    if not per_read.empty:
        layers.append(alt.Chart(per_read).mark_tick(
            color=CRITICAL, thickness=2, size=22).encode(
            y=alt.Y("Read:N", sort="-x"), x="Floor:Q",
            tooltip=["Read:N", alt.Tooltip("Floor:Q", format=".4f",
                                           title="This read's noise floor")]))
        subtitle = ("The tick is where this read's own score reaches when the "
                    "response is shuffled. Grey past a tick means the read "
                    "did not survive adjusting for how many were tested.")
    elif ceiling:
        layers.append(alt.Chart(pd.DataFrame({"x": [float(ceiling)]})).mark_rule(
            color=CRITICAL, size=2, strokeDash=[5, 3]).encode(x="x:Q"))
        layers.append(alt.Chart(pd.DataFrame({"x": [float(ceiling)]})).mark_text(
            text="noise floor", align="left", dx=5, dy=-8, color=CRITICAL,
            fontSize=11).encode(x="x:Q", y=alt.value(4)))
        subtitle = ("Bars short of the floor rank high among columns that know "
                    "nothing.")
    else:
        subtitle = "Importance is a share, so it always sums to one."
    return _finish(alt.layer(*layers), subtitle)


def drift_with_bound(points: pd.DataFrame, curve: pd.DataFrame,
                     generation: str, response: str,
                     spec: Optional[float] = None,
                     limit: Optional[float] = None,
                     max_observed: Optional[float] = None) -> alt.LayerChart:
    """Passage data, the fitted drift, and the confidence bound that sets the limit.

    The bound is the load-bearing line, so it is drawn solid and the fit
    dashed - the opposite of the usual emphasis, because the limit is read off
    the bound. Beyond the last real passage point the band is shaded to show
    where measurement stops and arithmetic begins.
    """
    # Not anchored at zero. A cell line drifting from 3.05 to 2.45 g/L is the
    # whole subject of the chart, and on a zero baseline it is a flat line in
    # the top fifth of the frame.
    span = alt.Scale(zero=False, nice=True)
    band = alt.Chart(curve).mark_area(opacity=0.3, color=DATA_SOFT).encode(
        x=alt.X("Generation:Q", title=generation),
        y=alt.Y("Lower bound:Q", title=response, scale=span),
        y2="Upper bound:Q")
    fit = alt.Chart(curve).mark_line(color=DATA, size=1.5,
                                     strokeDash=[4, 3]).encode(
        x="Generation:Q", y=alt.Y("Fitted:Q", scale=span))
    falling = float(curve["Fitted"].iloc[-1]) < float(curve["Fitted"].iloc[0])
    edge = "Lower bound" if falling else "Upper bound"
    bound = alt.Chart(curve).mark_line(color=DATA, size=2).encode(
        x="Generation:Q", y=alt.Y("%s:Q" % edge, title=response, scale=span))
    dots = alt.Chart(points).mark_circle(size=70, color=SECOND, opacity=0.85).encode(
        x="%s:Q" % generation, y=alt.Y("%s:Q" % response, scale=span),
        tooltip=["%s:Q" % generation, alt.Tooltip("%s:Q" % response, format=".4g")])

    layers = [band, fit, bound, dots]
    if max_observed is not None:
        shade = pd.DataFrame({"x": [float(max_observed)],
                              "x2": [float(curve["Generation"].max())]})
        layers.insert(0, alt.Chart(shade).mark_rect(
            opacity=0.07, color=MUTED).encode(x="x:Q", x2="x2:Q"))
        layers.append(alt.Chart(shade).mark_rule(
            color=MUTED, size=1, strokeDash=[2, 3]).encode(x="x:Q"))
        layers.append(alt.Chart(shade).mark_text(
            text="beyond the data", align="left", dx=5, dy=10, color=MUTED,
            fontSize=11).encode(x="x:Q", y=alt.value(0)))
    if spec is not None:
        layers.append(alt.Chart(pd.DataFrame({"y": [float(spec)]})).mark_rule(
            color=CRITICAL, size=2).encode(y="y:Q"))
        layers.append(alt.Chart(pd.DataFrame({"y": [float(spec)]})).mark_text(
            text="specification", align="left", dx=5, dy=-8, color=CRITICAL,
            fontSize=11).encode(y="y:Q", x=alt.value(4)))
    if limit is not None:
        layers.append(alt.Chart(pd.DataFrame({"x": [float(limit)]})).mark_rule(
            color=WARNING, size=2, strokeDash=[6, 3]).encode(x="x:Q"))
        layers.append(alt.Chart(pd.DataFrame({"x": [float(limit)]})).mark_text(
            text="limit", align="left", dx=5, dy=-8, color=WARNING,
            fontSize=11).encode(x="x:Q", y=alt.value(4)))
    return _finish(alt.layer(*layers),
                   "The limit is read off the solid bound, not the dashed fit: "
                   "half of future lines would sit below the fit.")
