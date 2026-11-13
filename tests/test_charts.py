"""The charts.

Charts had no tests at all until a zero-anchored y axis squashed a passage
study into the top fifth of its frame — a defect invisible to every unit test
in the suite and obvious the moment anyone looked at it. These are the
mechanical checks: that each chart serialises, that measurement axes are not
anchored at zero, and that the few charts carrying a verdict cannot contradict
themselves.

Serialising matters on its own. An Altair chart is built lazily, so a mistake
in an encoding raises nothing until Streamlit converts it — which happens on
the page, in front of whoever is using it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from analysis_tools import charts

RNG = np.random.default_rng(5)


@dataclass
class FakeEffect:
    term: str
    effect: float
    significant: bool


@dataclass
class FakeRead:
    name: str
    importance: float
    above_noise: bool
    noise_ceiling: float
    p_adjusted: Optional[float] = 0.01


def grouped():
    return pd.DataFrame({
        "Resin": ["A"] * 8 + ["B"] * 8 + ["C"] * 8,
        "Operator": (["Sam", "Lee"] * 12),
        "Part": [str(i % 6) for i in range(24)],
        "Yield": RNG.normal(90, 2, 24),
        "Temp": RNG.normal(35, 3, 24),
    })


def effects():
    return [FakeEffect("Temp", 2.4, True), FakeEffect("pH", -1.1, True),
            FakeEffect("Temp*pH", 0.3, False), FakeEffect("Time", -0.2, False)]


def reads():
    return [FakeRead("Day14", 0.62, True, 0.35, 0.01),
            FakeRead("Day7", 0.21, False, 0.38, 0.4),
            FakeRead("Seed", 0.17, False, 0.37, 0.9)]


def curve():
    x = np.linspace(0, 120, 200)
    fitted = 3.0 - 0.006 * x
    spread = 0.03 + 0.0008 * np.abs(x - 40)
    return pd.DataFrame({"Generation": x, "Fitted": fitted,
                         "Lower bound": fitted - spread,
                         "Upper bound": fitted + spread})


def every_chart():
    """Each chart, built the way the pages build it."""
    frame = grouped()
    yield "scatter_with_fit", charts.scatter_with_fit(frame, "Temp", "Yield")
    yield "residuals_vs_fitted", charts.residuals_vs_fitted(
        RNG.normal(90, 2, 24), RNG.normal(0, 1, 24))
    yield "box_by_group", charts.box_by_group(frame, "Resin", "Yield")
    yield "main_effects", charts.main_effects(frame, "Resin", "Yield")
    yield "anom", charts.anom(frame, "Resin", "Yield")
    yield "capability_histogram", charts.capability_histogram(
        RNG.normal(100, 2, 60), lsl=94.0, usl=106.0, target=100.0)
    yield "control_chart", charts.control_chart(
        RNG.normal(10, 1, 40), centre=10.0, upper=13.0, lower=7.0,
        flagged=[7, 22])
    yield "variability_chart", charts.variability_chart(
        frame, "Part", "Operator", "Yield")
    yield "effects_pareto", charts.effects_pareto(effects(), margin=0.9)
    yield "effects_normal_plot", charts.effects_normal_plot(effects())
    yield "interaction_plot", charts.interaction_plot(
        frame, "Resin", "Operator", "Yield")
    yield "importance_bars", charts.importance_bars(reads(), 0.40, "Final")
    yield "drift_with_bound", charts.drift_with_bound(
        pd.DataFrame({"Generation": [0, 20, 40, 60], "Titer": [3.0, 2.9, 2.8, 2.6]}),
        curve(), "Generation", "Titer", spec=2.5, limit=95.0, max_observed=60.0)


CHARTS = list(every_chart())


@pytest.mark.parametrize("name,chart", CHARTS, ids=[n for n, _ in CHARTS])
def test_every_chart_serialises(name, chart):
    """The failure this catches happens on the page, not in the tests."""
    assert chart is not None, "%s returned nothing" % name
    spec = chart.to_dict()
    assert spec.get("layer") or spec.get("mark") or spec.get("hconcat") \
        or spec.get("vconcat"), "%s produced an empty spec" % name


def _rows(spec, layer):
    """A layer's data, whether inline or hoisted into the datasets block."""
    data = layer.get("data", {})
    if "values" in data:
        return data["values"]
    return spec.get("datasets", {}).get(data.get("name"), [])


def _scales(spec, axis):
    """Every scale on one axis, anywhere in a layered spec."""
    found = []
    def walk(node):
        if isinstance(node, dict):
            encoding = node.get("encoding", {})
            channel = encoding.get(axis)
            if isinstance(channel, dict) and channel.get("type") == "quantitative":
                found.append(channel.get("scale") or {})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(spec)
    return found


#: Charts whose measurement axis must not be anchored at zero. A yield of 90
#: against a zero baseline is a flat line near the top of the frame, and the
#: variation the chart exists to show is a few pixels tall.
MEASUREMENT_AXIS = {
    "scatter_with_fit": "y", "residuals_vs_fitted": "y", "box_by_group": "y",
    "main_effects": "y", "anom": "y", "control_chart": "y",
    "variability_chart": "y", "drift_with_bound": "y",
}


@pytest.mark.parametrize("name,chart", CHARTS, ids=[n for n, _ in CHARTS])
def test_measurement_axes_are_not_anchored_at_zero(name, chart):
    axis = MEASUREMENT_AXIS.get(name)
    if axis is None:
        return                       # bars and histograms belong on a zero base
    scales = _scales(chart.to_dict(), axis)
    assert scales, "%s has no quantitative %s axis to check" % (name, axis)
    # Layered charts add rules and labels whose encodings carry no scale of
    # their own; Vega-Lite resolves one scale for the layer. So the question
    # is whether the chart declares a non-zero baseline anywhere and
    # contradicts it nowhere - which is exactly what the defect looked like:
    # drift_with_bound declared nothing at all and defaulted to zero.
    assert any(s.get("zero") is False for s in scales), (
        "%s never declares zero=False on its %s axis — the data will sit "
        "squashed against one edge" % (name, axis))
    assert not any(s.get("zero") is True for s in scales), (
        "%s forces its %s axis to include zero" % (name, axis))


def test_a_bar_chart_does_start_at_zero():
    """The converse. A share of explanation read off a truncated axis is a
    misleading bar, which is the opposite mistake."""
    spec = charts.importance_bars(reads(), 0.40, "Final").to_dict()
    scales = _scales(spec, "x")
    assert not any(s.get("zero") is False for s in scales)


def test_importance_bars_draw_a_floor_for_every_read():
    """One line across the chart was the family-wise floor while the colour
    was a per-read verdict; the two could disagree, and a grey bar past the
    line reads as a contradiction."""
    spec = charts.importance_bars(reads(), 0.40, "Final").to_dict()
    ticks = [layer for layer in spec["layer"]
             if layer.get("mark", {}).get("type") == "tick"]
    assert len(ticks) == 1
    values = _rows(spec, ticks[0])
    assert len(values) == len(reads())
    assert {v["Read"] for v in values} == {r.name for r in reads()}


def test_importance_bars_survive_reads_without_a_floor():
    bare = [FakeRead("A", 0.5, True, None), FakeRead("B", 0.5, False, None)]
    charts.importance_bars(bare, 0.4, "Final").to_dict()
    charts.importance_bars(bare, None, "Final").to_dict()


def test_drift_chart_marks_where_the_data_stops():
    """Everything right of the last passage point is arithmetic, and the
    chart has to say so."""
    spec = charts.drift_with_bound(
        pd.DataFrame({"Generation": [0, 20, 40, 60], "Titer": [3.0, 2.9, 2.8, 2.6]}),
        curve(), "Generation", "Titer", spec=2.5, limit=95.0,
        max_observed=60.0).to_dict()
    text = str(spec)
    assert "beyond the data" in text
    assert "specification" in text
    assert "limit" in text


def test_drift_chart_without_a_limit_or_spec_still_draws():
    charts.drift_with_bound(
        pd.DataFrame({"Generation": [0, 20, 40], "Titer": [3.0, 2.9, 2.8]}),
        curve(), "Generation", "Titer").to_dict()


def test_drift_chart_tracks_the_upper_bound_when_the_trend_rises():
    """A measurement drifting up towards an upper limit is read off the upper
    bound, not the lower one."""
    rising = curve()
    rising["Fitted"] = 2.0 + 0.006 * rising["Generation"]
    rising["Lower bound"] = rising["Fitted"] - 0.05
    rising["Upper bound"] = rising["Fitted"] + 0.05
    spec = charts.drift_with_bound(
        pd.DataFrame({"Generation": [0, 20], "Titer": [2.0, 2.1]}),
        rising, "Generation", "Titer", spec=2.9).to_dict()
    solid = [layer for layer in spec["layer"]
             if layer.get("mark", {}).get("type") == "line"
             and layer["mark"].get("strokeDash") is None]
    assert solid, "no solid bound line drawn"
    assert solid[0]["encoding"]["y"]["field"] == "Upper bound"


def test_the_palette_is_the_validated_one():
    """These exact values were checked with the palette validator: worst
    colour-vision-deficient delta-E 9.2, worst normal-vision 24.0. Changing
    one without re-running it is how a chart quietly stops being readable."""
    assert charts.DATA == "#2a78d6"
    assert charts.SECOND == "#eb6834"
    assert charts.THIRD == "#1baf7a"
    assert charts.CRITICAL == "#d03b3b"


def test_status_is_never_carried_by_colour_alone():
    """A reader who cannot separate red from blue still has to see which
    point the chart is talking about."""
    spec = charts.control_chart(RNG.normal(10, 1, 30), centre=10.0, upper=13.0,
                                lower=7.0, flagged=[4, 19]).to_dict()
    text = str(spec)
    assert "shape" in text, "flagged points are distinguished by colour only"
