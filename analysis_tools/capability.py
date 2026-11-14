"""Is the process stable, and is it capable?

In that order, and the order is the point. A capability index computed on a
process that is not in control describes a sample, not a process: the mean and
the spread it rests on are not properties of anything that will still be true
next month. The capability guidance in the Six Sigma project guide asks the
stability question first for exactly this reason, so this module refuses to
present capability without presenting stability beside it.

The second refusal is about the parts-per-million figures. Cpk is a ratio of
distances and survives mild non-normality; a PPM or a Z.bench is a statement
about the *tails*, which is precisely where the normal model fails first. A
right-skewed yield distribution can show Cpk 1.4 and a true defect rate ten
times what the normal model predicts. So those figures are withheld when the
data fails a normality test, and the reason is printed where the number would
have been.

That is not this tool being unusually strict. JMP does the same: "Benchmark Z
statistics are available only for capability analyses based on the normal
distribution."

Within and overall
------------------
Two families, and they answer different questions:

* **Cp, Cpk** use the *within-subgroup* standard deviation - the short-term
  spread, what the process could do if the only variation were the noise
  between consecutive units.
* **Pp, Ppk** use the *overall* standard deviation - everything, including the
  drift between shifts and lots. This is what the customer actually receives.

JMP labels them this way by default (its AIAG labelling preference), so the
names here match what a user will see when they check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

from analysis_tools.assumptions import check_normality

#: d2 for a moving range of two consecutive observations. The constant that
#: turns an average moving range into a standard deviation.
D2_N2 = 1.128
#: D4 for the moving-range chart's upper limit at n = 2.
D4_N2 = 3.267

#: The conventional floor. JMP's help gives the same number.
MINIMUM_CAPABLE = 1.33

#: How often a *perfectly stable* normal process trips at least one of the four
#: rules, by sample size. Measured here, over 1,500 simulated stable processes
#: per size, rather than taken from a table - and it is the single most
#: important number on this page.
#:
#: Every rule added makes a chart more sensitive and also more likely to cry
#: wolf, and the rates compound with the number of points. At 200 observations
#: four out of five stable processes signal somewhere. A tool that announces
#: "not stable" on that evidence, with no indication of how often it would say
#: so by chance, teaches people to ignore it - which costs them the signals
#: that were real.
#:
#: So the rate is reported beside the signals. Rule 1 alone measures 14.5% at
#: n = 60 against a theoretical 14.9%, which is how we know the rules
#: themselves are implemented correctly.
FALSE_ALARM_BY_N = {20: 0.081, 30: 0.173, 50: 0.303, 60: 0.345,
                    100: 0.543, 200: 0.817}


def chance_of_a_signal(n: int) -> float:
    """Roughly how often a stable process of this length trips a rule.

    Linear interpolation between the measured points, held flat outside them.
    """
    sizes = sorted(FALSE_ALARM_BY_N)
    if n <= sizes[0]:
        return FALSE_ALARM_BY_N[sizes[0]] * n / sizes[0]
    if n >= sizes[-1]:
        return FALSE_ALARM_BY_N[sizes[-1]]
    for low, high in zip(sizes, sizes[1:]):
        if low <= n <= high:
            span = (n - low) / (high - low)
            return (FALSE_ALARM_BY_N[low]
                    + span * (FALSE_ALARM_BY_N[high] - FALSE_ALARM_BY_N[low]))
    return FALSE_ALARM_BY_N[sizes[-1]]

VERIFY_CAPABILITY = "Analyze > Quality and Process > Process Capability"
VERIFY_CONTROL = "Analyze > Quality and Process > Control Chart > Individuals & Moving Range"


@dataclass
class Signal:
    """One out-of-control signal on a control chart."""

    rule: str
    description: str
    points: list


@dataclass
class Stability:
    """Whether the process is behaving consistently over time."""

    n: int
    centre: float
    sigma_within: float
    upper: float
    lower: float
    mr_bar: float
    mr_upper: float
    signals: list = field(default_factory=list)
    plain: str = ""
    verify_in: str = VERIFY_CONTROL

    @property
    def stable(self) -> bool:
        return not self.signals


@dataclass
class Capability:
    """Capability indices, and what could not honestly be computed."""

    n: int
    mean: float
    sigma_within: Optional[float]
    sigma_overall: float
    lsl: Optional[float]
    usl: Optional[float]
    target: Optional[float]
    indices: dict = field(default_factory=dict)
    intervals: dict = field(default_factory=dict)
    observed_out: Optional[float] = None
    predicted_ppm: Optional[float] = None
    z_bench: Optional[float] = None
    withheld: list = field(default_factory=list)
    plain: str = ""
    cautions: list = field(default_factory=list)
    normality = None
    stability: Optional[Stability] = None
    verify_in: str = VERIFY_CAPABILITY


def moving_range_sigma(values: np.ndarray) -> tuple:
    """Short-term sigma from the average moving range.

    The moving range between consecutive observations captures only the
    variation that happens *between neighbours*, which is what makes it a
    short-term estimate: a slow drift across a month inflates the overall
    standard deviation and leaves this one alone. The gap between the two is
    itself the finding.
    """
    if values.size < 2:
        return None, None
    moving = np.abs(np.diff(values))
    mr_bar = float(moving.mean())
    return (mr_bar / D2_N2 if mr_bar > 0 else None), mr_bar


def assess_stability(values: Sequence[float]) -> Optional[Stability]:
    """An individuals and moving-range chart, with the standard signal rules.

    Four rules, which is the set that earns its place. Every rule added makes
    the chart more sensitive and also more likely to cry wolf: with all eight
    Western Electric rules running, a perfectly stable process signals roughly
    every 90 points by chance alone. These four are the ones whose patterns
    have a mechanism behind them.
    """
    x = np.asarray([v for v in values if v == v], dtype=float)
    if x.size < 8:
        return None
    centre = float(x.mean())
    sigma, mr_bar = moving_range_sigma(x)
    if not sigma:
        return None
    upper, lower = centre + 3 * sigma, centre - 3 * sigma

    signals = []
    beyond = [i for i, v in enumerate(x) if v > upper or v < lower]
    if beyond:
        signals.append(Signal(
            "1 point beyond 3 sigma",
            "A single point outside the control limits. The most direct "
            "evidence that something changed at that point.", beyond))

    side = np.sign(x - centre)
    run, runs = 1, []
    for i in range(1, x.size):
        if side[i] != 0 and side[i] == side[i - 1]:
            run += 1
            if run >= 9:
                runs.append(i)
        else:
            run = 1
    if runs:
        signals.append(Signal(
            "9 points in a row on one side of the centre",
            "The process has shifted. Nine in a row on one side happens by "
            "chance about once in 250 points.", runs))

    diffs = np.diff(x)
    trend_up = trend_down = 1
    trends = []
    for i, d in enumerate(diffs):
        trend_up = trend_up + 1 if d > 0 else 1
        trend_down = trend_down + 1 if d < 0 else 1
        if max(trend_up, trend_down) >= 6:
            trends.append(i + 1)
    if trends:
        signals.append(Signal(
            "6 points in a row increasing or decreasing",
            "A trend. Something is drifting rather than jumping - tool wear, "
            "fouling, a reagent ageing.", trends))

    alternating = []
    if x.size >= 14:
        for i in range(13, x.size):
            window = np.diff(x[i - 13:i + 1])
            if np.all(window[:-1] * window[1:] < 0):
                alternating.append(i)
    if alternating:
        signals.append(Signal(
            "14 points alternating up and down",
            "Over-adjustment, or two sources alternating - two heads, two "
            "analysts, two shifts.", alternating))

    chance = chance_of_a_signal(int(x.size))
    if signals:
        listed = " ".join("%s (%d occurrence%s)."
                          % (sig.rule, len(sig.points),
                             "" if len(sig.points) == 1 else "s")
                          for sig in signals)
        strength = ("Several rules fired, which is harder to explain by chance "
                    "than one." if len(signals) > 1 else
                    "One rule fired.")
        plain = ("**Signals on the control chart.** %s %s\n\nRead that against "
                 "how often it happens anyway: with %d observations and these "
                 "four rules, about **%.0f%% of perfectly stable processes** trip "
                 "at least one signal by chance — measured, not quoted. So a "
                 "single signal is weak evidence on its own. A signal you can "
                 "tie to something that happened — a lot change, a new analyst, "
                 "a maintenance date — is not, and neither is a cluster of them. "
                 "If the signals are real, capability computed from this data "
                 "describes the sample you happen to have rather than a process."
                 % (listed, strength, x.size, 100 * chance))
    else:
        plain = ("No signals over these %d observations: nothing outside the "
                 "control limits, and no runs, trends or alternation. Worth "
                 "knowing that with this many points a stable process would have "
                 "shown something about %.0f%% of the time, so a clean chart here "
                 "is meaningful." % (x.size, 100 * chance))

    return Stability(n=int(x.size), centre=centre, sigma_within=float(sigma),
                     upper=float(upper), lower=float(lower), mr_bar=float(mr_bar),
                     mr_upper=float(D4_N2 * mr_bar), signals=signals, plain=plain)


def assess_capability(values: Sequence[float], lsl: Optional[float] = None,
                      usl: Optional[float] = None,
                      target: Optional[float] = None,
                      name: str = "the measurement") -> Capability:
    """Capability indices, with the tail figures gated on normality."""
    x = np.asarray([v for v in values if v == v], dtype=float)
    n = x.size
    if n < 2 or (lsl is None and usl is None):
        return Capability(n=int(n), mean=float(x.mean()) if n else float("nan"),
                          sigma_within=None, sigma_overall=float("nan"),
                          lsl=lsl, usl=usl, target=target,
                          plain="Capability needs at least two measurements and "
                                "at least one specification limit.")

    mean = float(x.mean())
    sigma_overall = float(x.std(ddof=1))
    sigma_within, _ = moving_range_sigma(x)
    stability = assess_stability(x)
    normality = check_normality(x, name)

    indices, intervals = {}, {}
    if sigma_within and sigma_within > 0:
        indices.update(_indices(mean, sigma_within, lsl, usl, target, "C"))
    if sigma_overall > 0:
        indices.update(_indices(mean, sigma_overall, lsl, usl, target, "P"))
        # A confidence interval on Pp follows from the chi-square distribution
        # of the sample variance. Quoting Ppk to three decimals from 20 points
        # without one invites the reader to believe all three.
        if "Pp" in indices and n > 1:
            chi_low = stats.chi2.ppf(0.025, n - 1)
            chi_high = stats.chi2.ppf(0.975, n - 1)
            intervals["Pp"] = (indices["Pp"] * np.sqrt(chi_low / (n - 1)),
                               indices["Pp"] * np.sqrt(chi_high / (n - 1)))
        if "Ppk" in indices:
            ppk = indices["Ppk"]
            se = np.sqrt(1.0 / (9 * n) + ppk ** 2 / (2 * (n - 1)))
            z = stats.norm.ppf(0.975)
            intervals["Ppk"] = (ppk - z * se, ppk + z * se)

    below = int((x < lsl).sum()) if lsl is not None else 0
    above = int((x > usl).sum()) if usl is not None else 0
    observed_out = (below + above) / n

    withheld, predicted_ppm, z_bench = [], None, None
    if normality.holds and sigma_overall > 0:
        p_low = stats.norm.cdf(lsl, mean, sigma_overall) if lsl is not None else 0.0
        p_high = stats.norm.sf(usl, mean, sigma_overall) if usl is not None else 0.0
        p_out = float(p_low + p_high)
        predicted_ppm = 1e6 * p_out
        z_bench = float(stats.norm.isf(p_out)) if 0 < p_out < 1 else None
    else:
        withheld.append(
            "**Parts-per-million and Z.bench are not shown.** %s A PPM figure is "
            "a statement about the tails of the distribution, and the tails are "
            "exactly where the normal model fails first — a skewed distribution "
            "can show a comfortable Cpk and a true defect rate many times what "
            "the normal model predicts. JMP withholds Z.bench for non-normal "
            "data for the same reason. The indices above are ratios of distances "
            "and survive mild non-normality; the tail figures do not."
            % normality.plain
        )

    cautions = []
    if stability and not stability.stable:
        cautions.append(
            "The control chart shows %d signal type(s). If they are real rather "
            "than chance, these indices describe this sample rather than a "
            "process — see the stability section for how often a stable process "
            "signals anyway at this sample size."
            % len(stability.signals)
        )
    if n < 30:
        cautions.append(
            "%d measurements. Capability indices from fewer than about 30 are "
            "imprecise — see the confidence interval on Pp, which is usually "
            "wider than people expect." % n
        )
    if sigma_within and sigma_overall > 0 and sigma_overall > 1.5 * sigma_within:
        cautions.append(
            "The overall spread (%.4g) is much wider than the short-term spread "
            "(%.4g). The process drifts between observations: the Cp family "
            "describes what it could do, the Pp family what it does."
            % (sigma_overall, sigma_within)
        )

    headline = indices.get("Ppk", indices.get("Cpk"))
    if headline is None:
        plain = "Capability could not be computed from this data."
    elif headline >= MINIMUM_CAPABLE:
        plain = ("**Capable.** Ppk = %.2f against the usual minimum of %.2f. "
                 "%d of %d measurements fall outside the specification."
                 % (headline, MINIMUM_CAPABLE, below + above, n))
    else:
        plain = ("**Not capable.** Ppk = %.2f, below the usual minimum of %.2f. "
                 "%d of %d measurements fall outside the specification."
                 % (headline, MINIMUM_CAPABLE, below + above, n))

    result = Capability(
        n=int(n), mean=mean, sigma_within=sigma_within, sigma_overall=sigma_overall,
        lsl=lsl, usl=usl, target=target, indices=indices, intervals=intervals,
        observed_out=observed_out, predicted_ppm=predicted_ppm, z_bench=z_bench,
        withheld=withheld, plain=plain, cautions=cautions, stability=stability,
    )
    result.normality = normality
    return result


def _indices(mean: float, sigma: float, lsl, usl, target, prefix: str) -> dict:
    """The index family for one sigma estimate. C for within, P for overall."""
    out = {}
    upper_name = "Cpu" if prefix == "C" else "Ppu"
    lower_name = "Cpl" if prefix == "C" else "Ppl"
    both_name = "Cp" if prefix == "C" else "Pp"
    k_name = "Cpk" if prefix == "C" else "Ppk"

    if usl is not None:
        out[upper_name] = (usl - mean) / (3 * sigma)
    if lsl is not None:
        out[lower_name] = (mean - lsl) / (3 * sigma)
    if lsl is not None and usl is not None:
        out[both_name] = (usl - lsl) / (6 * sigma)
    sided = [v for k, v in out.items() if k in (upper_name, lower_name)]
    if sided:
        out[k_name] = min(sided)
    # Cpm penalises being off target as well as being spread out, which Cpk
    # does not. It is only defined when a target was given.
    if target is not None and lsl is not None and usl is not None and prefix == "C":
        tau = np.sqrt(sigma ** 2 + (mean - target) ** 2)
        out["Cpm"] = (usl - lsl) / (6 * tau)
    return out
