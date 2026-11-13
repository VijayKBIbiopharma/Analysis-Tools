"""Vector selection.

Most of these tests are about the refusals. A model that returns a number for
a protein unlike anything in the history is the failure mode that matters
here, because the number looks exactly like a good one - and the interval
beside it is *narrow*, since nothing in the residuals knows the question
changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis_tools.cld_vector import (CANDIDATE_CAP, assess_transferability,
                                       enumerate_candidates, level_coverage,
                                       plan_vector, read_catalogue,
                                       replicates_for)

PROMOTERS = ["CMV", "EF1a", "CAG", "SV40"]
MARKERS = ["Puromycin", "Hygromycin", "Blasticidin"]
HELPERS = ["pHelper-A", "pHelper-B", "pHelper-C"]
CLASSES = ["mAb", "bispecific", "Fc-fusion", "enzyme", "VHH"]
MW = {"mAb": 148.0, "bispecific": 104.0, "Fc-fusion": 126.0, "enzyme": 62.0,
      "VHH": 28.0}

CATALOGUE = {"Promoter": PROMOTERS, "Marker": MARKERS, "Helper": HELPERS}
PROTEIN = ["ProteinClass", "MW_kDa"]


def history(n=90, classes=CLASSES, seed=7, promoters=PROMOTERS,
            helpers=HELPERS, noise=0.25):
    """A plausible construct history: CAG is the best promoter, VHH expresses
    well, bispecifics badly, and everything else is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        protein = classes[i % len(classes)]
        promoter = promoters[rng.integers(len(promoters))]
        marker = MARKERS[rng.integers(len(MARKERS))]
        helper = helpers[rng.integers(len(helpers))]
        titre = (2.5 + 0.5 * (promoter == "CAG") + 0.2 * (marker == "Puromycin")
                 + 0.3 * (protein == "VHH") - 0.3 * (protein == "bispecific")
                 + rng.normal(0, noise))
        rows.append({"ProteinClass": protein, "MW_kDa": MW.get(protein, 100.0),
                     "Promoter": promoter, "Marker": marker, "Helper": helper,
                     "Titer": round(titre, 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# What the history can and cannot say

def test_one_protein_cannot_transfer():
    """Four hundred runs of one molecule are one protein, not four hundred."""
    data = history(80, classes=["mAb"])
    out = assess_transferability(data, PROTEIN)
    assert out.n_proteins == 1
    assert out.can_transfer is False
    assert "same protein" in out.why


def test_no_varying_protein_attribute_cannot_transfer():
    data = history(60)
    data["ProteinClass"] = "mAb"
    data["MW_kDa"] = 148.0
    out = assess_transferability(data, PROTEIN)
    assert out.can_transfer is False
    assert out.varying == []
    assert set(out.constant) == set(PROTEIN)


def test_two_proteins_is_a_line_through_two_points():
    data = history(60, classes=["mAb", "VHH"])
    out = assess_transferability(data, PROTEIN, programme={"ProteinClass": "mAb",
                                                          "MW_kDa": 148.0})
    assert out.n_proteins == 2
    assert out.can_transfer is True            # arithmetically possible
    assert out.per_programme[0]["standing"] == "thin"   # and worth very little


def test_familiar_protein_is_supported():
    out = assess_transferability(history(), PROTEIN,
                                 programme={"ProteinClass": "mAb", "MW_kDa": 148.0})
    assert out.n_proteins == 5
    assert out.per_programme[0]["standing"] == "supported"
    assert out.per_programme[0]["unseen"] == []


def test_unseen_protein_class_is_flagged():
    out = assess_transferability(history(), PROTEIN,
                                 programme={"ProteinClass": "nanobody-fusion",
                                            "MW_kDa": 41.0})
    detail = out.per_programme[0]
    assert detail["standing"] == "untested"
    assert any("nanobody-fusion" in m for m in detail["unseen"])


def test_molecular_weight_outside_the_historical_range_is_flagged():
    """310 kDa against a history topping out at 148."""
    out = assess_transferability(history(), PROTEIN,
                                 programme={"ProteinClass": "mAb", "MW_kDa": 310.0})
    detail = out.per_programme[0]
    assert detail["standing"] == "untested"
    assert any("MW_kDa" in m and "outside" in m for m in detail["unseen"])


def test_attribute_the_history_does_not_record_at_all():
    out = assess_transferability(history(), PROTEIN,
                                 programme={"ProteinClass": "mAb", "MW_kDa": 148.0,
                                            "GlycoProfile": "afucosylated"})
    assert any("GlycoProfile" in m for m in out.per_programme[0]["novel"])


def test_constant_attribute_gives_no_evidence_rather_than_a_number():
    data = history()
    data["Host"] = "CHO-K1"
    out = assess_transferability(data, PROTEIN + ["Host"],
                                 programme={"ProteinClass": "mAb", "MW_kDa": 148.0,
                                            "Host": "CHO-S"})
    assert any("Host" in m for m in out.per_programme[0]["no_evidence"])


# --------------------------------------------------------------------------
# Candidates

def test_full_enumeration_when_affordable():
    frame, total, sampled = enumerate_candidates(CATALOGUE)
    assert total == 4 * 3 * 3 == len(frame)
    assert sampled is False
    assert len(frame.drop_duplicates()) == len(frame)


def test_sampling_is_flagged_and_distinct():
    big = {"F%d" % i: list(range(6)) for i in range(8)}      # 1.68 million
    frame, total, sampled = enumerate_candidates(big, cap=500)
    assert total == 6 ** 8
    assert sampled is True
    assert len(frame) == 500
    assert len(frame.drop_duplicates()) == 500


def test_empty_catalogue_yields_nothing():
    frame, total, sampled = enumerate_candidates({})
    assert total == 0 and frame.empty


def test_cap_is_large_enough_for_a_real_catalogue():
    """Six factors at four levels each is 4,096 - a site catalogue should be
    enumerated in full, not sampled."""
    assert CANDIDATE_CAP > 4 ** 6


# --------------------------------------------------------------------------
# Coverage: a level nobody has built cannot be placed

def test_never_built_level_is_reported():
    data = history()
    rows = level_coverage(data, {"Helper": HELPERS + ["pHelper-D"]})
    assert rows[0]["never_built"] == ["pHelper-D"]


def test_thin_level_is_reported():
    data = history()
    data.loc[data.index[:2], "Promoter"] = "SV40"
    data = data[(data["Promoter"] != "SV40") | (data.index < 2)]
    rows = level_coverage(data, {"Promoter": PROMOTERS})
    thin = [t["level"] for row in rows for t in row["thin"]]
    assert "SV40" in thin


def test_fully_covered_factor_is_not_reported():
    assert level_coverage(history(), {"Promoter": PROMOTERS}) == []


# --------------------------------------------------------------------------
# Planning

def test_thin_history_is_refused():
    plan = plan_vector(history(8), CATALOGUE, "Titer", PROTEIN)
    assert plan.failed is not None
    assert "8 complete historical runs" in plan.failed
    assert plan.sheet is None


def test_catalogue_unrelated_to_the_history_is_refused():
    plan = plan_vector(history(), {"Enhancer": ["E1", "E2"]}, "Titer", PROTEIN)
    assert plan.failed is not None
    assert "catalogue factors" in plan.failed


def test_factor_with_no_built_level_is_refused_not_guessed():
    catalogue = dict(CATALOGUE, Helper=["pHelper-X", "pHelper-Y"])
    plan = plan_vector(history(), catalogue, "Titer", PROTEIN)
    assert plan.failed is not None
    assert "ever been built" in plan.failed


def test_never_built_level_is_excluded_and_said_so():
    catalogue = dict(CATALOGUE, Helper=HELPERS + ["pHelper-D"])
    plan = plan_vector(history(), catalogue, "Titer", PROTEIN,
                       programme={"ProteinClass": "mAb", "MW_kDa": 148.0})
    assert plan.failed is None
    assert "pHelper-D" not in set(plan.candidates["Helper"])
    assert any("pHelper-D" in c and "never been built" in c for c in plan.cautions)


def test_prediction_for_an_untested_protein_carries_the_warning():
    plan = plan_vector(history(), CATALOGUE, "Titer", PROTEIN,
                       programme={"ProteinClass": "nanobody-fusion", "MW_kDa": 41.0})
    assert plan.failed is None
    assert plan.best is not None                       # a number is still given
    assert any("outside what the history covers" in c for c in plan.cautions)
    assert any("does not widen" in c for c in plan.cautions)


def test_single_protein_history_warns_before_predicting():
    plan = plan_vector(history(60, classes=["mAb"]), CATALOGUE, "Titer", PROTEIN,
                       programme={"ProteinClass": "VHH", "MW_kDa": 28.0})
    assert plan.failed is None
    assert any("No prediction for a different protein can be defended"
               in c for c in plan.cautions)


# --------------------------------------------------------------------------
# The run sheet

@pytest.mark.parametrize("wells", [6, 12, 24, 48])
def test_sheet_fills_exactly_the_wells_offered(wells):
    plan = plan_vector(history(), CATALOGUE, "Titer", PROTEIN,
                       programme={"ProteinClass": "mAb", "MW_kDa": 148.0},
                       wells=wells)
    assert len(plan.sheet) == wells
    assert list(plan.sheet["Well"]) == list(range(1, wells + 1))


def test_sheet_concentrates_rather_than_spreading():
    """Running everything once was the worst of the five strategies measured,
    in both the poor-prediction and good-prediction columns."""
    plan = plan_vector(history(), CATALOGUE, "Titer", PROTEIN,
                       programme={"ProteinClass": "mAb", "MW_kDa": 148.0},
                       wells=24)
    factors = ["Promoter", "Marker", "Helper"]
    builds = plan.sheet[factors].astype(str).agg("|".join, axis=1)
    assert builds.nunique() <= 8
    assert builds.value_counts().min() >= 3


def test_sheet_is_randomised():
    """Position on a plate is a factor nobody controls, so the order has to be
    scrambled rather than running the best candidate in the first wells."""
    plan = plan_vector(history(), CATALOGUE, "Titer", PROTEIN,
                       programme={"ProteinClass": "mAb", "MW_kDa": 148.0},
                       wells=24)
    assert not plan.sheet["Predicted"].is_monotonic_decreasing


def test_replicates_rise_with_how_much_the_model_can_be_trusted():
    assert replicates_for(0.0) == 3
    assert replicates_for(None) == 3
    assert replicates_for(0.30) == 4          # r = 0.55
    assert replicates_for(0.80) == 6          # r = 0.89
    assert replicates_for(-5.0) == 3          # a negative cross-validated R2
    assert replicates_for(2.0) == 6           # clamped rather than crashing


# --------------------------------------------------------------------------
# The interval

def test_bounds_bracket_the_prediction():
    plan = plan_vector(history(), CATALOGUE, "Titer", PROTEIN,
                       programme={"ProteinClass": "mAb", "MW_kDa": 148.0})
    frame = plan.candidates
    assert (frame["Lower bound"] < frame["Predicted"]).all()
    assert (frame["Predicted"] < frame["Upper bound"]).all()
    assert plan.interval_width > 0


def test_safest_choice_maximises_the_lower_bound():
    plan = plan_vector(history(), CATALOGUE, "Titer", PROTEIN,
                       programme={"ProteinClass": "mAb", "MW_kDa": 148.0})
    assert plan.safest["Lower bound"] == plan.candidates["Lower bound"].max()
    assert plan.best["Predicted"] == plan.candidates["Predicted"].max()


def test_a_wider_interval_is_asked_for_and_given():
    narrow = plan_vector(history(), CATALOGUE, "Titer", PROTEIN, confidence=0.80)
    wide = plan_vector(history(), CATALOGUE, "Titer", PROTEIN, confidence=0.95)
    assert wide.interval_width > narrow.interval_width


def test_the_interval_covers_what_it_claims():
    """Measured, not asserted.

    Fit on 70 runs, predict the 30 the model never saw, and count how often the
    truth lands inside the 90% band. Over 60 held-out runs across two seeds:

        nominal   measured   half-width
          80%       86.7%      0.386 g/L
          90%       95.0%      0.533 g/L
          95%      100.0%      0.727 g/L

    Conservative in every row, which is the direction a tool should err in. An
    earlier split-conformal version was conservative to the point of
    uselessness - 100% coverage at a nominal 90% - because its errors came
    from a model fitted on three quarters of the history while the prediction
    came from one fitted on all of it.
    """
    data = history(100, seed=11)
    train, held = data.iloc[:70], data.iloc[70:]
    inside = []
    for _, row in held.iterrows():
        plan = plan_vector(train, CATALOGUE, "Titer", PROTEIN,
                           programme={"ProteinClass": row["ProteinClass"],
                                      "MW_kDa": row["MW_kDa"]},
                           confidence=0.90)
        match = plan.candidates
        for name in ("Promoter", "Marker", "Helper"):
            match = match[match[name].astype(str) == str(row[name])]
        if match.empty:
            continue
        low, high = match.iloc[0]["Lower bound"], match.iloc[0]["Upper bound"]
        inside.append(low <= row["Titer"] <= high)
    coverage = float(np.mean(inside))
    assert len(inside) >= 25
    assert 0.85 <= coverage <= 1.0, "90%% band covered %.1f%%" % (100 * coverage)


# --------------------------------------------------------------------------
# Housekeeping

def test_the_plan_is_reproducible():
    one = plan_vector(history(), CATALOGUE, "Titer", PROTEIN, seed=3)
    two = plan_vector(history(), CATALOGUE, "Titer", PROTEIN, seed=3)
    pd.testing.assert_frame_equal(one.candidates, two.candidates)
    pd.testing.assert_frame_equal(one.sheet, two.sheet)


def test_a_jmp_path_is_offered_for_independent_checking():
    plan = plan_vector(history(), CATALOGUE, "Titer", PROTEIN)
    assert "Fit Model" in plan.verify_in


def test_catalogue_sheet_accepts_either_separator_and_a_numeric_range():
    frame = pd.DataFrame({
        "Factor": ["Promoter", "Helper", "Copies"],
        "Levels": ["CMV; EF1a; CAG", "pH-A| pH-B", None],
        "Min": [None, None, 1], "Max": [None, None, 4], "Step": [None, None, 1]})
    out = read_catalogue(frame)
    assert out["Promoter"] == ["CMV", "EF1a", "CAG"]
    assert out["Helper"] == ["pH-A", "pH-B"]
    assert out["Copies"] == [1, 2, 3, 4]


def test_the_shipped_example_catalogue_reads():
    out = read_catalogue(pd.read_csv("examples/catalogue.csv"))
    assert out["Copies"] == [1, 2, 3, 4]
    assert "pHelper-D" in out["Helper"]
