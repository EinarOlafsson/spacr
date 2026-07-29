"""Invasion assay (Toxoplasma red/green stain): ``spacr.submodules.analyze_invasion``.

Before permeabilisation an antibody reaches only the parasites still *outside*
the host cell; the cells are then permeabilised and a second antibody stains
*all* parasites. So **attached** is a positive observation in the outside-stain
channel and **invaded** is the *absence* of one — and absence is the unreliable
direction. Every failure of the outside stain (poor penetration, a focal plane
off the parasite's equator, photobleaching, a dim strain) turns a parasite that
is genuinely outside into a parasite scored as invaded, and therefore inflates
invasion efficiency. Nothing plausible pushes the error the other way.

Every fixture below writes a ``measurements.db`` whose answer is known by
construction: the per-parasite outside-channel intensity is set literally, so
the attached/invaded split and the efficiency can be written down before the
code runs.

The single most important test in this file is
``test_a_weak_outside_stain_is_scored_invaded_and_inflates_efficiency`` — it
constructs the assay's characteristic failure, asserts the *direction* of the
error, and asserts that the QC says so instead of emitting a confident number.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# analyze_invasion imports spacr.io / spacr.plot / spacr.settings lazily inside
# the call; pull the heavy chain in at collection time so it is not charged to
# whichever test happens to run first.
import spacr.io  # noqa: E402,F401
import spacr.plot  # noqa: E402,F401
import spacr.settings  # noqa: E402,F401
import spacr.sp_stats  # noqa: E402,F401
import spacr.submodules  # noqa: E402,F401


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_blocking_show_and_clean_figs(monkeypatch):
    """Never let a figure window open, never let figures accumulate."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


PARASITE_AREA = 100.0
OUTSIDE_CHANNEL = 1
TOTAL_CHANNEL = 0


def write_db(root, fields, extra_cell_wells=(), statistic="percentile_95",
             total_intensity=200.0):
    """Write ``<root>/measurements/measurements.db`` from an explicit field spec.

    ``fields`` is a list of dicts::

        {'row': 'r1', 'column': 'c1', 'field': 'f1',
         'outside': [10.0, 10.0, 100.0], 'host': True}

    Each entry becomes one field holding ``len(outside)`` parasites whose
    outside-stain statistic is exactly the value given. ``host`` False (or a
    per-parasite list of bools) makes the parasite extracellular — ``cell_id``
    0, no host cell.

    ``extra_cell_wells`` is a list of ``(row, column)`` pairs that get host
    cells but no parasites at all: an uninfected well.
    """
    measurements = os.path.join(str(root), "measurements")
    os.makedirs(measurements, exist_ok=True)
    db = os.path.join(measurements, "measurements.db")

    pathogen_rows, cell_rows = [], []
    label = 0
    cells_seen = set()

    for spec in fields:
        row, column = spec["row"], spec["column"]
        field = spec.get("field", "f1")
        prcf = f"plate1_{row}_{column}_{field}"
        outside = list(spec["outside"])
        host = spec.get("host", True)
        if not isinstance(host, (list, tuple)):
            host = [host] * len(outside)

        for index, (value, has_host) in enumerate(zip(outside, host)):
            label += 1
            cell = (index + 1) if has_host else 0
            pathogen_rows.append({
                "object_label": label,
                "cell_id": cell,
                "plateID": "plate1", "rowID": row, "columnID": column,
                "fieldID": field, "prcf": prcf,
                "pathogen_area": PARASITE_AREA,
                f"pathogen_channel_{TOTAL_CHANNEL}_mean_intensity":
                    total_intensity,
                f"pathogen_channel_{OUTSIDE_CHANNEL}_{statistic}": float(value),
            })
            if has_host and (prcf, cell) not in cells_seen:
                cells_seen.add((prcf, cell))
                cell_rows.append({
                    "object_label": cell,
                    "plateID": "plate1", "rowID": row, "columnID": column,
                    "fieldID": field, "prcf": prcf,
                    "cell_area": 20000.0,
                })

    for i, (row, column) in enumerate(extra_cell_wells):
        cell_rows.append({
            "object_label": 900 + i,
            "plateID": "plate1", "rowID": row, "columnID": column,
            "fieldID": "f1", "prcf": f"plate1_{row}_{column}_f1",
            "cell_area": 20000.0,
        })

    with sqlite3.connect(db) as con:
        pd.DataFrame(pathogen_rows).to_sql("pathogen", con, index=False,
                                           if_exists="replace")
        if cell_rows:
            pd.DataFrame(cell_rows).to_sql("cell", con, index=False,
                                           if_exists="replace")
    return str(root)


def settings_for(src, **overrides):
    """Baseline settings: two conditions in columns c1/c2, nothing saved."""
    settings = {
        "src": src,
        "outside_channel": OUTSIDE_CHANNEL,
        "total_channel": TOTAL_CHANNEL,
        "cell_types": None,
        "cell_plate_metadata": None,
        "pathogen_types": ["dmso", "drug"],
        "pathogen_plate_metadata": [["c1"], ["c2"]],
        "treatments": None,
        "treatment_plate_metadata": None,
        "save": False,
        "verbose": False,
    }
    settings.update(overrides)
    return settings


def split(n_invaded, n_attached, low=10.0, high=100.0):
    """Return an outside-stain value list with a known invaded/attached split."""
    return [low] * n_invaded + [high] * n_attached


def well(out, prc):
    """Return the single well row for ``prc``."""
    rows = out["wells"][out["wells"]["prc"] == prc]
    assert len(rows) == 1, f"{prc}: {len(rows)} rows"
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Choosing the statistic: an outside stain is a rim stain
# ---------------------------------------------------------------------------

def _frame(**columns):
    return pd.DataFrame({key: [float(v)] for key, v in columns.items()})


def test_auto_statistic_prefers_the_boundary_measurement_over_the_object_mean():
    """The outside antibody coats the parasite surface, so the signal lives on
    the boundary. measure.py's periphery statistic reads only the boundary
    ring; the object mean divides that rim by the whole area, so a bigger
    parasite reads dimmer than a smaller one stained identically — a
    size-dependent bias in the direction that manufactures invaded calls."""
    from spacr.submodules import _resolve_invasion_intensity_column

    everything = _frame(**{
        "pathogen_channel_1_periphery_95_percentile": 90.0,
        "pathogen_channel_1_percentile_95": 80.0,
        "pathogen_channel_1_mean_intensity": 20.0,
    })
    column, name = _resolve_invasion_intensity_column(everything, "pathogen", 1)
    assert (column, name) == ("pathogen_channel_1_periphery_95_percentile",
                              "periphery_95")

    # No periphery ring measured: the 95th percentile of the object's own
    # pixels still samples the rim, and averages enough of them to be stable.
    no_periphery = _frame(**{
        "pathogen_channel_1_percentile_95": 80.0,
        "pathogen_channel_1_mean_intensity": 20.0,
    })
    assert _resolve_invasion_intensity_column(no_periphery, "pathogen", 1) == (
        "pathogen_channel_1_percentile_95", "percentile_95")


def test_falling_back_to_the_object_mean_warns_about_rim_dilution(capsys):
    from spacr.submodules import _resolve_invasion_intensity_column

    only_mean = _frame(**{"pathogen_channel_1_mean_intensity": 20.0})
    column, name = _resolve_invasion_intensity_column(only_mean, "pathogen", 1)
    assert (column, name) == ("pathogen_channel_1_mean_intensity", "mean")
    printed = capsys.readouterr().out
    assert "rim stain" in printed and "diluted" in printed


def test_an_all_nan_statistic_column_is_skipped_like_a_missing_one():
    from spacr.submodules import _resolve_invasion_intensity_column

    frame = pd.DataFrame({
        "pathogen_channel_1_periphery_95_percentile": [np.nan, np.nan],
        "pathogen_channel_1_percentile_95": [80.0, 81.0],
    })
    assert _resolve_invasion_intensity_column(frame, "pathogen", 1)[1] == \
        "percentile_95"


def test_no_outside_channel_statistic_at_all_names_what_was_tried():
    from spacr.submodules import _resolve_invasion_intensity_column

    with pytest.raises(KeyError, match="outside_channel"):
        _resolve_invasion_intensity_column(_frame(x=1.0), "pathogen", 1)


def test_named_statistic_resolves_and_a_missing_one_names_its_column():
    from spacr.submodules import _resolve_invasion_intensity_column

    frame = _frame(**{"pathogen_channel_2_max_intensity": 5.0, "custom": 1.0})
    assert _resolve_invasion_intensity_column(frame, "pathogen", 2, "max") == (
        "pathogen_channel_2_max_intensity", "max")
    # A literal column name is honoured verbatim.
    assert _resolve_invasion_intensity_column(frame, "pathogen", 2, "custom") == (
        "custom", "custom")
    with pytest.raises(KeyError, match="pathogen_channel_2_median_intensity"):
        _resolve_invasion_intensity_column(frame, "pathogen", 2, "median")
    with pytest.raises(KeyError, match="nonsense"):
        _resolve_invasion_intensity_column(frame, "pathogen", 2, "nonsense")


def test_local_background_column_is_the_ring_outside_the_object_not_the_stain():
    """measure.py's ``outside_*`` columns are the intensity of a five-pixel
    ring outside the object's own mask — a local background estimate. They are
    not the outside *stain*, which is a whole channel."""
    from spacr.submodules import _resolve_invasion_background_column

    frame = _frame(**{
        "pathogen_channel_1_outside_50_percentile": 3.0,
        "pathogen_channel_1_outside_mean": 4.0,
        "mine": 1.0,
    })
    assert _resolve_invasion_background_column(frame, "pathogen", 1, "auto") == \
        "pathogen_channel_1_outside_50_percentile"
    assert _resolve_invasion_background_column(frame, "pathogen", 1, "mine") == \
        "mine"
    for off in (None, False, "none", ""):
        assert _resolve_invasion_background_column(frame, "pathogen", 1, off) is None
    with pytest.raises(KeyError, match="ghost"):
        _resolve_invasion_background_column(frame, "pathogen", 1, "ghost")


def test_auto_background_falls_back_to_the_mean_ring_then_gives_up(capsys):
    from spacr.submodules import _resolve_invasion_background_column

    mean_only = _frame(**{"pathogen_channel_1_outside_mean": 4.0})
    assert _resolve_invasion_background_column(mean_only, "pathogen", 1, "auto") \
        == "pathogen_channel_1_outside_mean"

    assert _resolve_invasion_background_column(_frame(x=1.0), "pathogen", 1,
                                               "auto") is None
    assert "raw intensities" in capsys.readouterr().out


def test_auto_background_prefers_the_canonical_percentile_name():
    from spacr.submodules import _resolve_invasion_background_column

    frame = _frame(**{
        "pathogen_channel_1_outside_percentile_50": 3.0,
        "pathogen_channel_1_outside_50_percentile": 4.0,
    })
    assert _resolve_invasion_background_column(frame, "pathogen", 1, "auto") \
        == "pathogen_channel_1_outside_percentile_50"


def test_background_subtraction_shifts_every_object(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame["pathogen_channel_1_outside_50_percentile"] = 4.0
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    out = analyze_invasion(settings_for(src, background_correction="auto"))
    parasites = out["parasites"]
    assert set(parasites["outside_background"]) == {4.0}
    assert sorted(set(parasites["outside_intensity"])) == [6.0, 96.0]
    # Subtracting a constant moves the threshold with the data, so the split
    # is unchanged.
    assert well(out, "plate1_r1_c1")["invasion_efficiency"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Bimodality: is there anything here to threshold?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_low,n_high", [(50, 50), (90, 10), (10, 90)])
def test_a_two_population_mixture_scores_one_whatever_the_mixing_ratio(n_low,
                                                                      n_high):
    from spacr.submodules import _bimodality_coefficient

    values = np.array([10.0] * n_low + [100.0] * n_high)
    assert _bimodality_coefficient(values) == pytest.approx(1.0)


def test_a_single_population_scores_about_a_third_and_misses_the_cutoff():
    from spacr.submodules import _bimodality_coefficient

    values = np.random.default_rng(0).normal(50.0, 5.0, 200)
    coefficient = _bimodality_coefficient(values)
    assert 0.25 < coefficient < 0.5
    assert coefficient < 5.0 / 9.0


def test_bimodality_refuses_a_sample_too_small_to_support_it():
    """The uncorrected coefficient exceeds the 5/9 cutoff on a genuinely
    unimodal sample about 45% of the time at n=10, so below min_objects it
    returns NaN — 'cannot tell' rather than a confident wrong answer."""
    from spacr.submodules import _bimodality_coefficient

    values = np.array([10.0] * 5 + [100.0] * 5)
    assert np.isnan(_bimodality_coefficient(values, min_objects=30))
    assert _bimodality_coefficient(values, min_objects=4) == pytest.approx(1.0)
    assert np.isnan(_bimodality_coefficient(np.array([1.0, 2.0])))


def test_one_value_repeated_is_one_population():
    from spacr.submodules import _bimodality_coefficient

    assert _bimodality_coefficient(np.full(60, 7.0)) == 0.0


# ---------------------------------------------------------------------------
# Threshold placement
# ---------------------------------------------------------------------------

def test_the_threshold_is_recentred_in_the_gap_without_moving_the_split():
    """skimage returns a histogram bin centre, which lands on the upper edge of
    the dim population. Recentring puts it in the empty space between the two
    populations — the same split, the widest margin."""
    from spacr.submodules import _invasion_centre_threshold, _invasion_threshold

    values = np.array([10.0] * 20 + [100.0] * 20)
    assert _invasion_threshold(values, "otsu") == pytest.approx(55.0)

    # Recentring never changes which side an object falls on.
    raw = 10.5
    centred = _invasion_centre_threshold(values, raw)
    assert np.array_equal(values > raw, values > centred)

    # Nothing on one side: leave the threshold where it is.
    assert _invasion_centre_threshold(values, 1000.0) == 1000.0
    assert _invasion_centre_threshold(values, -1.0) == -1.0
    assert _invasion_centre_threshold(np.array([]), 5.0) == 5.0
    assert np.isnan(_invasion_centre_threshold(values, float("nan")))


@pytest.mark.parametrize("method", ["otsu", "triangle", "li", "yen", "mean"])
def test_every_threshold_method_separates_a_clean_two_population_field(method):
    from spacr.submodules import _invasion_threshold

    values = np.array([10.0] * 30 + [100.0] * 30)
    threshold = _invasion_threshold(values, method)
    assert 10.0 < threshold < 100.0


def test_a_constant_field_supports_no_threshold_and_says_so():
    from spacr.submodules import _invasion_threshold

    assert np.isnan(_invasion_threshold(np.full(40, 7.0)))
    assert np.isnan(_invasion_threshold(np.array([])))
    assert np.isnan(_invasion_threshold(np.array([np.nan, np.nan])))
    assert np.isnan(_invasion_threshold(np.array([]), "mean"))


def test_an_unknown_threshold_method_lists_the_ones_that_exist():
    from spacr.submodules import _invasion_threshold

    with pytest.raises(ValueError, match="otsu"):
        _invasion_threshold(np.array([1.0, 2.0]), "kittler")


def test_relative_difference_is_symmetric_and_survives_a_zero_threshold():
    from spacr.submodules import _invasion_relative_difference

    assert _invasion_relative_difference(60.0, 6.0) == pytest.approx(0.9)
    assert _invasion_relative_difference(6.0, 60.0) == pytest.approx(0.9)
    assert _invasion_relative_difference(0.0, 0.0) == 0.0
    assert np.isnan(_invasion_relative_difference(np.nan, 1.0))
    assert np.isnan(_invasion_relative_difference(1.0, np.nan))


def test_threshold_span_uses_the_data_scale_when_the_threshold_is_zero():
    from spacr.submodules import _invasion_threshold_span

    low, high = _invasion_threshold_span(40.0, [1.0, 2.0], 0.25)
    assert (low, high) == pytest.approx((30.0, 50.0))

    values = np.array([-10.0, 10.0])
    low, high = _invasion_threshold_span(0.0, values, 0.5)
    assert (low, high) == pytest.approx((-5.0, 5.0))

    assert all(np.isnan(v) for v in _invasion_threshold_span(np.nan, [1.0], 0.25))
    assert _invasion_threshold_span(0.0, [], 0.25) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# The core readout
# ---------------------------------------------------------------------------

def test_a_clean_bimodal_field_gives_the_exact_split_and_efficiency(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(36, 24)}])
    out = analyze_invasion(settings_for(src))

    row = well(out, "plate1_r1_c1")
    assert (row["n_invaded"], row["n_attached"], row["n_total"]) == (36, 24, 60)
    assert row["invasion_efficiency"] == pytest.approx(36 / 60)
    assert row["threshold_median"] == pytest.approx(55.0)
    assert row["threshold_source"] == "field"
    assert row["bimodality_coefficient"] == pytest.approx(1.0)
    assert row["qc_flags"] == "" and row["qc_pass"]

    parasites = out["parasites"]
    calls = parasites.groupby(parasites["outside_intensity"])["invasion_class"]
    assert set(calls.get_group(10.0)) == {"invaded"}
    assert set(calls.get_group(100.0)) == {"attached"}


def test_efficiency_is_reported_with_its_denominator_and_the_class_counts(tmp_path):
    """A proportion without a denominator is not a result: 90% of ten and 90%
    of four thousand are the same ratio and different evidence."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(9, 1)},
        {"row": "r2", "column": "c1", "outside": split(3600, 400)},
    ])
    out = analyze_invasion(settings_for(src))

    small = well(out, "plate1_r1_c1")
    large = well(out, "plate1_r2_c1")
    assert small["invasion_efficiency"] == pytest.approx(0.9)
    assert large["invasion_efficiency"] == pytest.approx(0.9)
    assert small["n_total"] == 10 and large["n_total"] == 4000

    # Same number, wildly different trust: only the small one is flagged.
    assert small["qc_flag_low_total"]
    assert "low_total" in small["qc_flags"]
    assert not large["qc_flag_low_total"]


# ---------------------------------------------------------------------------
# THE asymmetry: absence of the outside stain is the unreliable direction
# ---------------------------------------------------------------------------

def _weak_stain_src(tmp_path):
    """One field: 30 truly invaded, 29 brightly attached, 1 weakly attached.

    Ground truth is 30 attached and 30 invaded, so the honest efficiency is
    exactly 0.5. The weakly stained parasite at 40 is genuinely *outside*; it
    is the one every failure mode of this assay loses.
    """
    control = list(np.linspace(4.0, 6.0, 60))
    return write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1",
         "outside": [5.0] * 30 + [40.0] + [100.0] * 29},
        {"row": "r1", "column": "c12", "outside": control},
    ])


def test_a_weak_outside_stain_is_scored_invaded_and_inflates_efficiency(tmp_path):
    """The assay's characteristic failure, in the direction it always fails.

    A parasite that is genuinely outside but stained weakly falls below the
    threshold and is scored invaded, so invasion efficiency goes *up*. Three
    runs on the same 60 parasites, whose truth is 30 attached / 30 invaded:

    * control wells put the threshold just above the negative distribution and
      recover the truth;
    * the automatic threshold, with no controls to anchor it, isolates the
      bright cluster and loses the weak parasite — the error is upward;
    * a threshold deliberately set too high loses it too, and the QC says the
      threshold disagrees with the control-derived one rather than reporting a
      confident number.
    """
    from spacr.submodules import analyze_invasion

    src = _weak_stain_src(tmp_path)
    base = dict(pathogen_types=["dmso"], pathogen_plate_metadata=[["c1"]])

    truth = analyze_invasion(settings_for(src, control_wells=["c12"], **base))
    truthful = well(truth, "plate1_r1_c1")
    assert truthful["n_attached"] == 30 and truthful["n_invaded"] == 30
    assert truthful["invasion_efficiency"] == pytest.approx(0.5)
    assert truthful["threshold_source"] == "control"
    assert not truthful["qc_flag_threshold_disagrees"]

    # The weakly stained parasite is correctly on the attached side.
    weak = truth["parasites"]
    weak = weak[weak["outside_intensity"] == 40.0].iloc[0]
    assert weak["invasion_class"] == "attached"

    # Without controls the automatic cut shaves the weak parasite off the
    # bright cluster, and the error runs upward — never downward.
    automatic = analyze_invasion(settings_for(src, control_wells=None, **base))
    auto_row = well(automatic, "plate1_r1_c1")
    assert auto_row["invasion_efficiency"] > truthful["invasion_efficiency"]
    assert auto_row["invasion_efficiency"] == pytest.approx(31 / 60)
    lost = automatic["parasites"]
    lost = lost[lost["outside_intensity"] == 40.0].iloc[0]
    assert lost["invasion_class"] == "invaded"

    # A threshold set too high does the same, and now the QC has a reference
    # to measure it against and flags the disagreement.
    too_high = analyze_invasion(settings_for(
        src, control_wells=["c12"], outside_threshold=60.0, **base))
    high_row = well(too_high, "plate1_r1_c1")
    assert high_row["invasion_efficiency"] > truthful["invasion_efficiency"]
    assert high_row["threshold_source"] == "fixed"
    assert high_row["qc_flag_threshold_disagrees"]
    assert "threshold_disagrees" in high_row["qc_flags"]
    assert not high_row["qc_pass"]


def test_efficiency_is_monotone_in_the_threshold(tmp_path):
    """Raising the outside-channel threshold can only turn attached into
    invaded, so the sensitivity bracket is ordered by construction and the
    upward move is the only dangerous one."""
    from spacr.submodules import analyze_invasion

    src = _weak_stain_src(tmp_path)
    efficiencies = []
    for threshold in (20.0, 60.0, 150.0):
        out = analyze_invasion(settings_for(
            src, outside_threshold=threshold, pathogen_types=["dmso"],
            pathogen_plate_metadata=[["c1"]], control_wells=["c12"]))
        efficiencies.append(well(out, "plate1_r1_c1")["invasion_efficiency"])
    assert efficiencies == sorted(efficiencies)
    assert efficiencies[0] == pytest.approx(0.5)
    # Above every parasite's signal, nothing is attached any more.
    assert efficiencies[-1] == pytest.approx(1.0)


def test_the_sensitivity_bracket_reports_only_the_dangerous_direction(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    out = analyze_invasion(settings_for(src))
    row = well(out, "plate1_r1_c1")

    assert (row["invasion_efficiency_low_threshold"]
            <= row["invasion_efficiency"]
            <= row["invasion_efficiency_high_threshold"])
    # A threshold sitting in a real gap cannot be nudged into the data.
    assert row["invasion_efficiency_inflation"] == pytest.approx(0.0)
    assert not row["qc_flag_threshold_inflates"]


def test_a_threshold_sitting_inside_the_data_is_flagged_as_inflating(tmp_path):
    from spacr.submodules import analyze_invasion

    values = list(np.random.default_rng(0).normal(50.0, 5.0, 200))
    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": values}])
    out = analyze_invasion(settings_for(src))
    row = well(out, "plate1_r1_c1")

    assert row["invasion_efficiency_inflation"] > 0.05
    assert row["qc_flag_threshold_inflates"]
    assert "threshold_inflates" in row["qc_flags"]


# ---------------------------------------------------------------------------
# Unimodal: no two populations means no classification
# ---------------------------------------------------------------------------

def test_a_unimodal_outside_distribution_is_flagged_not_silently_split(tmp_path):
    """Otsu will happily split one smear of signal down the middle and return a
    confident number. The bimodality coefficient is what says it has done so."""
    from spacr.submodules import analyze_invasion

    unimodal = list(np.random.default_rng(0).normal(50.0, 5.0, 200))
    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": unimodal},
        {"row": "r2", "column": "c1", "outside": split(100, 100)},
    ])
    out = analyze_invasion(settings_for(src))

    smear = well(out, "plate1_r1_c1")
    clean = well(out, "plate1_r2_c1")

    # A number still comes out — but it comes out flagged.
    assert np.isfinite(smear["invasion_efficiency"])
    assert smear["bimodality_coefficient"] < 5.0 / 9.0
    assert smear["qc_flag_unimodal"]
    assert "unimodal" in smear["qc_flags"]

    assert clean["bimodality_coefficient"] == pytest.approx(1.0)
    assert not clean["qc_flag_unimodal"]

    # And the flag is raised per field as well as per well.
    fields = out["fields"].set_index("prcf")
    assert fields.loc["plate1_r1_c1_f1", "qc_flag_unimodal"]
    assert not fields.loc["plate1_r2_c1_f1", "qc_flag_unimodal"]


def test_a_well_too_small_to_test_for_two_populations_is_flagged_too(tmp_path):
    """Below min_objects_for_bimodality the coefficient is NaN, which means
    'cannot demonstrate two populations' — and the assay's claim needs them
    demonstrated, so it flags."""
    from spacr.submodules import analyze_invasion

    # Twelve a side: enough objects for a threshold (min_objects_for_threshold
    # is 10), too few to demonstrate two populations (min_objects_for_
    # bimodality is 30).
    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(12, 12)}])
    out = analyze_invasion(settings_for(src))
    row = well(out, "plate1_r1_c1")

    assert row["n_total"] == 24
    assert row["invasion_efficiency"] == pytest.approx(0.5)
    assert np.isnan(row["bimodality_coefficient"])
    assert row["qc_flag_unimodal"]


# ---------------------------------------------------------------------------
# Per-field thresholding
# ---------------------------------------------------------------------------

def test_thresholds_are_per_field_so_illumination_is_not_read_as_invasion(tmp_path):
    """Two fields of one well, the second imaged twenty times brighter. Each
    holds fifteen attached and fifteen invaded parasites, so the well's honest
    efficiency is 0.5. A single plate-wide cut lands between the two *fields*
    rather than between the two *populations*, and reports the dim field as
    entirely invaded — an illumination gradient read out as an invasion
    gradient."""
    from spacr.submodules import _invasion_threshold, analyze_invasion

    dim = split(15, 15, low=10.0, high=100.0)
    bright = split(15, 15, low=200.0, high=2000.0)
    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "field": "f1", "outside": dim},
        {"row": "r1", "column": "c1", "field": "f2", "outside": bright},
    ])

    out = analyze_invasion(settings_for(src))
    row = well(out, "plate1_r1_c1")
    assert row["n_attached"] == 30 and row["n_invaded"] == 30
    assert row["invasion_efficiency"] == pytest.approx(0.5)

    fields = out["fields"].set_index("prcf")
    assert fields.loc["plate1_r1_c1_f1", "threshold"] == pytest.approx(55.0)
    assert fields.loc["plate1_r1_c1_f2", "threshold"] == pytest.approx(1100.0)
    assert fields.loc["plate1_r1_c1_f1", "invasion_efficiency"] == pytest.approx(0.5)
    assert fields.loc["plate1_r1_c1_f2", "invasion_efficiency"] == pytest.approx(0.5)

    # The single global threshold this data would produce, applied globally.
    global_threshold = _invasion_threshold(np.array(dim + bright), "otsu")
    assert global_threshold == pytest.approx(1100.0)
    pooled = analyze_invasion(settings_for(
        src, outside_threshold=float(global_threshold)))
    pooled_row = well(pooled, "plate1_r1_c1")
    assert pooled_row["invasion_efficiency"] == pytest.approx(45 / 60)
    pooled_fields = pooled["fields"].set_index("prcf")
    # The dim field is wiped out entirely; the bright one is untouched.
    assert pooled_fields.loc["plate1_r1_c1_f1", "invasion_efficiency"] == 1.0
    assert pooled_fields.loc["plate1_r1_c1_f2", "invasion_efficiency"] == \
        pytest.approx(0.5)


def test_a_field_too_small_to_threshold_borrows_the_well_then_the_plate(tmp_path):
    """Otsu on four parasites is not a threshold. The fallback is announced in
    ``automatic_source`` rather than being silent."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "field": "f1", "outside": split(20, 20)},
        {"row": "r1", "column": "c1", "field": "f2", "outside": [10.0, 100.0]},
    ])
    out = analyze_invasion(settings_for(src))
    fields = out["fields"].set_index("prcf")

    assert fields.loc["plate1_r1_c1_f1", "automatic_source"] == "field"
    assert fields.loc["plate1_r1_c1_f2", "automatic_source"] == "well"
    assert fields.loc["plate1_r1_c1_f2", "threshold"] == pytest.approx(55.0)
    assert fields.loc["plate1_r1_c1_f2", "invasion_efficiency"] == pytest.approx(0.5)


def test_a_plate_with_no_variation_anywhere_leaves_the_objects_unclassified(tmp_path):
    """No threshold exists, so none is invented: the parasites are reported
    unclassified and the well's efficiency is NaN rather than a coin flip."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": [42.0] * 40}])
    out = analyze_invasion(settings_for(src))

    fields = out["fields"].iloc[0]
    assert fields["threshold_source"] == "none"
    assert fields["qc_flag_no_threshold"]

    row = well(out, "plate1_r1_c1")
    assert row["n_unclassified"] == 40 and row["n_total"] == 0
    assert np.isnan(row["invasion_efficiency"])
    assert set(out["parasites"]["invasion_class"].astype(str)) == {"unclassified"}


# ---------------------------------------------------------------------------
# Control wells
# ---------------------------------------------------------------------------

def test_control_wells_override_the_automatic_threshold_and_say_so(tmp_path, capsys):
    """Wells whose parasites carry no outside stain give the honest negative
    distribution, which beats any automatic method. The report names the
    source so a control run cannot be mistaken for an automatic one."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)},
        {"row": "r1", "column": "c12", "outside": list(np.linspace(28.0, 32.0, 60))},
    ])
    base = dict(pathogen_types=["dmso"], pathogen_plate_metadata=[["c1"]])

    controlled = analyze_invasion(settings_for(src, control_wells=["c12"], **base))
    printed = capsys.readouterr().out
    assert "threshold taken from the control wells" in printed

    row = well(controlled, "plate1_r1_c1")
    assert row["threshold_source"] == "control"
    assert row["threshold_median"] == pytest.approx(
        np.quantile(np.linspace(28.0, 32.0, 60), 0.99))
    assert controlled["control_thresholds"]["plate1"] == pytest.approx(
        row["threshold_median"])
    # The control well is a staining control, not a condition: it leaves the
    # results and lands in its own table.
    assert "plate1_r1_c12" not in set(controlled["wells"]["prc"])
    assert len(controlled["controls"]) == 60

    automatic = analyze_invasion(settings_for(src, control_wells=None, **base))
    auto_row = well(automatic, "plate1_r1_c1")
    assert auto_row["threshold_source"] == "field"
    assert auto_row["threshold_median"] == pytest.approx(55.0)
    assert automatic["control_thresholds"] == {}


@pytest.mark.parametrize("spec", ["c12", "r1_c12", "plate1_r1_c12"])
def test_control_wells_accept_the_house_well_vocabulary(tmp_path, spec):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)},
        {"row": "r1", "column": "c12", "outside": [5.0] * 30},
    ])
    out = analyze_invasion(settings_for(
        src, control_wells=[spec], pathogen_types=["dmso"],
        pathogen_plate_metadata=[["c1"]]))
    assert len(out["controls"]) == 30
    assert well(out, "plate1_r1_c1")["threshold_source"] == "control"


def test_too_few_control_objects_falls_back_and_warns(tmp_path, capsys):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)},
        {"row": "r1", "column": "c12", "outside": [5.0, 5.0, 6.0]},
    ])
    out = analyze_invasion(settings_for(
        src, control_wells=["c12"], pathogen_types=["dmso"],
        pathogen_plate_metadata=[["c1"]]))

    assert "min_control_objects" in capsys.readouterr().out
    assert out["control_thresholds"] == {}
    assert well(out, "plate1_r1_c1")["threshold_source"] == "field"


def test_a_fixed_threshold_beats_the_controls_but_they_become_its_reference(
        tmp_path, capsys):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)},
        {"row": "r1", "column": "c12", "outside": [5.0] * 30},
    ])
    out = analyze_invasion(settings_for(
        src, control_wells=["c12"], outside_threshold=55.0,
        pathogen_types=["dmso"], pathogen_plate_metadata=[["c1"]]))

    assert "'outside_threshold' is set" in capsys.readouterr().out
    row = well(out, "plate1_r1_c1")
    assert row["threshold_source"] == "fixed"
    assert row["threshold_median"] == pytest.approx(55.0)
    assert row["reference_threshold_median"] == pytest.approx(5.0)


def test_control_wells_swallowing_the_whole_plate_is_an_error(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5)}])
    with pytest.raises(ValueError, match="control_wells"):
        analyze_invasion(settings_for(src, control_wells=["c1"]))


# ---------------------------------------------------------------------------
# Statistics: the well is the unit of replication
# ---------------------------------------------------------------------------

def _per_well_efficiency_src(tmp_path, left, right, per_well=1000):
    """Six wells of ``per_well`` parasites with exactly the given efficiencies."""
    fields = []
    for column, efficiencies in (("c1", left), ("c2", right)):
        for index, efficiency in enumerate(efficiencies):
            invaded = int(round(per_well * efficiency))
            fields.append({"row": f"r{index + 1}", "column": column,
                           "outside": split(invaded, per_well - invaded)})
    return write_db(tmp_path / "p", fields)


def test_the_test_uses_the_well_not_the_parasite_as_the_unit(tmp_path):
    """Three wells a side, a six-point efficiency difference and a thousand
    parasites per well. Pooling the parasites calls it overwhelming; the
    per-well test — the honest one, because parasites in a well share a
    coverslip, an antibody bath and a focal plane — cannot reach 0.05 with
    three replicates, and says so."""
    from spacr.submodules import analyze_invasion

    src = _per_well_efficiency_src(tmp_path, [0.50, 0.52, 0.48],
                                   [0.56, 0.58, 0.54])
    out = analyze_invasion(settings_for(src))
    comparison = out["comparisons"].iloc[0]

    assert comparison["unit_of_replication"] == "well"
    assert comparison["test"].startswith("Mann-Whitney U on per-well")
    assert comparison["n_wells_1"] == 3 and comparison["n_wells_2"] == 3
    assert comparison["n_parasites_1"] == 3000

    # Pooling every parasite as an independent observation: p ~ 4e-6.
    assert comparison["pooled_chi_squared_p_value"] < 1e-4
    # The well as the unit: three against three cannot beat 0.05.
    assert comparison["p_value"] == pytest.approx(0.1)
    assert comparison["p_value"] > 0.05

    assert comparison["mean_efficiency_1"] == pytest.approx(0.50)
    assert comparison["mean_efficiency_2"] == pytest.approx(0.56)
    assert comparison["efficiency_difference"] == pytest.approx(-0.06)
    assert comparison["rank_biserial"] == pytest.approx(-1.0)


def test_the_summary_separates_the_per_well_mean_from_the_pooled_proportion(tmp_path):
    """Unequal wells make the two disagree; the pooled number is the one a
    chi-squared on raw counts is implicitly about."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(90, 10)},
        {"row": "r2", "column": "c1", "outside": split(100, 900)},
    ])
    out = analyze_invasion(settings_for(src))
    summary = out["summary"].set_index("condition").loc["dmso"]

    assert summary["n_wells"] == 2
    assert summary["invasion_efficiency"] == pytest.approx((0.9 + 0.1) / 2)
    assert summary["invasion_efficiency_pooled"] == pytest.approx(190 / 1100)
    assert summary["invasion_efficiency_median"] == pytest.approx(0.5)
    assert summary["invasion_efficiency_sd"] == pytest.approx(
        np.std([0.9, 0.1], ddof=1))
    assert summary["n_total"] == 1100


def test_one_condition_returns_an_empty_comparison_table_with_its_columns(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    out = analyze_invasion(settings_for(
        src, pathogen_types=["dmso"], pathogen_plate_metadata=[["c1"]]))

    comparisons = out["comparisons"]
    assert len(comparisons) == 0
    for column in ("group1", "p_value", "unit_of_replication",
                   "pooled_chi_squared_p_value", "p_value_adj"):
        assert column in comparisons.columns


def test_a_single_well_per_condition_cannot_be_tested_and_returns_nan(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)},
        {"row": "r1", "column": "c2", "outside": split(50, 10)},
    ])
    out = analyze_invasion(settings_for(src))
    comparison = out["comparisons"].iloc[0]

    assert np.isnan(comparison["p_value"])
    assert comparison["n_wells_1"] == 1 and comparison["n_wells_2"] == 1
    # The pooled table is still computable, and still the wrong unit.
    assert np.isfinite(comparison["pooled_chi_squared_p_value"])
    assert comparison["p_value_adj"] is None or np.isnan(comparison["p_value_adj"])


def test_a_condition_with_nothing_to_pool_gets_nan_rather_than_a_crash(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(
        tmp_path / "p",
        [{"row": "r1", "column": "c1", "outside": split(30, 30)}],
        extra_cell_wells=[("r1", "c2")])
    out = analyze_invasion(settings_for(src))
    comparison = out["comparisons"].iloc[0]

    assert comparison["n_parasites_2"] == 0
    assert np.isnan(comparison["pooled_chi_squared_p_value"])
    assert np.isnan(comparison["p_value"])


# ---------------------------------------------------------------------------
# Degenerate wells
# ---------------------------------------------------------------------------

def test_a_well_with_no_parasites_reports_a_zero_denominator_and_nan(tmp_path):
    """NaN, not 0.0: the well has not observed zero invasion, it has observed
    nothing, and 0.0 would be averaged into every downstream number."""
    from spacr.submodules import analyze_invasion

    src = write_db(
        tmp_path / "p",
        [{"row": "r1", "column": "c1", "outside": split(30, 30)}],
        extra_cell_wells=[("r2", "c1")])
    out = analyze_invasion(settings_for(src))

    empty = well(out, "plate1_r2_c1")
    assert (empty["n_attached"], empty["n_invaded"], empty["n_total"]) == (0, 0, 0)
    assert np.isnan(empty["invasion_efficiency"])
    assert empty["qc_flag_low_total"]
    assert np.isnan(empty["bimodality_coefficient"])
    assert empty["threshold_source"] == "none"

    summary = out["summary"].set_index("condition").loc["dmso"]
    assert summary["n_wells"] == 2 and summary["n_wells_scored"] == 1


def test_all_attached_and_all_invaded_wells_reach_zero_and_one(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": [100.0] * 60},
        {"row": "r2", "column": "c1", "outside": [10.0] * 60},
    ])
    out = analyze_invasion(settings_for(src, outside_threshold=50.0))

    attached = well(out, "plate1_r1_c1")
    invaded = well(out, "plate1_r2_c1")
    assert attached["invasion_efficiency"] == 0.0
    assert attached["n_attached"] == 60 and attached["n_invaded"] == 0
    assert invaded["invasion_efficiency"] == 1.0
    assert invaded["n_attached"] == 0 and invaded["n_invaded"] == 60

    # Both are legitimately one population, and both say so: a well where
    # everything invaded carries no internal evidence that its own threshold
    # was right.
    assert attached["qc_flag_unimodal"]
    assert invaded["qc_flag_unimodal"]


# ---------------------------------------------------------------------------
# Extracellular parasites
# ---------------------------------------------------------------------------

def test_a_parasite_with_no_host_cell_is_attached_by_default(tmp_path):
    """It cannot have invaded anything. Different labs score this differently,
    so it is a setting — but the default is the literal reading, and the count
    is reported either way."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1",
         "outside": [10.0] * 30 + [100.0] * 25 + [10.0] * 5,
         "host": [True] * 55 + [False] * 5},
    ])

    default = well(analyze_invasion(settings_for(src)), "plate1_r1_c1")
    assert default["n_no_host_cell"] == 5
    # The five extracellular parasites look invaded by stain and are scored
    # attached anyway.
    assert default["n_attached"] == 30 and default["n_invaded"] == 30
    assert default["invasion_efficiency"] == pytest.approx(0.5)

    stained = well(analyze_invasion(settings_for(src,
                                                 extracellular_class="classify")),
                   "plate1_r1_c1")
    assert stained["n_invaded"] == 35
    assert stained["invasion_efficiency"] == pytest.approx(35 / 60)

    dropped = well(analyze_invasion(settings_for(src,
                                                 extracellular_class="exclude")),
                   "plate1_r1_c1")
    assert dropped["n_objects"] == 55
    assert dropped["n_no_host_cell"] == 0
    assert dropped["invasion_efficiency"] == pytest.approx(30 / 55)


def test_excluding_every_parasite_says_which_setting_did_it(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5), "host": False}])
    with pytest.raises(ValueError, match="extracellular_class='exclude'"):
        analyze_invasion(settings_for(src, extracellular_class="exclude"))


def test_an_unknown_extracellular_class_is_rejected_before_any_work(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5)}])
    with pytest.raises(ValueError, match="extracellular_class"):
        analyze_invasion(settings_for(src, extracellular_class="maybe"))


def test_a_table_with_no_cell_column_lets_the_stain_decide_everything(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame = frame.drop(columns=["cell_id"])
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    out = analyze_invasion(settings_for(src))
    row = well(out, "plate1_r1_c1")
    assert row["n_no_host_cell"] == 0
    assert row["invasion_efficiency"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Object filters
# ---------------------------------------------------------------------------

def test_area_filters_drop_debris_and_clumps(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame.loc[:9, "pathogen_area"] = 5.0        # debris, all invaded
        frame.loc[50:, "pathogen_area"] = 100000.0  # clumps, all attached
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    trimmed = analyze_invasion(settings_for(src, min_parasite_area=50,
                                            max_parasite_area=1000))
    assert well(trimmed, "plate1_r1_c1")["n_total"] == 40
    assert analyze_invasion(settings_for(src))["wells"].iloc[0]["n_total"] == 60


def test_filtering_everything_away_names_the_filters(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5)}])
    with pytest.raises(ValueError, match="min_parasite_area"):
        analyze_invasion(settings_for(src, min_parasite_area=10 ** 9))


def test_min_total_intensity_drops_objects_the_total_stain_never_lit(tmp_path):
    """The post-permeabilisation channel stains every parasite, so an object
    dark in it is debris in the pathogen mask, not a parasite."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame.loc[:9, "pathogen_channel_0_mean_intensity"] = 1.0
        frame.to_sql("pathogen", con, index=False, if_exists="replace")

    out = analyze_invasion(settings_for(src, min_total_intensity=50.0))
    assert well(out, "plate1_r1_c1")["n_total"] == 50

    with pytest.raises(KeyError, match="total_channel"):
        analyze_invasion(settings_for(src, min_total_intensity=50.0,
                                      total_channel=7))


# ---------------------------------------------------------------------------
# Table shape errors
# ---------------------------------------------------------------------------

def test_a_table_that_is_not_a_spacr_measurements_table_is_named(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5)}])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame.drop(columns=["fieldID"]).to_sql("pathogen", con, index=False,
                                               if_exists="replace")
    with pytest.raises(ValueError, match="fieldID"):
        analyze_invasion(settings_for(src))


def test_a_missing_group_column_lists_what_is_available(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5)}])
    with pytest.raises(KeyError, match="not_a_column"):
        analyze_invasion(settings_for(src, group_column="not_a_column"))


def test_a_well_map_that_matches_nothing_says_which_maps_to_check(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5)}])
    with pytest.raises(ValueError, match="pathogen_plate_metadata"):
        analyze_invasion(settings_for(src, pathogen_plate_metadata=[["c9"],
                                                                    ["c8"]]))


def test_prcf_is_rebuilt_when_the_table_does_not_carry_it(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "field": "f1", "outside": split(15, 15)},
        {"row": "r1", "column": "c1", "field": "f2", "outside": split(15, 15)},
    ])
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql_query("SELECT * FROM pathogen", con)
        frame.drop(columns=["prcf"]).to_sql("pathogen", con, index=False,
                                            if_exists="replace")
    out = analyze_invasion(settings_for(src))
    assert sorted(out["fields"]["prcf"]) == ["plate1_r1_c1_f1",
                                             "plate1_r1_c1_f2"]


def test_change_plate_relabels_two_sources_and_keeps_their_fields_apart(tmp_path):
    from spacr.submodules import analyze_invasion

    src_a = write_db(tmp_path / "a", [
        {"row": "r1", "column": "c1", "outside": split(20, 20)}])
    src_b = write_db(tmp_path / "b", [
        {"row": "r1", "column": "c1", "outside": split(10, 30)}])
    out = analyze_invasion(settings_for([src_a, src_b], change_plate=True))

    assert sorted(out["wells"]["plateID"]) == ["plate1", "plate2"]
    assert sorted(out["fields"]["prcf"]) == ["plate1_r1_c1_f1",
                                             "plate2_r1_c1_f1"]
    assert well(out, "plate1_r1_c1")["invasion_efficiency"] == pytest.approx(0.5)
    assert well(out, "plate2_r1_c1")["invasion_efficiency"] == pytest.approx(0.25)


def test_seeding_wells_from_the_cell_table_can_be_switched_off(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(
        tmp_path / "p",
        [{"row": "r1", "column": "c1", "outside": split(30, 30)}],
        extra_cell_wells=[("r2", "c1")])
    out = analyze_invasion(settings_for(src, seed_wells_from_cells=False))
    assert set(out["wells"]["prc"]) == {"plate1_r1_c1"}


def test_a_database_without_a_cell_table_still_runs(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30), "host": False}])
    # No host cells anywhere means write_db wrote no cell table at all.
    with sqlite3.connect(os.path.join(src, "measurements",
                                      "measurements.db")) as con:
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"pathogen"}
    out = analyze_invasion(settings_for(src))
    assert well(out, "plate1_r1_c1")["n_total"] == 60


# ---------------------------------------------------------------------------
# Output files and figures
# ---------------------------------------------------------------------------

def test_save_writes_the_csvs_next_to_the_other_submodule_outputs(tmp_path):
    """Sibling submodules write to <src>/results/<function name>/; so does
    this one."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(36, 24)},
        {"row": "r1", "column": "c2", "outside": split(12, 48)},
    ])
    out = analyze_invasion(settings_for(src, save=True))

    output_dir = os.path.join(src, "results", "analyze_invasion")
    assert os.path.isdir(output_dir)

    parasites = pd.read_csv(os.path.join(output_dir, "parasite_calls.csv"))
    assert len(parasites) == len(out["parasites"]) == 120
    for column in ("plateID", "rowID", "columnID", "fieldID", "prcf",
                   "object_label", "condition", "outside_intensity",
                   "outside_intensity_raw", "threshold", "threshold_source",
                   "invasion_class", "no_host_cell"):
        assert column in parasites.columns, column
    assert set(parasites["invasion_class"]) == {"attached", "invaded"}

    fields = pd.read_csv(os.path.join(output_dir, "field_thresholds.csv"))
    for column in ("prcf", "threshold", "threshold_source",
                   "automatic_threshold", "reference_threshold",
                   "bimodality_coefficient", "qc_flag_unimodal",
                   "n_attached", "n_invaded", "n_total",
                   "invasion_efficiency"):
        assert column in fields.columns, column
    assert len(fields) == 2

    wells = pd.read_csv(os.path.join(output_dir, "well_invasion.csv"))
    for column in ("prc", "condition", "n_attached", "n_invaded", "n_total",
                   "invasion_efficiency", "invasion_efficiency_inflation",
                   "bimodality_coefficient", "threshold_median",
                   "threshold_source", "qc_flag_low_total",
                   "qc_flag_unimodal", "qc_flag_threshold_disagrees",
                   "qc_flag_threshold_inflates", "qc_flags", "qc_pass"):
        assert column in wells.columns, column
    assert len(wells) == 2
    assert sorted(wells["invasion_efficiency"]) == pytest.approx([0.2, 0.6])

    summary = pd.read_csv(os.path.join(output_dir, "condition_summary.csv"))
    assert set(summary["condition"]) == {"dmso", "drug"}
    assert "invasion_efficiency_pooled" in summary.columns

    comparisons = pd.read_csv(os.path.join(output_dir,
                                           "condition_comparisons.csv"))
    assert len(comparisons) == 1
    assert set(comparisons["unit_of_replication"]) == {"well"}

    for name in ("chi_squared_results.csv", "chi_squared_pairwise_results.csv"):
        assert os.path.getsize(os.path.join(output_dir, name)) > 0

    for name in ("invasion_per_well.pdf", "invasion_by_condition.pdf",
                 "outside_stain_thresholds.pdf"):
        path = os.path.join(output_dir, name)
        assert os.path.getsize(path) > 0
        with open(path, "rb") as handle:
            assert handle.read(4) == b"%PDF"

    # The settings snapshot lands where every other pipeline puts it.
    assert os.path.isfile(os.path.join(src, "settings", "analyze_invasion.csv"))


def test_figures_are_created_and_closed(tmp_path):
    """Three figures come back in the output dict and none is left open in
    pyplot's registry — a batch over 40 plates must not leak 120 figures."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(36, 24)},
        {"row": "r1", "column": "c2", "outside": split(12, 48)},
    ])
    plt.close("all")
    assert plt.get_fignums() == []

    out = analyze_invasion(settings_for(src))

    assert set(out["figures"]) == {"per_well", "by_condition", "thresholds"}
    assert plt.get_fignums() == [], "analyze_invasion leaked a figure"

    per_well = out["figures"]["per_well"]
    by_condition = out["figures"]["by_condition"]
    assert per_well.axes[0].get_title() == "Invasion — per well"
    assert by_condition.axes[0].get_title() == "Invasion — by condition"
    for figure in (per_well, by_condition):
        assert figure.axes[0].get_legend().get_title().get_text() == \
            "Parasite class"

    labels = sorted(t.get_text() for t in per_well.axes[0].get_xticklabels())
    assert labels == ["plate1_r1_c1", "plate1_r1_c2"]


def test_every_bar_is_annotated_with_its_denominator(tmp_path):
    """A stacked proportion bar hides its n; the assay's whole point is that
    the n matters, so it is written on the bar."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(9, 1)},
        {"row": "r1", "column": "c2", "outside": split(300, 300)},
    ])
    out = analyze_invasion(settings_for(src))
    texts = {t.get_text() for t in out["figures"]["per_well"].axes[0].texts}
    assert {"n=10", "n=600"} <= texts


def test_the_threshold_figure_shows_the_distribution_it_was_taken_from(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)},
        {"row": "r2", "column": "c1", "outside": split(20, 40)},
    ])
    out = analyze_invasion(settings_for(src))
    figure = out["figures"]["thresholds"]

    drawn = [axis for axis in figure.axes if axis.get_title()]
    assert len(drawn) == 2
    titles = " ".join(axis.get_title() for axis in drawn)
    assert "plate1_r1_c1" in titles and "n=60" in titles and "BC=1.00" in titles
    # A vertical line at the threshold each panel used.
    for axis in drawn:
        positions = [line.get_xdata()[0] for line in axis.lines]
        assert pytest.approx(55.0) in positions


def test_the_threshold_figure_caps_how_many_panels_it_draws(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": f"r{i}", "column": "c1", "outside": split(30, 30)}
        for i in range(1, 7)])
    out = analyze_invasion(settings_for(src, qc_plot_max_panels=2))
    figure = out["figures"]["thresholds"]

    drawn = [axis for axis in figure.axes if axis.get_title()]
    assert len(drawn) == 2
    assert "first 2 wells" in figure._suptitle.get_text()


def test_a_single_condition_still_draws_bars_with_empty_stats(tmp_path):
    """One group leaves no pair to chi-square, so the shared plot helper is
    bypassed; the figures are descriptive either way."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    out = analyze_invasion(settings_for(
        src, pathogen_types=["dmso"], pathogen_plate_metadata=[["c1"]]))

    assert np.isnan(out["chi_squared"]["p_value"].iloc[0])
    assert len(out["chi_squared_pairwise"]) == 0
    assert out["figures"]["by_condition"].axes[0].get_title() == \
        "Invasion — by condition"


@pytest.mark.parametrize("level", ["object", "well", "plate"])
def test_every_aggregation_level_runs(tmp_path, level):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)},
        {"row": "r2", "column": "c1", "outside": split(20, 40)},
        {"row": "r1", "column": "c2", "outside": split(45, 15)},
        {"row": "r2", "column": "c2", "outside": split(50, 10)},
    ])
    out = analyze_invasion(settings_for(src, level=level))
    assert len(out["wells"]) == 4
    assert out["comparisons"].iloc[0]["n_wells_1"] == 2


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_settings_module_defaults_win_over_the_local_fallback(tmp_path,
                                                              monkeypatch):
    """spacr.settings owns every pipeline's defaults; the copy in submodules is
    only a gap-filler for running the assay before the GUI knobs exist."""
    import spacr.settings as spacr_settings
    from spacr.submodules import analyze_invasion

    def fake_defaults(settings):
        settings.setdefault("outside_threshold", 55.0)
        settings.setdefault("min_parasites_per_well", 1)
        return settings

    monkeypatch.setattr(spacr_settings, "set_analyze_invasion_defaults",
                        fake_defaults, raising=False)

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(3, 3)}])
    out = analyze_invasion(settings_for(src))
    row = well(out, "plate1_r1_c1")
    assert row["threshold_source"] == "fixed"
    assert not row["qc_flag_low_total"]


def test_the_local_fallback_fills_every_key_the_assay_reads():
    from spacr.submodules import _set_analyze_invasion_defaults

    settings = _set_analyze_invasion_defaults({"src": "/tmp/x"})
    for key in ("parasite_table", "compartment", "outside_channel",
                "total_channel", "intensity_statistic",
                "outside_threshold_method", "background_correction",
                "outside_threshold", "control_wells", "control_quantile",
                "min_control_objects", "min_parasite_area",
                "max_parasite_area", "min_total_intensity",
                "seed_wells_from_cells", "qc_plot_max_panels", "cmap",
                "change_plate",
                "min_objects_for_threshold", "min_objects_for_bimodality",
                "bimodality_cutoff", "threshold_agreement_tolerance",
                "threshold_sensitivity", "inflation_warn",
                "min_parasites_per_well", "extracellular_class",
                "group_column", "level", "save", "verbose"):
        assert key in settings, key
    # Caller values are never overwritten.
    assert _set_analyze_invasion_defaults({"src": "x",
                                           "outside_channel": 3})["outside_channel"] == 3


def test_verbose_prints_the_threshold_of_every_field_and_the_flagged_wells(
        tmp_path, capsys):
    """The threshold is the number the whole assay rests on, so it is
    reportable rather than internal."""
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(5, 5)}])
    analyze_invasion(settings_for(src, verbose=True))
    printed = capsys.readouterr().out

    assert "Outside-stain statistic" in printed
    assert "Per-field thresholds" in printed
    assert "plate1_r1_c1_f1" in printed
    assert "low_total" in printed


def test_the_intensity_column_actually_used_is_returned(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p", [
        {"row": "r1", "column": "c1", "outside": split(30, 30)}])
    out = analyze_invasion(settings_for(src))
    assert out["intensity_column"] == "pathogen_channel_1_percentile_95"
    assert out["intensity_statistic"] == "percentile_95"


def test_a_periphery_column_wins_end_to_end(tmp_path):
    from spacr.submodules import analyze_invasion

    src = write_db(tmp_path / "p",
                   [{"row": "r1", "column": "c1", "outside": split(30, 30)}],
                   statistic="periphery_95_percentile")
    out = analyze_invasion(settings_for(src))
    assert out["intensity_statistic"] == "periphery_95"
    assert well(out, "plate1_r1_c1")["invasion_efficiency"] == pytest.approx(0.5)
