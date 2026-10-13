"""Round-6 coverage for the analysis modules' remaining untaken turns.

Four modules, and in every case the branch left open is the *quiet* half of a
decision -- the arm that produces no output, or the arm that a guarantee made
earlier in the same function has already closed off:

* :mod:`spacr.illumination` -- a corrector told to ``skip`` a plate it has no
  field for counts the skip whether or not it is allowed to say so, and stops
  saying so after ``_MAX_CLIP_WARNINGS``; the radial QC profile drops a ring
  no pixel falls in rather than plotting a NaN; and ``illumination_qc: False``
  really does mean no QC figure while the hook is still installed.

* :mod:`spacr.submodules` -- a cross-validation CSV with no column key skips
  the well filter and then names the key it needed, instead of filtering on a
  column that is not there; and an invasion run in which nothing invaded still
  reports ``n_invaded = 0`` rather than losing the column.

* :mod:`spacr.hit_attribution` -- identifiers containing the ``|`` that
  ``_group_series`` joins on collapse eight distinct wells into one group, and
  the cross-fitter refuses the design instead of handing GroupKFold a single
  bag.

* :mod:`spacr.guide_attribution` -- option C's per-cell shift before the
  exponential means no cell can come back with an all-zero row, so the
  fall-back-to-the-prior guard behind it cannot fire; the test pins the
  guarantee in the regime that would otherwise underflow.

Two of those branches are unreachable and are recorded as such, with the proof
in a comment above the test that pins the guarantee instead:
``submodules.py:4900`` (the invasion class column) and
``guide_attribution.py:728`` (the all-zero-row fallback).
"""
from __future__ import annotations

import itertools
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spacr import guide_attribution as ga  # noqa: E402
from spacr import hit_attribution as ha  # noqa: E402
from spacr import illumination as ill  # noqa: E402
from spacr import measure_hooks as mh  # noqa: E402
from spacr.measure_hooks import PreprocessingContext  # noqa: E402


@pytest.fixture(autouse=True)
def _no_figures_and_no_hooks(monkeypatch):
    """No hook, no environment and no figure survives a test in this file.

    The hook registry and the illumination environment variables are both
    process-global; one leaking out would change what a later test measures.
    """
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    mh.clear_measurement_hooks()
    yield
    mh.clear_measurement_hooks()
    plt.close("all")
    for name in (mh.HOOKS_ENV_VAR, ill.MODEL_ENV_VAR, ill.ON_MISSING_ENV_VAR):
        os.environ.pop(name, None)


# ---------------------------------------------------------------------------
# spacr.illumination
# ---------------------------------------------------------------------------

def _flat(shape=(5, 5)):
    """A mild horizontal vignette normalised to mean 1."""
    field = np.linspace(1.2, 0.8, shape[1], dtype=np.float32)
    field = np.tile(field, (shape[0], 1))
    return (field / field.mean()).astype(np.float32)


def _model(flat, *, plate="plate1"):
    """A one-plate, one-channel model built by hand -- no estimation."""
    field = ill.IlluminationField(
        plate=plate, channels=(0,), flatfield=np.stack([flat]),
        dark=np.zeros(1, np.float32), n_fields=3,
        estimator="polynomial", degree=4, bin_size=1)
    return ill.IlluminationModel(fields={plate: field}, meta={})


def _merged(folder, flat, *, plate="plate1", n_fields=3):
    """Merged single-channel fields of flat signal under ``flat``."""
    os.makedirs(folder, exist_ok=True)
    plane = np.rint(1000.0 * flat).astype(np.uint16)
    for index in range(n_fields):
        np.save(os.path.join(folder, f"{plate}_A01_F{index:03d}.npy"),
                np.stack([plane], axis=-1))
    return str(folder)


def test_a_skip_is_counted_whether_or_not_the_corrector_may_announce_it(capsys):
    """``verbose=False`` silences the warning; it does not silence the count.

    ``on_missing='skip'`` measures a plate the model does not cover
    uncorrected, which is a fact about the table that has to survive into
    ``stats`` even when the run asked for a quiet corrector. And a loud one
    stops after ``_MAX_CLIP_WARNINGS``, because one line per field on a plate
    the model never saw is thousands of identical lines.
    """
    model = _model(_flat((4, 4)), plate="plateA")
    array = np.full((4, 4, 1), 100, np.uint16)

    def elsewhere(index):
        return PreprocessingContext(file_name=f"plateB_A01_F{index:03d}",
                                    channels=[0], settings={})

    quiet = ill.IlluminationCorrector(model, on_missing="skip", verbose=False)
    result = quiet(array, elsewhere(0))

    assert result is array, "a skipped field must come back untouched"
    assert quiet.stats["skipped"] == 1
    assert capsys.readouterr().out == ""

    # The same skip from a corrector that is allowed to speak: it says
    # UNCORRECTED once per field until the cap, and then stops.
    loud = ill.IlluminationCorrector(model, on_missing="skip", verbose=True)
    calls = ill._MAX_CLIP_WARNINGS + 2
    for index in range(calls):
        loud(array, elsewhere(index))

    printed = capsys.readouterr().out
    assert printed.count("UNCORRECTED") == ill._MAX_CLIP_WARNINGS
    assert "plateB" in printed
    assert loud.stats["skipped"] == calls, (
        "the cap is on the warnings, not on the counter")


def test_the_qc_profile_drops_a_ring_that_holds_no_pixel(tmp_path):
    """A field smaller than the ring count leaves rings with nothing in them.

    ``_radial_profile`` bins pixels by normalised corner-radius into a fixed
    number of rings. A 5x5 field has 25 pixels for 20 rings and their radii
    are far from uniform, so several rings are empty -- and an empty ring
    contributes no point rather than a NaN that would break the QC plot's
    line. The profile is drawn and never returned, so the ring count is
    asserted on the helper directly; the public call above proves the figure
    is really produced from a field that has those empty rings.
    """
    merged = _merged(tmp_path / "merged", _flat((5, 5)))
    save_dir = str(tmp_path / "qc")

    report = ill.illumination_qc(_model(_flat((5, 5))), merged,
                                 save_dir=save_dir, verbose=False)

    assert os.path.isfile(os.path.join(save_dir,
                                       "illumination_qc_plate1.png"))
    assert report["_figures"]["plate1"].endswith("illumination_qc_plate1.png")
    assert np.isfinite(report["plate1"][0]["slope_before"])

    small = np.arange(1, 26, dtype=float).reshape(5, 5)
    radii, means = ill._radial_profile(small, 1, (5, 5))
    assert len(radii) == len(means)
    assert 0 < len(radii) < 20, "a 5x5 field cannot fill twenty rings"
    assert np.all(np.diff(radii) > 0)
    assert np.isfinite(means).all()

    # A field large enough to put a pixel in every ring returns all twenty,
    # so the short answer above is the empty-ring case and not the only one.
    large = np.random.default_rng(0).random((64, 64)) + 1.0
    radii_full, means_full = ill._radial_profile(large, 1, (64, 64))
    assert len(radii_full) == 20 and len(means_full) == 20


def test_declining_the_qc_writes_no_figure_but_still_installs_the_hook(tmp_path):
    """``illumination_qc: False`` skips the report, not the correction.

    The QC figure is the expensive half of ``prepare_illumination_correction``
    -- it re-reads every merged field -- so a re-run that already trusts its
    model turns it off. What must not be turned off with it is the hook that
    makes ``measure_crop`` correct anything at all.
    """
    merged = _merged(tmp_path / "merged", _flat())
    saved = str(tmp_path / "saved.npz")
    _model(_flat()).save(saved)
    folder = tmp_path / "illumination"

    settings = {"illumination_correction": True, "src": merged,
                "illumination_model": saved, "channels": [0],
                "illumination_qc": False, "verbose": False}
    model = ill.prepare_illumination_correction(settings)

    assert isinstance(model, ill.IlluminationModel)
    assert not folder.exists(), "the QC step created its own output folder"
    assert [hook.name for hook in mh.preprocessing_hooks()] == [ill.HOOK_NAME]

    # The same call with the QC left on writes the figure into that folder,
    # so the absence above is the setting and not a broken path.
    mh.clear_measurement_hooks()
    settings["illumination_qc"] = True
    ill.prepare_illumination_correction(settings)

    assert sorted(os.listdir(folder)) == ["illumination_qc_plate1.png"]
    assert [hook.name for hook in mh.preprocessing_hooks()] == [ill.HOOK_NAME]


# ---------------------------------------------------------------------------
# spacr.submodules -- generate_score_heatmap
# ---------------------------------------------------------------------------

ROWS = [f"r{i}" for i in range(1, 9)]
COL = "c3"
OTHER_COL = "c2"


def _write_scores_csv(path, seed, value_column="pred", with_column_key=True):
    """Per-object scores, one ``c3`` row and one ``c2`` row per plate row.

    ``with_column_key=False`` writes the same scores with no column key at
    all, which is the input the well filter cannot apply.
    """
    rng = np.random.default_rng(seed)
    records = []
    for row in ROWS:
        for column in (COL, OTHER_COL):
            record = {"rowID": row,
                      value_column: round(float(rng.uniform(0, 1)), 6)}
            if with_column_key:
                record["columnID"] = column
            records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)
    return {record["rowID"]: record[value_column] for record in records
            if not with_column_key or record["columnID"] == COL}


def _counts(index):
    return {"A": 10 + 3 * index, "B": 40 - 2 * index, "C": 1000}


def _write_mixed_csv(path):
    """Per-well sgRNA read counts for the mixed control condition."""
    records = []
    for index, row in enumerate(ROWS):
        counts = _counts(index)
        for name, key in (("sgA", "A"), ("sgB", "B"), ("sgC", "C")):
            records.append({"columnID": COL, "rowID": row,
                            "grna_name": name, "count": counts[key]})
    pd.DataFrame(records).to_csv(path, index=False)


def _model_folder(tmp_path):
    """A folder of per-model sub-folders, each holding a ``scores.csv``."""
    folder = tmp_path / "models"
    folder.mkdir()
    for index, name in enumerate(("modelA", "modelB")):
        (folder / name).mkdir()
        _write_scores_csv(str(folder / name / "scores.csv"), seed=17 + index)
    return folder


def _heatmap_settings(folder, mixed, cv, dst):
    return {"folders": [str(folder)], "csv_name": "scores.csv",
            "data_column": "pred", "csv": str(mixed), "cv_csv": str(cv),
            "data_column_cv": "pred_cv", "plateID": 1, "columnID": COL,
            "control_sgrnas": ["sgA", "sgB"], "fraction_grna": "sgA",
            "cmap": "coolwarm", "dst": str(dst)}


def test_a_cv_table_with_no_column_key_names_the_key_it_needed(tmp_path):
    """The well filter is skipped rather than applied to a column that is not there.

    ``read_table`` canonicalises every spelling of the column key, so a CV CSV
    that reaches ``group_cv_score`` without a ``columnID`` genuinely has no
    column axis: there is nothing to filter ``settings['columnID']`` against.
    Skipping the filter and letting the per-well groupby raise ``KeyError:
    'columnID'`` names the missing key; filtering on a fabricated one would
    have silently kept every column of the plate in a per-well mean.
    """
    from spacr.submodules import generate_score_heatmap

    folder = _model_folder(tmp_path)
    mixed = tmp_path / "mixed.csv"
    _write_mixed_csv(str(mixed))
    dst = tmp_path / "out"
    dst.mkdir()

    keyless = tmp_path / "cv_without_column.csv"
    _write_scores_csv(str(keyless), seed=3, value_column="pred_cv",
                      with_column_key=False)

    with pytest.raises(KeyError) as caught:
        generate_score_heatmap(
            _heatmap_settings(folder, mixed, keyless, dst))
    assert "columnID" in str(caught.value)

    # The same CV table WITH the key filters to c3 and the run completes, so
    # the refusal above is the missing key and not a broken harness.
    keyed = tmp_path / "cv.csv"
    truth = _write_scores_csv(str(keyed), seed=3, value_column="pred_cv")
    out = generate_score_heatmap(_heatmap_settings(folder, mixed, keyed, dst))

    assert len(out) == len(ROWS)
    assert set(out["columnID"]) == {COL}
    by_row = dict(zip(out["rowID"], out["pred_cv"]))
    assert by_row == pytest.approx(truth), (
        "the c2 rows leaked into the per-well CV mean")


# ---------------------------------------------------------------------------
# spacr.submodules -- analyze_invasion
# ---------------------------------------------------------------------------

PARASITE_AREA = 100.0
OUTSIDE_CHANNEL = 1
TOTAL_CHANNEL = 0


def _write_invasion_db(root, outside, *, host):
    """One well of parasites whose outside-stain statistic is set literally."""
    measurements = os.path.join(str(root), "measurements")
    os.makedirs(measurements, exist_ok=True)
    db = os.path.join(measurements, "measurements.db")
    prcf = "plate1_r1_c1_f1"

    pathogens, cells = [], []
    for index, value in enumerate(outside, start=1):
        cell = index if host else 0
        pathogens.append({
            "object_label": index, "cell_id": cell,
            "plateID": "plate1", "rowID": "r1", "columnID": "c1",
            "fieldID": "f1", "prcf": prcf, "pathogen_area": PARASITE_AREA,
            f"pathogen_channel_{TOTAL_CHANNEL}_mean_intensity": 200.0,
            f"pathogen_channel_{OUTSIDE_CHANNEL}_percentile_95": float(value),
        })
        if host:
            cells.append({"object_label": cell, "plateID": "plate1",
                          "rowID": "r1", "columnID": "c1", "fieldID": "f1",
                          "prcf": prcf, "cell_area": 20000.0})
    if not host:
        # An uninfected host cell so the cell table exists at all.
        cells.append({"object_label": 900, "plateID": "plate1", "rowID": "r1",
                      "columnID": "c1", "fieldID": "f1", "prcf": prcf,
                      "cell_area": 20000.0})

    with sqlite3.connect(db) as con:
        pd.DataFrame(pathogens).to_sql("pathogen", con, index=False,
                                       if_exists="replace")
        pd.DataFrame(cells).to_sql("cell", con, index=False,
                                   if_exists="replace")
    return str(root)


def _invasion_settings(src):
    return {"src": src, "outside_channel": OUTSIDE_CHANNEL,
            "total_channel": TOTAL_CHANNEL, "cell_types": None,
            "cell_plate_metadata": None, "pathogen_types": ["dmso", "drug"],
            "pathogen_plate_metadata": [["c1"], ["c2"]], "treatments": None,
            "treatment_plate_metadata": None, "save": False, "verbose": False}


# UNREACHABLE: submodules.py:4900 (`field_counts[name] = 0`) and the
# 4899 -> 4900 arc into it cannot be taken. `_invasion_classify` sets
# `invasion_class` to a `pd.Categorical` whose categories are
# `_INVASION_CLASSES + ['unclassified']` (submodules.py:4046-4049), and
# `SeriesGroupBy.value_counts()` over a categorical enumerates EVERY category,
# not only the observed ones -- so the `unstack` at 4897 always yields one
# column per category and neither name in `_INVASION_CLASSES` can be absent
# from `field_counts.columns`. The guard is a defensive re-check after a step
# that already guarantees its condition. What is testable is the guarantee
# itself, which is what this test pins.
def test_a_field_where_nothing_invaded_still_reports_a_zero_invaded_count(
        tmp_path):
    """No invaded parasite is a count of zero, never a missing column.

    Every parasite here overlaps no host cell, and ``extracellular_class``
    defaults to ``'attached'`` -- a parasite outside a cell has not invaded
    anything whatever the stain says. So the whole field scores attached and
    the invaded class is observed nowhere. It still has to reach the fields
    and wells tables as ``n_invaded = 0`` with an efficiency of 0.0, because a
    missing column there would be a ``KeyError`` in the merge that builds the
    per-field table and a dropped denominator in every mean below it.
    """
    from spacr.submodules import analyze_invasion

    out = analyze_invasion(_invasion_settings(_write_invasion_db(
        tmp_path / "none", [5.0, 6.0, 7.0, 8.0, 9.0, 10.0], host=False)))

    assert set(out["parasites"]["invasion_class"]) == {"attached"}
    fields = out["fields"]
    assert list(fields["n_invaded"]) == [0]
    assert list(fields["n_attached"]) == [6]
    well = out["wells"].iloc[0]
    assert (well["n_invaded"], well["n_attached"], well["n_total"]) == (0, 6, 6)
    assert well["invasion_efficiency"] == pytest.approx(0.0)

    # A well that really does contain invaded parasites counts them, so the
    # zero above is the observation and not a column that is always empty.
    both = analyze_invasion(_invasion_settings(_write_invasion_db(
        tmp_path / "some", [10.0] * 4 + [100.0] * 6, host=True)))
    row = both["wells"].iloc[0]
    assert (row["n_invaded"], row["n_attached"]) == (4, 6)
    assert row["invasion_efficiency"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# spacr.hit_attribution
# ---------------------------------------------------------------------------

def _colliding_wells(n=8):
    """Well keys that are all distinct and all join to the same group string.

    ``_group_series`` identifies a well by ``"|".join`` of plate, row and
    column, so identifiers that themselves contain ``|`` are ambiguous: every
    way of cutting ``a|a|a|a|a|a`` into three pieces is a different well and
    the same group key.
    """
    tokens = ["a"] * 6
    return [("|".join(tokens[:i]), "|".join(tokens[i:j]), "|".join(tokens[j:]))
            for i, j in itertools.combinations(range(1, 6), 2)][:n]


def _candidate_frame(keys, *, seed=5, cells_per_well=8):
    """Cells for ``keys``, alternating target and control wells."""
    rng = np.random.default_rng(seed)
    rows = []
    for index, (plate, row, column) in enumerate(keys):
        target = index % 2 == 0
        for cell in range(cells_per_well):
            rows.append({
                "plateID": plate, "rowID": row, "columnID": column,
                "prcfo": f"{index}_{cell}", "target_well": target,
                "cell_area": rng.normal(2.0 if target else 0.0, 0.5),
                "cell_texture": rng.normal(1.0 if target else 0.0, 0.5)})
    return pd.DataFrame(rows)


def test_well_keys_that_collapse_into_one_group_are_an_insufficient_design():
    """Eight wells that share one group key are one bag, and one bag is refused.

    The well-count gate upstream sees eight distinct ``(plate, row, column)``
    triples and is satisfied. It is the *group* series that collapses them,
    because it identifies a well by joining the three ids with ``|`` and these
    ids contain ``|`` themselves. Cross-fitting then has a single held-out
    group, which is not cross-fitting at all -- so it is refused by name
    rather than handed to GroupKFold, which would fail with a message about
    n_splits that says nothing about the design.
    """
    keys = _colliding_wells()
    assert len(set(keys)) == 8, "the eight wells must be distinct"
    assert len({"|".join(key) for key in keys}) == 1, (
        "the eight wells must share one group key")

    with pytest.raises(ha.InsufficientDesignError) as caught:
        ha.crossfit_candidate_probabilities(_candidate_frame(keys),
                                            prefer_plate=False)
    assert "at least two groups" in str(caught.value)

    # The same eight wells with unambiguous ids cross-fit normally, so the
    # refusal is the collision and not the size of the design.
    plain = [tuple(part.replace("|", "-") for part in key) for key in keys]
    scored, features, level, warnings = ha.crossfit_candidate_probabilities(
        _candidate_frame(plain), prefer_plate=False)

    assert level == "well"
    assert len(set(scored["attribution_fold"])) >= 2
    assert scored["candidate_probability"].notna().all()
    assert "cell_area" in features and warnings == []


# ---------------------------------------------------------------------------
# spacr.guide_attribution
# ---------------------------------------------------------------------------

# UNREACHABLE: guide_attribution.py:728 (`density[dead, :] = weights`) and the
# 727 -> 728 arc into it cannot be taken. Two lines above, line 722 subtracts
# each row's own maximum from `log_density`, so every row contains an entry
# that is exactly 0.0 and `density = np.exp(log_density)` (line 723) contains a
# 1.0 in that position -- every row therefore sums to at least 1. The entries
# themselves cannot be NaN or infinite either: each addend is
# `np.log(np.maximum(density, 1e-300))` (line 716) over a `nan_to_num`-ed
# density (line 715), so it lies in [-690.8, +inf) with the +inf excluded by
# `posinf=0.0`, and `factor` (line 692) is a finite number in (0, 1]. So
# `density.sum(axis=1) <= 0` is false for every row. The guard is a defensive
# re-check after the shift that already guarantees its condition; this test
# pins the guarantee in the regime that would underflow without it.
def test_two_hundred_correlated_measurements_still_give_a_real_distribution():
    """The correlation correction scales the evidence; it must not zero it.

    Option C reads every measurement, and 200 near-duplicate columns are worth
    about one independent one -- so ``effective_dimension`` collapses to 1 and
    the summed log density is scaled by 1/200. With every guide 300 sigma from
    every measurement the raw product of densities is zero in double
    precision, and only the per-cell shift before the exponential keeps the
    row from being all zeros and the normalisation from dividing by zero.

    A well the measurements CAN separate does not land on the prior, so the
    fallback-shaped answer below is what the evidence says and not a floor.
    """
    n_columns = 200
    rng = np.random.default_rng(0)
    shared = rng.normal(0.0, 1.0, (4, 1))
    values = np.repeat(shared, n_columns, axis=1) + rng.normal(
        0.0, 1e-3, (4, n_columns))
    priors = {"g1": 0.6, "g2": 0.4}
    hopeless = {"g1": [300.0] * n_columns, "g2": [-300.0] * n_columns}

    posterior, guides, report = ga.posterior_multivariate(
        values, priors, hopeless,
        centres=[0.0] * n_columns, scales=[1.0] * n_columns)

    assert guides == ("g1", "g2")
    assert report["n_measurements"] == 200.0
    assert report["effective_dimension"] == pytest.approx(1.0, abs=1e-3)
    assert report["scale_factor"] == pytest.approx(1.0 / n_columns, rel=1e-3)
    assert np.isfinite(posterior).all() and (posterior > 0).all()
    assert posterior.sum(axis=1) == pytest.approx(np.ones(4))
    # Equally impossible for both guides means the sequencing decides.
    assert posterior[:, 0] == pytest.approx(np.full(4, 0.6), abs=1e-9)

    separable = np.repeat(np.array([[2.0], [-2.0], [2.0], [-2.0]]),
                          n_columns, axis=1)
    told, _guides, _report = ga.posterior_multivariate(
        separable, {"g1": 0.5, "g2": 0.5},
        {"g1": [2.0] * n_columns, "g2": [-2.0] * n_columns},
        centres=[0.0] * n_columns, scales=[1.0] * n_columns)

    assert told[0, 0] > 0.99 and told[1, 1] > 0.99
    assert np.isfinite(told).all()
