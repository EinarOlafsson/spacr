"""Tests for :mod:`spacr.seg_qc`, the segmentation scorecard.

The core of this file is synthetic masks with *known* defects: a clean field, a
field whose objects were welded together, a field shattered into fragments, a
field whose objects are mostly on the edge, an empty field. Each one has to be
called what it is, and — just as important — the clean and merely-dense fields
have to be left alone, because a QC report that cries wolf is a QC report
nobody reads.

Two of the tests are about the statistics rather than the flags:

* :func:`test_mad_outliers_catch_debris_that_defeats_mean_and_standard_deviation`
  computes both rules on the same field and shows the mean/std one is blind to
  exactly the case this check exists for.
* :func:`test_border_objects_are_excluded_from_the_size_statistics` shows that
  truncated objects would otherwise fake a second size population.

Everything here is CPU-only, offline, deterministic and unmarked.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from spacr.seg_qc import (
    FLAG_BORDER,
    FLAG_EMPTY,
    FLAG_HIGH_COUNT,
    FLAG_LOW_COUNT,
    FLAG_NEAR_EMPTY,
    FLAG_OUTLIERS,
    FLAG_OVER,
    FLAG_UNDER,
    FLAG_UNREADABLE,
    QC_DEFAULTS,
    SETTING_KEYS,
    FieldQC,
    format_scorecard,
    qc_mode,
    run_segmentation_qc,
    score_field,
    score_masks,
    summarize_qc,
    thresholds_from_settings,
    write_scorecard,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# synthetic mask builders — every defect below is put there on purpose
# ---------------------------------------------------------------------------

def _disc(labels, cy, cx, radius, value):
    """Paint one filled disc of label ``value`` into ``labels``."""
    h, w = labels.shape
    y0, y1 = max(0, int(cy - radius) - 1), min(h, int(cy + radius) + 2)
    x0, x1 = max(0, int(cx - radius) - 1), min(w, int(cx + radius) + 2)
    if y1 <= y0 or x1 <= x0:
        raise AssertionError(f"disc at ({cy}, {cx}) falls outside a {h}x{w} field")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    hit = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius * radius
    sub = labels[y0:y1, x0:x1]
    sub[hit] = value
    return labels


def _grid_field(shape=(512, 512), radius=10, spacing=50, margin=40, jitter=0, seed=0):
    """A grid of well-separated discs, none touching the border.

    ``jitter`` perturbs the radii so the size distribution is not degenerate —
    a MAD of exactly zero would make the outlier test too easy to pass.
    """
    rng = np.random.default_rng(seed)
    labels = np.zeros(shape, np.int32)
    value = 0
    for cy in range(margin, shape[0] - margin, spacing):
        for cx in range(margin, shape[1] - margin, spacing):
            value += 1
            r = radius + (int(rng.integers(-jitter, jitter + 1)) if jitter else 0)
            _disc(labels, cy, cx, r, value)
    return labels


def _fused_field(shape=(384, 384), radius=16, step=26, cluster=3, block=96):
    """Clusters of overlapping discs, each cluster welded into ONE label.

    This is what a confluent monolayer segmented with too large a diameter
    produces: nine cells, one mask object, a field two-thirds covered.
    """
    labels = np.zeros(shape, np.int32)
    value = 0
    for by in range(0, shape[0], block):
        for bx in range(0, shape[1], block):
            value += 1
            cy0 = by + block // 2 - (cluster - 1) * step // 2
            cx0 = bx + block // 2 - (cluster - 1) * step // 2
            for i in range(cluster):
                for j in range(cluster):
                    _disc(labels, cy0 + i * step, cx0 + j * step, radius, value)
    return labels


def _dense_but_separated_field(shape=(512, 512), radius=20, spacing=50, margin=30):
    """Almost half the field is foreground, and every object is its own label.

    The both-halves rule has to leave this alone: density on its own is not
    fusion.
    """
    return _grid_field(shape=shape, radius=radius, spacing=spacing, margin=margin)


def _shattered_field(shape=(512, 512), radius=2, spacing=20, margin=20):
    """Hundreds of 4-px fragments — one cell blown apart into shards."""
    return _grid_field(shape=shape, radius=radius, spacing=spacing, margin=margin)


def _border_field(shape=(300, 300), radius=8, n_border=16, n_interior=5):
    """A field whose objects are mostly sitting on the image edge."""
    labels = np.zeros(shape, np.int32)
    value = 0
    step = (shape[1] - 2 * radius) // max(1, n_border // 2)
    for k in range(n_border // 2):
        value += 1
        _disc(labels, 0, radius + k * step, radius, value)
        value += 1
        _disc(labels, shape[0] - 1, radius + k * step, radius, value)
    for k in range(n_interior):
        value += 1
        _disc(labels, shape[0] // 2, 40 + k * 50, radius, value)
    return labels


def _write_plate(tmp_path, fields, subdir="cell_mask_stack"):
    """Write ``{name: mask}`` as the ``.npy`` folder spacr.object produces."""
    folder = tmp_path / subdir
    folder.mkdir(parents=True, exist_ok=True)
    for name, mask in fields.items():
        np.save(folder / f"{name}.npy", mask.astype(np.uint16))
    return str(folder)


# ---------------------------------------------------------------------------
# per-field verdicts
# ---------------------------------------------------------------------------

def test_a_clean_field_is_called_clean():
    """No flag, 'ok', and the object count is the one that was painted."""
    mask = _grid_field(jitter=2)
    qc = score_field(mask, "cell", "clean_field")
    assert qc.flags == []
    assert qc.severity == "ok"
    assert qc.n_objects == int(mask.max())
    assert qc.metrics["border_fraction"] == 0.0
    assert "Nothing wrong" in qc.note


def test_a_fused_field_flags_under_segmentation():
    """Nine overlapping cells welded into one label, on a 68%-covered field."""
    qc = score_field(_fused_field(), "cell", "fused_field")
    assert FLAG_UNDER in qc.flags
    assert qc.severity == "fail"
    # both halves of the signature must be present in the metrics
    assert qc.metrics["foreground_fraction"] >= QC_DEFAULTS["foreground_fraction"]
    assert qc.metrics["split_ratio"] >= QC_DEFAULTS["split_ratio"]
    assert "fused" in qc.note


def test_a_dense_but_correctly_separated_field_is_not_called_fused():
    """Density alone is not fusion — the rule diameter.py argues, tested.

    This field is 45% foreground, well past the confluence threshold, and every
    object is separately labelled. Flagging it would make the check useless on
    exactly the confluent plates it is meant to serve.
    """
    mask = _dense_but_separated_field()
    qc = score_field(mask, "cell", "dense_field")
    assert qc.metrics["foreground_fraction"] >= QC_DEFAULTS["foreground_fraction"]
    assert FLAG_UNDER not in qc.flags
    assert qc.metrics["split_ratio"] < QC_DEFAULTS["split_ratio"]


def test_a_shattered_field_flags_over_segmentation():
    """Every object is 4 px across, which is a fragment, not a cell."""
    qc = score_field(_shattered_field(), "cell", "shattered_field")
    assert FLAG_OVER in qc.flags
    assert qc.severity == "fail"
    assert qc.metrics["tiny_fraction"] == pytest.approx(1.0)


def test_border_touching_objects_are_flagged_with_the_right_fraction():
    """16 of 21 objects sit on the edge: 76%, and the number is reported."""
    qc = score_field(_border_field(), "cell", "border_field")
    assert FLAG_BORDER in qc.flags
    assert qc.n_objects == 21
    assert qc.metrics["border_fraction"] == pytest.approx(16 / 21)
    assert qc.metrics["n_interior"] == 5.0
    assert "truncated" in qc.note


def test_an_empty_field_is_flagged_not_crashed_on():
    """Zero objects is a verdict, not an exception."""
    qc = score_field(np.zeros((64, 64), np.uint16), "cell", "empty_field")
    assert qc.flags == [FLAG_EMPTY]
    assert qc.severity == "fail"
    assert qc.n_objects == 0
    assert qc.metrics["foreground_fraction"] == 0.0
    assert "nothing here to crop or measure" in qc.note


def test_a_single_object_field_raises_no_spurious_size_outlier():
    """One object cannot be an outlier: n is too small for robust statistics."""
    mask = _disc(np.zeros((256, 256), np.int32), 128, 128, 10, 1)
    qc = score_field(mask, "cell", "single_object")
    assert qc.n_objects == 1
    assert FLAG_OUTLIERS not in qc.flags
    assert FLAG_OVER not in qc.flags
    assert FLAG_UNDER not in qc.flags
    assert qc.flags == [FLAG_NEAR_EMPTY]         # the honest complaint
    assert np.isnan(qc.metrics["outlier_fraction"])


def test_mad_outliers_catch_debris_that_defeats_mean_and_standard_deviation():
    """The whole reason the range is median +/- k*MAD and not mean +/- k*std.

    Twenty real objects plus five big pieces of debris. Debris inflates the
    standard deviation so much that the same k rejects nothing, while the
    median and MAD barely move and the debris lands far outside.
    """
    labels = np.zeros((1024, 1024), np.int32)
    value = 0
    rng = np.random.default_rng(3)
    for cy in (100, 250, 400, 550):                    # 4 rows x 5 columns = 20
        for cx in (100, 250, 400, 550, 700):
            value += 1
            _disc(labels, cy, cx, 10 + int(rng.integers(-1, 2)), value)
    for cx in (120, 300, 480, 660, 840):               # 5 lumps of debris
        value += 1
        _disc(labels, 900, cx, 50, value)

    qc = score_field(labels, "cell", "debris_field")
    assert qc.n_objects == 25
    assert FLAG_OUTLIERS in qc.flags
    assert qc.metrics["outlier_fraction"] == pytest.approx(5 / 25)

    # The same field, judged the naive way.
    areas = np.bincount(labels.ravel())[1:]
    diameters = 2.0 * np.sqrt(areas[areas > 0] / np.pi)
    k = QC_DEFAULTS["outlier_mad"]
    mean_std_outliers = np.abs(diameters - diameters.mean()) > k * diameters.std()
    assert not mean_std_outliers.any(), (
        "mean/std flagged something; the contrast this test exists to make is gone"
    )
    median = np.median(diameters)
    mad_outliers = np.abs(diameters - median) > k * 1.4826 * np.median(
        np.abs(diameters - median)
    )
    assert mad_outliers.sum() == 5


def test_border_objects_are_excluded_from_the_size_statistics():
    """A truncated object's area is a lie, so it may not fake a size outlier.

    Thirty whole discs and eight half-discs cut by the edge. Counting the
    halves as objects would put 21% of the field outside the robust range and
    raise a size_outliers flag that means nothing.
    """
    labels = np.zeros((600, 600), np.int32)
    value = 0
    for cy in range(100, 550, 80):
        for cx in range(100, 550, 80):
            if value >= 30:
                break
            value += 1
            _disc(labels, cy, cx, 12, value)
    for k, cx in enumerate(range(40, 560, 65)):
        if k >= 8:
            break
        value += 1
        _disc(labels, 0, cx, 12, value)

    qc = score_field(labels, "cell", "edge_cut_field")
    assert qc.n_objects == 38
    assert qc.metrics["n_interior"] == 30.0
    assert qc.metrics["outlier_fraction"] == 0.0
    assert FLAG_OUTLIERS not in qc.flags
    assert FLAG_BORDER not in qc.flags          # 21% is below the 30% threshold


@pytest.mark.parametrize("caster", [lambda m: m.astype(bool), lambda m: m.astype(np.float32)])
def test_boolean_and_float_masks_are_understood(caster):
    """Masks arrive as bools from thresholding and as floats from a resize."""
    mask = _grid_field(jitter=1)
    qc = score_field(caster(mask), "cell", "cast_field")
    assert qc.n_objects == int(mask.max())
    assert qc.flags == []


def test_a_mask_that_is_not_a_label_image_is_reported_not_raised():
    """A 3-D array is a caller error, and it comes back as a flag."""
    qc = score_field(np.zeros((3, 8, 8), np.uint16), "cell", "wrong_shape")
    assert qc.flags == [FLAG_UNREADABLE]
    assert qc.severity == "fail"
    assert "2-D label image" in qc.note


# ---------------------------------------------------------------------------
# plate context
# ---------------------------------------------------------------------------

def _healthy_plate(n=6, **kwargs):
    return {f"plate1_A0{i + 1}_f01": _grid_field(jitter=2, seed=i, **kwargs) for i in range(n)}


def test_the_plate_rollup_names_the_bad_fields():
    """Six good fields, one nearly empty, one shattered."""
    fields = _healthy_plate()
    fields["plate1_B01_f01"] = _disc(np.zeros((512, 512), np.int32), 256, 256, 10, 1)
    fields["plate1_B02_f01"] = _shattered_field()

    qcs = score_masks(fields, "cell")
    summary = summarize_qc(qcs)

    assert summary["n_fields"] == 8
    assert set(summary["failing_fields"]) == {"plate1_B01_f01", "plate1_B02_f01"}
    assert summary["n_ok"] == 6
    assert summary["verdict"] == "fail"
    assert summary["median_objects_per_field"] == pytest.approx(81.0)
    by_field = {q.field: q for q in qcs}
    assert FLAG_LOW_COUNT in by_field["plate1_B01_f01"].flags
    assert FLAG_HIGH_COUNT in by_field["plate1_B02_f01"].flags
    assert FLAG_OVER in by_field["plate1_B02_f01"].flags


def test_a_field_with_three_objects_where_the_plate_median_is_hundreds_fails():
    """The user's own example: 3 objects against a median of 300 is not a data point."""
    fields = _healthy_plate()
    bad = np.zeros((512, 512), np.int32)
    for k, cx in enumerate((100, 250, 400)):
        _disc(bad, 256, cx, 10, k + 1)
    fields["plate1_C01_f01"] = bad

    by_field = {q.field: q for q in score_masks(fields, "cell")}
    qc = by_field["plate1_C01_f01"]
    assert FLAG_LOW_COUNT in qc.flags
    assert qc.severity == "fail"
    assert qc.metrics["count_ratio"] < QC_DEFAULTS["count_ratio"]
    assert "plate median" in qc.note


def test_plate_relative_fusion_needs_both_fewer_and_bigger_objects():
    """Half the count at 1.4x the diameter is the fused-pair signature.

    This field is not dense, so the per-field distance-transform check cannot
    see it; only the comparison against the rest of the plate can.
    """
    fields = _healthy_plate()
    fused = np.zeros((512, 512), np.int32)
    value = 0
    for cy in range(40, 470, 100):
        for cx in range(40, 470, 50):
            value += 1
            _disc(fused, cy, cx, 15, value)
    fields["plate1_D01_f01"] = fused

    by_field = {q.field: q for q in score_masks(fields, "cell")}
    qc = by_field["plate1_D01_f01"]
    assert qc.metrics["foreground_fraction"] < QC_DEFAULTS["foreground_fraction"]
    assert qc.metrics["diameter_ratio"] >= QC_DEFAULTS["size_ratio"]
    assert qc.metrics["count_ratio"] <= 1 / QC_DEFAULTS["size_ratio"]
    assert FLAG_UNDER in qc.flags
    # and the healthy fields are still healthy
    assert all(q.severity == "ok" for f, q in by_field.items() if f != "plate1_D01_f01")


def test_a_sparse_plate_demotes_its_empty_fields_to_warnings():
    """With a plate median of two pathogens per field, an empty field is the assay."""
    fields = {}
    for i in range(6):
        mask = np.zeros((256, 256), np.int32)
        _disc(mask, 80, 40 + 30 * i, 6, 1)
        _disc(mask, 170, 60 + 25 * i, 6, 2)
        fields[f"sparse_{i}"] = mask
    fields["sparse_empty"] = np.zeros((256, 256), np.int32)

    by_field = {q.field: q for q in score_masks(fields, "pathogen")}
    empty = by_field["sparse_empty"]
    assert FLAG_EMPTY in empty.flags
    assert empty.severity == "warn"
    assert "the assay" in empty.note
    assert summarize_qc(list(by_field.values()))["verdict"] == "warn"


def test_plate_context_is_not_applied_to_one_or_two_fields():
    """Two fields cannot have a median that means anything, so none is claimed."""
    fields = {"a": _grid_field(jitter=2), "b": _shattered_field()}
    qcs = score_masks(fields, "cell")
    assert all(np.isnan(q.metrics["count_ratio"]) for q in qcs)
    # the per-field verdict still stands on its own evidence
    assert FLAG_OVER in {f for q in qcs if q.field == "b" for f in q.flags}


def test_an_unreadable_mask_does_not_cost_the_plate_its_scorecard(tmp_path):
    """One corrupt .npy is one bad field, not a lost report."""
    folder = _write_plate(tmp_path, _healthy_plate(n=4))
    (Path(folder) / "corrupt.npy").write_bytes(b"not a numpy file at all")

    qcs = score_masks(folder, "cell")
    by_field = {q.field: q for q in qcs}
    assert by_field["corrupt"].flags == [FLAG_UNREADABLE]
    assert by_field["corrupt"].severity == "fail"
    assert len(qcs) == 5
    # the unreadable field must not drag the plate median down to zero
    assert summarize_qc(qcs)["median_objects_per_field"] > 10


def test_score_masks_reads_a_folder_of_npy_and_keeps_the_field_names(tmp_path):
    folder = _write_plate(tmp_path, _healthy_plate(n=3))
    qcs = score_masks(folder, "cell")
    assert [q.field for q in qcs] == ["plate1_A01_f01", "plate1_A02_f01", "plate1_A03_f01"]
    assert all(q.object_type == "cell" for q in qcs)


def test_score_masks_accepts_a_stack_and_an_empty_folder(tmp_path):
    stack = np.stack([_grid_field(jitter=2, seed=i) for i in range(3)])
    qcs = score_masks(stack, "nucleus")
    assert [q.field for q in qcs] == ["field_0000", "field_0001", "field_0002"]

    empty_folder = tmp_path / "no_masks"
    empty_folder.mkdir()
    assert score_masks(str(empty_folder), "cell") == []


def test_summarize_qc_says_so_when_nothing_was_scored():
    summary = summarize_qc([])
    assert summary["verdict"] == "empty"
    assert summary["n_fields"] == 0
    assert "nothing to say" in summary["message"]


# ---------------------------------------------------------------------------
# the card and the CSV
# ---------------------------------------------------------------------------

def _plate_with_two_bad_fields():
    fields = _healthy_plate()
    fields["plate1_B01_f01"] = _shattered_field()
    fields["plate1_B02_f01"] = np.zeros((512, 512), np.int32)
    return fields


def test_the_csv_has_one_row_per_field_and_the_numbers_behind_the_flags(tmp_path):
    qcs = score_masks(_plate_with_two_bad_fields(), "cell")
    path = write_scorecard(qcs, str(tmp_path), "cell")

    assert path == os.path.join(str(tmp_path), "qc", "segmentation_qc_cell.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(qcs) == 8
    assert [r["field"] for r in rows] == [q.field for q in qcs]
    for column in ("n_objects", "severity", "flags", "note",
                   "border_fraction", "foreground_fraction", "median_diameter"):
        assert column in rows[0]
    with open(path, newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert len(header) == len(set(header)), f"duplicate CSV column: {header}"
    bad = next(r for r in rows if r["field"] == "plate1_B01_f01")
    assert FLAG_OVER in bad["flags"]
    assert bad["severity"] == "fail"
    assert float(bad["n_objects"]) > 0
    empty = next(r for r in rows if r["field"] == "plate1_B02_f01")
    assert empty["n_objects"] == "0"
    assert empty["flags"] == FLAG_EMPTY


def test_format_scorecard_names_the_failing_fields():
    qcs = score_masks(_plate_with_two_bad_fields(), "cell")
    card = format_scorecard(qcs)

    assert "failing fields: " in card
    assert "plate1_B01_f01" in card
    assert "plate1_B02_f01" in card
    assert "FAIL" in card
    # clean fields are counted, not listed: a 1536-field plate must not scroll
    assert "plate1_A01_f01" not in card
    assert "6 ok" in card


def test_format_scorecard_survives_an_empty_plate_and_a_clean_one():
    assert "nothing scored" in format_scorecard([])
    clean = format_scorecard(score_masks(_healthy_plate(), "cell"))
    assert "OK" in clean
    assert "failing fields" not in clean


def test_format_scorecard_stops_listing_after_max_rows():
    """A plate where everything failed must not scroll the terminal."""
    fields = {
        f"bad_{i:03d}": _shattered_field(shape=(160, 160), radius=2, spacing=20, margin=20)
        for i in range(30)
    }
    card = format_scorecard(score_masks(fields, "cell"), max_rows=5)
    assert "and 25 more flagged field(s)" in card
    assert "+25 more" in card


# ---------------------------------------------------------------------------
# the entry point spacr.object calls
# ---------------------------------------------------------------------------

def test_seg_qc_off_does_no_work_and_writes_nothing(tmp_path, monkeypatch):
    """Off means off: no mask is opened, nothing is printed, nothing is written."""
    import spacr.seg_qc as Q

    folder = _write_plate(tmp_path, _plate_with_two_bad_fields())
    monkeypatch.setattr(
        Q, "score_masks", lambda *a, **k: pytest.fail("seg_qc='off' scored a field")
    )
    printed = []
    assert run_segmentation_qc(folder, "cell", str(tmp_path), mode="off",
                               print_fn=printed.append) is None
    assert printed == []
    assert not (tmp_path / "qc").exists()


def test_report_mode_writes_the_card_and_prints_it(tmp_path):
    folder = _write_plate(tmp_path, _plate_with_two_bad_fields())
    printed = []
    result = run_segmentation_qc(folder, "cell", str(tmp_path), mode="report",
                                 print_fn=printed.append)

    assert result["csv_path"] == str(tmp_path / "qc" / "segmentation_qc_cell.csv")
    assert os.path.exists(result["csv_path"])
    assert result["summary"]["verdict"] == "fail"
    assert result["flags_path"] is None
    assert not (tmp_path / "qc" / "segmentation_qc_cell_flags.json").exists()
    assert any("failing fields" in line for line in printed)


def test_flag_mode_records_the_per_field_flags_for_downstream_use(tmp_path):
    folder = _write_plate(tmp_path, _plate_with_two_bad_fields())
    result = run_segmentation_qc(folder, "cell", str(tmp_path), mode="flag",
                                 print_fn=lambda *_: None)

    with open(result["flags_path"], encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["object_type"] == "cell"
    assert payload["verdict"] == "fail"
    assert FLAG_OVER in payload["fields"]["plate1_B01_f01"]
    assert result["flags"]["plate1_B02_f01"] == [FLAG_EMPTY]
    # 'flag' still changes nothing on disk beyond the report
    assert sorted(os.listdir(folder)) == sorted(f"{n}.npy" for n in _plate_with_two_bad_fields())


def test_a_quiet_run_still_prints_the_verdict(tmp_path):
    folder = _write_plate(tmp_path, _plate_with_two_bad_fields())
    printed = []
    run_segmentation_qc(folder, "cell", str(tmp_path), mode="report", verbose=False,
                        print_fn=printed.append)
    assert len(printed) == 1
    assert "FAIL" in printed[0]


def test_an_empty_mask_folder_is_reported_not_written(tmp_path):
    folder = tmp_path / "cell_mask_stack"
    folder.mkdir()
    printed = []
    result = run_segmentation_qc(str(folder), "cell", str(tmp_path), print_fn=printed.append)
    assert result["field_qcs"] == []
    assert result["csv_path"] is None
    assert not (tmp_path / "qc").exists()
    assert "no mask found" in printed[0]


# ---------------------------------------------------------------------------
# thresholds are settings, not magic numbers
# ---------------------------------------------------------------------------

def test_every_threshold_is_a_documented_setting_with_a_default_and_a_tooltip():
    from spacr.settings import expected_types, set_default_settings_preprocess_generate_masks, tooltips

    defaults = set_default_settings_preprocess_generate_masks({})
    for key, name in SETTING_KEYS.items():
        assert name in QC_DEFAULTS
        assert key in defaults, f"{key} has no default"
        assert defaults[key] == pytest.approx(QC_DEFAULTS[name]), f"{key} disagrees with QC_DEFAULTS"
        assert key in expected_types, f"{key} has no declared type"
        assert key in tooltips, f"{key} has no tooltip"
    assert defaults["seg_qc"] == "report"
    assert "seg_qc" in tooltips and "seg_qc" in expected_types


def test_thresholds_from_settings_reads_the_seg_qc_knobs_and_ignores_the_rest():
    th = thresholds_from_settings({
        "seg_qc_border_fraction": 0.9,
        "seg_qc_min_objects": "4",          # settings CSVs hand back strings
        "seg_qc_outlier_mad": None,         # unset stays default
        "seg_qc": "report",                 # the mode is not a threshold
        "cell_diameter": 30,
    })
    assert th == {"border_fraction": 0.9, "min_objects": 4.0}


def test_raising_a_threshold_silences_the_flag_it_governs():
    """The knobs are load-bearing: the same mask, two verdicts."""
    mask = _border_field()
    assert FLAG_BORDER in score_field(mask, "cell", "f").flags
    assert FLAG_BORDER not in score_field(mask, "cell", "f", border_fraction=0.9).flags


def test_a_misspelled_threshold_raises_instead_of_doing_nothing():
    with pytest.raises(TypeError, match="unknown segmentation-QC threshold"):
        score_field(np.zeros((8, 8), np.uint16), "cell", "f", boarder_fraction=0.5)


@pytest.mark.parametrize("raw,expected", [
    (None, "off"), (False, "off"), ("off", "off"), ("", "off"), ("False", "off"),
    (True, "report"), ("report", "report"), ("REPORT", "report"), ("nonsense", "report"),
    ("flag", "flag"), (" Flag ", "flag"),
])
def test_qc_mode_normalises_what_a_settings_csv_round_trip_produces(raw, expected):
    assert qc_mode({"seg_qc": raw}) == expected
    assert qc_mode({}) == "report"


# ---------------------------------------------------------------------------
# the wiring into spacr.object
# ---------------------------------------------------------------------------

def test_object_writes_the_card_next_to_measurements(tmp_path):
    """spacr.object hands the plate folder, not the mask folder, as the destination."""
    import spacr.object as O

    plate = tmp_path / "plate"
    mask_src = plate / "masks"
    _write_plate(mask_src, _plate_with_two_bad_fields())
    settings = {"seg_qc": "report", "verbose": False}

    result = O._run_seg_qc(str(mask_src), settings, "cell")
    assert result["csv_path"] == str(plate / "qc" / "segmentation_qc_cell.csv")
    assert os.path.exists(result["csv_path"])
    assert result["summary"]["n_fields"] == 8
    assert "seg_qc_flags" not in settings


def test_object_honours_seg_qc_off_and_flag(tmp_path):
    import spacr.object as O

    plate = tmp_path / "plate"
    mask_src = plate / "masks"
    _write_plate(mask_src, _plate_with_two_bad_fields())

    assert O._run_seg_qc(str(mask_src), {"seg_qc": "off"}, "cell") is None
    assert not (plate / "qc").exists()

    settings = {"seg_qc": "flag", "verbose": False}
    result = O._run_seg_qc(str(mask_src), settings, "cell")
    assert result["mode"] == "flag"
    assert settings["seg_qc_flags"]["cell"]["plate1_B02_f01"] == [FLAG_EMPTY]


def test_object_passes_the_seg_qc_thresholds_through(tmp_path):
    """A threshold set in settings has to reach the scorer."""
    import spacr.object as O

    plate = tmp_path / "plate"
    mask_src = plate / "masks"
    _write_plate(mask_src, {"only_field": _border_field()})

    strict = O._run_seg_qc(str(mask_src), {"seg_qc": "report", "verbose": False}, "cell")
    assert FLAG_BORDER in strict["field_qcs"][0].flags

    lax = O._run_seg_qc(
        str(mask_src),
        {"seg_qc": "report", "verbose": False, "seg_qc_border_fraction": 0.95},
        "cell",
    )
    assert FLAG_BORDER not in lax["field_qcs"][0].flags


def test_a_qc_failure_never_kills_a_segmentation_run(tmp_path, monkeypatch, capsys):
    """Hours of segmentation must not be lost to a scorecard bug."""
    import spacr.object as O
    import spacr.seg_qc as Q

    def boom(*args, **kwargs):
        raise RuntimeError("scorecard exploded")

    monkeypatch.setattr(Q, "run_segmentation_qc", boom)
    assert O._run_seg_qc(str(tmp_path), {"seg_qc": "report"}, "cell") is None
    assert "Segmentation QC skipped for cell" in capsys.readouterr().out


def test_the_mask_generators_all_call_the_scorecard():
    """Every path that writes masks has to be wired, not just the SAM one."""
    import inspect

    import spacr.object as O

    for func in (O.generate_cellpose_masks_sam,
                 O.generate_cellpose_masks,
                 O.generate_organelle_masks_sam):
        assert "_run_seg_qc(src, settings, object_type)" in inspect.getsource(func), func.__name__


# ---------------------------------------------------------------------------
# the cost guarantee: no torch, no cellpose
# ---------------------------------------------------------------------------

def test_a_call_does_not_pull_in_torch_or_cellpose(tmp_path):
    """In-process guard: scoring a plate must not *add* the model stack."""
    folder = _write_plate(tmp_path, _healthy_plate(n=3))
    before = {m.split(".")[0] for m in list(sys.modules)}
    run_segmentation_qc(folder, "cell", str(tmp_path), print_fn=lambda *_: None)
    after = {m.split(".")[0] for m in list(sys.modules)}
    added = (after - before) & {"torch", "torchvision", "cellpose", "tensorflow"}
    assert not added, f"segmentation QC imported {sorted(added)}"


def test_neither_torch_nor_cellpose_is_in_sys_modules_after_a_call(tmp_path):
    """The real guarantee, checked in a fresh interpreter.

    In-process this cannot be proven: the pytest session has already imported
    half of spaCR (this very file imports spacr.object), and the coverage
    runner pre-imports torch through a sitecustomize shim. So a clean
    interpreter is started with PYTHONPATH set to the repo alone and asked what
    it ended up with. The point is not purity: this module runs on every field
    of every plate, and torch plus cellpose cost seconds and hundreds of MB
    before a single mask has been read.
    """
    folder = _write_plate(tmp_path, _healthy_plate(n=3))
    np.save(Path(folder) / "fused.npy", _fused_field().astype(np.uint16))

    code = textwrap.dedent(
        """
        import sys
        from spacr.seg_qc import run_segmentation_qc

        result = run_segmentation_qc(sys.argv[1], "cell", sys.argv[2], mode="flag")
        assert result["summary"]["n_fields"] == 4, result["summary"]

        heavy = sorted({m.split(".")[0] for m in sys.modules}
                       & {"torch", "torchvision", "cellpose", "tensorflow"})
        print("HEAVY:" + ",".join(heavy))
        """
    )
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["MPLBACKEND"] = "Agg"
    env["QT_QPA_PLATFORM"] = "offscreen"

    proc = subprocess.run(
        [sys.executable, "-c", code, folder, str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    heavy_line = next(line for line in proc.stdout.splitlines() if line.startswith("HEAVY:"))
    assert heavy_line == "HEAVY:", f"heavy modules imported: {heavy_line}"


def test_the_dataclass_is_the_documented_shape():
    """The card's public contract: other code reads these six attributes."""
    qc = FieldQC(field="f", object_type="cell", n_objects=3)
    assert qc.flags == [] and qc.metrics == {}
    assert qc.severity == "ok" and qc.note == ""
    assert not qc.failed
    assert "3 objects" in str(qc)
    qc.flags.append(FLAG_EMPTY)
    qc.severity = "fail"
    assert qc.failed
    assert "empty_field" in str(qc)
