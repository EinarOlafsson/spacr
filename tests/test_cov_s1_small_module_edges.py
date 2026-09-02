"""Edges of the small readers: a value that arrived in an unexpected shape.

Each case here is a frame, a settings dict or a plate address the happy path
does not describe -- a crop table written for a different crop mode, a channel
list typed as words, a stack file that has already been deleted, a well with
no row letter, a measurement whose scale overflows. None of them is an error;
every one of them used to be answered with the wrong number or an exception.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# png_list -- a crop table written for another crop mode
# ---------------------------------------------------------------------------

def test_a_crop_table_from_another_mode_still_finds_its_object_column(
        tmp_path):
    """``png_list`` carries only the id column of the mode that wrote it.

    Asking for ``cell_id`` on a nucleus crop table and finding nothing would
    drop every row -- silently, because an empty result is a valid one.
    """
    from spacr.png_list import crop_rows_from_png_list

    frame = pd.DataFrame({
        "nucleus_id": ["o5", "o7", "omulti"],
        "path_name": ["/merged/f1.npy"] * 3,
        "png_path": ["/crops/a.png", "/crops/b.png", "/crops/c.png"],
    })

    out = crop_rows_from_png_list(str(tmp_path / "measurements.db"), frame,
                                  object_type="cell", verbose=False)

    assert list(out["object_label"]) == [5, 7]
    assert list(out["object_type"]) == ["cell", "cell"]
    assert len(out) == 2


# ---------------------------------------------------------------------------
# _v1_v2_bridge -- settings and files that do not read
# ---------------------------------------------------------------------------

def test_a_channel_list_of_words_falls_through_to_the_default_plate():
    """``channels`` is a hand-edited list. One unparseable entry costs that
    entry; a list of nothing usable costs the list, not the run."""
    from spacr._v1_v2_bridge import v2_channels_from_settings

    channels, names = v2_channels_from_settings({"channels": ["red", 2]})
    assert channels == [2]
    assert names == ["ch1"]

    channels, names = v2_channels_from_settings({"channels": ["red", "green"]})
    assert channels == [0, 1, 2, 3]
    assert names == ["ch0", "ch1", "ch2", "ch3"]


def test_the_disk_report_survives_files_that_are_no_longer_there(tmp_path,
                                                                 caplog):
    """A saving reported as 12,288 bytes is not a saving anybody can weigh,
    and a stack deleted between the run and the report is not a reason to
    lose the whole figure -- it is one file's bytes."""
    from spacr._v1_v2_bridge import report_disk_savings

    stack = tmp_path / "merged" / "plate1.npy"
    stack.parent.mkdir()
    stack.write_bytes(b"\x00" * 4096)

    class _Stack:
        def __init__(self, path):
            self.path = str(path)

    stacks = [_Stack(stack), _Stack(tmp_path / "merged" / "deleted.npy")]
    with caplog.at_level(logging.INFO, logger="spacr.pipeline_v2.bridge"):
        report = report_disk_savings(tmp_path, stacks)

    assert report["v2_bytes"] == 4096
    assert report["v1_estimated_bytes"] == 4 * 4096
    assert report["saved_pct"] == 75.0
    logged = caplog.text
    assert "4.10 KB" in logged
    assert "16.38 KB" in logged
    assert "12.29 KB" in logged


def test_a_sidecar_on_a_mount_that_will_not_answer_is_not_a_crashed_report(
        tmp_path, monkeypatch):
    """The sidecars are looked for beside the plate root, which is routinely
    a network share. A stale handle there raises instead of answering, and
    this is a log line at the end of a finished run -- it may not take the
    run with it over three files that are only worth a few kilobytes."""
    from pathlib import Path

    from spacr._v1_v2_bridge import report_disk_savings

    stack = tmp_path / "plate1.npy"
    stack.write_bytes(b"\x00" * 2048)
    (tmp_path / "filename_map.csv").write_text("a,b\n")
    real_exists = Path.exists

    def _stale(self, *args, **kwargs):
        if self.name == "filename_map.csv":
            raise OSError(116, "Stale file handle")
        return real_exists(self, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", _stale)

    class _Stack:
        path = str(stack)

    report = report_disk_savings(tmp_path, [_Stack()])

    assert report["v2_bytes"] == 2048
    assert report["saved_bytes"] == 3 * 2048


# ---------------------------------------------------------------------------
# organelle_types -- a key that belongs to no slot
# ---------------------------------------------------------------------------

def test_a_key_outside_the_slots_comes_back_unchanged():
    """A caller runs a whole settings dict through the translator, so a key
    that names no slot has to survive the trip rather than be mangled into
    an ``organelle_`` spelling that nothing reads."""
    from spacr.organelle_types import primary_setting

    assert primary_setting("cell_cellprob_threshold") == "cell_cellprob_threshold"
    assert primary_setting("src") == "src"
    assert primary_setting("") == ""


# ---------------------------------------------------------------------------
# seg_qc -- an address with no row letter
# ---------------------------------------------------------------------------

def test_a_plate_gradient_is_still_found_when_the_rows_cannot_be_read(
        monkeypatch):
    """Half an address is not a reason to abandon the other half.

    A field whose well parses but whose row letter does not is simply not a
    member of any row group. Counting it under a blank key would invent a
    row that holds every field on the plate and hide the real column step.
    """
    from spacr import seg_qc

    def _no_row(name):
        stem = str(name).split("_")
        return seg_qc.FieldAddress(field=str(name), plate=stem[0],
                                   well=stem[1], row="",
                                   column=int(stem[1][1:]))

    monkeypatch.setattr(seg_qc, "parse_field_name", _no_row)
    fields = [
        seg_qc.FieldQC(field=f"plate1_A{column:02d}_f1", object_type="cell",
                       n_objects=10 if column <= 4 else 40,
                       metrics={"median_diameter": 20.0})
        for column in range(1, 9)
    ]

    findings = seg_qc._gradient_findings(fields, seg_qc.GRADIENT_RATIO,
                                         seg_qc.MIN_FIELDS_PER_HALF)

    assert [f.kind for f in findings] == ["count_gradient"]
    assert "columns 5-8 hold 4.0x" in findings[0].headline
    assert "rows" not in findings[0].headline


# ---------------------------------------------------------------------------
# guide_attribution -- a measurement whose scale overflows
# ---------------------------------------------------------------------------

def test_a_measurement_that_overflows_its_own_scale_counts_as_no_dimension():
    """The effective dimension divides by the columns' spread.

    A column held in units so large that its variance overflows to infinity
    standardises to all zeros, and the correlation matrix it produces has no
    eigenvalues at all. Returning ``total**2 / 0`` there would put a NaN into
    the scale factor that every posterior is multiplied by.
    """
    from spacr.guide_attribution import effective_dimension

    matrix = np.array([[1e200, 1e200], [-1e200, -1e200]], dtype=float)

    with np.errstate(over="ignore", invalid="ignore"):
        assert effective_dimension(matrix) == 0.0


def test_the_effective_dimension_counts_copies_of_a_column_once():
    from spacr.guide_attribution import effective_dimension

    rng = np.random.default_rng(0)
    column = rng.normal(size=(200, 1))
    copies = np.hstack([column] * 5)
    independent = rng.normal(size=(200, 5))

    assert effective_dimension(copies) == pytest.approx(1.0, abs=1e-6)
    assert effective_dimension(independent) > 3.0
