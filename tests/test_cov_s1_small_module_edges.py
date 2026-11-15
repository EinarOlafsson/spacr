"""Edges of the small readers: a value that arrived in an unexpected shape.

Each case here is a frame, a settings dict or a plate address the happy path
does not describe -- a channel list typed as words, a stack file that has
already been deleted, a well with no row letter, a measurement whose scale overflows. None of them is an error;
every one of them used to be answered with the wrong number or an exception.
"""
from __future__ import annotations


import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _v1_v2_bridge -- settings and files that do not read
# ---------------------------------------------------------------------------

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
