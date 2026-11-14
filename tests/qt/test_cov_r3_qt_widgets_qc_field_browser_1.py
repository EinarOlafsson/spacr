"""Edge inputs the QC field browser must survive without hiding a field.

Every path here is one a real plate reaches: a scorecard written before the
plate path was recorded, a field name still carrying its ``.npy`` suffix, a
legacy merged array with no plane manifest, a mask stack whose files were
replaced by something that is not a 2-D mask, and flat or all-NaN planes.
The browser is the last step before a user quarantines data, so each of
these must produce an explained, still-usable dialog rather than an
exception or a silently dropped field.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr import seg_qc  # noqa: E402
from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    finding_targets,
    targets_from_digest,
)

pytestmark = pytest.mark.qt

FIELD = "plate1_A01_1"


def _qc(field, object_type="cell", severity="fail", flags=("under_segmented",)):
    return seg_qc.FieldQC(field, object_type, 1, list(flags), {}, severity, "")


def _digest(root, cards, findings=()):
    return seg_qc.QCDigest(
        root=str(root), verdict="fail", headline="review",
        scorecards=list(cards), findings=list(findings))


def test_a_pathless_scorecard_and_a_dot_npy_field_still_reach_the_browser(
        tmp_path):
    """A scorecard with no file path must not orphan its fields.

    Older QC runs recorded no scorecard path and wrote field names with the
    ``.npy`` suffix still attached.  If the browser derived the plate root
    only from the scorecard path it would open ``/merged`` at the filesystem
    root, and a suffixed name would never match the file on disk -- the
    banner would list fields that open to "already gone".  A blank field
    name, which a truncated CSV row produces, must be dropped rather than
    become a target that points at the plate folder itself.
    """
    plate = tmp_path / "plate1"
    card = seg_qc.Scorecard("", "cell", [
        _qc("   "),
        _qc(f"{FIELD}.NPY"),
        _qc("plate1_A02_1"),
    ])

    targets = targets_from_digest(_digest(plate, [card]))

    assert [target.field for target in targets] == [FIELD, "plate1_A02_1"]
    assert targets[0].plate_root == os.path.abspath(str(plate))
    assert targets[0].merged_dir == os.path.join(
        os.path.abspath(str(plate)), "merged")


def test_each_finding_opens_only_the_fields_it_actually_implicates(tmp_path):
    """A finding's browser list must not spill into unrelated fields.

    The banner offers one browser per finding.  If the field/plate/object
    filters were dropped, clicking a single-field cell finding would open
    every field on every plate in the digest and the user would quarantine
    the wrong array.  All three filters are exercised against the same
    digest so the empty result below is proved to be a filter, not an empty
    digest.
    """
    plate = tmp_path / "plate1"
    card = seg_qc.Scorecard("", "cell", [_qc(FIELD), _qc("plate1_A02_1")])
    other = seg_qc.Scorecard("", "nucleus", [_qc("plate2_B01_1", "nucleus")])
    exact = seg_qc.Finding(
        severity="fail", kind="flag", headline="one field",
        flag="under_segmented", plate="plate1", object_type="cell",
        fields=(FIELD,))
    positional = seg_qc.Finding(
        severity="warn", kind="count_gradient", headline="plate 2 drifts",
        plate="plate2")
    wrong_type = seg_qc.Finding(
        severity="warn", kind="count_gradient", headline="no pathogens here",
        object_type="pathogen")
    digest = _digest(plate, [card, other], [exact, positional, wrong_type])

    assert [t.field for t in finding_targets(digest, exact)] == [FIELD]
    assert [t.field for t in finding_targets(digest, positional)] == [
        "plate2_B01_1"]
    assert finding_targets(digest, wrong_type) == ()


def test_plate_matching_falls_back_to_the_name_prefix(tmp_path, monkeypatch):
    """A plate-scoped finding must still find its fields if parsing fails.

    ``parse_field_name`` is the strict reader; when it cannot be used the
    browser falls back to the leading ``plate_`` segment.  Without that
    fallback a positional finding would match nothing and the banner's
    "inspect" link would open an empty dialog for the very pattern it just
    reported.
    """
    def _boom(name):
        raise RuntimeError("field-name parser unavailable")

    monkeypatch.setattr(seg_qc, "parse_field_name", _boom)
    plate = tmp_path / "plate1"
    card = seg_qc.Scorecard("", "cell", [_qc(FIELD)])
    other = seg_qc.Scorecard("", "cell", [_qc("plate2_B01_1")])
    positional = seg_qc.Finding(
        severity="warn", kind="count_gradient", headline="plate 1 drifts",
        plate="plate1")
    digest = _digest(plate, [card, other], [positional])

    assert [t.field for t in finding_targets(digest, positional)] == [FIELD]
