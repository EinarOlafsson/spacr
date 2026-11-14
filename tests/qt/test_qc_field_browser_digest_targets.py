"""Which fields the QC field browser opens for a segmentation-QC digest.

``targets_from_digest`` turns the in-memory :class:`spacr.seg_qc.QCDigest`
into one browser target per field, and ``finding_targets`` narrows that to
the fields one banner finding implicates. Both must survive the shapes old
and hand-built digests take: a scorecard with no file path (the plate root
then comes from ``digest.root``), field names still carrying ``.npy``,
blank names from a truncated row, and a field-name parser that cannot be
used (plate matching then falls back to the ``plate_`` name prefix).
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

from spacr import seg_qc  # noqa: E402
from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldVerdict,
    finding_targets,
    targets_from_digest,
)

pytestmark = pytest.mark.qt

FIELD = "plate1_A01_1"


def _qc(field, object_type="cell", severity="fail", flags=("under_segmented",),
        note=""):
    return seg_qc.FieldQC(field, object_type, 1, list(flags), {}, severity,
                          note)


def _digest(root, cards, findings=()):
    return seg_qc.QCDigest(root=str(root), verdict="fail", headline="review",
                           scorecards=list(cards), findings=list(findings))


def _summary(targets):
    """(field, plate_root, merged_dir, verdict object types) per target."""
    return [(t.field, t.plate_root, t.merged_dir,
             [v.object_type for v in t.verdicts]) for t in targets]


def test_a_pathless_digest_still_reaches_the_real_merged_folder(subtests,
                                                                tmp_path):
    """targets_from_digest: digest.root fallback, .npy suffix, blank names, grouping."""
    root = os.path.abspath(str(tmp_path / "plate1"))
    merged = os.path.join(root, "merged")
    cases = [
        # (id, cards, expected summary)
        ("a suffixed name and a blank row on a pathless card",
         [seg_qc.Scorecard("", "cell", [_qc("   "), _qc(f"{FIELD}.NPY"),
                                        _qc("plate1_A02_1")])],
         [(FIELD, root, merged, ["cell"]),
          ("plate1_A02_1", root, merged, ["cell"])]),
        ("one field scored for two object types is one target",
         [seg_qc.Scorecard("", "cell", [_qc("   "), _qc(f"{FIELD}.npy"),
                                        _qc(FIELD, object_type="nucleus")])],
         [(FIELD, root, merged, ["cell", "nucleus"])]),
        ("a clean field with no flags is not a target",
         [seg_qc.Scorecard("", "cell", [_qc(FIELD, severity="ok", flags=()),
                                        _qc("plate1_A02_1")])],
         [("plate1_A02_1", root, merged, ["cell"])]),
    ]
    for case_id, cards, expected in cases:
        with subtests.test(msg=case_id):
            targets = targets_from_digest(_digest(tmp_path / "plate1", cards))
            assert _summary(targets) == expected

    with subtests.test(msg="a card with a path roots its fields beside it"):
        card_path = tmp_path / "plateX" / seg_qc.CARD_DIR / "card.csv"
        card = seg_qc.Scorecard(str(card_path), "cell", [_qc(FIELD, note="n")])
        [target] = targets_from_digest(_digest(tmp_path / "elsewhere", [card]))
        plate_x = os.path.abspath(str(tmp_path / "plateX"))
        assert (target.plate_root, target.merged_dir) == (
            plate_x, os.path.join(plate_x, "merged"))
        assert target.verdicts == (
            QCFieldVerdict("cell", "fail", ("under_segmented",), "n"),)


def test_plate_matching_falls_back_to_the_name_prefix(subtests, tmp_path,
                                                      monkeypatch):
    """finding_targets: plate filter with and without the field-name parser."""
    cards = [seg_qc.Scorecard("", "cell", [_qc(FIELD)]),
             seg_qc.Scorecard("", "cell", [_qc("plate2_B01_1")])]
    positional = seg_qc.Finding(severity="warn", kind="count_gradient",
                                headline="plate 1 drifts", plate="plate1")
    digest = _digest(tmp_path / "plate1", cards, [positional])

    def _boom(_name):
        raise RuntimeError("field-name parser unavailable")

    for case_id, broken in (("with the strict parser", False),
                            ("with the parser unavailable", True)):
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            if broken:
                patch.setattr(seg_qc, "parse_field_name", _boom)
            targets = finding_targets(digest, positional)
            assert [t.field for t in targets] == [FIELD]
            assert targets[0].reasons == ("count_gradient",)
            assert targets[0].finding_texts == ("plate 1 drifts",)
