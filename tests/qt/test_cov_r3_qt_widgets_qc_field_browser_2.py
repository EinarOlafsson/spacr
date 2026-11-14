"""Edge conditions of the QC field browser: grouping, legacy arrays, blank canvases.

Each test drives a narrow path the triage dialog reaches when a plate is not
textbook -- a scorecard with no file, a merged array written before the plane
manifest existed, a mask stack whose shape no longer matches, a flat channel,
an empty target list -- where the browser either explains itself or crashes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPixmap  # noqa: E402

from spacr import seg_qc  # noqa: E402
from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldBrowser,
    QCFieldTarget,
    QCFieldVerdict,
    _FieldView,
    finding_targets,
    load_qc_field,
    targets_from_digest,
)

from tests.qt.test_qc_field_browser import _digest  # noqa: E402

pytestmark = pytest.mark.qt

FIELD = "plate1_A01_1"


def _finding(**kwargs):
    base = dict(severity="fail", kind="flag", headline="Two fields need review.",
                flag="under_segmented", plate="", object_type="", fields=(),
                n_fields=1)
    base.update(kwargs)
    return seg_qc.Finding(**base)


def _legacy_plate(root: Path, planes: int, stacks=()) -> QCFieldTarget:
    """A merged array written before the plane manifest existed."""
    merged = root / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:16, :20]
    array = np.zeros((16, 20, planes), dtype=np.uint16)
    for plane in range(planes):
        array[..., plane] = (xx + yy + plane) * 7
    label = np.zeros((16, 20), dtype=np.uint16)
    label[4:10, 5:12] = 3
    for plane in (plane for plane in (4, 5) if planes > plane):
        array[..., plane] = label
    np.save(merged / f"{FIELD}.npy", array)
    for name in stacks:
        folder = root / "norm_channel_stack" / f"{name}_mask_stack"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{FIELD}.npy", label)
    return QCFieldTarget(
        field=FIELD, plate_root=str(root), merged_dir=str(merged))


def test_a_finding_only_claims_the_fields_and_plate_it_names(tmp_path):
    """Clicking one finding must not open every field on the bench.

    ``finding_targets`` keeps each banner link honest: a flag finding names
    its fields exactly, and a finding about another plate owns none of these.
    Losing either filter invites the user to quarantine an untouched array.
    """
    digest, _flag = _digest(tmp_path / "plate1")
    both = [target.field for target in targets_from_digest(digest)]

    named = finding_targets(digest, _finding(fields=(FIELD,)))
    elsewhere = finding_targets(digest, _finding(plate="plate9"))

    assert both == [FIELD, "plate1_A02_1"]
    assert [target.field for target in named] == [FIELD]
    assert elsewhere == ()


def test_an_object_type_and_a_clean_verdict_narrow_a_finding(tmp_path):
    """A pathogen finding must not open cell-only fields, and clean means clean.

    One filter drops fields never scored for the finding's object type; the
    other drops positional matches for a ``clean`` finding, which says nothing
    is wrong.  Without them "no problems found" still offers a quarantine.
    """
    digest, _flag = _digest(tmp_path / "plate1")

    cells = finding_targets(digest, _finding(object_type="cell"))
    pathogens = finding_targets(digest, _finding(object_type="pathogen"))
    gradient = finding_targets(digest, _finding(kind="count_gradient", flag=""))
    clean = finding_targets(digest, _finding(kind="clean", flag=""))

    assert [target.field for target in cells] == [FIELD, "plate1_A02_1"]
    assert pathogens == ()
    assert [target.field for target in gradient] == [FIELD, "plate1_A02_1"]
    assert clean == ()


def test_two_findings_saying_the_same_thing_are_recorded_once(tmp_path):
    """The verdict panel and the quarantine ledger must not stutter.

    Diagnose can raise the same flag twice for one plate, and the reason list
    is written verbatim into the quarantine sidecar, so a missing dedup puts
    ``cell:under_segmented`` in the audit record twice.
    """
    digest, _flag = _digest(tmp_path / "plate1")
    twice = _finding(fields=(FIELD,), object_type="cell")
    digest.findings = [twice, _finding(fields=(FIELD,), object_type="cell")]

    target = targets_from_digest(digest)[0]

    assert target.field == FIELD
    assert target.reasons == ("cell:under_segmented",)
    assert target.finding_texts == ("Two fields need review.",)
    assert target.audit_flags.count("cell:under_segmented") == 1


def test_a_merged_plane_substitutes_for_a_missing_mask_stack(tmp_path):
    """An archived mask stack must not cost the user the outlines.

    The merged array carries its own mask planes, so a plate whose stack
    folders were archived still shows outlines.  An object type with neither
    a stack nor a plane must say so, or an empty overlay reads as "fine here".
    """
    target = _legacy_plate(tmp_path / "plate1", 6)
    target = QCFieldTarget(
        field=target.field, plate_root=target.plate_root,
        merged_dir=target.merged_dir,
        verdicts=(QCFieldVerdict("cell", "fail", ("empty_field",)),
                  QCFieldVerdict("pathogen", "warn", ("tiny_objects",))))

    payload = load_qc_field(target)

    assert set(payload.masks) == {"cell", "nucleus"}
    assert int(payload.masks["cell"].max()) == 3
    assert any("pathogen" in warning for warning in payload.warnings)


def test_a_null_pixmap_leaves_the_canvas_empty_without_fitting_it(qtbot):
    """Clearing the image must not ask Qt to fit a zero-sized scene.

    A render that produced nothing still reaches the canvas.  Fitting the
    view to an empty rectangle leaves the transform undefined, so the next
    real image opens at an arbitrary zoom; the null case is skipped instead.
    """
    view = _FieldView()
    qtbot.addWidget(view)

    view.set_pixmap(QPixmap())
    assert view._item is not None
    assert view._item.pixmap().isNull() is True
    assert view.transform().m11() == pytest.approx(1.0)

    view.set_pixmap(QPixmap(40, 32))
    assert view._item.pixmap().isNull() is False
    assert view._scene.sceneRect().width() == pytest.approx(40.0)


def test_a_browser_with_no_targets_says_so_and_stays_inert(qtbot, tmp_path):
    """An empty digest must open a dialog that explains itself, not an empty one.

    ``finding_targets`` can legitimately return nothing.  The dialog still
    opens, so it must say there is nothing to browse and disable every
    control; showing a target it lacks must be a no-op, not an IndexError.
    """
    empty = QCFieldBrowser([], threaded=False)
    qtbot.addWidget(empty)
    digest, _flag = _digest(tmp_path / "plate1")
    loaded = QCFieldBrowser(targets_from_digest(digest), threaded=False)
    qtbot.addWidget(loaded)

    assert empty.current_target is None
    assert empty.current_field == ""
    assert "No flagged fields" in empty._field_title.text()
    assert not empty._quarantine.isEnabled()
    assert not empty._channel.isEnabled()

    empty._show_target()
    assert "No flagged fields" in empty._field_title.text()
    assert "flagged field 1 of 2" in loaded._field_title.text()


def test_the_verdict_panel_repeats_neither_a_finding_nor_an_empty_note(
        qtbot, tmp_path):
    """Each sentence in the verdict panel has to earn its line.

    A field carries one verdict row per object type plus every plate-level
    finding that implicated it, and the same headline can arrive twice.
    Repeating it turns a short summary into a wall the user stops reading.
    """
    headline = "Counts step across the plate."
    target = QCFieldTarget(
        field=FIELD, plate_root=str(tmp_path),
        merged_dir=str(tmp_path / "merged"),
        verdicts=(QCFieldVerdict("cell", "warn", ("tiny_objects",), ""),),
        finding_texts=("", headline, headline))

    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)

    lines = browser._verdict.text().splitlines()
    assert lines == ["[WARN] cell: tiny_objects",
                     f"Plate-level finding: {headline}"]
    assert "—" not in browser._verdict.text()


def test_clearing_the_layer_row_removes_spacers_as_well_as_checkboxes(
        qtbot, tmp_path):
    """Moving to the next field must leave no stale mask toggle behind.

    The layer row is rebuilt per field because a plate can have cell and
    nucleus here and only cell there.  It also holds non-widget layout items;
    a clear that only handled widgets leaves the checkbox count creeping up.
    """
    digest, _flag = _digest(tmp_path / "plate1")
    browser = QCFieldBrowser(targets_from_digest(digest), threaded=False)
    qtbot.addWidget(browser)
    assert set(browser._layer_checks) == {"cell", "nucleus"}
    browser._layers.addStretch(1)
    assert browser._layers.count() == 3

    browser._clear_layers()

    assert browser._layers.count() == 0
    assert browser._layer_checks == {}


# ---------------------------------------------------------------------------
# 352: the arms that run when a field, a render or a filesystem misbehaves.
# ---------------------------------------------------------------------------

def test_the_paths_for_no_target_are_empty_rather_than_invented():
    """A move with no field must not address one.

    `_paths_for` takes its target as an argument precisely because the
    completion handler of a move has to address the field the move was
    STARTED for -- the user may have been sent elsewhere since. With no
    target there is no such field, and returning a plausible path would
    let a completion handler act on somebody else's data.
    """
    from spacr.qt.widgets.qc_field_browser import QCFieldBrowser

    assert QCFieldBrowser._paths_for(None) == ("", "")


def test_the_quarantine_path_is_a_sibling_of_the_merged_folder(tmp_path):
    """Built by string joins, deliberately, and this pins the shape.

    Not `quarantine_dir_for`, which resolves the plate folder:
    `Path.resolve()` is a realpath walk over every component, and on one
    `/nas_mnt` autofs mount one component was enough to park the GUI
    thread. So the layout is asserted here rather than trusted to a
    resolver nobody may call.
    """
    from spacr.qt.widgets.qc_field_browser import (QCFieldTarget,
                                                   QUARANTINE_DIRNAME,
                                                   QCFieldBrowser)

    merged = tmp_path / "plate1" / "merged"
    target = QCFieldTarget(field="A01_f01", plate_root=str(tmp_path / "plate1"),
                           merged_dir=str(merged))
    active, quarantined = QCFieldBrowser._paths_for(target)
    assert active == os.path.join(str(merged), "A01_f01.npy")
    assert quarantined == os.path.join(str(tmp_path / "plate1"),
                                       QUARANTINE_DIRNAME, "A01_f01.npy")
    # The quarantine folder is a SIBLING of merged/, inside the plate.
    assert os.path.dirname(os.path.dirname(quarantined)) == str(
        tmp_path / "plate1")
