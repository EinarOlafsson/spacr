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
    QCFieldImage,
    QCFieldTarget,
    QCFieldVerdict,
    _FieldView,
    finding_targets,
    load_qc_field,
    render_qc_field,
    targets_from_digest,
)

from tests.qt.test_qc_field_browser import _digest, _write_field  # noqa: E402

pytestmark = pytest.mark.qt

FIELD = "plate1_A01_1"


def _qc(field, object_type="cell", flags=("under_segmented",), note="n"):
    return seg_qc.FieldQC(field, object_type, 1, list(flags), {}, "fail", note)


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


def test_a_scorecard_with_no_file_path_still_names_its_plate_root(tmp_path):
    """A digest built in memory must still point at a real merged folder.

    A digest assembled without files has only ``digest.root``; without that
    fallback the browser computes a ``merged_dir`` of ``""`` and every field
    reports itself as already gone.  The same pass normalises the ``.npy``
    suffix and drops nameless rows, so no field is listed under two spellings.
    """
    root = tmp_path / "project"
    root.mkdir()
    card = seg_qc.Scorecard("", "cell", [
        _qc("   "), _qc(f"{FIELD}.npy"), _qc(FIELD, object_type="nucleus")])
    digest = seg_qc.QCDigest(
        root=str(root), verdict="fail", headline="h", scorecards=[card])

    targets = targets_from_digest(digest)

    assert [target.field for target in targets] == [FIELD]
    assert targets[0].plate_root == os.path.abspath(str(root))
    assert targets[0].merged_dir == os.path.join(
        os.path.abspath(str(root)), "merged")
    assert [v.object_type for v in targets[0].verdicts] == ["cell", "nucleus"]


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


def test_a_legacy_merged_array_infers_how_many_channels_it_has(tmp_path):
    """Plates merged before the manifest existed still open with real channels.

    With no manifest the loader counts backwards from the mask stacks on disk
    and falls back to spaCR's four-channel default.  Get it wrong and a label
    plane is offered to the user as an intensity channel.
    """
    from_stacks = load_qc_field(_legacy_plate(
        tmp_path / "stacked", 5, stacks=("cell", "nucleus")))
    wide = load_qc_field(_legacy_plate(tmp_path / "wide", 6))
    narrow = load_qc_field(_legacy_plate(tmp_path / "narrow", 3))

    assert from_stacks.channel_names == ("1", "2", "3")
    assert from_stacks.intensities.shape == (16, 20, 3)
    assert set(from_stacks.masks) == {"cell", "nucleus"}
    assert wide.channel_names == ("1", "2", "3", "4")
    assert narrow.channel_names == ("1", "2", "3")
    assert narrow.intensities.shape == (16, 20, 3)


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


def test_an_unreadable_mask_stack_is_reported_per_object_type(tmp_path):
    """One broken stack must not take the whole field's overlays down.

    A trailing singleton axis is fine and gets squeezed, but a three-plane
    array or a mask from a differently-cropped run is not this field's mask.
    Each is named so the user knows which outline is missing.
    """
    plate = tmp_path / "plate1"
    _write_field(plate, FIELD)
    stacks = plate / "norm_channel_stack"
    cell = np.zeros((32, 40), dtype=np.uint16)
    cell[5:20, 6:24] = 1
    np.save(stacks / "cell_mask_stack" / f"{FIELD}.npy", cell[..., None])
    np.save(stacks / "nucleus_mask_stack" / f"{FIELD}.npy",
            np.zeros((32, 40, 3), dtype=np.uint16))
    (stacks / "pathogen_mask_stack").mkdir()
    np.save(stacks / "pathogen_mask_stack" / f"{FIELD}.npy",
            np.zeros((10, 10), dtype=np.uint16))

    payload = load_qc_field(QCFieldTarget(
        field=FIELD, plate_root=str(plate), merged_dir=str(plate / "merged")))

    assert set(payload.masks) == {"cell"}
    assert payload.masks["cell"].shape == (32, 40)
    assert int(payload.masks["cell"].max()) == 1
    joined = " ".join(payload.warnings)
    assert "nucleus" in joined and "pathogen" in joined


def test_a_two_dimensional_merged_array_is_explained_not_rendered(tmp_path):
    """A single-plane .npy in merged/ must produce a sentence, not a traceback.

    A plain 2-D image copied into ``merged/`` has no channel axis, and
    indexing it as if it did raises inside the loader thread.  The dialog
    reports the shape it found instead.
    """
    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    np.save(merged / f"{FIELD}.npy", np.zeros((8, 9), dtype=np.uint16))

    payload = load_qc_field(QCFieldTarget(
        field=FIELD, plate_root=str(tmp_path / "plate1"),
        merged_dir=str(merged)))

    assert payload.intensities is None
    assert "(8, 9)" in payload.error
    assert FIELD in payload.error
    assert payload.path.endswith(f"{FIELD}.npy")


def test_a_flat_or_all_nan_channel_renders_black_instead_of_dividing_by_zero():
    """A blank channel is a real QC result and must display, not crash.

    An empty field, a dead laser line or an all-NaN plane gives a percentile
    window of zero width; scaling by it divides by zero and paints noise.  A
    black plane is returned instead, which is what an empty channel looks
    like, while a channel with real range still scales to full contrast.
    """
    flat = QCFieldImage(intensities=np.full((4, 4, 2), 7, dtype=np.uint16))
    empty = QCFieldImage(
        intensities=np.full((4, 4, 2), np.nan, dtype=np.float32))
    real = QCFieldImage(intensities=np.dstack([
        np.arange(16, dtype=np.uint16).reshape(4, 4)] * 2))

    assert int(render_qc_field(flat, 0).max()) == 0
    assert int(render_qc_field(empty, 0).max()) == 0
    assert int(render_qc_field(real, 0).max()) == 255


def test_a_single_channel_field_composites_as_grey():
    """A one-channel plate must not render as a red-only image.

    The composite fills red, green and blue from the first three channels.
    With one channel that would leave two planes at zero and tint the field
    red, which reads as a channel-mapping bug.  It is greyscale instead.
    """
    payload = QCFieldImage(intensities=np.arange(
        16, dtype=np.uint16).reshape(4, 4, 1))

    rgb = render_qc_field(payload, -1)

    assert rgb.shape == (4, 4, 3)
    assert int(rgb.max()) == 255
    assert np.array_equal(rgb[..., 0], rgb[..., 1])
    assert np.array_equal(rgb[..., 0], rgb[..., 2])


def test_render_refuses_impossible_requests_and_skips_unusable_masks():
    """The render worker must fail loudly on nonsense and quietly on absence.

    Re-raising a failed payload's reason keeps it on screen instead of a bare
    AttributeError; a channel index past the end is a stale combo-box
    selection.  A missing or resized mask is routine and is skipped instead.
    """
    good = np.zeros((4, 4), dtype=np.uint16)
    good[1:3, 1:3] = 1
    payload = QCFieldImage(
        intensities=np.dstack([np.arange(16, dtype=np.uint16).reshape(4, 4)] * 2),
        masks={"cell": good, "nucleus": np.ones((2, 2), dtype=np.uint16)})

    with pytest.raises(ValueError, match="no active"):
        render_qc_field(QCFieldImage(error="no active copy"))
    with pytest.raises(IndexError, match="outside"):
        render_qc_field(payload, 5)

    plain = render_qc_field(payload, 0, ())
    outlined = render_qc_field(payload, 0, ("cell",))
    with_junk = render_qc_field(payload, 0, ("cell", "nucleus", "pathogen"))
    assert np.any(outlined != plain)
    assert np.array_equal(with_junk, outlined)


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


def test_a_stale_banner_link_falls_back_to_the_first_flagged_field(
        qtbot, tmp_path):
    """A link naming a field this digest no longer has must still open.

    Banner links are built from a digest that can be re-computed while the
    dialog opens, and two plates can hold the same field stem.  With no match
    the browser opens on its first field rather than at a bogus index.
    """
    digest, _flag = _digest(tmp_path / "plate1")
    targets = targets_from_digest(digest)

    unknown = QCFieldBrowser(targets, initial_field="plate9_Z99_9",
                             threaded=False)
    qtbot.addWidget(unknown)
    wrong_plate = QCFieldBrowser(
        targets, initial_field="plate1_A02_1",
        initial_plate_root=str(tmp_path / "other"), threaded=False)
    qtbot.addWidget(wrong_plate)
    exact = QCFieldBrowser(targets, initial_field="plate1_A02_1",
                           initial_plate_root=targets[1].plate_root,
                           threaded=False)
    qtbot.addWidget(exact)

    assert unknown.current_field == FIELD
    assert wrong_plate.current_field == FIELD
    assert exact.current_field == "plate1_A02_1"


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


def test_previous_stops_at_the_first_field_instead_of_wrapping(
        qtbot, tmp_path):
    """Holding the left arrow at the start of the list must not walk off it.

    Triage is a keyboard loop, so the arrow keys repeat.  A negative index
    would silently wrap to the last field and the user would review the plate
    in the wrong order without noticing.
    """
    digest, _flag = _digest(tmp_path / "plate1")
    browser = QCFieldBrowser(targets_from_digest(digest), threaded=False)
    qtbot.addWidget(browser)

    browser.previous_field()
    assert browser.current_field == FIELD

    browser.next_field()
    assert browser.current_field == "plate1_A02_1"
    browser.previous_field()
    assert browser.current_field == FIELD


def test_a_custom_object_role_gets_a_stable_distinct_outline_colour():
    """Outline colour is how the user tells two overlaid masks apart.

    A numbered organelle role inherits the organelle colour so organelles
    read as one family; a project-specific role gets a colour from its own
    name -- stable, and not one already spoken for by another mask.
    """
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[1:3, 1:3] = 1
    payload = QCFieldImage(
        intensities=np.arange(16, dtype=np.uint16).reshape(4, 4, 1),
        masks={"organelle_2": mask, "mito": mask})

    organelle = render_qc_field(payload, 0, ("organelle_2",))
    custom = render_qc_field(payload, 0, ("mito",))

    assert tuple(int(v) for v in organelle[1, 1]) == (35, 205, 235)
    assert tuple(int(v) for v in custom[1, 1]) not in {
        (35, 205, 235), (45, 220, 105), (210, 80, 255), (255, 145, 45)}
    assert np.array_equal(custom, render_qc_field(payload, 0, ("mito",)))
