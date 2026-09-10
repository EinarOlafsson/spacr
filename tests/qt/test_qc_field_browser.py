"""The Measure QC banner reaches a fast, reversible field-triage loop."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QLabel, QWidget  # noqa: E402

from spacr import seg_qc  # noqa: E402
from spacr.qt import prerun  # noqa: E402
from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldBrowser,
    QCFieldTarget,
    QCFieldVerdict,
    finding_targets,
    load_qc_field,
    render_qc_field,
    targets_from_digest,
)

pytestmark = pytest.mark.qt


def _write_field(plate: Path, field: str, shift: int = 0) -> None:
    merged = plate / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:32, :40]
    array = np.zeros((32, 40, 5), dtype=np.uint16)
    array[..., 0] = xx * 100 + shift
    array[..., 1] = yy * 120 + shift
    array[..., 2] = (xx + yy) * 70 + shift
    cell = np.zeros((32, 40), dtype=np.uint16)
    cell[5:20, 6:24] = 1
    nucleus = np.zeros((32, 40), dtype=np.uint16)
    nucleus[9:15, 11:18] = 1
    array[..., 3] = cell
    array[..., 4] = nucleus
    np.save(merged / f"{field}.npy", array)
    (merged / ".spacr_plane_layout.json").write_text(json.dumps({
        "version": 1,
        "intensity_channels": ["DNA", "green", "red"],
        "mask_plane_order": ["cell", "nucleus"],
        "mask_dims": {"cell": 3, "nucleus": 4},
    }), encoding="utf-8")
    for name, mask in (("cell", cell), ("nucleus", nucleus)):
        stack = plate / "norm_channel_stack" / f"{name}_mask_stack"
        stack.mkdir(parents=True, exist_ok=True)
        np.save(stack / f"{field}.npy", mask)


def _digest(plate: Path):
    first = "plate1_A01_1"
    second = "plate1_A02_1"
    _write_field(plate, first)
    _write_field(plate, second, shift=50)
    qc_dir = plate / "qc"
    qc_dir.mkdir()
    cell_rows = [
        seg_qc.FieldQC(
            first, "cell", 1, ["under_segmented"], {}, "fail",
            "One cell region covers most of the field."),
        seg_qc.FieldQC(
            second, "cell", 2, ["tiny_objects"], {}, "warn",
            "Most cell regions are unusually small."),
    ]
    nucleus_rows = [
        seg_qc.FieldQC(
            first, "nucleus", 1, ["empty_field"], {}, "fail",
            "The nucleus mask is effectively empty."),
    ]
    cards = [
        seg_qc.Scorecard(
            str(qc_dir / "segmentation_qc_cell.csv"), "cell", cell_rows),
        seg_qc.Scorecard(
            str(qc_dir / "segmentation_qc_nucleus.csv"), "nucleus",
            nucleus_rows),
    ]
    finding = seg_qc.Finding(
        severity="fail", kind="flag", flag="under_segmented",
        headline="Two fields need visual review.", plate="plate1",
        object_type="cell", fields=(first, second), n_fields=2)
    return seg_qc.QCDigest(
        root=str(plate), verdict="fail", headline=finding.headline,
        scorecards=cards, findings=[finding]), finding


def test_targets_group_object_types_and_load_real_merged_masks(tmp_path):
    digest, finding = _digest(tmp_path / "plate1")

    targets = targets_from_digest(digest)

    assert [target.field for target in targets] == [
        "plate1_A01_1", "plate1_A02_1"]
    assert [verdict.object_type for verdict in targets[0].verdicts] == [
        "cell", "nucleus"]
    assert [target.field for target in finding_targets(digest, finding)] == [
        "plate1_A01_1", "plate1_A02_1"]
    payload = load_qc_field(targets[0])
    assert payload.error == ""
    assert payload.channel_names == ("DNA", "green", "red")
    assert payload.intensities.shape == (32, 40, 3)
    assert set(payload.masks) == {"cell", "nucleus"}
    assert int(payload.masks["cell"].max()) == 1
    composite = render_qc_field(payload, -1, ())
    cell_overlay = render_qc_field(payload, -1, ("cell",))
    single_channel = render_qc_field(payload, 1, ())
    assert composite.shape == (32, 40, 3)
    assert np.any(cell_overlay != composite), "the mask toggle changed no pixel"
    assert np.any(single_channel != composite), "the channel picker changed no pixel"


def test_a_positional_finding_browses_individually_clean_fields(tmp_path):
    digest, _finding = _digest(tmp_path / "plate1")
    for card in digest.scorecards:
        for qc in card.field_qcs:
            qc.flags = []
            qc.severity = "ok"
    gradient = seg_qc.Finding(
        severity="fail", kind="count_gradient",
        headline="Counts step across the plate.", plate="plate1",
        object_type="cell", n_fields=2)
    digest.findings = [gradient]

    targets = targets_from_digest(digest)

    assert [target.field for target in targets] == [
        "plate1_A01_1", "plate1_A02_1"]
    assert all("cell:count_gradient" in target.audit_flags
               for target in targets)


def test_clicking_a_banner_field_link_opens_at_that_exact_field(qtbot, tmp_path):
    digest, _finding = _digest(tmp_path / "plate1")
    screen = QWidget()
    screen._thread = None
    qtbot.addWidget(screen)
    banner = prerun.SegQCBanner(screen)
    banner._field_browser_factory = lambda targets, **kwargs: QCFieldBrowser(
        targets, threaded=False, **kwargs)
    banner._digest = digest
    banner._draw()

    links = banner.findChildren(QLabel, "QCFieldLinks")
    assert len(links) == 1
    assert "plate1_A01_1" in links[0].text()
    assert "plate1_A02_1" in links[0].text()
    links[0].linkActivated.emit("1")

    browser = banner._field_browser
    assert isinstance(browser, QCFieldBrowser)
    assert browser.current_field == "plate1_A02_1"
    assert "tiny_objects" in browser._verdict.text()
    assert browser._view._item is not None
    assert set(browser._layer_checks) == {"cell", "nucleus"}

    # A loader may still be winding down when the user leaves Measure. Hiding
    # the owner closes it through the dialog's bounded shutdown path instead
    # of deleting a live QThread with the widget tree.
    screen.show()
    screen.hide()
    assert banner._field_browser is None
    assert browser.isVisible() is False


def test_left_right_and_q_make_a_complete_keyboard_triage_loop(qtbot, tmp_path):
    digest, _finding = _digest(tmp_path / "plate1")
    targets = targets_from_digest(digest)
    browser = QCFieldBrowser(targets, threaded=False)
    qtbot.addWidget(browser)
    browser.show()
    browser.activateWindow()
    browser.setFocus()

    assert browser.current_field == "plate1_A01_1"
    browser._view.setFocus()
    qtbot.keyClick(browser._view, Qt.Key_Right)
    assert browser.current_field == "plate1_A02_1"
    assert browser._quarantine.isEnabled()

    qtbot.keyClick(browser._view, Qt.Key_Q)
    merged = Path(targets[1].merged_dir)
    quarantined = merged.parent / "merged_quarantined" / "plate1_A02_1.npy"
    assert quarantined.is_file()
    assert not (merged / "plate1_A02_1.npy").exists()
    record = json.loads(Path(
        f"{quarantined}.quarantine.json").read_text(encoding="utf-8"))
    assert "cell:tiny_objects" in record["qc_flags"]

    qtbot.keyClick(browser._view, Qt.Key_Q)
    assert (merged / "plate1_A02_1.npy").is_file()
    assert not quarantined.exists()
    qtbot.keyClick(browser._view, Qt.Key_Left)
    assert browser.current_field == "plate1_A01_1"


def test_production_threaded_loading_delivers_a_pixmap_on_the_gui_thread(
        qtbot, tmp_path):
    digest, _finding = _digest(tmp_path / "plate1")
    browser = QCFieldBrowser(
        targets_from_digest(digest), initial_field="plate1_A02_1",
        threaded=True)
    qtbot.addWidget(browser)
    browser.show()

    qtbot.waitUntil(
        lambda: (not browser._jobs.is_busy()
                 and not browser._render_jobs.is_busy()
                 and browser._view._item is not None),
        timeout=10000)

    assert browser.current_field == "plate1_A02_1"
    assert browser._view._item.pixmap().isNull() is False
    assert browser._jobs.active_jobs() <= 1


def test_a_missing_stale_field_is_explained_instead_of_crashing(qtbot, tmp_path):
    merged = tmp_path / "plate1" / "merged"
    merged.mkdir(parents=True)
    target = QCFieldTarget(
        field="plate1_A01_9", plate_root=str(tmp_path / "plate1"),
        merged_dir=str(merged),
        verdicts=(QCFieldVerdict("cell", "fail", ("unreadable",), "gone"),))

    browser = QCFieldBrowser([target], threaded=False)
    qtbot.addWidget(browser)

    assert "already gone" in browser._load_status.text().lower()
    assert "out of date" in browser._action_status.text().lower()
    assert not browser._quarantine.isEnabled()


def test_quarantine_is_disabled_while_measure_is_in_flight(qtbot, tmp_path):
    digest, _finding = _digest(tmp_path / "plate1")
    running = {"value": True}
    browser = QCFieldBrowser(
        targets_from_digest(digest), threaded=False,
        run_active=lambda: running["value"])
    qtbot.addWidget(browser)

    assert not browser._quarantine.isEnabled()
    assert "Measure is running" in browser._action_status.text()
    running["value"] = False
    browser._sync_action()
    assert browser._quarantine.isEnabled()
    assert prerun.BLOCKS_RUN is False
