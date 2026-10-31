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


def test_real_async_quarantine_and_restore_preserve_masks_and_gui_delivery(qtbot, tmp_path, monkeypatch, qapp):
    import threading
    from PySide6.QtCore import QThread
    from spacr.qt.widgets import qc_field_browser as module

    plate = tmp_path / "plate1"
    field = "plate1_A01_1"
    _write_field(plate, field)
    active = plate / "merged" / f"{field}.npy"
    original_bytes = active.read_bytes()
    mask_paths = tuple((plate / "norm_channel_stack").rglob("*.npy"))
    masks = {path: path.read_bytes() for path in mask_paths}
    target = QCFieldTarget(field=field, plate_root=str(plate), merged_dir=str(active.parent),
                           verdicts=(QCFieldVerdict("cell", "fail", ("under_segmented",), ""),))
    dialog = QCFieldBrowser([target], threaded=True, run_active=lambda: False)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog._jobs.is_busy() and not dialog._render_jobs.is_busy()
                    and dialog._quarantine.isEnabled())
    changes, workers = [], []
    started, release = threading.Event(), threading.Event()
    original = module._move_field

    def move(*args, **kwargs):
        import sys
        previous = sys.gettrace()
        inherited = threading.gettrace() if previous is None else None
        if inherited is not None:
            sys.settrace(inherited)
        try:
            workers.append(QThread.currentThread())
            if len(workers) == 1:
                started.set()
                assert release.wait(5)
            return original(*args, **kwargs)
        finally:
            if inherited is not None:
                sys.settrace(previous)

    monkeypatch.setattr(module, "_move_field", move)
    dialog.quarantineChanged.connect(lambda name, state: changes.append(
        (name, state, QThread.currentThread())))
    quarantined = plate / "merged_quarantined" / f"{field}.npy"
    try:
        qtbot.keyClick(dialog._view, Qt.Key_Q)
        qtbot.waitUntil(started.is_set)
        assert dialog._move_jobs.is_busy() and not dialog._quarantine.isEnabled()
        qtbot.keyClick(dialog._view, Qt.Key_Q)
        assert len(workers) == 1 and active.is_file() and not quarantined.exists()
        release.set()
        qtbot.waitUntil(lambda: len(changes) == 1 and not dialog._move_jobs.is_busy()
                        and not dialog._jobs.is_busy() and dialog._quarantine.isEnabled())
        assert changes == [(field, True, qapp.thread())]
        assert not active.exists() and quarantined.read_bytes() == original_bytes
        record = json.loads(Path(f"{quarantined}.quarantine.json").read_text(encoding="utf-8"))
        assert record["field"] == field
        qtbot.keyClick(dialog._view, Qt.Key_Q)
        qtbot.waitUntil(lambda: len(changes) == 2 and not dialog._move_jobs.is_busy())
        assert changes == [(field, True, qapp.thread()), (field, False, qapp.thread())]
        assert active.read_bytes() == original_bytes and not quarantined.exists()
        assert all(worker != qapp.thread() for worker in workers) and len(workers) == 2
        assert {path: path.read_bytes() for path in mask_paths} == masks
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog._move_jobs.is_busy())
        dialog.close()


def test_empty_field_browser_has_no_file_state_or_triage_action(qtbot):
    dialog = QCFieldBrowser([], threaded=False)
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.current_target is None
    assert dialog._field_paths() == ("", "")
    assert dialog._file_state() == (False, False)
    dialog._recheck_files()
    assert not dialog._quarantine.isEnabled()
    assert dialog._jobs.active_jobs() == 0
    dialog.close()


def test_an_older_real_move_completion_preserves_the_newer_pending_field(
        qtbot, tmp_path, monkeypatch, qapp):
    import threading

    plate = tmp_path / "plate1"
    names = ("plate1_A01_1", "plate1_A02_1")
    for offset, name in enumerate(names):
        _write_field(plate, name, shift=50 * offset)
    targets = tuple(QCFieldTarget(field=name, plate_root=str(plate),
                    merged_dir=str(plate / "merged"), verdicts=()) for name in names)
    originals = {name: (plate / "merged" / f"{name}.npy").read_bytes() for name in names}
    masks = {path: path.read_bytes() for path in (plate / "norm_channel_stack").rglob("*.npy")}
    dialog = QCFieldBrowser(targets, threaded=True, run_active=lambda: False)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: not dialog._jobs.is_busy() and not dialog._render_jobs.is_busy()
                    and dialog._quarantine.isEnabled(), timeout=15000)
    real_submit = dialog._move_jobs.submit
    completions = []
    entered, release = threading.Event(), threading.Event()
    submitted = 0

    def hold_delivery(read, receiver, *args, **kwargs):
        nonlocal submitted
        submitted += 1
        number = submitted
        def actual_move():
            if number == 2:
                entered.set()
                if not release.wait(15):
                    raise TimeoutError("the triage test did not release the real second move")
            return read()
        def completed(value):
            completions.append((receiver, value))
        return real_submit(actual_move, completed, *args, **kwargs)

    monkeypatch.setattr(dialog._move_jobs, "submit", hold_delivery)
    changes = []
    dialog.quarantineChanged.connect(lambda field, state: changes.append((field, state)))
    try:
        qtbot.keyClick(dialog._view, Qt.Key_Q)
        qtbot.waitUntil(lambda: len(completions) == 1 and not dialog._move_jobs.is_busy(), timeout=15000)
        first_sink = dialog._pending_move
        first_away = plate / "merged_quarantined" / f"{names[0]}.npy"
        assert first_away.read_bytes() == originals[names[0]]
        assert json.loads(Path(str(first_away) + ".quarantine.json").read_text())["field"] == names[0]
        assert not (plate / "merged" / f"{names[0]}.npy").exists()
        qtbot.keyClick(dialog._view, Qt.Key_Right)
        qtbot.waitUntil(lambda: dialog.current_target is targets[1] and not dialog._jobs.is_busy()
                        and not dialog._render_jobs.is_busy() and dialog._quarantine.isEnabled(), timeout=15000)
        payload = dialog._payload
        image = dialog._view._item.pixmap().cacheKey()
        qtbot.keyClick(dialog._view, Qt.Key_Q)
        qtbot.waitUntil(entered.is_set, timeout=15000)
        pending = dialog._pending_move
        assert pending is not first_sink and dialog._move_jobs.is_busy()
        receiver, value = completions[0]
        receiver(value)
        assert dialog._pending_move is pending
        assert dialog.current_target is targets[1] and dialog._payload is payload
        assert dialog._view._item.pixmap().cacheKey() == image
        assert changes == [(names[0], True)]
        assert (plate / "merged" / f"{names[1]}.npy").read_bytes() == originals[names[1]]
        release.set()
        qtbot.waitUntil(lambda: len(completions) == 2 and not dialog._move_jobs.is_busy(), timeout=15000)
        receiver, value = completions[1]
        receiver(value)
        qtbot.waitUntil(lambda: not dialog._jobs.is_busy() and not dialog._render_jobs.is_busy()
                        and dialog._quarantine.isEnabled(), timeout=15000)
        assert dialog._pending_move is None and changes == [(name, True) for name in names]
        second_away = plate / "merged_quarantined" / f"{names[1]}.npy"
        assert second_away.read_bytes() == originals[names[1]]
        assert json.loads(Path(str(second_away) + ".quarantine.json").read_text())["field"] == names[1]
        for expected_count, name, direction in ((3, names[1], None), (4, names[0], Qt.Key_Left)):
            if direction is not None:
                qtbot.keyClick(dialog._view, direction)
                qtbot.waitUntil(lambda: dialog.current_field == name and not dialog._jobs.is_busy()
                                and not dialog._render_jobs.is_busy() and dialog._quarantine.isEnabled(), timeout=15000)
            qtbot.keyClick(dialog._view, Qt.Key_Q)
            qtbot.waitUntil(lambda: len(completions) == expected_count and not dialog._move_jobs.is_busy(), timeout=15000)
            receiver, value = completions[-1]
            receiver(value)
            qtbot.waitUntil(lambda: not dialog._jobs.is_busy() and not dialog._render_jobs.is_busy()
                            and dialog._quarantine.isEnabled(), timeout=15000)
            assert dialog._pending_move is None
            assert (plate / "merged" / f"{name}.npy").read_bytes() == originals[name]
        assert changes == [(names[0], True), (names[1], True), (names[1], False), (names[0], False)]
        assert all(not (plate / "merged_quarantined" / f"{name}.npy").exists() for name in names)
        assert {path: path.read_bytes() for path in masks} == masks
    finally:
        release.set()
        qtbot.waitUntil(lambda: not dialog._move_jobs.is_busy(), timeout=15000)


def test_an_unrelated_background_probe_preserves_the_current_triage_field(qtbot, tmp_path):
    from spacr.qt import path_probe

    digest, _finding = _digest(tmp_path / "plate1")
    targets = targets_from_digest(digest)
    browser = QCFieldBrowser(targets, threaded=False)
    qtbot.addWidget(browser)
    browser.show()
    assert browser.current_field == targets[0].field
    assert browser._quarantine.isEnabled()
    picture = browser._view._item.pixmap().toImage()
    title = browser._field_title.text()
    verdict = browser._verdict.text()
    source = Path(targets[0].merged_dir) / f"{targets[0].field}.npy"
    original = source.read_bytes()
    unrelated = str(tmp_path / "another-project" / "missing.npy")

    with qtbot.waitSignal(path_probe.probes.answered, timeout=5000,
                          check_params_cb=lambda path, present: path == unrelated and not present):
        path_probe.exists(unrelated)

    assert path_probe.known(unrelated) is False
    assert browser.current_field == targets[0].field
    assert browser._field_title.text() == title
    assert browser._verdict.text() == verdict
    assert browser._view._item.pixmap().toImage() == picture
    assert browser._quarantine.isEnabled()
    assert source.read_bytes() == original
    assert not (source.parent.parent / "merged_quarantined" / source.name).exists()
    browser.close()
