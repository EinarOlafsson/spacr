"""The QC field browser: loading a field, rendering it, stepping and quarantining.

The browser is the last step before a user quarantines data, so every path a
real plate reaches -- a truncated merged array, a legacy array with no plane
manifest, a mask stack that is not this field's mask, a flat or all-NaN
channel, the ends of the field list, a Measure run in flight, a field in both
folders, a quarantine that cannot be written -- must produce an explained,
still-usable dialog.  Loader warnings and errors are asserted verbatim.
"""
from __future__ import annotations

import shutil
import threading

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets import qc_field_browser as browser_module  # noqa: E402
from spacr.qt.widgets.qc_field_browser import (  # noqa: E402
    QCFieldBrowser,
    QCFieldImage,
    QCFieldTarget,
    QCFieldVerdict,
    load_qc_field,
    render_qc_field,
    targets_from_digest,
)
from tests.qt.test_qc_field_browser import _digest, _write_field  # noqa: E402

pytestmark = pytest.mark.qt

FIELD = "plate1_A01_1"
SECOND = "plate1_A02_1"
ORGANELLE = (35, 205, 235)


def _legacy_plate(root, planes, stacks=None, manifest=None):
    """A merged (8, 10, planes) ramp array with no manifest, plus the given mask stacks."""
    plate = root / "plate1"
    merged = plate / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    ramp = np.arange(80, dtype=np.uint16).reshape(8, 10)
    array = np.dstack([ramp * (index + 1) for index in range(planes)])
    np.save(merged / f"{FIELD}.npy", array)
    if manifest is not None:
        (merged / ".spacr_plane_layout.json").write_text(manifest, encoding="utf-8")
    for object_type, mask in (stacks or {}).items():
        folder = plate / "norm_channel_stack" / f"{object_type}_mask_stack"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{FIELD}.npy", mask)
    return plate, array


def _target(plate, object_types=("cell",), field=FIELD):
    return QCFieldTarget(
        field=field, plate_root=str(plate), merged_dir=str(plate / "merged"),
        verdicts=tuple(QCFieldVerdict(name, "fail", ("under_segmented",), "")
                       for name in object_types))


def _cell_mask(shape=(8, 10)):
    mask = np.zeros(shape, dtype=np.uint16)
    mask[2:6, 3:7] = 1
    return mask


def test_load_qc_field_explains_a_merged_array_it_cannot_use(tmp_path, subtests):
    """A truncated file or a 2-D image in merged/ comes back as a sentence naming the field."""
    cases = {
        "truncated": (lambda path: path.write_bytes(b"\x93NUMPY truncated"),
                      "Could not read plate1_A01_1: "),
        "two-dimensional-8x10": (
            lambda path: np.save(path, np.zeros((8, 10), dtype=np.uint16)),
            "Expected a merged (height, width, channels) array for plate1_A01_1; "
            "found shape (8, 10)."),
        "two-dimensional-8x9": (
            lambda path: np.save(path, np.zeros((8, 9), dtype=np.uint16)),
            "Expected a merged (height, width, channels) array for plate1_A01_1; "
            "found shape (8, 9)."),
    }
    for case, (write, error) in cases.items():
        with subtests.test(msg=case):
            plate = tmp_path / case / "plate1"
            (plate / "merged").mkdir(parents=True)
            write(plate / "merged" / f"{FIELD}.npy")
            payload = load_qc_field(_target(plate))
            assert payload.intensities is None
            assert payload.path == str(plate / "merged" / f"{FIELD}.npy")
            if case == "truncated":
                assert payload.error.startswith(error)
            else:
                assert payload.error == error


def test_load_qc_field_infers_a_legacy_layout_and_names_each_missing_mask(
        tmp_path, subtests):
    """Mask stacks set the legacy channel count; default planes stand in; bad stacks are named."""
    cell = _cell_mask()
    cases = {
        # planes, stacks, object types, channels, masks, warnings
        "one-stack-five-intensities": (
            6, {"cell": cell}, ("cell", "nucleus", "mito"), 5,
            {"cell", "nucleus"}, ("No mito mask exists for this field.",)),
        "two-stacks-three-intensities": (
            5, {"cell": cell, "nucleus": cell}, ("cell", "nucleus"), 3,
            {"cell", "nucleus"}, ()),
        "no-stacks-default-planes": (6, {}, ("cell", "nucleus"), 4,
                                     {"cell", "nucleus"}, ()),
        "short-array-all-intensity": (3, {}, ("cell",), 3, set(),
                                      ("No cell mask exists for this field.",)),
        "broken-stacks-named": (
            5, {"cell": cell[..., None], "nucleus": np.zeros((2, 3, 4), np.uint16),
                "pathogen": np.zeros((4, 4), np.uint16)},
            ("cell", "nucleus", "pathogen"), 2, {"cell"},
            ("Could not read the nucleus mask: expected a 2-D mask, got shape "
             "(2, 3, 4)",
             "Could not read the pathogen mask: mask shape (4, 4) does not match "
             "image (8, 10)")),
    }
    for case, (planes, stacks, types, channels, masks, warnings) in cases.items():
        with subtests.test(msg=case):
            plate, array = _legacy_plate(tmp_path / case, planes, stacks)
            payload = load_qc_field(_target(plate, types))
            assert payload.error == ""
            assert payload.channel_names == tuple(
                str(i) for i in range(1, channels + 1))
            assert np.array_equal(payload.intensities, array[..., :channels])
            assert set(payload.masks) == masks
            assert payload.warnings == warnings
            if "cell" in stacks:
                assert np.array_equal(payload.masks["cell"], cell)
            if case == "no-stacks-default-planes":
                assert np.array_equal(payload.masks["cell"], array[..., 4])
                assert np.array_equal(payload.masks["nucleus"], array[..., 5])
            if case == "one-stack-five-intensities":
                assert np.array_equal(payload.masks["nucleus"], array[..., 5])
    with subtests.test(msg="corrupt-manifest-warns-and-opens"):
        plate, array = _legacy_plate(tmp_path / "manifest", 5, manifest="{not json")
        payload = load_qc_field(_target(plate))
        assert payload.error == ""
        assert payload.channel_names == ("1", "2", "3", "4")
        assert np.array_equal(payload.masks["cell"], array[..., 4])
        assert len(payload.warnings) == 1
        assert payload.warnings[0].startswith(
            "Plane-layout metadata could not be read: cannot read merged plane "
            "layout ")
    with subtests.test(msg="trailing-singleton-mask-on-a-real-field"):
        plate = tmp_path / "written" / "plate1"
        _write_field(plate, FIELD)
        stacks = plate / "norm_channel_stack"
        wide = _cell_mask((32, 40))
        np.save(stacks / "cell_mask_stack" / f"{FIELD}.npy", wide[..., None])
        np.save(stacks / "nucleus_mask_stack" / f"{FIELD}.npy",
                np.zeros((32, 40, 3), dtype=np.uint16))
        (stacks / "pathogen_mask_stack").mkdir()
        np.save(stacks / "pathogen_mask_stack" / f"{FIELD}.npy",
                np.zeros((10, 10), dtype=np.uint16))
        payload = load_qc_field(QCFieldTarget(
            field=FIELD, plate_root=str(plate), merged_dir=str(plate / "merged")))
        assert set(payload.masks) == {"cell"}
        assert np.array_equal(payload.masks["cell"], wide)
        joined = " ".join(payload.warnings)
        assert "nucleus" in joined and "pathogen" in joined


def test_render_qc_field_composites_blank_channels_and_colours_roles(subtests):
    """Flat/NaN planes go black, one channel goes grey, roles get stable colours, nonsense raises."""
    ramp = np.arange(16, dtype=np.uint16).reshape(4, 4)
    for case, intensities, channel in (
            ("flat", np.full((4, 4, 2), 7, dtype=np.uint16), 0),
            ("all-nan", np.full((4, 4, 2), np.nan, dtype=np.float32), 0),
            ("single-nan-composite", np.full((4, 4, 1), np.nan, np.float32), -1)):
        with subtests.test(msg=case):
            rgb = render_qc_field(QCFieldImage(intensities=intensities), channel, ())
            assert rgb.shape == (4, 4, 3)
            assert int(rgb.max()) == 0
    with subtests.test(msg="flat-beside-live"):
        payload = QCFieldImage(intensities=np.dstack(
            [np.full((4, 4), 7, np.uint16), ramp]))
        assert int(render_qc_field(payload, 0, ()).max()) == 0
        live = render_qc_field(payload, 1, ())
        assert (int(live.min()), int(live.max())) == (0, 255)
    for case, intensities in (("single-channel-uint16", ramp[..., None]),
                              ("single-channel-float", ramp[..., None].astype(
                                  np.float32))):
        with subtests.test(msg=case):
            rgb = render_qc_field(QCFieldImage(intensities=intensities), -1)
            assert (rgb.shape, int(rgb.min()), int(rgb.max())) == ((4, 4, 3), 0, 255)
            assert np.array_equal(rgb[..., 0], rgb[..., 1])
            assert np.array_equal(rgb[..., 0], rgb[..., 2])
    with subtests.test(msg="role-colours"):
        mask = np.zeros((4, 4), dtype=np.uint16)
        mask[1:3, 1:3] = 1
        for intensities in (np.zeros((4, 4, 3), np.uint16), ramp[..., None]):
            payload = QCFieldImage(intensities=intensities, masks={
                "organelle_b": mask, "organelle_2": mask, "mito": mask})
            for organelle in ("organelle_b", "organelle_2"):
                assert tuple(int(v) for v in render_qc_field(
                    payload, -1, (organelle,))[1, 1]) == ORGANELLE
            custom = render_qc_field(payload, -1, ("mito",))
            assert tuple(int(v) for v in custom[1, 1]) not in {
                ORGANELLE, (45, 220, 105), (210, 80, 255), (255, 145, 45), (0, 0, 0)}
            assert np.array_equal(custom, render_qc_field(payload, -1, ("mito",)))
        black = QCFieldImage(intensities=np.zeros((4, 4, 3), np.uint16),
                             masks={"mito": mask})
        assert tuple(int(v) for v in render_qc_field(
            black, -1, ("mito",))[1, 1]) == (161, 228, 105)
    with subtests.test(msg="refusals-and-unusable-masks"):
        good = np.zeros((4, 4), dtype=np.uint16)
        good[1:3, 1:3] = 1
        payload = QCFieldImage(intensities=np.dstack([ramp] * 2), masks={
            "cell": good, "nucleus": np.ones((2, 2), dtype=np.uint16)})
        with pytest.raises(ValueError, match="^no active copy$"):
            render_qc_field(QCFieldImage(error="no active copy"))
        with pytest.raises(IndexError, match="outside"):
            render_qc_field(payload, 5)
        plain = render_qc_field(payload, 0, ())
        outlined = render_qc_field(payload, 0, ("cell",))
        assert np.any(outlined != plain)
        assert np.array_equal(
            render_qc_field(payload, 0, ("cell", "nucleus", "pathogen")), outlined)


def _key(browser, key):
    event = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)
    QApplication.sendEvent(browser, event)
    return event


def test_the_browser_steps_without_wrapping_and_opens_exact_links(
        qtbot, tmp_path, subtests):
    """Arrows stop at both ends, links need field and plate, other keys reach QDialog, bad channels say why."""
    digest, _finding = _digest(tmp_path / "plate1")
    targets = targets_from_digest(digest)

    def open_browser(**kwargs):
        dialog = QCFieldBrowser(targets, threaded=False, **kwargs)
        qtbot.addWidget(dialog)
        return dialog

    with subtests.test(msg="arrows-stop-at-the-ends"):
        browser = open_browser()
        browser.previous_field()
        assert browser.current_field == FIELD
        browser.next_field()
        assert (browser.current_field, browser._next.isEnabled()) == (SECOND, False)
        browser.next_field()
        assert browser.current_field == SECOND
        assert "flagged field 2 of 2" in browser._field_title.text()
        browser.previous_field()
        assert browser.current_field == FIELD
    with subtests.test(msg="initial-link-falls-back-to-the-first-field"):
        for kwargs, expected in (
                ({"initial_field": "plate9_Z99_9"}, FIELD),
                ({"initial_field": SECOND,
                  "initial_plate_root": str(tmp_path / "other")}, FIELD),
                ({"initial_field": SECOND,
                  "initial_plate_root": targets[1].plate_root}, SECOND)):
            assert open_browser(**kwargs).current_field == expected, kwargs
    with subtests.test(msg="open-at-needs-an-exact-match"):
        browser = open_browser()
        for args, answer, current, position in (
                ((SECOND,), True, SECOND, 2), (("plate9_Z99_9",), False, SECOND, 2),
                ((FIELD, str(tmp_path / "elsewhere")), False, SECOND, 2),
                ((FIELD, targets[0].plate_root), True, FIELD, 1)):
            assert browser.open_at(*args) is answer, args
            assert browser.current_field == current
            assert f"flagged field {position} of 2" in browser._field_title.text()
    with subtests.test(msg="triage-keys-claimed-the-rest-passed-on"):
        browser = open_browser()
        browser.show()
        assert _key(browser, Qt.Key_Right).isAccepted()
        assert browser.current_field == SECOND
        _key(browser, Qt.Key_A)
        assert (browser.current_field, browser.isVisible()) == (SECOND, True)
        rejected = []
        browser.rejected.connect(lambda: rejected.append(True))
        _key(browser, Qt.Key_Escape)
        assert rejected == [True]
    with subtests.test(msg="a-stale-channel-says-why-the-canvas-froze"):
        plate = tmp_path / "single" / "plate1"
        _write_field(plate, FIELD)
        browser = QCFieldBrowser([_target(plate)], threaded=False)
        qtbot.addWidget(browser)
        first = browser._view._item
        browser._channel.setCurrentIndex(2)
        second = browser._view._item
        assert first is not None and second is not first
        browser._channel.addItem("Channel 9: stale", 9)
        browser._channel.setCurrentIndex(browser._channel.count() - 1)
        assert "Could not render this field" in browser._load_status.text()
        assert "outside 0..2" in browser._load_status.text()
        assert browser._view._item is second


def test_the_quarantine_button_refuses_every_unsafe_move(qtbot, tmp_path,
                                                         subtests):
    """A run in flight, duplicate copies, a blocked folder or no field refuse; clearing the cause re-arms it."""
    def plate_with_field(name):
        plate = tmp_path / name / "plate1"
        _write_field(plate, FIELD)
        return plate, plate / "merged" / f"{FIELD}.npy"

    def open_browser(targets, **kwargs):
        dialog = QCFieldBrowser(targets, threaded=False, **kwargs)
        qtbot.addWidget(dialog)
        changes = []
        dialog.quarantineChanged.connect(
            lambda field, moved: changes.append((field, moved)))
        return dialog, changes

    with subtests.test(msg="no-field-stays-disabled"):
        browser, _changes = open_browser([])
        assert not browser._quarantine.isEnabled()
        browser._quarantine.setEnabled(True)
        qtbot.waitUntil(lambda: not browser._quarantine.isEnabled(), timeout=5000)
        assert "No flagged fields" in browser._field_title.text()
        assert browser.current_target is None
    with subtests.test(msg="a-probe-that-raises-reads-as-no-run"):
        plate, _active = plate_with_field("probe")
        state = {"mode": "raise"}

        def run_active():
            if state["mode"] == "raise":
                raise RuntimeError("Internal C++ object already deleted")
            return state["mode"] == "running"

        browser, _changes = open_browser([_target(plate)], run_active=run_active)
        assert browser._quarantine.isEnabled()
        assert "reversible" in browser._action_status.text()
        state["mode"] = "running"
        qtbot.waitUntil(lambda: not browser._quarantine.isEnabled(), timeout=5000)
        assert "Measure is running" in browser._action_status.text()
    with subtests.test(msg="refused-outright-while-measure-runs"):
        plate, active = plate_with_field("running")
        running = {"value": True}
        browser, changes = open_browser([_target(plate)],
                                        run_active=lambda: running["value"])
        browser.toggle_quarantine()
        assert (changes, active.is_file()) == ([], True)
        assert "Measure is running" in browser._action_status.text()
        running["value"] = False
        browser.toggle_quarantine()
        assert changes == [(FIELD, True)]
        assert not active.exists()
        assert (plate / "merged_quarantined" / f"{FIELD}.npy").is_file()
    with subtests.test(msg="duplicate-copies-refuse-both-ways"):
        plate, active = plate_with_field("duplicate")
        copy = plate / "merged_quarantined" / f"{FIELD}.npy"
        copy.parent.mkdir()
        shutil.copy2(active, copy)
        browser, changes = open_browser([_target(plate)])
        assert "Resolve duplicate copies" in browser._quarantine.text()
        assert not browser._quarantine.isEnabled()
        assert "Nothing will be overwritten" in browser._action_status.text()
        browser.toggle_quarantine()
        assert (changes, active.is_file(), copy.is_file()) == ([], True, True)
        copy.unlink()
        qtbot.waitUntil(lambda: browser._quarantine.isEnabled(), timeout=5000)
        browser.toggle_quarantine()
        assert (changes, active.exists(), copy.is_file()) == (
            [(FIELD, True)], False, True)
    with subtests.test(msg="a-blocked-quarantine-leaves-the-field"):
        plate, active = plate_with_field("blocked")
        blocker = plate / "merged_quarantined"
        blocker.write_text("not a folder", encoding="utf-8")
        browser, changes = open_browser([_target(plate)])
        browser.toggle_quarantine()
        assert (changes, active.is_file(), blocker.is_file()) == ([], True, True)
        assert "Could not change quarantine" in browser._action_status.text()
        blocker.unlink()
        browser.toggle_quarantine()
        assert (changes, active.exists()) == ([(FIELD, True)], False)
        assert "Quarantined" in browser._action_status.text()


def test_a_pending_quarantine_disables_the_action_until_the_move_finishes(
        qtbot, tmp_path, monkeypatch):
    """A second triage action while a threaded move is in flight queues nothing."""
    plate = tmp_path / "plate1"
    _write_field(plate, FIELD)
    active = plate / "merged" / f"{FIELD}.npy"
    destination = plate / "merged_quarantined" / f"{FIELD}.npy"
    original_bytes = active.read_bytes()
    entered, release = threading.Event(), threading.Event()
    move_field = browser_module._move_field

    def gated_move(target, was_quarantined, sink):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release the quarantine worker")
        return move_field(target, was_quarantined, sink)

    monkeypatch.setattr(browser_module, "_move_field", gated_move)
    browser = QCFieldBrowser([_target(plate)], threaded=True)
    qtbot.addWidget(browser)
    runners = (browser._jobs, browser._render_jobs, browser._move_jobs)
    changes = []
    browser.quarantineChanged.connect(
        lambda field, moved: changes.append((field, moved)))
    try:
        qtbot.waitUntil(lambda: browser._quarantine.isEnabled()
                        and browser._view._item is not None
                        and not any(r.is_busy() for r in runners), timeout=10000)
        browser.toggle_quarantine()
        qtbot.waitUntil(entered.is_set, timeout=5000)
        pending = browser._pending_move
        assert pending is not None and browser._move_jobs.is_busy()
        assert not browser._quarantine.isEnabled()
        assert (active.read_bytes() == original_bytes, destination.exists(),
                changes) == (True, False, [])
        browser.toggle_quarantine()
        assert browser._pending_move is pending
        assert browser._move_jobs.pending_jobs() == 1
        release.set()
        qtbot.waitUntil(lambda: browser._pending_move is None
                        and browser._quarantine.isEnabled()
                        and not any(r.is_busy() for r in runners), timeout=10000)
        assert changes == [(FIELD, True)]
        assert not active.exists()
        assert destination.read_bytes() == original_bytes
        assert "Restore field" in browser._quarantine.text()
        assert "Quarantined" in browser._action_status.text()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not any(r._jobs for r in runners), timeout=10000)


def test_the_overview_wizard_plate_switch_acts_on_its_senders_plate(subtests):
    """A plate switch acts on its own plate (or the empty path); no sender changes nothing."""
    class _Sender:
        def __init__(self, path):
            self._path = path

        def property(self, name):
            assert name == "platePath"
            return self._path

    class _Stub:
        def __init__(self, sender):
            self._sender = sender
            self.calls = []

        def sender(self):
            return self._sender

        def _use_plate_subfolders(self, path, on):
            self.calls.append(("subfolders", path, on))

        def _apply_to_plate(self, path):
            self.calls.append(("apply", path))

    wizard = browser_module._OverviewMetadataWizard
    cases = {
        "plate-property": (_Sender("/screen/p1"), True,
                           [("subfolders", "/screen/p1", True)]),
        "no-plate-property": (_Sender(None), False, [("subfolders", "", False)]),
        "no-sender": (None, True, []),
    }
    for case, (sender, on, calls) in cases.items():
        with subtests.test(msg=case):
            stub = _Stub(sender)
            wizard._plate_subfolders_toggled(stub, on)
            if sender is None:
                wizard._apply_plate_button_clicked(stub)
            assert stub.calls == calls
