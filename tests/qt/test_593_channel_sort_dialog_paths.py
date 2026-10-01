"""The channel-sort dialogs' answers, refusals and hand assignment, offscreen.

Modal questions are answered by monkeypatching ``QMessageBox`` and
``exec``; ``_headless`` is patched off where a test needs the path a person
at the screen would take.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtWidgets import QDialog, QMessageBox

from spacr.qt.widgets import channel_sort_dialog as csd

pytestmark = pytest.mark.qt


def _tif(path: Path, array) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


def _drawn(root: Path, masks=True, shape=(12, 12)) -> Path:
    folder = root / "drawn"
    for i in range(1, 4):
        for stain, value in (("nuc", 10), ("cell", 100)):
            name = f"{stain}_img{i:02d}.tif"
            _tif(folder / name, np.full(shape, value + i, np.uint16))
            if masks:
                _tif(folder / "masks" / name, np.full(shape, i, np.uint16))
    return folder


def _names(folder: Path):
    return sorted(n for n in os.listdir(folder) if n.endswith(".tif"))


@pytest.fixture
def dialog(qtbot, qt_theme_applied, tmp_path):
    made = csd.ChannelSortDialog(str(_drawn(tmp_path)))
    qtbot.addWidget(made)
    yield made
    made._stop_thumbs()


def test_a_masked_icon_is_tinted_and_a_blank_one_is_grey(qt_theme_applied):
    grey = np.full((4, 4), 100, np.uint8)
    mask = np.zeros((4, 4), np.uint8)
    mask[0, 0] = 255
    image = csd._grey_icon(grey, mask).toImage()
    assert image.pixelColor(0, 0).red() == 200
    assert image.pixelColor(0, 0).green() == 50
    assert image.pixelColor(3, 3).red() == 100
    assert csd._grey_icon(None, size=8).width() == 8


def test_the_thumbnail_worker_reads_every_image_and_stops_when_asked(
        qt_theme_applied, tmp_path):
    folder = _drawn(tmp_path)
    worker = csd._ThumbWorker(str(folder), _names(folder))
    got = []
    worker.ready.connect(lambda name, image, mask: got.append(
        (name, image is not None, mask is not None)))
    worker.run()
    assert got == [(name, True, True) for name in _names(folder)]
    stopped = csd._ThumbWorker(str(folder), _names(folder))
    stopped.stop()
    seen = []
    stopped.ready.connect(lambda *args: seen.append(args))
    stopped.run()
    assert seen == []


def test_the_worker_survives_a_file_it_cannot_read(qt_theme_applied, tmp_path,
                                                   monkeypatch):
    folder = _drawn(tmp_path, masks=False)

    def broken(*_args):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(csd.cs, "mask_for", broken)
    worker = csd._ThumbWorker(str(folder), _names(folder)[:1])
    got = []
    worker.ready.connect(lambda name, image, mask: got.append((image, mask)))
    worker.run()
    assert got == [(None, None)]


def test_a_missing_thumbnail_leaves_the_icon_alone(dialog):
    item = dialog._items[dialog.names[0]]
    before = item.icon().cacheKey()
    dialog._take_thumbnail(dialog.names[0], None, None)
    dialog._take_thumbnail("not listed", np.zeros((4, 4), np.uint8), None)
    assert item.icon().cacheKey() == before


def test_channels_can_be_added_assigned_and_cleared(dialog):
    before = len(dialog._channel_buttons)
    assert dialog.add_channel() == dialog._channel_count
    assert len(dialog._channel_buttons) == before + 1
    first = dialog.names[0]
    dialog.list.item(0).setSelected(True)
    dialog.assign_selected(3)
    assert dialog.channels[first] == 3
    dialog.assign_selected(None)
    assert first not in dialog.channels


def test_detected_sets_replace_the_regex_view_until_a_hand_assignment(dialog):
    sets = {(("set", "000001"),): {1: dialog.names[0], 2: dialog.names[3]}}
    dialog.detected = sets
    dialog.refresh()
    found, text = dialog.current_sets()
    assert found is sets and "1 set(s) detected" in text
    dialog.list.item(0).setSelected(True)
    dialog.assign_selected(1)
    assert dialog.detected is None


def test_the_role_boxes_are_kept_when_the_masked_channels_do_not_change(
        dialog):
    boxes = dict(dialog._role_boxes)
    dialog.refresh()
    assert dialog._role_boxes == boxes
    for box in boxes.values():
        box.setCurrentIndex(csd.ROLE_CHOICES.index("none"))
    assert dialog.mask_roles() == {}


def test_the_regex_window_reports_a_bad_regex(qtbot, qt_theme_applied, tmp_path):
    folder = _drawn(tmp_path)
    window = csd.RegexWindow(str(folder), _names(folder), {}, "(")
    qtbot.addWidget(window)
    window.run_test()
    assert "not valid" in window.summary.text()
    assert not window.use_button.isEnabled()


def test_auto_regex_says_when_nothing_fits(qtbot, qt_theme_applied, tmp_path,
                                           monkeypatch):
    folder = _drawn(tmp_path)
    window = csd.RegexWindow(str(folder), _names(folder), {}, "")
    qtbot.addWidget(window)
    monkeypatch.setattr(csd.cs, "infer_regex", lambda *_a: None)
    assert window.auto_regex() is None
    assert "found no regex" in window.summary.text()


def test_detect_needs_two_channels_and_reports_when_nothing_pairs(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    folder = _drawn(tmp_path)
    names = _names(folder)
    window = csd.RegexWindow(str(folder), names, {}, "")
    qtbot.addWidget(window)
    assert window.detect() is None
    assert "at least two channels" in window.summary.text()
    channels = {n: 1 if n.startswith("nuc") else 2 for n in names}
    window = csd.RegexWindow(str(folder), names, channels, "")
    qtbot.addWidget(window)
    empty = csd.cs.DetectedSets()
    empty.problems.append("nothing pairs")
    monkeypatch.setattr(csd.cs, "detect_sets", lambda *_a: empty)
    assert window.detect() is None
    assert window.summary.text() == "nothing pairs"


def test_detected_sets_are_kept_only_when_confirmed(qtbot, qt_theme_applied,
                                                   tmp_path):
    folder = _drawn(tmp_path)
    names = _names(folder)
    channels = {n: 1 if n.startswith("nuc") else 2 for n in names}
    window = csd.RegexWindow(str(folder), names, channels, "")
    qtbot.addWidget(window)
    assert window._ask_examples({}) is False
    window.confirm_examples = lambda _sets: False
    assert window.detect() is None and window.detected is None


def test_examples_are_asked_on_a_screen(qtbot, qt_theme_applied, tmp_path,
                                        monkeypatch):
    folder = _drawn(tmp_path)
    names = _names(folder)
    channels = {n: 1 if n.startswith("nuc") else 2 for n in names}
    window = csd.RegexWindow(str(folder), names, channels, "")
    qtbot.addWidget(window)
    sets = csd.cs.detect_sets(str(folder), channels).sets
    monkeypatch.setattr(csd, "_headless", lambda: False)
    monkeypatch.setattr(csd.ExampleSetsDialog, "exec",
                        lambda self: QDialog.Accepted)
    assert window._ask_examples(sets) is True


def test_use_keeps_the_typed_regex(qtbot, qt_theme_applied, tmp_path):
    folder = _drawn(tmp_path)
    window = csd.RegexWindow(str(folder), _names(folder), {}, "")
    qtbot.addWidget(window)
    window.regex_edit.setText(r"(?P<chanID>nuc|cell)_img(?P<fieldID>\d+)\.tif")
    window.detected = {"stale": {}}
    window._use()
    assert window.regex.startswith("(?P<chanID>")
    assert window.detected is None
    assert window.result() == QDialog.Accepted


def test_the_regex_window_hands_back_a_regex_or_detected_sets(
        dialog, monkeypatch):
    assert dialog._open_regex_window() is False
    monkeypatch.setattr(csd, "_headless", lambda: False)
    typed = r"(?P<chanID>nuc|cell)_img(?P<fieldID>\d+)\.tif"

    def accept_regex(self):
        self.regex = typed
        return QDialog.Accepted

    monkeypatch.setattr(csd.RegexWindow, "exec", accept_regex)
    assert dialog._open_regex_window() is True
    assert dialog.regex == typed and dialog.detected is None
    sets = {(("set", "000001"),): {1: dialog.names[0], 2: dialog.names[3]}}

    def accept_detected(self):
        self.regex_edit.setText("x")
        self.detected = sets
        return QDialog.Accepted

    monkeypatch.setattr(csd.RegexWindow, "exec", accept_detected)
    assert dialog._open_regex_window() is True
    assert dialog.detected == sets and dialog.regex == "x"
    monkeypatch.setattr(csd.RegexWindow, "exec", lambda self: QDialog.Rejected)
    assert dialog._open_regex_window() is False


def test_apply_asks_to_change_the_regex_and_retries(dialog, monkeypatch):
    good = dialog.regex
    dialog.set_regex("")
    monkeypatch.setattr(csd, "_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *_a: QMessageBox.Yes)

    def fix():
        dialog.set_regex(good)
        return True

    dialog.open_regex_window = fix
    dialog._on_apply()
    assert dialog.plan is not None and dialog.plan.ok


def test_apply_stops_when_the_move_is_declined(dialog, monkeypatch):
    monkeypatch.setattr(csd, "_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "question", lambda *_a: QMessageBox.No)
    dialog._on_apply()
    assert dialog.plan is None
    dialog.set_regex("")
    dialog._on_apply()
    assert dialog.plan is None


def test_a_plan_with_problems_is_shown_not_applied(dialog, monkeypatch):
    warned = []
    monkeypatch.setattr(csd, "_headless", lambda: False)
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *args: warned.append(args[2]))
    os.makedirs(os.path.join(dialog.folder, csd.cs.DEFAULT_DEST_NAME))
    original = csd.cs.unused_folder
    monkeypatch.setattr(csd.cs, "unused_folder",
                        lambda parent, name: os.path.join(parent, name))
    dialog._on_apply()
    monkeypatch.setattr(csd.cs, "unused_folder", original)
    assert dialog.plan is None
    assert warned and "already exists" in warned[0]
    assert "already exists" in dialog.status.text()


def test_rgb_images_are_converted_when_the_user_agrees(qtbot, qt_theme_applied,
                                                       tmp_path, monkeypatch):
    folder = tmp_path / "rgb"
    for i in range(1, 11):
        for stain in ("nuc", "cell"):
            _tif(folder / f"{stain}_img{i:02d}.tif",
                 np.full((6, 6, 3), i, np.uint8))
    made = csd.ChannelSortDialog(str(folder))
    qtbot.addWidget(made)
    asked = []
    monkeypatch.setattr(csd, "_headless", lambda: False)

    def answer(_parent, title, text):
        asked.append((title, text))
        return QMessageBox.Yes

    monkeypatch.setattr(QMessageBox, "question", answer)
    made._on_apply()
    made._stop_thumbs()
    assert "…and 12 more" in asked[0][1]
    assert made.plan is not None and made.plan.ok
    assert all(row.convert == "rgb" for row in made.plan.rows)
