"""Pressing Add folders or Add plates and choosing folders adds their plates.

The maintainer reported both buttons doing nothing. These tests drive the
whole path the way a user does, under the app's style sheet and glass: the
button is clicked, the real picker opens modally, one or several plate
folders are Ctrl-clicked in its list, Choose is clicked, and the chosen
plates must reach the metadata step's plate table or the open Overview.
The picker must also open at a sensible size: at least its layout's
minimum, so every button shows, and at most 960 by 640 or three fifths of
the screen.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEventLoop, QItemSelectionModel, QPoint, Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QFileDialog, QListView, QPushButton

from spacr.qt import theme
from spacr.qt.i18n import tr
from spacr.qt.widgets import glass
from spacr.qt.widgets import qc_field_browser as browser

pytestmark = pytest.mark.qt

PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"


def _modal_exec(dialog):
    """``QDialog.exec`` itself: shown application-modal, a local loop until the dialog finishes.

    The suite refuses the real ``exec`` so a forgotten modal cannot hang it;
    here a :class:`_Chooser` with a deadline always answers the picker.
    """
    loop = QEventLoop()
    dialog.finished.connect(loop.quit)
    dialog.setWindowModality(Qt.ApplicationModal)
    dialog.show()
    loop.exec()
    return dialog.result()


@pytest.fixture
def glassed_app(monkeypatch):
    """The app's dark sheet and glass filter, as spaCR runs with them."""
    app = QApplication.instance()
    from spacr.qt.widgets import setup_card

    monkeypatch.setattr(setup_card.SetupCard, "_tick", lambda self: None)
    monkeypatch.setattr(QFileDialog, "exec", _modal_exec)
    previous = app.styleSheet()
    app.setStyleSheet(theme.stylesheet(theme="dark"))
    installed_here = glass.install_glass_everywhere(app)
    yield app
    if installed_here:
        glass.uninstall_glass_everywhere(app)
    app.setStyleSheet(previous)


def _plate(folder, wells=("A01", "A02")):
    folder.mkdir(parents=True, exist_ok=True)
    for index, well in enumerate(wells):
        for channel in (1, 2):
            tifffile.imwrite(folder / f"p1-{well}-f1-c{channel}.tif",
                             np.full((24, 32), 100 * (index + 1) + channel, dtype=np.uint16))
    return folder


class _Chooser:
    """Waits for the modal picker, Ctrl-clicks folders by name in it and clicks Choose."""

    def __init__(self, qtbot, folder, names):
        self.qtbot, self.folder, self.names = qtbot, str(folder), tuple(names)
        self.seen, self.problems = [], []

    def arm(self, tries=200):
        QTimer.singleShot(50, lambda: self.step(tries))

    def step(self, tries):
        dialog = QApplication.activeModalWidget()
        if not isinstance(dialog, QFileDialog) or not dialog.isVisible():
            if tries:
                QTimer.singleShot(50, lambda: self.step(tries - 1))
            else:
                self.problems.append("the picker never opened modally")
            return
        if not self.seen:
            self.seen.append(_measure(dialog))
            dialog.setDirectory(self.folder)
        view = dialog.findChild(QListView, "listView")
        model, root = view.model(), view.rootIndex()
        rows = {model.index(row, 0, root).data(): model.index(row, 0, root) for row in range(model.rowCount(root))}
        if not set(self.names) <= set(rows):
            if tries:
                QTimer.singleShot(50, lambda: self.step(tries - 1))
            else:
                self.problems.append(f"the picker never listed {self.names}: {sorted(rows)}")
                dialog.reject()
            return
        view.selectionModel().clearSelection()
        for position, name in enumerate(self.names):
            view.scrollTo(rows[name])
            point = view.visualRect(rows[name]).center()
            modifier = Qt.ControlModifier if position else Qt.NoModifier
            self.qtbot.mouseClick(view.viewport(), Qt.LeftButton, modifier, point)
        QApplication.processEvents()
        chosen = sorted(index.data() for index in view.selectionModel().selectedRows())
        if chosen != sorted(self.names):
            view.selectionModel().clearSelection()
            for name in self.names:
                view.selectionModel().select(rows[name], QItemSelectionModel.Select | QItemSelectionModel.Rows)
            self.problems.append(f"clicking selected {chosen}")
        choose = dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open)
        if choose is None or not choose.isEnabled():
            self.problems.append("Choose is not enabled with the folders selected")
            dialog.reject()
            return
        self.qtbot.mouseClick(choose, Qt.LeftButton)


def _measure(dialog):
    screen = dialog.screen().availableGeometry()
    least = dialog.minimumSizeHint()
    buttons = {}
    for button in dialog.findChildren(QPushButton):
        if button.isHidden():
            continue
        corner = button.mapTo(dialog, QPoint(button.width() - 1, button.height() - 1))
        buttons[button.text().replace("&", "")] = (
            button.isVisible() and dialog.rect().contains(button.mapTo(dialog, QPoint(0, 0)))
            and dialog.rect().contains(corner))
    return {"size": (dialog.width(), dialog.height()), "least": (least.width(), least.height()),
            "screen": (screen.width(), screen.height()), "buttons": buttons,
            "resizable": getattr(dialog, "_spacr_resizer", None) is not None or dialog.isSizeGripEnabled()}


def _assert_sensible(record):
    width, height = record["size"]
    least_width, least_height = record["least"]
    screen_width, screen_height = record["screen"]
    assert width >= least_width and height >= least_height, record
    assert width <= max(least_width, min(960, int(screen_width * .6))), record
    assert height <= max(least_height, min(640, int(screen_height * .6))), record
    assert record["buttons"].get("Choose") and record["buttons"].get("Cancel"), record["buttons"]
    assert record["resizable"]


def _click(qtbot, button):
    qtbot.mouseClick(button, Qt.LeftButton)


@pytest.mark.parametrize("names", [("plate2",), ("plate2", "plate3")])
def test_add_folders_on_the_metadata_step_adds_the_chosen_folders(glassed_app, qtbot, tmp_path, names):
    first = _plate(tmp_path / "plate1")
    more = tmp_path / "more"
    for name in ("plate2", "plate3"):
        _plate(more / name)
    wizard = browser._OverviewMetadataWizard(str(first), {"custom_regex": PATTERN})
    wizard._startup_timer.stop()
    qtbot.addWidget(wizard)
    wizard.show()
    chooser = _Chooser(qtbot, more, names)
    chooser.arm()
    _click(qtbot, wizard.findChild(QPushButton, "OverviewWizardAddFolders"))
    assert not chooser.problems
    _assert_sensible(chooser.seen[0])
    sources = browser._overview_source_paths(wizard.source.text())
    assert sources == [str(first)] + [str(more / name) for name in names]
    qtbot.waitUntil(lambda: {plate["path"] for plate in wizard._detected_plates}
                    == {str(first)} | {str(more / name) for name in names}, timeout=15000)


def _retire(view, qtbot):
    view._cancel()
    runners = (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
               view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
               view._native_jobs, view._image_stack_jobs, view._group_jobs)
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in runners), timeout=15000)
    for runner in runners:
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.mark.parametrize("names", [("plate2",), ("plate2", "plate3")])
def test_add_plates_on_the_overview_adds_the_chosen_plates(glassed_app, qtbot, tmp_path, names):
    first = _plate(tmp_path / "plate1")
    more = tmp_path / "more"
    for name in ("plate2", "plate3"):
        _plate(more / name)
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    view = browser._FieldOverview(settings={
        "source": str(first), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (3, 3), "representatives": 1})
    qtbot.addWidget(view, before_close_func=lambda widget: _retire(widget, qtbot))
    view.resize(1500, 1000)
    view.show()
    view._scan()
    qtbot.waitUntil(lambda: bool(view._fields) and not view._index_jobs.is_busy(), timeout=15000)
    before = {field.plate for field in view._all_fields}
    button = next(b for b in view.findChildren(QPushButton) if b.text() == tr("Add plates…"))
    chooser = _Chooser(qtbot, more, names)
    chooser.arm()
    _click(qtbot, button)
    assert not chooser.problems
    _assert_sensible(chooser.seen[0])
    qtbot.waitUntil(lambda: {field.source for field in view._all_fields}
                    == {str(first)} | {str(more / name) for name in names}, timeout=15000)
    assert len({field.plate for field in view._all_fields}) == len(before) + len(names)
    assert browser._overview_source_paths(view.source.text()) == [str(first)] + [str(more / name)
                                                                                for name in names]
