"""Make Masks' split button, folder jobs, uncertainty actions and SAM2 entry."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QFileDialog, QMenu  # noqa: E402

from spacr.qt.screens import make_masks as mm  # noqa: E402


def test_a_folder_job_keeps_its_error():
    def broken():
        raise RuntimeError("disk gone")

    job = mm._FolderJobWorker(broken)
    job.run()
    assert isinstance(job.error, RuntimeError)
    ok = mm._FolderJobWorker(lambda: 7)
    ok.run()
    assert ok.result == 7


@pytest.fixture
def split(qtbot):
    button = mm._SplitMenuButton("Load test data…")
    qtbot.addWidget(button)
    button.resize(160, 30)
    button.show()
    return button


def _press(button, x):
    point = QPointF(x, 10)
    return QMouseEvent(QEvent.MouseButtonPress, point, button.mapToGlobal(point),
                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)


def test_the_arrow_the_right_click_and_alt_down_open_the_menu(split,
                                                              monkeypatch):
    opened = []
    menu = QMenu(split)
    monkeypatch.setattr(menu, "popup", lambda point: opened.append(point))
    split.set_split_menu(menu)
    split.mousePressEvent(_press(split, split.width() - 2))
    split.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse,
                                             QPoint(5, 5)))
    split.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_Down,
                                  Qt.AltModifier))
    assert len(opened) == 3
    split.setEnabled(False)
    split._pop_menu()
    assert len(opened) == 3


def test_a_plain_button_presses_normally(split):
    clicked = []
    split.clicked.connect(lambda: clicked.append(True))
    split.mousePressEvent(_press(split, 5))
    split.keyPressEvent(QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
    split.contextMenuEvent(QContextMenuEvent(QContextMenuEvent.Mouse,
                                             QPoint(5, 5)))
    split.repaint()
    assert split.isDown() and clicked == []


def test_the_heatmap_falls_back_to_grey_and_flattens_colour_fields(monkeypatch):
    import matplotlib

    uncertainty = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    image = np.ones((4, 4, 3), np.uint8) * 50
    coloured = mm._uncertainty_heatmap(uncertainty, image)
    assert coloured.shape == (4, 4, 3)

    class _NoMaps:
        def __getitem__(self, name):
            raise KeyError(name)

    monkeypatch.setattr(matplotlib, "colormaps", _NoMaps())
    grey = mm._uncertainty_heatmap(uncertainty)
    assert grey.shape == (4, 4, 3)


@pytest.fixture
def screen(qtbot):
    widget = mm.MakeMasksScreen()
    qtbot.addWidget(widget)
    return widget


def test_a_separated_navigation_group_has_a_divider(screen):
    from PySide6.QtWidgets import QFrame

    group, _row = screen._nav_button_group(separated=True)
    assert group.findChild(QFrame, "NavGroupSeparator") is not None


def test_sam2_tracking_cancelled_opens_nothing(screen, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    opened = []
    monkeypatch.setattr(mm, "_Sam2SeedDialog",
                        lambda *a, **k: opened.append(True))
    screen._on_sam2_tracking()
    assert opened == []


def test_sam2_tracking_opens_the_seed_dialog_on_the_movie(screen, monkeypatch,
                                                          tmp_path):
    import tifffile

    movie = tmp_path / "movie.tif"
    tifffile.imwrite(movie, np.zeros((3, 8, 8), np.uint16))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(movie), "")))

    class _Dialog:
        shown = []

        def __init__(self, frames, parent=None, **kwargs):
            _Dialog.shown.append(np.asarray(frames).shape[0])

        def exec(self):
            return 0

    monkeypatch.setattr(mm, "_Sam2SeedDialog", _Dialog)
    screen._on_sam2_tracking()
    assert _Dialog.shown == [3]


def test_the_ensemble_model_is_picked_from_a_file(screen, monkeypatch):
    if not hasattr(screen, "_uncertainty_ensemble"):
        screen._build_uncertainty_ensemble_setting()
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("/models/second.pth", "")))
    screen._pick_uncertainty_ensemble()
    assert screen._uncertainty_ensemble.currentText() == "/models/second.pth"


def test_a_busy_uncertainty_run_refuses_a_second_and_errors_are_kept(
        screen, monkeypatch):
    taken = []
    monkeypatch.setattr(screen, "_take_uncertainty", taken.append)
    screen._uncertainty_request = {"busy": True}
    assert screen._start_uncertainty({}, threaded=False) is False
    screen._uncertainty_request = None

    def broken(request, models=None):
        raise RuntimeError("model failed")

    monkeypatch.setattr(mm, "_uncertainty_snapshot", broken)
    assert screen._start_uncertainty({"x": 1}, threaded=False) is True
    assert isinstance(taken[0][2], RuntimeError)


def test_a_threaded_uncertainty_run_is_queued_on_its_worker(screen,
                                                           monkeypatch):
    submitted = []

    class _Worker:
        def __init__(self, work, done, name=None):
            pass

        def submit(self, request):
            submitted.append(request)

    monkeypatch.setattr(mm, "_NewestRequestWorker", _Worker)
    screen._uncertainty_request = None
    screen._uncertainty_worker = None
    assert screen._start_uncertainty({"x": 2}, threaded=True) is True
    assert submitted == [{"x": 2}]
    screen._uncertainty_request = None


def test_a_delivery_after_the_screen_is_gone_is_dropped(screen, monkeypatch):
    class _Signal:
        def emit(self, payload):
            raise RuntimeError("deleted")

    monkeypatch.setattr(screen, "_uncertainty_delivered", _Signal())
    assert screen._uncertainty_done({}, None, None) is None
