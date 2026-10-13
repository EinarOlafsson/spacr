"""An arriving figure's raster is decoded on the run's worker, not the GUI.

Lesson 08's Measure run (clip batch_20_batch_settings) held the GUI thread
98-269 ms each time a figure arrived. Measured offscreen on 2026-10-13 at
2560x1440 with a real run, a 2 ms heartbeat and a sampler of the GUI
thread's stack: :meth:`FigureQueue.add_figure` loaded the PNG the worker had
written (up to 4034 x 3591 px) and smooth-scaled it to the strip thumbnail
on the GUI thread, 23-33 ms alone and several times that while the run's
worker drew its next figure.

The bridge now calls :func:`_decode_display_png` on the worker after
:func:`render_figure_to_png`; the gallery wraps the decoded picture and
thumbnail. These tests hold the pictures to those the GUI thread made from
the same file, and time the GUI thread's share against the old one.
"""
from __future__ import annotations

import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                  # noqa: E402
import pytest                                                       # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtGui import QImage, QPixmap                           # noqa: E402

from spacr.qt.hidpi import scaled_for                               # noqa: E402
from spacr.qt.widgets import figure_queue as fq                     # noqa: E402
from spacr.qt.widgets.figure_queue import THUMB_SIZE, FigureQueue   # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _empty_store(monkeypatch):
    monkeypatch.setattr(fq, "_DECODED", type(fq._DECODED)())


def _write_png(path, transparent=False, panels=3, side=600):
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, panels, figsize=(3 * panels, 3))
    for ax in np.atleast_1d(axes):
        ax.imshow(rng.random((side, side)), cmap="gray")
        ax.set_title("crop")
    fig.savefig(str(path), dpi=100, transparent=transparent,
                facecolor="none" if transparent else "#101820")
    plt.close(fig)
    return str(path)


def _on_a_worker(function, *args):
    box = {}

    def run():
        box["value"] = function(*args)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    return box["value"]


def _same(a: QImage, b: QImage) -> bool:
    return a.format() == b.format() and a == b


@pytest.mark.parametrize("transparent", [False, True])
def test_the_decoded_picture_is_the_pixmap_the_gui_thread_loads(
        qapp, tmp_path, transparent):
    path = _write_png(tmp_path / "f.png", transparent=transparent)
    assert _on_a_worker(fq._decode_display_png, path) is True
    image, _, _ = fq._take_decoded(path)
    assert _same(QPixmap.fromImage(image).toImage(), QPixmap(path).toImage())
    assert fq._take_decoded(path) is None


def test_the_decoded_thumbnail_is_the_one_the_strip_scales(qtbot, tmp_path):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    path = _write_png(tmp_path / "f.png")
    _on_a_worker(fq._decode_display_png, path)
    decoded = fq._take_decoded(path)
    assert decoded[1] is not None
    assert decoded[2] == fq.device_ratio(queue._list)
    gui = scaled_for(QPixmap(path), queue._list, THUMB_SIZE)
    worker = QPixmap.fromImage(decoded[1])
    assert worker.devicePixelRatio() == gui.devicePixelRatio()
    assert _same(worker.toImage(), gui.toImage())


def test_add_figure_shows_the_decoded_picture_and_files_the_png(
        qtbot, tmp_path, monkeypatch):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    path = _write_png(tmp_path / "f.png")
    expected = QPixmap(path).toImage()
    _on_a_worker(fq._decode_display_png, path)
    loads = []
    real = fq.QPixmap

    class _Counting(real):
        def __init__(self, *args):
            if args and isinstance(args[0], str):
                loads.append(args[0])
            super().__init__(*args)

    monkeypatch.setattr(fq, "QPixmap", _Counting)
    fig = plt.figure()
    index = queue.add_figure(fig, prerendered_png=path)
    assert loads == []
    assert fq._take_decoded(path) is None
    stored = queue._png_paths[index]
    assert _same(QImage(stored).convertToFormat(expected.format()), expected)
    assert _same(queue._ram[index].toImage(), expected)
    plt.close(fig)


def test_a_thumbnail_scaled_at_another_ratio_is_scaled_again(qtbot, tmp_path):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    path = _write_png(tmp_path / "f.png")
    pixmap = QPixmap(path)
    stale = scaled_for(QImage(path), fq._Ratio(3.0), THUMB_SIZE)
    icon = queue._thumb_icon(pixmap, (pixmap.toImage(), stale, 3.0))
    want = scaled_for(pixmap, queue._list, THUMB_SIZE)
    got = icon.pixmap(THUMB_SIZE, want.devicePixelRatio())
    assert got.size() == want.size()


def test_only_the_newest_decoded_pictures_are_kept(qapp, tmp_path):
    paths = [_write_png(tmp_path / f"f{n}.png", panels=1, side=50)
             for n in range(fq._DECODED_KEEP + 2)]
    for path in paths:
        assert fq._decode_display_png(path)
    assert list(fq._DECODED) == paths[-fq._DECODED_KEEP:]


def test_an_unreadable_raster_is_left_to_the_gui_thread(qapp, tmp_path):
    assert fq._decode_display_png(str(tmp_path / "missing.png")) is False
    assert fq._take_decoded(None) is None
    assert fq._take_decoded(str(tmp_path / "missing.png")) is None


def test_a_format_pixmaps_do_not_convert_is_passed_through(qapp):
    image = QImage(4, 4, QImage.Format_Grayscale8)
    assert fq._as_pixmap_image(image) is image


def test_adopting_a_figure_holds_the_gui_thread_far_less(qtbot, tmp_path):
    """Self-calibrated: the GUI thread's old share against its new one."""
    queue = FigureQueue()
    qtbot.addWidget(queue)
    path = _write_png(tmp_path / "big.png", panels=6, side=1200)

    def old():
        started = time.perf_counter()
        pixmap = QPixmap(path)
        scaled_for(pixmap, queue._list, THUMB_SIZE)
        return time.perf_counter() - started

    def new():
        _on_a_worker(fq._decode_display_png, path)
        decoded = fq._take_decoded(path)
        started = time.perf_counter()
        QPixmap.fromImage(decoded[0])
        queue._thumb_icon(None, decoded)
        return time.perf_counter() - started

    before = sorted(old() for _ in range(5))[2]
    now = sorted(new() for _ in range(5))[2]
    assert now < before / 3, (now, before)


def test_the_gallerys_own_renders_are_not_user_runs(qtbot):
    """A vector-page refinement must not build Home a run banner."""
    queue = FigureQueue()
    qtbot.addWidget(queue)
    assert queue._jobs._user_visible is False
    assert queue._jobs._app_key == "figures"


class _Handle:
    def __init__(self, app_key, user_visible=True):
        self.app_key = app_key
        self.user_visible = user_visible


def _fake_registry():
    from PySide6.QtCore import QObject, Signal

    class _Registry(QObject):
        changed = Signal()

        def __init__(self):
            super().__init__()
            self.handles = []

        def active(self):
            return list(self.handles)

        def start(self, key, **kwargs):
            handle = _Handle(key, **kwargs)
            self.handles.append(handle)
            self.changed.emit()
            return handle

        def finish(self, handle):
            self.handles.remove(handle)
            self.changed.emit()

    return _Registry()


def test_a_housekeeping_job_does_not_walk_the_widgets_for_fractals(
        qapp, monkeypatch):
    """A refinement starting mid-run used to cost the animation policy a
    walk over every widget in the application, 115 ms on the GUI thread."""
    from spacr.qt import bridge
    from spacr.qt.widgets import ambient

    fake = _fake_registry()
    monkeypatch.setattr(bridge, "registry", lambda: fake)
    monkeypatch.setattr(ambient, "_analysis_gpu_present", lambda: False)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    walks = []
    monkeypatch.setattr(ambient, "_live_spaceout_fractals",
                        lambda: walks.append(1) or [])
    policy = ambient._animation_resource_policy()
    try:
        run = fake.start("measure")
        assert policy.pauses_animation()
        before = len(walks)
        refine = fake.start("figures", user_visible=False)
        fake.finish(refine)
        assert len(walks) == before
        second = fake.start("measure")
        assert len(walks) == before + 1
        fake.finish(second)
        fake.finish(run)
        assert not policy.pauses_animation()
    finally:
        monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
