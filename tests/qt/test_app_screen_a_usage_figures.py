"""The resource strip and the figure views of ``AppScreen``.

The usage bars paint what the sampler read and nothing when it read nothing
(an empty sample must not read as an idle machine); GPUtil is never entered
on a CPU-only host. Figures arriving from a run reveal the figures card, and
the run-folder figure grid finds a run's pictures without walking into
sibling runs and drops anything it cannot draw rather than leaving a null
tile that would shift every caption after it.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt                                      # noqa: E402
from PySide6.QtGui import QImage                                   # noqa: E402

from spacr.qt.screens import app_screen                            # noqa: E402
from spacr.qt.screens.app_screen import (                          # noqa: E402
    AppScreen,
    QtGui_QListWidgetItem_helper,
)

from tests.qt.test_app_screen_a_construction import (               # noqa: E402,F401
    _reset_hover_tooltip_around_each_test,
    built,
    settle,
)

pytestmark = pytest.mark.qt


def _pct(usage_bar) -> int:
    """The integer percentage a UsageBar shows (its label must agree)."""
    assert usage_bar._pct.text() == f"{usage_bar._bar.value()}%"
    return usage_bar._bar.value()


def _gputil(gpus):
    fake = types.ModuleType("GPUtil")
    fake.getGPUs = gpus
    return fake


class _Gpu:
    load = 0.25
    memoryUtil = 0.5


def test_usage_bars_paint_what_the_sampler_read_and_nothing_else(
        qtbot, subtests):
    """A real refresh paints RAM ~ psutil and every bar in 0..100; an absent
    GPU pins GPU/VRAM to 0, a present one shows its load and memory; a
    CPU-only host never calls GPUtil; a psutil that raises and an empty
    sample both leave the bars where they were. The CPU toggle creates one
    labelled bar per logical core (C00..Cnn), fills each in 0..100 and reuses
    them when reopened; an unreadable core count opens the strip with no bars.
    """
    import psutil

    with subtests.test(case="real sample"), built("mask") as scr:
        scr._refresh_usage()
        settle(qtbot, scr)
        assert _pct(scr._usage_ram) == pytest.approx(
            psutil.virtual_memory().percent, abs=5.0)
        for bar in (scr._usage_cpu, scr._usage_gpu, scr._usage_vram):
            assert 0 <= _pct(bar) <= 100

    for name, gpus, expected in (("no GPU", lambda: [], (0, 0)),
                                 ("one GPU", lambda: [_Gpu()], (25, 50))):
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            mp.setitem(sys.modules, "GPUtil", _gputil(gpus))
            mp.setattr(app_screen, "_nvidia_smi_available", lambda: True)
            mp.setattr(app_screen, "_nvml_gpu_usage", lambda: None)
            with built("mask") as scr:
                settle(qtbot, scr)
                scr._usage_gpu.set_value(55)
                scr._usage_vram.set_value(55)
                scr._refresh_usage()
                settle(qtbot, scr)
                assert (_pct(scr._usage_gpu), _pct(scr._usage_vram)) == expected

    with subtests.test(case="CPU-only host"), pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "GPUtil", _gputil(lambda: (_ for _ in ()).throw(
            AssertionError("GPUtil must not run without nvidia-smi"))))
        mp.setattr(app_screen, "_nvidia_smi_available", lambda: False)
        sample = app_screen._sample_usage(False)
        assert (sample["gpu"], sample["vram"]) == (0, 0)

    with built("mask") as scr:
        settle(qtbot, scr)
        with subtests.test(case="psutil raises"), \
                pytest.MonkeyPatch.context() as mp:
            fake = types.ModuleType("psutil")

            def _boom(*a, **k):
                raise RuntimeError("no /proc")

            fake.virtual_memory = fake.cpu_percent = fake.cpu_count = _boom
            scr._usage_ram.set_value(42)
            mp.setitem(sys.modules, "psutil", fake)
            scr._refresh_usage()
            settle(qtbot, scr)
            assert _pct(scr._usage_ram) == 42
        with subtests.test(case="empty sample"):
            scr._usage_ram.set_value(42)
            scr._apply_usage({})
            assert _pct(scr._usage_ram) == 42

    n = int(psutil.cpu_count(logical=True) or 0)
    assert n > 0

    with subtests.test(case=f"{n} cores"), built("mask") as scr:
        assert scr._per_core_bars == [] and scr._per_core_wrap.isHidden()
        scr._btn_cpu_toggle.setChecked(True)
        assert len(scr._per_core_bars) == n
        assert not scr._per_core_wrap.isHidden()
        assert scr._per_core_bars[0]._label.text() == "C00"
        assert scr._per_core_bars[-1]._label.text() == f"C{n - 1:02d}"
        scr._refresh_usage()
        settle(qtbot, scr)
        assert all(0 <= _pct(bar) <= 100 for bar in scr._per_core_bars)
        wanted = [(7 * i) % 101 for i in range(n)]
        scr._apply_usage({"per_core": wanted})
        assert [_pct(bar) for bar in scr._per_core_bars] == wanted
        scr._btn_cpu_toggle.setChecked(False)
        assert scr._per_core_wrap.isHidden()
        scr._btn_cpu_toggle.setChecked(True)
        assert len(scr._per_core_bars) == n

    with subtests.test(case="unknown core count"), built("mask") as scr, \
            pytest.MonkeyPatch.context() as mp:
        fake = types.ModuleType("psutil")
        fake.cpu_count = lambda logical=True: (_ for _ in ()).throw(
            RuntimeError("nope"))
        mp.setitem(sys.modules, "psutil", fake)
        scr._on_toggle_per_core(True)
        assert scr._per_core_bars == []
        assert not scr._per_core_wrap.isHidden()


def _fig():
    from matplotlib.figure import Figure
    fig = Figure(figsize=(2, 2))
    fig.subplots().plot([0, 1, 2], [0, 1, 4])
    return fig


class _Garbage:
    def savefig(self, buf, **k):
        buf.write(b"definitely not a png")

    def get_facecolor(self):
        return "white"


class _Unrenderable:
    def savefig(self, *a, **k):
        raise RuntimeError("no canvas")

    def get_facecolor(self):
        return "white"


def test_run_figures_reach_the_queue_as_numbered_thumbnails(qtbot, subtests):
    """The first figure reveals the hidden figures card and each one is
    queued; a thumbnail item is numbered from 1, centred, and has an icon
    only when the figure rendered to a decodable PNG.
    """
    with subtests.test(case="figures card"), built("mask") as scr:
        assert scr._figures_card.isHidden()
        assert scr._figure_queue.count() == 0
        scr._on_figure_ready(_fig(), "")
        assert not scr._figures_card.isHidden()
        assert scr._figure_queue.count() == 1
        scr._on_figure_ready(_fig())
        assert scr._figure_queue.count() == 2

    with subtests.test(case="collapsed figures pane keeps its header floor"), \
            built("mask") as scr:
        from spacr.qt.gui_scale import _actual_minimum_height
        from spacr.qt.screens.app_screen import _RuntimeViewport
        scr._on_figure_ready(_fig(), "")
        split = scr._runtime_splitter
        pane = split._pane_of(scr._figures_card)
        assert pane is not None and pane.mode == "header"
        split.set_collapsed(pane.name, True)
        assert pane.is_collapsed()
        (viewport,) = scr.findChildren(_RuntimeViewport)
        viewport._sync_floor()
        assert _actual_minimum_height(scr._figures_card) == \
            split._collapsed_extent(pane)

    for name, fig, index, has_icon in (("real figure", _fig(), 4, True),
                                       ("undecodable bytes", _Garbage(), 2,
                                        False),
                                       ("savefig raises", _Unrenderable(), 0,
                                        False)):
        with subtests.test(case=name):
            item = QtGui_QListWidgetItem_helper(fig, index)
            assert item.text() == f"#{index + 1}"
            assert item.icon().isNull() is not has_icon
            if has_icon:
                assert item.textAlignment() == int(Qt.AlignCenter)


def _png(path, size=8):
    image = QImage(size, size, QImage.Format.Format_RGB32)
    image.fill(0x336699)
    assert image.save(str(path), "PNG")


def test_the_figure_grid_shows_only_this_runs_drawable_pictures(
        tmp_path, subtests):
    """A run folder is walked recursively, a screen folder is not (its
    subfolders are sibling runs), only image files count, an unlistable
    folder has none; the screen folders above a run are results/ and the
    screen root, a bare folder gives its parent, and the climb stops at /.
    ``_pictures_from`` returns one non-null pixmap per drawable file titled
    with its relative path; an undecodable image, a PDF whose renderer raises
    and a PDF that renders to nothing are dropped (captions are positional).
    """
    run = tmp_path / "run"
    (run / "regression_qc").mkdir(parents=True)
    _png(run / "volcano.png")
    _png(run / "regression_qc" / "residuals.png")
    (run / "coefficients.csv").write_text("a\n", encoding="utf-8")
    flat = tmp_path / "flat"
    flat.mkdir()
    _png(flat / "volcano.png")
    (flat / "coefficients.csv").write_text("a\n", encoding="utf-8")
    (flat / "settings.json").write_text("{}", encoding="utf-8")
    screen = tmp_path / "screen_root"
    (screen / "regression_2").mkdir(parents=True)
    _png(screen / "fraction_threshold.png")
    _png(screen / "regression_2" / "volcano.png")
    nested = tmp_path / "screen" / "results" / "regression_1"
    nested.mkdir(parents=True)
    bare = tmp_path / "somewhere" / "my_run"
    bare.mkdir(parents=True)
    gone = str(tmp_path / "never-existed")

    names = [
        ("run folder recursive", str(run), {"recursive": True},
         [os.path.join("regression_qc", "residuals.png"), "volcano.png"]),
        ("only pictures count", str(flat), {}, ["volcano.png"]),
        ("screen folder flat", str(screen), {"recursive": False},
         ["fraction_threshold.png"]),
        ("missing flat", gone, {"recursive": False}, []),
        ("missing recursive", gone, {"recursive": True}, []),
    ]
    for name, folder, kwargs, expected in names:
        with subtests.test(case=name):
            assert AppScreen._figure_names_under(folder, **kwargs) == expected

    above = [
        ("under results/", str(nested),
         [str(tmp_path / "screen" / "results"), str(tmp_path / "screen")]),
        ("bare folder", str(bare), [str(tmp_path / "somewhere")]),
        ("filesystem root", "/", []),
        ("empty", "", []),
    ]
    for name, folder, expected in above:
        with subtests.test(case=f"above {name}"):
            assert AppScreen._screen_folders_above(folder) == expected

    from spacr.qt.widgets import figure_queue

    pics = tmp_path / "pics"
    qc = pics / "regression_qc"
    qc.mkdir(parents=True)
    _png(qc / "residuals.png")
    _png(pics / "good.png")
    (pics / "broken.png").write_text("not a png", encoding="utf-8")
    (pics / "summary.pdf").write_bytes(b"%PDF-1.4\n")

    def refusing(_path):
        raise RuntimeError("no PDF backend is available")

    cases = [
        ("readable image", None,
         [os.path.join("regression_qc", "residuals.png")], ["regression_qc"]),
        ("undecodable image", None, ["broken.png", "good.png"], ["good"]),
        ("pdf renderer raises", refusing, ["summary.pdf", "good.png"],
         ["good"]),
        ("pdf renders nothing", lambda _p: None, ["summary.pdf"], []),
    ]
    for name, renderer, files, title_parts in cases:
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            if renderer is not None:
                mp.setattr(figure_queue, "render_pdf_to_image", renderer)
            pixmaps, titles = AppScreen._pictures_from(str(pics), files)
            assert len(pixmaps) == len(titles) == len(title_parts)
            assert all(not p.isNull() for p in pixmaps)
            for title, part in zip(titles, title_parts):
                assert part in title




def test_run_folder_and_page_identity_never_guess(tmp_path, subtests, qtbot):
    """Two spellings of one folder are the same run; a missing or non-path
    folder matches nothing. A page is "on screen" only when the container
    has a current widget that is it or contains it.
    """
    from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

    run = tmp_path / "results" / "regression_1"
    run.mkdir(parents=True)
    same = [
        ("dot-dot spelling", str(run),
         os.path.join(str(run), "..", "regression_1"), True),
        ("empty", "", "/data/run", False),
        ("None left", None, "/data/run", False),
        ("None right", "/data/run", None, False),
        ("not a path", ["/data/run"], "/data/run", False),
    ]
    for name, a, b, expected in same:
        with subtests.test(case=f"same run: {name}"):
            assert AppScreen._same_run_folder(a, b) is expected

    stack = QStackedWidget()
    qtbot.addWidget(stack)
    page, inner = QWidget(), QWidget()
    QVBoxLayout(page).addWidget(inner)
    stack.addWidget(page)
    stack.setCurrentWidget(page)
    empty = types.SimpleNamespace(currentWidget=lambda: None)
    pages = [
        ("no container", None, object(), False),
        ("no page", object(), None, False),
        ("empty container", empty, object(), False),
        ("the page", stack, page, True),
        ("inside the page", stack, inner, True),
    ]
    for name, container, widget, expected in pages:
        with subtests.test(case=f"page: {name}"):
            assert AppScreen._is_the_page(container, widget) is expected
