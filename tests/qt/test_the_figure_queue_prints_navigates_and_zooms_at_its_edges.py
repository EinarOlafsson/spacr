"""The figure queue at its edges: print copies, navigation, vector zoom.

Pinned here, each as what the user sees or gets:

* a print copy whose first write fails is written again on a fresh Agg
  canvas; when that fails too, no file is claimed; a PDF that fails is
  removed rather than left half-written, and the PNG still counts;
* a queue that has never shown a figure follows the one that arrives, and
  a figure arriving while the user looks elsewhere does not move them even
  when the navigation strip cannot be redrawn;
* Next moves one figure on; the counts of figures held in RAM, spilled to
  disk, and crisp renders in flight read what the queue holds;
* a figure whose vector page was never written is marked failed and keeps
  its raster; zooming a figure with no raster or no view does nothing;
* a wheel turn outside the data does nothing; inside, the canvas redraws.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt.widgets import figure_queue as fq  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def prefs(monkeypatch, tmp_path_factory):
    from spacr.qt import preferences as preferences_module

    store = tmp_path_factory.mktemp("figq_edges_prefs") / "prefs.ini"
    monkeypatch.setattr(
        preferences_module, "_settings",
        lambda: QSettings(str(store), QSettings.Format.IniFormat))
    preferences_module.set_figure_format("png")
    return preferences_module


def _fig(seed: int = 0) -> Figure:
    figure = Figure(figsize=(3.0, 2.0))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [seed, seed + 1, seed])
    return figure


def _queue(qtbot):
    queue = fq.FigureQueue(ram_cap=100)
    qtbot.addWidget(queue)
    queue.set_live_canvas_enabled(False)
    return queue


SCREEN = ("#101418", "#e6e6e6", "#e6e6e6")


# ---------------------------------------------------------------------------
# Print copies
# ---------------------------------------------------------------------------

def test_a_print_copy_is_written_again_on_a_fresh_canvas(tmp_path,
                                                         monkeypatch):
    real_write = fq._write_print_file
    attempts = []

    def flaky(figure, path, fmt, dpi):
        attempts.append(fmt)
        if len(attempts) == 1:
            raise RuntimeError("canvas was closed")
        return real_write(figure, path, fmt, dpi)

    monkeypatch.setattr(fq, "_write_print_file", flaky)
    png = tmp_path / "fig.png"
    assert fq._render_print_copy(_fig(), str(png), 50, 0, False, SCREEN)
    assert attempts == ["png", "png"]
    assert png.is_file() and png.stat().st_size > 0


def test_a_print_copy_that_cannot_be_written_claims_no_file(tmp_path,
                                                           monkeypatch):
    def broken(figure, path, fmt, dpi):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fq, "_write_print_file", broken)
    png = tmp_path / "fig.png"
    assert fq._render_print_copy(_fig(), str(png), 50, 0, True,
                                 SCREEN) is False
    assert not png.exists()


@pytest.mark.parametrize("leave_a_stub", [True, False])
def test_a_failed_print_pdf_is_removed_and_the_png_kept(tmp_path, monkeypatch,
                                                        caplog, leave_a_stub):
    real_write = fq._write_print_file

    def no_pdf(figure, path, fmt, dpi):
        if fmt == "pdf":
            if leave_a_stub:
                open(path, "wb").close()
            raise RuntimeError("no PDF backend")
        return real_write(figure, path, fmt, dpi)

    monkeypatch.setattr(fq, "_write_print_file", no_pdf)
    png = tmp_path / "fig.png"
    with caplog.at_level("WARNING", logger=fq.LOG.name):
        assert fq._render_print_copy(_fig(), str(png), 50, 0, True, SCREEN)
    assert png.is_file()
    assert not fq._sibling_pdf(png).exists()
    assert "print PDF export failed" in caplog.text


# ---------------------------------------------------------------------------
# Navigation and counts
# ---------------------------------------------------------------------------

def test_a_queue_that_has_shown_nothing_follows_the_new_figure(qtbot):
    queue = _queue(qtbot)
    queue._current = None
    assert queue._following_the_tail(3) is True


def test_a_figure_arriving_elsewhere_does_not_move_the_user(qtbot,
                                                            monkeypatch):
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue.show_index(0)

    def broken():
        raise RuntimeError("navigation strip is being rebuilt")

    monkeypatch.setattr(queue, "_refresh_nav", broken)
    queue.add_figure(_fig(2))
    assert queue.count() == 3
    assert queue._current == 0


def test_next_moves_one_on_and_the_counts_read_what_is_held(qtbot):
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.add_figure(_fig(1))
    queue.show_index(0)

    queue.show_next()
    assert queue._current == 1
    queue.show_next()
    assert queue._current == 1

    assert queue.ram_resident() == len(queue._ram) >= 1
    assert queue.spilled_count() == queue.count() - queue.ram_resident()
    assert queue.active_jobs() == queue._jobs.active_jobs()
    assert queue.is_busy() is queue._jobs.is_busy()


# ---------------------------------------------------------------------------
# The vector page
# ---------------------------------------------------------------------------

def test_a_figure_whose_vector_page_was_never_written_keeps_its_raster(
        qtbot, monkeypatch, caplog):
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.show_index(0)
    monkeypatch.setattr(queue, "_figure_format_is_pdf", lambda: True)
    pdf = fq._sibling_pdf(queue._png_paths[0])
    if pdf.exists():
        pdf.unlink()
    queue._pdf_state.pop(0, None)

    with caplog.at_level("WARNING", logger=fq.LOG.name):
        queue._request_pdf_refinement(0)
    assert queue._pdf_state[0] == "failed"
    assert "has no vector page" in caplog.text
    assert not queue.is_busy()


def test_zooming_a_figure_with_no_raster_renders_nothing(qtbot):
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.show_index(0)
    queue._png_paths.pop(0)
    before = dict(queue._pdf_render_px)
    queue._on_view_zoomed(3.0)
    assert queue._pdf_render_px == before
    assert not queue.is_busy()


def test_zooming_when_the_view_is_gone_renders_nothing(qtbot, monkeypatch):
    queue = _queue(qtbot)
    queue.add_figure(_fig(0))
    queue.show_index(0)
    fq._sibling_pdf(queue._png_paths[0]).write_bytes(b"%PDF-1.4\n")

    class Gone:
        def viewport(self):
            raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(queue, "_view", Gone())
    before = dict(queue._pdf_render_px)
    queue._on_view_zoomed(3.0)
    assert queue._pdf_render_px == before
    assert not queue.is_busy()


# ---------------------------------------------------------------------------
# The live canvas wheel
# ---------------------------------------------------------------------------

def test_a_wheel_turn_outside_the_data_does_nothing(qtbot):
    queue = _queue(qtbot)
    figure = _fig(0)
    axes = figure.axes[0]
    limits = (axes.get_xlim(), axes.get_ylim())
    event = types.SimpleNamespace(inaxes=axes, xdata=None, ydata=1.0,
                                  button="up")
    queue._on_canvas_scroll(event)
    assert (axes.get_xlim(), axes.get_ylim()) == limits


def test_a_wheel_turn_in_the_data_zooms_and_redraws(qtbot, monkeypatch):
    queue = _queue(qtbot)
    figure = _fig(0)
    axes = figure.axes[0]
    axes.set_xlim(0, 2)
    redraws = []
    monkeypatch.setattr(queue, "_canvas",
                        types.SimpleNamespace(
                            draw_idle=lambda: redraws.append(True)),
                        raising=False)
    event = types.SimpleNamespace(inaxes=axes, xdata=1.0, ydata=1.0,
                                  button="up")
    queue._on_canvas_scroll(event)
    left, right = axes.get_xlim()
    assert right - left < 2.0
    assert redraws == [True]
