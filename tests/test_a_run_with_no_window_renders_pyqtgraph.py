"""A pipeline with no display still writes its pyqtgraph figures.

The screen draws every regression plot in pyqtgraph and the run wrote the
same plots again in matplotlib -- two pictures of one number, from two code
paths that can disagree. The move to one renderer only works if the RUN can
use it, and a run has no window: `spacr-run` from a terminal and a notebook
on a server both reach this code with no display at all.

So the contract under test is the one the request turns on: a figure the
pipeline renders offscreen is a real file, in the format the user asked for,
announced to the gallery, and when Qt cannot start at all the refusal SAYS
so instead of writing nothing and reporting success.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.figures import headless                              # noqa: E402


def _frame(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "gene": ["nc"] * 25 + ["pc"] * 25 + ["GRA14"] * 25,
        "fraction": np.r_[rng.normal(0.2, 0.05, 25),
                          rng.normal(0.8, 0.05, 25),
                          rng.normal(0.5, 0.10, 25)],
    })


def _spec(frame=None):
    from spacr.qt.widgets.grouped_plot import PlotSpec

    return PlotSpec(frame=frame if frame is not None else _frame(),
                    value="fraction", group="gene", unit="well",
                    title="fraction by gene", x_label="gene",
                    y_label="fraction")


# ---------------------------------------------------------------------------
# It writes a real file
# ---------------------------------------------------------------------------

def test_a_render_with_no_display_writes_a_real_figure(tmp_path, qapp):
    path = headless.render_offscreen(_spec(), str(tmp_path / "fraction.pdf"),
                           publish=False)
    assert path, "the renderer wrote nothing and said nothing"
    assert os.path.exists(path)
    # NOT AN EMPTY PAGE. A PDF header alone is a few hundred bytes; a drawn
    # scene is thousands. The assertion is deliberately loose -- it is
    # guarding against "the widget was exported before it was laid out",
    # which produces a valid, empty file.
    assert os.path.getsize(path) > 2000, os.path.getsize(path)


def test_the_extension_follows_the_format_preference(tmp_path, qapp,
                                                     monkeypatch):
    """A PNG written to a .pdf name is a file no viewer opens."""
    from spacr import plot as plot_module

    monkeypatch.setattr(plot_module, "figure_output_preferences",
                        lambda: ("png", 150))
    path = headless.render_offscreen(_spec(), str(tmp_path / "fraction.pdf"),
                           publish=False)
    assert path and path.endswith(".png"), path
    assert not os.path.exists(str(tmp_path / "fraction.pdf"))


def test_an_explicit_format_still_wins(tmp_path, qapp, monkeypatch):
    from spacr import plot as plot_module

    monkeypatch.setattr(plot_module, "figure_output_preferences",
                        lambda: ("png", 150))
    path = headless.render_offscreen(_spec(), str(tmp_path / "fraction.png"),
                           fmt="pdf", publish=False)
    assert path and path.endswith(".pdf"), path


def test_the_parent_directory_is_created(tmp_path, qapp):
    """A run folder's figure subdirectory does not exist before the first
    figure goes into it."""
    target = tmp_path / "results" / "figures" / "fraction.pdf"
    path = headless.render_offscreen(_spec(), str(target), publish=False)
    assert path and os.path.exists(path)


# ---------------------------------------------------------------------------
# Saved and visible are the same event
# ---------------------------------------------------------------------------

def test_the_rendered_file_reaches_the_gallery(tmp_path, qapp):
    from spacr import figure_sink

    seen = []
    figure_sink.set_file_sink(lambda path, title: seen.append((path, title)))
    try:
        path = headless.render_offscreen(_spec(), str(tmp_path / "fraction.pdf"))
    finally:
        figure_sink.clear_sink()
    assert seen, "the figure was written and never announced"
    assert seen[0][0] == path
    assert seen[0][1] == "fraction by gene", "the tile has no name"


def test_publishing_can_be_turned_off_for_an_intermediate(tmp_path, qapp):
    """Not every render is a deliverable; a bundle publishes once, not per
    file it happens to write on the way."""
    from spacr import figure_sink

    seen = []
    figure_sink.set_file_sink(lambda path, title: seen.append(path))
    try:
        headless.render_offscreen(_spec(), str(tmp_path / "fraction.pdf"),
                        publish=False)
    finally:
        figure_sink.clear_sink()
    assert seen == []


def test_a_sink_that_raises_does_not_lose_the_file(tmp_path, qapp):
    """The file is already on disk; a GUI that has gone away must not take
    the run's output with it."""
    from spacr import figure_sink

    def _angry(path, title):
        raise RuntimeError("the gallery is gone")

    figure_sink.set_file_sink(_angry)
    try:
        path = headless.render_offscreen(_spec(), str(tmp_path / "fraction.pdf"))
    finally:
        figure_sink.clear_sink()
    assert path and os.path.exists(path)


# ---------------------------------------------------------------------------
# The bundle, from a run that has no screen
# ---------------------------------------------------------------------------

def test_a_run_writes_the_same_bundle_the_screen_does(tmp_path, qapp):
    folder = headless.render_bundle(_spec(), str(tmp_path), "fraction")
    assert folder, "no bundle was written"
    written = sorted(os.listdir(folder))
    assert "data.csv" in written
    assert "statistics.csv" in written
    assert "settings.json" in written
    assert any(f.endswith(".pdf") for f in written), written
    assert any(f.endswith(".png") for f in written), written


def test_the_bundle_statistics_are_the_comparison_on_screen(tmp_path, qapp):
    """The numbers beside the figure describe the groups the figure draws --
    which is the whole reason the pair is written together."""
    folder = headless.render_bundle(_spec(), str(tmp_path), "fraction")
    text = open(os.path.join(folder, "statistics.csv"),
                encoding="utf-8").read()
    for label in ("nc", "pc", "GRA14"):
        assert f"n [{label}]" in text, label
    assert "unit,well" in text, "the replicate unit is not recorded"


# ---------------------------------------------------------------------------
# When it cannot, it says so
# ---------------------------------------------------------------------------

def test_no_qt_at_all_refuses_out_loud(tmp_path, monkeypatch, caplog):
    """Instruction 106. A run that silently stops writing figures is the
    worst outcome here, so the refusal names the fix."""
    monkeypatch.setattr(headless, "application",
                        lambda: (None, headless.NO_QT))
    with caplog.at_level("WARNING"):
        path = headless.render_offscreen(_spec(), str(tmp_path / "fraction.pdf"))
    assert path is None
    assert "cannot be rendered" in caplog.text
    assert "pip install spacr" in caplog.text


def test_the_bundle_refuses_the_same_way(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(headless, "application",
                        lambda: (None, headless.NO_PLATFORM))
    with caplog.at_level("WARNING"):
        folder = headless.render_bundle(_spec(), str(tmp_path), "fraction")
    assert folder is None
    assert "QT_QPA_PLATFORM=offscreen" in caplog.text, (
        "the refusal has to name what would fix it")


def test_an_existing_application_is_reused_not_replaced(qapp):
    """A module called from the desktop app must not build a second
    QApplication -- Qt allows exactly one."""
    app, refusal = headless.application()
    assert refusal == ""
    assert app is qapp


def test_a_frame_with_nothing_in_it_writes_nothing(tmp_path, qapp):
    """An empty table is not a figure. Writing a blank page for it puts a
    tile in the gallery that says nothing."""
    empty = pd.DataFrame({"gene": [], "fraction": []})
    assert headless.render_offscreen(_spec(empty), str(tmp_path / "empty.pdf")) is None


def test_a_worker_thread_refuses_rather_than_building_a_widget(tmp_path,
                                                               qapp):
    """A QWidget built off the GUI thread lives on a thread about to end.

    The regression QC suite runs on the run's own worker thread under a live
    application, and building a scene there segfaulted the process twice in
    places that had nothing to do with it. The readiness check answers this,
    and this module has to be asking it -- which is the whole point of not
    starting a QApplication of its own.
    """
    import threading

    got = {}

    def _try():
        got["path"] = headless.render_offscreen(_spec(), str(tmp_path / "worker.pdf"),
                                      publish=False)
        got["reason"] = headless.application()[1]

    worker = threading.Thread(target=_try, name="a-run-worker")
    worker.start()
    worker.join(30)
    assert got.get("path") is None, "a widget was built off the GUI thread"
    assert "GUI thread" in got.get("reason", ""), got.get("reason")
