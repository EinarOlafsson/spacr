"""Pipeline figures ignore the GUI figure preferences (maintainer, 2026-11-19).

With a title size, a font size and a DPI set in Preferences, a figure saved
inside ``pipeline_figures()`` (or with ``save_figure(preferences=False)``) is
byte-identical to the same figure saved with no preferences at all, while an
ordinary save still applies them. ``spacr-run`` and ``run_ops`` save inside
the context, and a process started from one inherits it. In the GUI process
the context changes nothing: a figure saved by a GUI-launched run keeps the
preferences (maintainer refinement, 2026-11-19 12:45).
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure  # noqa: E402

from spacr.figures import style  # noqa: E402
from spacr.plot import save_figure  # noqa: E402

PREFERENCES = {"title_size": 13.6, "font_size": 10.8, "dpi": 154}


@pytest.fixture
def chosen(monkeypatch):
    """Preferences as a user who changed the title size, font size and DPI."""
    from spacr.qt import preferences

    state = {"style": dict(PREFERENCES)}
    monkeypatch.setattr(preferences, "get_figure_style",
                        lambda: dict(state["style"]))
    monkeypatch.setattr(preferences, "get_figure_style_per_graph", lambda: {})
    monkeypatch.setattr(preferences, "get_figure_format", lambda: "png")
    monkeypatch.setattr(preferences, "get_figure_png_dpi", lambda: 300)
    monkeypatch.delenv(style._PIPELINE_FIGURES_ENV, raising=False)
    monkeypatch.setattr(style, "_gui_process", lambda: False)
    return state


def _draw(path, **save):
    """The tuning-curve shape: 5 x 3.8 in, a 9 pt title, saved at 150 dpi."""
    with style.figure_style("print"):
        figure = Figure(figsize=(5.0, 3.8), dpi=150)
        axis = figure.add_subplot(111)
        axis.plot([0, 1, 2], [3, 1, 2])
        axis.set_title("A1: decode auto thresholds", fontsize=9)
        written = save_figure(figure, str(path), fmt="png", dpi=150, **save)
    return written, axis.title.get_size()


def test_a_pipeline_figure_equals_one_drawn_without_preferences(tmp_path,
                                                                chosen):
    with style.pipeline_figures():
        piped, piped_title = _draw(tmp_path / "piped.png")
    keyword, _ = _draw(tmp_path / "keyword.png", preferences=False)
    chosen["style"] = {}
    plain, plain_title = _draw(tmp_path / "plain.png")
    assert piped_title == plain_title == 9
    data = {name: open(path, "rb").read()
            for name, path in (("piped", piped), ("keyword", keyword),
                               ("plain", plain))}
    assert data["piped"] == data["plain"] == data["keyword"]


def test_an_ordinary_save_still_applies_the_preferences(tmp_path, chosen):
    styled, title = _draw(tmp_path / "styled.png")
    assert title == pytest.approx(13.6)
    with style.pipeline_figures():
        piped, _ = _draw(tmp_path / "piped.png")
    assert open(styled, "rb").read() != open(piped, "rb").read()


def test_the_context_reaches_child_processes_and_ends_with_the_run(chosen):
    with style.pipeline_figures():
        assert style._pipeline_figures_active()
        assert os.environ[style._PIPELINE_FIGURES_ENV] == "1"
        child = subprocess.run(
            [sys.executable, "-c",
             "from spacr.figures import style; "
             "print(style._pipeline_figures_active())"],
            capture_output=True, text=True, check=True)
    assert child.stdout.strip() == "True"
    assert not style._pipeline_figures_active()
    assert style._PIPELINE_FIGURES_ENV not in os.environ


def test_a_gui_launched_run_keeps_the_preferences(tmp_path, chosen,
                                                  monkeypatch):
    monkeypatch.setattr(style, "_gui_process", lambda: True)
    with style.pipeline_figures():
        assert not style._pipeline_figures_active()
        assert style._PIPELINE_FIGURES_ENV not in os.environ
        gui, title = _draw(tmp_path / "gui.png")
    assert title == pytest.approx(13.6)
    styled, _ = _draw(tmp_path / "styled.png")
    assert open(gui, "rb").read() == open(styled, "rb").read()


def test_run_ops_and_the_cli_run_inside_the_context():
    from spacr import cli, ops_engine
    from spacr.qt import bridge

    assert ops_engine._in_pipeline_figures(style._pipeline_figures_active)()
    assert ops_engine.run_ops.__wrapped__.__name__ == "run_ops"
    import inspect

    assert "pipeline_figures()" in inspect.getsource(cli.cmd_run)
    assert "pipeline_figures()" not in inspect.getsource(bridge.PipelineWorker.run)
