"""Saving a figure and showing it are the same event.

Instruction 139 C, reported 2026-08-18: "all graphs should be sable and
observable in the software, currently several graphs are saved but I cannot
see them in the software".

THE CAUSE. A figure reached the GUI by one route -- `spacr/qt/bridge.py`
replaces `matplotlib.pyplot.show` and emits everything in
`plt.get_fignums()`. So a figure was visible if and only if it was IN pyplot's
registry AND somebody called `show`.

`spacr.regression_qc` fails both halves, which is why its ~19-panel report was
invisible: it builds bare `matplotlib.figure.Figure` objects -- the correct
thing for a library to do -- and writes them with savefig. Every panel on
disk, none in the application.
"""
from __future__ import annotations

import os

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

from spacr import figure_sink  # noqa: E402


@pytest.fixture(autouse=True)
def _no_sink_left_behind():
    """A sink is global; a test that leaks one breaks the next."""
    figure_sink.clear_sink()
    yield
    figure_sink.clear_sink()


def _figure():
    fig = Figure()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    return fig


def test_a_published_figure_is_written_and_announced(tmp_path):
    seen = []
    figure_sink.set_sink(lambda fig, path: seen.append(path))

    written = figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))

    assert written and os.path.isfile(written)
    assert seen == [written]


def test_a_figure_pyplot_never_saw_is_still_announced(tmp_path):
    """The whole point. `plt.get_fignums()` is empty and it still arrives."""
    import matplotlib.pyplot as plt

    plt.close("all")
    seen = []
    figure_sink.set_sink(lambda fig, path: seen.append(path))

    figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))

    assert plt.get_fignums() == [], "the fixture figure must not be in pyplot"
    assert len(seen) == 1


def test_headless_still_writes_the_file(tmp_path):
    """`spacr-run` and a notebook install no sink. The run's output must not
    depend on a GUI being attached."""
    assert figure_sink.sink() is None
    written = figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))
    assert written and os.path.isfile(written)


def test_a_sink_that_raises_does_not_lose_the_file(tmp_path):
    """A GUI that has gone away must not take the run's output with it."""
    def angry(fig, path):
        raise RuntimeError("the window is gone")

    figure_sink.set_sink(angry)
    written = figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))
    assert written and os.path.isfile(written)


def test_the_save_honours_the_figure_format_preference(tmp_path, monkeypatch):
    """Through `spacr.plot.save_figure`, not a literal extension -- a
    complaint this project has already had twice."""
    from spacr import plot

    calls = []
    real = plot.save_figure
    monkeypatch.setattr(plot, "save_figure",
                        lambda *a, **k: calls.append((a, k)) or real(*a, **k))

    figure_sink.publish(_figure(), str(tmp_path / "panel.pdf"))
    assert calls, "publish did not go through save_figure"


def test_publish_without_a_path_announces_and_writes_nothing(tmp_path):
    seen = []
    figure_sink.set_sink(lambda fig, path: seen.append(path))
    assert figure_sink.publish(_figure()) is None
    assert seen == [None]


def test_close_happens_after_the_sink_has_had_the_figure(tmp_path):
    """A cleared figure has nothing left to render."""
    axes_seen = []
    figure_sink.set_sink(lambda fig, path: axes_seen.append(len(fig.axes)))

    figure_sink.publish(_figure(), str(tmp_path / "p.pdf"), close=True)

    assert axes_seen == [1], "the sink got an already-cleared figure"


def test_set_sink_hands_back_the_previous_one():
    first, second = (lambda *a: None), (lambda *a: None)
    assert figure_sink.set_sink(first) is None
    assert figure_sink.set_sink(second) is first
    assert figure_sink.sink() is second


def test_the_qc_report_goes_through_the_sink(tmp_path):
    """The suite the report was about. `_save` is its one funnel."""
    import inspect

    from spacr import regression_qc

    source = inspect.getsource(regression_qc._save)
    assert "publish(fig, path" in source
    # THE CALL, not the word. The docstring explains what it replaced, so a
    # bare substring match finds its own prose and passes on a function that
    # still writes directly.
    assert "fig.savefig(" not in source, (
        "the QC panels still write directly, so they are still invisible")


def test_the_bridge_installs_and_clears_the_sink():
    """A sink left installed after a run holds the worker alive and emits
    into a dead signal on the next one."""
    import inspect

    from spacr.qt import bridge

    source = inspect.getsource(bridge)
    assert "set_sink(_publish_figure)" in source
    assert "clear_sink()" in source


def test_a_figure_that_cannot_be_cleared_is_not_an_error(tmp_path):
    """`close=True` is best-effort. A panel that parked something odd on its
    figure must not take down the run that was only trying to save it."""
    class Awkward(Figure):
        def clf(self, *args, **kwargs):
            raise RuntimeError("this figure refuses to be cleared")

    fig = Awkward()
    fig.add_subplot(111).plot([0, 1], [0, 1])
    written = figure_sink.publish(fig, str(tmp_path / "p.pdf"), close=True)
    assert written and os.path.isfile(written)
