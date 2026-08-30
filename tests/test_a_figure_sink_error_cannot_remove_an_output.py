"""Publishing a figure: the file is written before the sink is told.

The docstring states the ordering rule and its reason -- "Saving occurs before
the best-effort sink notification, so a display-sink error cannot remove an
output file that was written successfully". Every guard here protects that: a
sink that raises, a close that fails, a caller with no path.

A figure that reached disk and then vanished because a GUI panel was closing is
the failure this ordering prevents, and it is the kind nobody reproduces.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def a_figure():
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    yield figure
    plt.close(figure)


@pytest.fixture(autouse=True)
def _no_sink():
    """Leave the module-level sink exactly as it was found."""
    from spacr import figure_sink

    figure_sink.clear_sink()
    yield
    figure_sink.clear_sink()


def test_no_figure_writes_nothing(tmp_path):
    """The first guard: a caller that produced no figure gets None.

    Returning a path for a figure that does not exist would have the caller
    report an output it cannot open.
    """
    from spacr.figure_sink import publish

    assert publish(None, path=str(tmp_path / "x.png")) is None


def test_no_path_publishes_to_the_sink_without_writing(a_figure, tmp_path):
    """The path guard: a live panel wants the figure, not a file.

    This is the GUI's normal case, and it must not write anything to disk --
    a run that quietly littered a screen folder with every previewed figure
    would fill it.
    """
    from spacr import figure_sink

    seen = []
    figure_sink.set_sink(lambda fig, written: seen.append((fig, written)))

    assert figure_sink.publish(a_figure) is None
    assert len(seen) == 1
    assert seen[0][1] is None
    assert not list(tmp_path.iterdir())


def test_a_sink_that_raises_does_not_lose_the_written_file(a_figure, tmp_path):
    """The ordering rule, stated as a test.

    The sink is best-effort and the file is not. A panel closing mid-run
    raises here, and the path must still come back -- otherwise the caller
    reports no output for a figure that is on disk.
    """
    from spacr import figure_sink

    def refuse(_fig, _written):
        raise RuntimeError("the display panel is gone")

    figure_sink.set_sink(refuse)
    target = tmp_path / "figure.png"

    written = figure_sink.publish(a_figure, path=str(target))

    assert written and os.path.isfile(written)


def test_a_figure_that_will_not_clear_still_returns_its_path(a_figure,
                                                             tmp_path,
                                                             monkeypatch):
    """The close guard, which runs after everything that matters.

    ``clf`` on a figure whose canvas is already gone raises, and by then the
    file is written and the sink is told. Losing the return value there would
    discard work that entirely succeeded.
    """
    from spacr import figure_sink

    def refuse():
        raise RuntimeError("this figure has no canvas any more")

    monkeypatch.setattr(a_figure, "clf", refuse)
    target = tmp_path / "figure.png"

    written = figure_sink.publish(a_figure, path=str(target), close=True)

    assert written and os.path.isfile(written)


def test_the_sink_is_told_the_path_that_was_written(a_figure, tmp_path):
    """What the sink receives, which is how a panel offers "open in folder"."""
    from spacr import figure_sink

    seen = []
    figure_sink.set_sink(lambda fig, written: seen.append(written))
    target = tmp_path / "figure.png"

    figure_sink.publish(a_figure, path=str(target))

    assert seen and seen[0] and os.path.isfile(seen[0])


def test_clearing_the_sink_stops_notifications(a_figure, tmp_path):
    """clear_sink, so the fixture's cleanup is itself covered."""
    from spacr import figure_sink

    seen = []
    figure_sink.set_sink(lambda fig, written: seen.append(written))
    figure_sink.clear_sink()

    figure_sink.publish(a_figure, path=str(tmp_path / "figure.png"))

    assert seen == []


# ---------------------------------------------------------------------------
# publish_file — the half publish() cannot cover
# ---------------------------------------------------------------------------

def test_a_file_somebody_else_wrote_is_announced(tmp_path):
    """The route bug 139 C was filed for.

    A pyqtgraph scene exported by FastPlot.export is a finished file and never
    was a matplotlib Figure, so the figure sink has nothing to render. Without
    this route, moving a plot to the fast renderer silently took it out of the
    gallery -- the figure was on disk and the user could not see it.
    """
    from spacr import figure_sink

    seen = []
    figure_sink.set_file_sink(lambda path, title: seen.append((path, title)))
    target = tmp_path / "volcano.png"
    target.write_bytes(b"\x89PNG")

    returned = figure_sink.publish_file(target, title="Volcano")

    assert returned == str(target)
    assert seen == [(str(target), "Volcano")]


def test_a_file_sink_that_raises_does_not_lose_the_path(tmp_path):
    """The same ordering rule as publish, for a file already on disk.

    The announcement is best-effort and the file is not. A GUI that has gone
    away must not take the run's output with it.
    """
    from spacr import figure_sink

    def refuse(_path, _title):
        raise RuntimeError("the gallery is gone")

    figure_sink.set_file_sink(refuse)
    target = tmp_path / "volcano.png"
    target.write_bytes(b"\x89PNG")

    assert figure_sink.publish_file(target) == str(target)


def test_no_path_announces_nothing():
    """The first guard: there is no file to announce."""
    from spacr import figure_sink

    seen = []
    figure_sink.set_file_sink(lambda path, title: seen.append(path))

    assert figure_sink.publish_file(None) is None
    assert figure_sink.publish_file("") is None
    assert seen == []


def test_a_path_with_no_sink_installed_is_still_returned(tmp_path):
    """Arc 79 -> 84: nothing to tell, and the caller still gets its path.

    Headless runs have no gallery, and the caller uses the return value to
    record what it wrote.
    """
    from spacr import figure_sink

    figure_sink.clear_sink()
    target = tmp_path / "volcano.png"
    target.write_bytes(b"\x89PNG")

    assert figure_sink.publish_file(target) == str(target)


def test_clearing_removes_both_sinks():
    """The docstring's reason: two routes into the gallery, and a run that left
    one installed would keep announcing into a screen that has moved on."""
    from spacr import figure_sink

    figure_sink.set_sink(lambda fig, written: None)
    figure_sink.set_file_sink(lambda path, title: None)

    figure_sink.clear_sink()

    assert figure_sink.sink() is None
    assert figure_sink.file_sink() is None


def test_setting_a_sink_returns_the_previous_one():
    """What makes a caller able to restore what it displaced."""
    from spacr import figure_sink

    first = lambda path, title: None
    figure_sink.set_file_sink(first)

    previous = figure_sink.set_file_sink(None)

    assert previous is first
