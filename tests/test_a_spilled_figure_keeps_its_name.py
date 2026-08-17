"""A figure's caption on the grid outlives its matplotlib Figure.

FOUND WHILE RULING THE FIGURE QUEUE OUT of the stacked-volcano report
(instruction 128 P item 2). The check that brief asked for was to run the
house-style panel path three times and compare the pixmaps the queue holds.
The pixmaps were fine -- three runs, three sections of seven, one volcano
each, no stacking anywhere in the queue -- but only two of the three volcanoes
could be FOUND by name, and the missing one was run 1's.

`figure_titles` read the label off the live Figure every time it was asked, so
a caption survived exactly as long as its Figure did. `_trim_live_figures`
spills everything past the "Editable figures kept" preference (20 by default)
and closes it, and from that moment the grid captioned the tile with the temp
file's stem -- `fig_00000` -- which is precisely what that method's own
docstring says a caption must never be: "a temp file's stem is an
implementation detail of how the picture got to the screen".

Three runs is 21 figures and already crosses the cap. The maintainer's screen
has done twelve.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

pytestmark = pytest.mark.qt


def _named_figure(name):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [1, 0])
    figure.set_label(name)
    figure._spacr_title = name
    return figure


@pytest.fixture()
def queue(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    widget = FigureQueue()
    qtbot.addWidget(widget)
    return widget


def _fill(queue, count, prefix="volcano"):
    for index in range(count):
        queue.add_figure(_named_figure(f"{prefix} {index}"))


def test_a_figure_past_the_live_cap_still_has_its_name(queue, monkeypatch):
    """The defect, at the smallest size that shows it: a cap of 2 and three
    figures, so the first is spilled and closed before the captions are
    asked for."""
    monkeypatch.setattr(type(queue), "live_figure_cap", lambda self: 2)
    _fill(queue, 3)

    assert queue.figure_titles() == ["volcano 0", "volcano 1", "volcano 2"]


def test_the_spill_really_happened(queue, monkeypatch):
    """A guard on the test above: if nothing were ever spilled it would pass
    for the wrong reason and stop protecting anything."""
    monkeypatch.setattr(type(queue), "live_figure_cap", lambda self: 2)
    _fill(queue, 3)

    assert queue.live_figure_count() == 2
    assert not queue.has_live_figure(0)


def test_three_runs_of_house_style_panels_keep_every_caption(queue,
                                                             monkeypatch):
    """At the shape the report came from: 21 figures against the default cap
    of 20, so exactly the first one is spilled -- which is run 1's volcano,
    the one that could not be found by name."""
    monkeypatch.setattr(type(queue), "live_figure_cap", lambda self: 20)
    for run in range(3):
        queue.mark_run(f"run {run + 1}")
        for panel in ("volcano", "effect", "p-values", "controls",
                      "guide support", "agreement", "plates"):
            queue.add_figure(_named_figure(panel))

    titles = queue.figure_titles()
    assert len(titles) == 21
    assert titles.count("volcano") == 3, titles[:8]
    assert not any(title.startswith("fig_") for title in titles), titles[:8]


def test_a_figure_that_never_had_a_name_still_falls_back_to_the_file(queue):
    """The fallback is not removed by remembering the ones that HAVE names:
    a picture can arrive from a pipeline that never labelled it."""
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(2, 1.5))
    ax.plot([0, 1], [1, 0])
    figure.set_label("")
    queue.add_figure(figure)

    assert queue.figure_titles() == ["fig_00000"]


def test_clearing_the_queue_forgets_the_names_too(queue):
    """A name left behind would caption the NEXT run's figure with the last
    run's panel, which is worse than `fig_00000` -- it is wrong rather than
    uninformative."""
    _fill(queue, 3)
    assert queue.figure_titles()[0] == "volcano 0"

    queue.clear()
    queue.add_figure(_named_figure("plate heatmap"))

    assert queue.figure_titles() == ["plate heatmap"]
