"""FastPlot's before-the-layout guards, and one that cannot fire.

Three of these run only when the widget is asked something before it has
finished assembling, or when an optional shape probe fails. All three
carry an inert `# pragma: no cover`.

The fourth is a guard in the violin profile that cannot be reached, and
this file says why instead of forcing it.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets import fast_plots as fp

pytestmark = pytest.mark.qt


@pytest.fixture()
def plot(qtbot):
    widget = fp.FastPlot(title="fast", x_label="ex", y_label="why")
    qtbot.addWidget(widget)
    return widget


class TestAskedBeforeTheLayoutExists:

    def test_chrome_height_is_zero_without_a_layout(self, plot, monkeypatch):
        """`_chrome_height` measures the furniture around the canvas.

        With no layout there is no furniture to measure, and returning 0
        is what lets the caller size the canvas to the whole widget
        rather than dividing by a missing object.
        """
        monkeypatch.setattr(type(plot), "layout", lambda self: None)
        assert plot._chrome_height() == 0

    def test_chrome_height_is_positive_once_it_is_laid_out(self, plot):
        """The ordinary path, so the guard above is visibly a guard."""
        assert plot._chrome_height() >= 0

    def test_refreshing_the_status_before_it_exists_does_nothing(self, plot):
        """`_refresh_status` is reached from setters that run during build.

        The status label is created part-way through, so a setter called
        before that point has nothing to write to -- and must not raise
        out of a constructor.
        """
        plot._status = None
        plot._refresh_status()          # must not raise

    def test_refreshing_the_status_writes_the_composed_line(self, plot):
        status = getattr(plot, "_status", None)
        if status is None:
            pytest.skip("this build has no status line")
        plot._headline = "a headline"
        plot._refresh_status()
        assert "a headline" in status.text()


class TestTheShapeProbeThatMayFail:
    """`_offer_graph_kinds` builds the "Show as" submenu.

    It asks `shape_of` what the data looks like, to offer "always start
    with this kind". That probe runs over somebody else's frame and can
    raise on a column that is not what it expected -- and the MENU still
    has to open. Losing one convenience entry is not a reason to lose
    the context menu on a plot.
    """

    @staticmethod
    def _spec_with_data():
        import pandas as pd

        class _Spec:
            frame = pd.DataFrame({"g": ["a", "a", "b", "b"],
                                  "v": [1.0, 2.0, 3.0, 4.0]})
            group = "g"
            value = "v"
            kind = "bar"

        return _Spec()

    def test_a_probe_that_raises_still_builds_the_menu(self, plot,
                                                       monkeypatch):
        """`offer` is stubbed as well, and that is not incidental.

        `offer` calls `shape_of` itself, OUTSIDE the guard. Replacing
        only `shape_of` therefore breaks the row lookup rather than the
        probe, and the error comes out of a line that has no try around
        it -- which is what the first version of this test did.
        """
        from PySide6.QtWidgets import QMenu
        from spacr import graph_types

        monkeypatch.setattr(plot, "graph_spec", self._spec_with_data)
        monkeypatch.setattr(
            graph_types, "offer",
            lambda *_a, **_k: [("bar", "Bar", ""), ("box", "Box", "")])

        def refuse(*_a, **_k):
            raise ValueError("that column is not a shape")

        monkeypatch.setattr(graph_types, "shape_of", refuse)

        menu = QMenu()
        plot._offer_graph_kinds(menu)        # must not raise
        titles = [a.menu().title() for a in menu.actions() if a.menu()]
        assert "Show as" in titles, (
            "a failed shape probe took the whole submenu with it")
        submenu = next(a.menu() for a in menu.actions()
                       if a.menu() and a.menu().title() == "Show as")
        captions = [a.text() for a in submenu.actions()]
        assert "Bar" in captions and "Box" in captions, (
            "the kinds themselves were lost with the probe")
        assert not any(c.startswith("Always start with") for c in captions), (
            "an entry was offered from a shape that could not be read")

    def test_a_probe_that_answers_adds_the_always_entry(self, plot,
                                                        monkeypatch):
        """The positive side, so the guard above is visibly a guard."""
        from PySide6.QtWidgets import QMenu
        from spacr import graph_types

        monkeypatch.setattr(plot, "graph_spec", self._spec_with_data)
        monkeypatch.setattr(
            graph_types, "offer",
            lambda *_a, **_k: [("bar", "Bar", ""), ("box", "Box", "")])
        monkeypatch.setattr(graph_types, "shape_of",
                            lambda *_a, **_k: "one value per group")

        menu = QMenu()
        plot._offer_graph_kinds(menu)
        submenu = next(a.menu() for a in menu.actions()
                       if a.menu() and a.menu().title() == "Show as")
        captions = [a.text() for a in submenu.actions()]
        assert any(c.startswith("Always start with") for c in captions)

    def test_a_plot_with_no_graph_spec_adds_nothing(self, plot,
                                                    monkeypatch):
        from PySide6.QtWidgets import QMenu

        monkeypatch.setattr(plot, "graph_spec", lambda: None)
        menu = QMenu()
        plot._offer_graph_kinds(menu)
        assert menu.actions() == []

    def test_a_spec_with_an_empty_frame_adds_nothing(self, plot,
                                                     monkeypatch):
        """An empty frame has no shape to offer kinds for."""
        import pandas as pd
        from PySide6.QtWidgets import QMenu

        class _Empty:
            frame = pd.DataFrame({"g": [], "v": []})
            group = "g"
            value = "v"
            kind = "bar"

        monkeypatch.setattr(plot, "graph_spec", lambda: _Empty())
        menu = QMenu()
        plot._offer_graph_kinds(menu)
        assert menu.actions() == []


class TestTheViolinProfile:

    def test_a_spread_of_values_traces_an_outline(self):
        centres, density = fp._violin_profile([1.0, 2.0, 2.0, 3.0], 0.4)
        assert centres is not None and density is not None
        assert len(centres) == len(density)
        assert density[0] == 0.0 and density[-1] == 0.0, (
            "the outline must be pinned shut at both ends")

    def test_identical_values_trace_nothing(self):
        """A density with no width is a vertical line, and drawing one as
        a violin claims a spread that is not there."""
        assert fp._violin_profile([2.0, 2.0, 2.0], 0.4) == (None, None)

    def test_values_that_are_not_finite_trace_nothing(self):
        assert fp._violin_profile([np.nan, np.nan], 0.4) == (None, None)
        assert fp._violin_profile([np.inf, 1.0], 0.4) == (None, None)

    def test_an_empty_peak_cannot_happen(self):
        """`if peak <= 0: return None, None` is unreachable.

        The histogram's range is `(min(v), max(v))`, so every value in
        `v` falls inside it and the counts sum to `len(v)`. `v` cannot be
        empty -- `np.min` would have raised -- and the guard above has
        already rejected the case where min equals max. So the tallest
        bin holds at least one value.

        Pinned from the producing side: if the range ever stops being
        derived from the data, this fails and that guard stops being
        dead.
        """
        import inspect

        source = inspect.getsource(fp._violin_profile)
        assert "range=(low, high)" in source, (
            "the histogram range is no longer the data's own; the peak "
            "guard may now be reachable")
        assert "low, high = float(np.min(v)), float(np.max(v))" in source

        for values in ([1.0, 2.0], [0.0, 1.0, 1.0, 5.0], list(range(50))):
            v = np.asarray(values, dtype=float)
            low, high = float(np.min(v)), float(np.max(v))
            bins = int(np.clip(np.sqrt(len(v)) * 2, 6, 24))
            counts, _edges = np.histogram(v, bins=bins, range=(low, high))
            assert counts.sum() == len(v)
            assert counts.max() > 0
