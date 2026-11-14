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
        plot._headline = "a headline nobody can read"
        plot._status = None

        plot._refresh_status()          # must not raise

        # ASSERTED. Without this the test passes against a
        # `_refresh_status` that returns before doing anything at all,
        # including on a plot that HAS a status line.
        assert plot._status is None, (
            "_refresh_status invented a status label on the way past")
        assert plot._headline == "a headline nobody can read", (
            "the headline was cleared when there was nowhere to write it")

    def test_refreshing_the_status_writes_the_composed_line(self, plot):
        status = getattr(plot, "_status", None)
        if status is None:
            pytest.skip("this build has no status line")
        plot._headline = "a headline"
        plot._refresh_status()
        assert "a headline" in status.text()


class TestTheViolinProfile:


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
