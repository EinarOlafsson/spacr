"""The training graphs are pyqtgraph and they APPEND (instruction 231).

"the graphs should be updated not regenerated epch after epoch."

Regenerating costs a full re-render of every point drawn so far, so the run
gets SLOWER THE LONGER IT GOES -- at exactly the moment the user is watching
it most closely. It also throws away anything they did to the view: a zoom
into the last twenty epochs undone by epoch twenty-one makes the live graph
unusable for the thing a live graph is for.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def monitor(app):
    from spacr.qt.widgets.training_monitor import TrainingMonitor

    return TrainingMonitor()


def _run(monitor, epochs=20):
    for epoch in range(1, epochs + 1):
        monitor.append(epoch, {
            "loss": 1.0 / epoch,
            "val_loss": 1.2 / epoch,
            "accuracy": min(0.99, 0.4 + epoch * 0.03),
            "class_nc": min(0.99, 0.3 + epoch * 0.03),
        })


class TestItIsPyqtgraph:

    def test_the_panels_are_pyqtgraph_widgets(self, monitor):
        import pyqtgraph as pg

        for panel in monitor.plots.values():
            assert isinstance(panel, pg.PlotWidget)

    def test_there_is_no_matplotlib_behind_it(self):
        """A matplotlib figure rendered to an image is the thing this
        replaces."""
        import inspect

        from spacr.qt.widgets import training_monitor

        source = inspect.getsource(training_monitor)
        assert "matplotlib" not in source
        assert "pyplot" not in source

    def test_there_are_three_panels(self):
        """The per-class panel answers 'is it learning ALL of it', which is
        the question a 96% aggregate hiding a class at 40% gets wrong."""
        from spacr.qt.widgets.training_monitor import PANELS

        assert len(PANELS) == 3
        assert "per_class" in {key for key, _t, _y in PANELS}


class TestEachEpochAppends:

    def test_the_curve_is_the_same_object_throughout(self, monitor):
        """Identity, not appearance: a rebuilt curve that looks the same IS
        the bug."""
        monitor.append(1, {"loss": 1.0})
        first = monitor.curves["loss"]
        _run(monitor, 30)
        assert monitor.curves["loss"] is first

    def test_every_series_keeps_its_own_object(self, monitor):
        _run(monitor, 5)
        before = dict(monitor.curves)
        _run(monitor, 5)
        for name, curve in before.items():
            assert monitor.curves[name] is curve

    def test_the_points_accumulate(self, monitor):
        _run(monitor, 12)
        xs, ys = monitor.points("loss")
        assert len(xs) == 12 and len(ys) == 12

    def test_one_curve_per_series_not_one_per_epoch(self, monitor):
        """`plot.plot()` ADDS an item every call, so calling it per epoch
        leaves n overlapping curves -- which looks like one curve and costs
        n times the render."""
        _run(monitor, 25)
        drawn = [i for i in monitor.plots["loss"].plotItem.items
                 if hasattr(i, "setData")]
        assert len(drawn) == 2, "train and val, not fifty"

    def test_the_cost_does_not_grow_with_the_epoch(self, monitor):
        """The property the instruction names. Asserted on the count of
        drawn items rather than on a timing, which would be flaky."""
        _run(monitor, 10)
        early = len(monitor.plots["loss"].plotItem.items)
        _run(monitor, 100)
        assert len(monitor.plots["loss"].plotItem.items) == early


class TestTheViewSurvives:

    def test_a_zoom_set_early_is_still_in_force_later(self, monitor):
        """"a zoom into the last twenty epochs is undone by epoch
        twenty-one"."""
        _run(monitor, 5)
        panel = monitor.plots["loss"]
        panel.setXRange(2, 4, padding=0)
        wanted = panel.viewRange()[0]
        _run(monitor, 50)
        assert panel.viewRange()[0] == pytest.approx(wanted, abs=0.01)


class TestTheSeriesGoOnTheRightPanel:

    def test_loss_and_val_loss_share_the_loss_panel(self, monitor):
        _run(monitor, 3)
        assert monitor._panel_for("loss") == "loss"
        assert monitor._panel_for("val_loss") == "loss"

    def test_accuracy_is_its_own_panel(self, monitor):
        assert monitor._panel_for("accuracy") == "accuracy"

    def test_anything_else_is_per_class(self, monitor):
        assert monitor._panel_for("class_nc") == "per_class"


class TestBadNumbers:

    def test_a_nan_epoch_is_a_gap_not_a_point(self, monitor):
        monitor.append(1, {"loss": 1.0})
        monitor.append(2, {"loss": float("nan")})
        monitor.append(3, {"loss": 0.5})
        xs, _ = monitor.points("loss")
        assert xs == (1.0, 3.0)

    def test_a_non_number_is_skipped(self, monitor):
        assert monitor.append(1, {"loss": "later"}) == 0

    def test_nothing_at_all_is_survivable(self, monitor):
        assert monitor.append(1, {}) == 0


class TestClearing:

    def test_it_is_the_only_place_the_curves_go(self, monitor):
        _run(monitor, 5)
        assert monitor.series()
        monitor.clear()
        assert monitor.series() == ()

    def test_a_new_run_starts_fresh(self, monitor):
        _run(monitor, 5)
        monitor.clear()
        _run(monitor, 3)
        xs, _ = monitor.points("loss")
        assert len(xs) == 3
