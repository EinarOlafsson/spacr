"""What the figure dialog does around the edit: cancel, propagate, redraw.

The dialog applies live, so Cancel is the only way out of an experiment.
Propagation is the other direction -- the values go back into the module's
settings panel -- and both have to survive a caller that does not behave.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PySide6")

from spacr.qt.widgets import figure_settings as fs  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def figure():
    fig = plt.figure(figsize=(6.0, 3.0))
    axis = fig.add_subplot(111)
    axis.plot([0, 1, 2], [1, 2, 3], label="one")
    axis.set_title("only")
    yield fig
    plt.close(fig)


def test_cancel_restores_the_size_and_the_ground_it_opened_with(qapp, figure):
    dialog = fs.FigureSettingsDialog(figure)
    try:
        figure.set_size_inches(11.0, 9.0)
        figure.patch.set_facecolor("#ff00ff")

        dialog.reject()
    finally:
        dialog.deleteLater()

    assert tuple(figure.get_size_inches()) == pytest.approx((6.0, 3.0)), (
        "live apply with no way out is a trap")
    assert matplotlib.colors.to_hex(figure.patch.get_facecolor()) != "#ff00ff"
    assert figure.axes, "the axes come back with it, not just the geometry"
    assert figure.axes[0].get_title() == "only"


def test_the_restored_axes_belong_to_the_figure_the_queue_holds(qapp, figure):
    """Restoring copies into the original figure; it does not swap the object."""
    dialog = fs.FigureSettingsDialog(figure)
    try:
        figure.axes[0].set_title("edited")
        dialog.reject()
    finally:
        dialog.deleteLater()

    assert figure.axes[0].figure is figure, (
        "everything else refers to the original figure by identity")


def test_propagation_is_offered_only_when_there_is_somewhere_to_write(qapp,
                                                                      figure):
    without = fs.FigureSettingsDialog(figure)
    with_panel = fs.FigureSettingsDialog(figure,
                                         propagate_callback=lambda _v: None)
    try:
        assert without._propagate_btn.isEnabled() is False
        assert with_panel._propagate_btn.isEnabled() is True
        without._propagate()   # the disabled route must still be harmless
    finally:
        without.deleteLater()
        with_panel.deleteLater()


def test_a_settings_panel_that_refuses_the_values_does_not_take_the_dialog(
        qapp, figure):
    asked = []

    def refuse(values):
        asked.append(values)
        raise RuntimeError("the panel has gone")

    dialog = fs.FigureSettingsDialog(figure, propagate_callback=refuse)
    try:
        dialog._propagate_btn.click()
    finally:
        dialog.deleteLater()

    assert asked, "the callback was reached"


def test_a_caller_that_does_not_know_about_previews_is_still_called(qapp,
                                                                    figure):
    """The preview keyword is an addition; older callers take no arguments."""
    calls = []
    dialog = fs.FigureSettingsDialog(figure,
                                     on_change=lambda: calls.append(1))
    try:
        dialog._redraw_now()
    finally:
        dialog.deleteLater()

    assert calls == [1]


def test_a_redraw_arriving_mid_render_is_coalesced_into_one_more(qapp,
                                                                 figure):
    """Renders must not stack: that is the hang this guard exists for."""
    calls = []
    dialog = None

    def on_change(preview=True):
        calls.append(preview)
        if len(calls) == 1:
            dialog._redraw_now()          # re-entrant, mid-render

    dialog = fs.FigureSettingsDialog(figure, on_change=on_change)
    try:
        dialog._redraw_now()
        assert calls == [True], "the second request only set the flag"
        assert dialog._dirty is False, "and was consumed by the timer restart"
        assert dialog._redraw.isActive(), "one final redraw is queued"
    finally:
        dialog.deleteLater()
