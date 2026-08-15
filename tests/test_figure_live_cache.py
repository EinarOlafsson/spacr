"""The Figures panel keeps recent figures EDITABLE, not just visible.

The panel displays a pixmap, so what the user sees is a picture of a figure
and has no legend to toggle or axis to rescale. The live matplotlib Figures
were retained -- but never capped, so a long run accumulated every one.

These tests pin both halves: the cap is the user's number, and a figure past
it is still viewable and (with dynamic figures on) still loads from its
vector page.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def queue(qtbot):
    from spacr.qt.widgets.figure_queue import FigureQueue

    widget = FigureQueue()
    qtbot.addWidget(widget)
    yield widget
    plt.close("all")


def _add(widget, n):
    for i in range(n):
        figure = plt.figure()
        figure.gca().plot([0, 1], [i, i])
        widget.add_figure(figure)


def test_the_live_figure_cap_is_the_users_number(queue, monkeypatch):
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 5)
    _add(queue, 12)
    assert queue.live_figure_count() == 5
    # The newest are the restylable ones -- those are the figures a user is
    # looking at when they want to change something.
    assert all(queue.has_live_figure(i) for i in range(7, 12))
    assert not any(queue.has_live_figure(i) for i in range(0, 5))


def test_an_evicted_figure_is_still_viewable(queue, monkeypatch):
    """Releasing the Figure must not lose the figure."""
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 3)
    _add(queue, 8)
    assert queue.count() == 8
    # Every slot still has its rendered page on disk.
    assert all(queue._png_paths.get(i) for i in range(8))


def test_raising_the_cap_retains_more(queue, monkeypatch):
    caps = {"n": 2}
    monkeypatch.setattr(queue, "live_figure_cap", lambda: caps["n"])
    _add(queue, 6)
    assert queue.live_figure_count() == 2
    caps["n"] = 10
    _add(queue, 1)
    assert queue.live_figure_count() == 3


def test_a_cap_of_one_is_legal(queue, monkeypatch):
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 1)
    _add(queue, 4)
    assert queue.live_figure_count() == 1


def test_the_preferences_round_trip():
    from spacr.qt.preferences import (
        MAX_FIG_LIVE_CACHE, MIN_FIG_LIVE_CACHE, get_figure_dynamic,
        get_figure_live_cache, set_figure_dynamic, set_figure_live_cache,
    )

    original, original_dynamic = get_figure_live_cache(), get_figure_dynamic()
    try:
        set_figure_live_cache(37)
        assert get_figure_live_cache() == 37
        set_figure_dynamic(False)
        assert get_figure_dynamic() is False
        set_figure_dynamic(True)
        assert get_figure_dynamic() is True
        # Out of range is refused rather than silently clamped on write.
        with pytest.raises(ValueError):
            set_figure_live_cache(MAX_FIG_LIVE_CACHE + 1)
        with pytest.raises(ValueError):
            set_figure_live_cache(MIN_FIG_LIVE_CACHE - 1)
    finally:
        set_figure_live_cache(original)
        set_figure_dynamic(original_dynamic)


def test_both_controls_exist_in_the_preferences_dialog(qtbot):
    """A setting the user cannot reach is not a setting."""
    from PySide6.QtWidgets import QCheckBox, QSpinBox

    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    spins = [w for w in dialog.findChildren(QSpinBox)
             if "restylable" in (w.toolTip() or "")]
    checks = [w for w in dialog.findChildren(QCheckBox)
              if "load its PDF page" in (w.toolTip() or "")]
    assert len(spins) == 1 and len(checks) == 1
    assert spins[0].minimum() >= 1
