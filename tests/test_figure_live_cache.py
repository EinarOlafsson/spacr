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


# ------------------------------------------------- restoring an old figure


def test_an_evicted_figure_comes_back_fully_editable(queue, monkeypatch):
    """The point of spilling a Figure rather than only its picture.

    A saved vector page allows a stroke to be recoloured, a width changed, a
    font resized. It does NOT allow anything data-bound: a log axis has to
    recompute every position. A restored Figure allows all of it.
    """
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 3)
    monkeypatch.setattr(queue, "dynamic_figures_enabled", lambda: True)
    _add(queue, 10)

    assert not queue.has_live_figure(0), "figure 0 should have been evicted"
    assert queue.is_restorable(0)

    figure = queue.figure_for(0)
    assert figure is not None
    axis = figure.axes[0]

    # Every appearance change the user asked for...
    axis.grid(False)
    axis.spines["left"].set_linewidth(3)
    axis.tick_params(labelsize=16)
    for line in axis.lines:
        line.set_color("crimson")
    # ...plus the data-bound one that a PDF could never give back.
    axis.set_yscale("log")
    assert axis.get_yscale() == "log"


def test_restoring_puts_the_figure_back_in_the_live_set(queue, monkeypatch):
    """Repeated edits must not re-read the disk each time."""
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 3)
    monkeypatch.setattr(queue, "dynamic_figures_enabled", lambda: True)
    _add(queue, 8)
    queue.figure_for(0)
    assert queue.has_live_figure(0)
    # And the cap still holds afterwards.
    assert queue.live_figure_count() <= 3


def test_dynamic_figures_off_does_not_restore(queue, monkeypatch):
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 2)
    monkeypatch.setattr(queue, "dynamic_figures_enabled", lambda: False)
    _add(queue, 6)
    assert queue.is_restorable(0), "the spill is still on disk"
    assert queue.figure_for(0) is None, "but the option says do not use it"


def test_an_unpicklable_figure_does_not_break_the_cap(queue, monkeypatch):
    """Failing to spill must never stop old figures being released."""
    monkeypatch.setattr(queue, "live_figure_cap", lambda: 2)
    monkeypatch.setattr(queue, "_spill_figure", lambda idx, fig: False)
    _add(queue, 7)
    assert queue.live_figure_count() == 2
    assert queue.figure_for(0) is None
    # It is still viewable from its rendered page.
    assert queue._png_paths.get(0)


# ------------------------------------------------------- restyling controls


def _rich_figure():
    import numpy as np

    rng = np.random.default_rng(0)
    figure = plt.figure(figsize=(6, 4))
    axis = figure.gca()
    axis.plot([0, 1, 2], [1, 2, 3], label="series A")
    axis.scatter(rng.normal(size=50), rng.normal(size=50), label="points B")
    axis.legend()
    axis.grid(True)
    axis.set_title("t")
    return figure


def _labels(form):
    from PySide6.QtWidgets import QFormLayout

    out = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.LabelRole)
        if item is not None and item.widget() is not None:
            out.append(item.widget().text())
    return out


def test_the_context_menu_offers_the_frequent_toggles(qtbot, queue):
    """Right-clicking a figure had no menu at all before this."""
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    queue.add_figure(_rich_figure())
    menu = build_figure_context_menu(queue, queue.figure_for(0))
    qtbot.addWidget(menu)
    texts = [a.text() for a in menu.actions() if a.text()]
    assert "Legend" in texts and "Grid" in texts
    assert any("settings" in t.lower() for t in texts)
    assert "Axis scale" in [m.title() for m in menu.findChildren(type(menu))]


def test_the_menu_says_so_when_a_figure_cannot_be_restyled(qtbot, queue):
    """Better than a menu whose entries silently do nothing."""
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    menu = build_figure_context_menu(queue, None)
    qtbot.addWidget(menu)
    assert len(menu.actions()) == 1
    assert not menu.actions()[0].isEnabled()


def test_the_settings_dialog_is_built_from_the_figure(qtbot, queue):
    """Controls follow what the figure has, not a fixed list.

    That is what makes "as many settings as possible, depending on the graph"
    true: a figure with two series gets two blocks of series controls, and a
    figure type added later is covered without editing the dialog.
    """
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    queue.add_figure(_rich_figure())
    dialog = FigureSettingsDialog(queue.figure_for(0), queue)
    qtbot.addWidget(dialog)

    tabs = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
    assert len(tabs) == 2, "one Figure tab plus one per axes"

    axes_form = dialog.tabs.widget(1).widget().layout()
    labels = _labels(axes_form)
    # Everything the user listed.
    for expected in ("X scale", "Y scale", "Grid", "Grid width", "Grid colour",
                     "Spine width", "Tick label size", "Legend",
                     "Legend text size", "Title", "X label"):
        assert expected in labels, expected
    # One block per series actually present: two series -> two colour rows.
    assert labels.count("  Colour") == 2
    assert labels.count("  Opacity") == 2


def test_a_figure_without_a_legend_gets_no_legend_controls(qtbot, queue):
    """A control that cannot do anything should not be offered."""
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = plt.figure()
    figure.gca().imshow([[1, 2], [3, 4]])
    queue.add_figure(figure)
    dialog = FigureSettingsDialog(queue.figure_for(0), queue)
    qtbot.addWidget(dialog)
    labels = _labels(dialog.tabs.widget(1).widget().layout())
    assert "Legend" not in labels
    # The axis controls that always apply are still there.
    assert "X scale" in labels and "Grid" in labels


def test_an_evicted_figure_can_still_be_restyled(qtbot, queue, monkeypatch):
    """The point of the whole spill mechanism, end to end."""
    from spacr.qt.widgets.figure_settings import build_figure_context_menu

    monkeypatch.setattr(queue, "live_figure_cap", lambda: 2)
    monkeypatch.setattr(queue, "dynamic_figures_enabled", lambda: True)
    for _ in range(6):
        queue.add_figure(_rich_figure())
    assert not queue.has_live_figure(0)
    menu = build_figure_context_menu(queue, queue.figure_for(0))
    qtbot.addWidget(menu)
    # A real menu, not the "cannot be restyled" one.
    assert len(menu.actions()) > 1


# ------------------------------------------------------ the dialog stays usable


def test_scrolling_the_dialog_does_not_edit_the_controls(qtbot, queue):
    """Qt gives spin boxes and combos the wheel by default.

    So scrolling to reach a control changed a dozen settings on the way past,
    each triggering a render. That is what made the dialog unusable.
    """
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    queue.add_figure(_rich_figure())
    dialog = FigureSettingsDialog(queue.figure_for(0), queue)
    qtbot.addWidget(dialog)

    spins = dialog.findChildren(QSpinBox) + dialog.findChildren(QDoubleSpinBox)
    assert spins, "the dialog should have numeric controls"
    assert all(w.focusPolicy() == Qt.StrongFocus for w in spins)

    box = spins[0]
    before = box.value()
    wheel = QWheelEvent(
        QPoint(5, 5), box.mapToGlobal(QPoint(5, 5)), QPoint(0, 0),
        QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    # Unfocused: the wheel belongs to the scroll area.
    assert dialog.eventFilter(box, wheel) is True
    assert box.value() == before
    # Focused: the user asked for it. Qt will not grant focus to a widget in
    # an unshown offscreen dialog, so what is under test here is the filter's
    # DECISION, not Qt's focus machinery.
    box.hasFocus = lambda: True
    assert dialog.eventFilter(box, wheel) is False


def test_rapid_changes_coalesce_into_one_redraw(qtbot, queue):
    """A full render is seconds on a large figure; per-change was a freeze."""
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    queue.add_figure(_rich_figure())
    rendered = {"n": 0}
    dialog = FigureSettingsDialog(
        queue.figure_for(0), queue,
        on_change=lambda preview=True: rendered.__setitem__(
            "n", rendered["n"] + 1))
    qtbot.addWidget(dialog)

    for _ in range(20):
        dialog._changed()
    assert rendered["n"] == 0, "nothing renders while changes are still coming"
    assert dialog._redraw.isActive()
    qtbot.wait(dialog.REDRAW_DELAY_MS + 150)
    assert rendered["n"] == 1, "twenty changes cost one render"


def test_a_preview_render_skips_the_vector_page(qtbot, queue):
    """The preview is what makes live feedback affordable."""
    queue.add_figure(_rich_figure())
    from pathlib import Path

    png = Path(queue._png_paths[0])
    pdf = png.with_suffix(".pdf")
    queue.refresh_current_figure(preview=False)
    stamp = pdf.stat().st_mtime_ns if pdf.exists() else None

    assert queue.refresh_current_figure(preview=True) is True
    # The vector page is untouched by a preview; it is rewritten on close.
    if stamp is not None:
        assert pdf.stat().st_mtime_ns == stamp
