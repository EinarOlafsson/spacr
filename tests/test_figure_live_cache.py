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


# ------------------------------------------- faults found by running the app


def test_every_matplotlib_colour_shape_converts(qtbot):
    """QColor accepts none of what matplotlib hands back.

    ``patch.get_facecolor()`` returns an RGBA tuple and a collection returns
    an ARRAY of RGBA rows. Passing either to QColor raised
    "TypeError: QVariant must be holding a QColor" the moment a colour button
    was clicked, so every colour control in the dialog was dead on arrival.
    """
    import numpy as np

    from spacr.qt.widgets.figure_settings import _as_hex

    figure = plt.figure()
    axis = figure.gca()
    line, = axis.plot([0, 1], [0, 1])
    scatter = axis.scatter([0, 1], [0, 1])

    for value in (figure.patch.get_facecolor(),      # RGBA tuple
                  line.get_color(),                  # named / hex
                  scatter.get_facecolor(),           # array of RGBA rows
                  np.array([0.1, 0.2, 0.3, 1.0]),    # single RGBA row
                  0.5,                               # float grey
                  "#ff0000"):
        assert _as_hex(value).startswith("#")
        assert len(_as_hex(value)) == 7
    # Unreadable input falls back rather than raising into the GUI.
    assert _as_hex(object()).startswith("#")


def test_turning_the_grid_off_turns_it_off(qtbot, queue):
    """matplotlib re-enables the grid when line properties accompany False.

    "First parameter to grid() is false, but line properties are supplied.
    The grid will be enabled." -- so the checkbox could not switch the grid
    off, which is the opposite of what it says.
    """
    from PySide6.QtWidgets import QCheckBox

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = _rich_figure()
    queue.add_figure(figure)
    dialog = FigureSettingsDialog(queue.figure_for(0), queue)
    qtbot.addWidget(dialog)

    form = dialog.tabs.widget(1).widget().layout()
    grid_box = None
    from PySide6.QtWidgets import QFormLayout
    for row in range(form.rowCount()):
        label = form.itemAt(row, QFormLayout.LabelRole)
        field = form.itemAt(row, QFormLayout.FieldRole)
        if (label and label.widget() and label.widget().text() == "Grid"
                and field and isinstance(field.widget(), QCheckBox)):
            grid_box = field.widget()
            break
    assert grid_box is not None

    axis = figure.axes[0]
    grid_box.setChecked(True)
    assert any(line.get_visible() for line in axis.get_xgridlines())
    grid_box.setChecked(False)
    assert not any(line.get_visible() for line in axis.get_xgridlines())


def test_the_legend_control_survives_an_unlabelled_figure(qtbot, queue):
    """legend() with no labelled artists warns and returns nothing.

    Calling it unconditionally destroyed the legend the figure already had.
    """
    figure = plt.figure()
    axis = figure.gca()
    axis.plot([0, 1], [0, 1])          # no label
    axis.legend(["manual"])            # a legend exists all the same
    queue.add_figure(figure)

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    dialog = FigureSettingsDialog(queue.figure_for(0), queue)
    qtbot.addWidget(dialog)
    # Building the dialog must not have emitted a warning or lost the legend.
    assert axis.get_legend() is not None


def test_many_series_get_a_rule_instead_of_a_control_each(qtbot, queue):
    """A volcano scatters once per compartment, so an axes holds ~27 of them.

    A control block each is 135 controls, and styling a screen one series at a
    time is not a thing anyone wants to do -- it reads as colouring individual
    data points. Past the threshold the dialog must offer a palette rule that
    reaches every series instead.
    """
    import numpy as np
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = Figure()
    axis = figure.add_subplot(111)
    rng = np.random.default_rng(0)
    for index in range(27):
        axis.scatter(rng.normal(size=20), rng.normal(size=20), label=f"c{index}")

    queue.add_figure(figure)
    live = queue.figure_for(0)
    dialog = FigureSettingsDialog(live, queue)
    qtbot.addWidget(dialog)

    # Not one block per series.
    combos = dialog.findChildren(QComboBox)
    assert len(combos) < 27, f"{len(combos)} combo boxes for 27 series"

    palette = [w for w in combos if w.findData("tab20") != -1]
    assert palette, "no palette rule offered for a many-series axes"

    live_axis = live.axes[0]
    before = [tuple(c.get_facecolor()[0]) for c in live_axis.collections]
    palette[0].setCurrentIndex(palette[0].findData("tab20"))
    after = [tuple(c.get_facecolor()[0]) for c in live_axis.collections]

    # The rule reaches the whole series set, not one member of it.
    assert sum(b != a for b, a in zip(before, after)) > 20
    assert len(set(after)) > 10, "the palette collapsed the series to one colour"


def test_few_series_are_still_named_individually(qtbot, queue):
    """The rule replaces per-series controls only when there are too many.

    Four lines on a QC plot are worth naming; the collapse must not eat them.
    """
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QComboBox

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = Figure()
    axis = figure.add_subplot(111)
    for index in range(4):
        axis.plot([0, 1], [index, index + 1], label=f"line{index}")

    queue.add_figure(figure)
    dialog = FigureSettingsDialog(queue.figure_for(0), queue)
    qtbot.addWidget(dialog)

    combos = dialog.findChildren(QComboBox)
    assert not [w for w in combos if w.findData("tab20") != -1], \
        "four series were collapsed into a rule"


def test_axis_limits_can_be_set(qtbot, queue):
    """"there is no x axis limits or y axis limits" -- there were none.

    Zooming to the part of the volcano with the hits in it is the most common
    thing anyone wants from a plot, and the dialog had no way to ask for it.
    """
    from matplotlib.figure import Figure
    from PySide6.QtWidgets import QDoubleSpinBox

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = Figure()
    axis = figure.add_subplot(111)
    axis.plot([0, 1, 2], [0, 5, 3])

    queue.add_figure(figure)
    live = queue.figure_for(0)
    live_axis = live.axes[0]
    dialog = FigureSettingsDialog(live, queue)
    qtbot.addWidget(dialog)

    boxes = dialog.findChildren(QDoubleSpinBox)
    start = live_axis.get_xlim()
    pair = [
        (boxes[i], boxes[i + 1]) for i in range(len(boxes) - 1)
        if abs(boxes[i].value() - start[0]) < 1e-9
        and abs(boxes[i + 1].value() - start[1]) < 1e-9
    ]
    assert pair, "no pair of spin boxes holds the current x limits"

    low, high = pair[0]
    low.setValue(-3.0)
    high.setValue(9.0)
    assert live_axis.get_xlim() == (-3.0, 9.0)


def test_a_render_in_flight_does_not_start_another(qtbot, queue):
    """This is the hang.

    A preview blocks the GUI thread for ~150 ms, and Qt keeps delivering
    events throughout -- spin-box auto-repeat, wheel, the debounce timer.
    Without a guard each lands another render behind the current one, the
    queue grows faster than it drains, and the window stops responding.
    """
    from matplotlib.figure import Figure

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    renders = []
    dialog = FigureSettingsDialog(
        figure, on_change=lambda preview=True: renders.append(preview))
    qtbot.addWidget(dialog)

    dialog._rendering = True
    for _ in range(40):
        dialog._changed()
        dialog._redraw_now()
    assert renders == [], f"{len(renders)} renders stacked behind one in flight"
    assert dialog._dirty, "the pending change was dropped instead of deferred"

    # ...and the picture still catches up once the thread is free.
    dialog._rendering = False
    dialog._redraw_now()
    assert len(renders) == 1


def test_the_live_preview_does_not_pay_for_a_tight_bbox(queue):
    """bbox_inches='tight' measures by doing a complete extra draw.

    On the volcano that is a flat ~125 ms on top of a ~150 ms render -- the
    largest single cost in the live path, for trimmed whitespace nobody is
    looking at mid-drag.
    """
    from pathlib import Path
    from unittest.mock import patch

    from matplotlib.figure import Figure

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])

    with patch.object(Figure, "savefig", autospec=True) as savefig:
        queue._render_preview(figure, Path("unused.png"))

    assert savefig.called
    assert "bbox_inches" not in savefig.call_args.kwargs


def test_the_preview_is_rendered_at_the_size_it_is_shown_at(queue):
    """"after applying settings the graph looks super pixelated".

    The preview was capped at 1100 px and Qt scaled it up to fill a larger
    view; the difference was made up by interpolation. Rendering at the size
    it will actually occupy costs no more and is sharp.
    """
    queue._view.resize(1400, 1000)
    assert queue._preview_target_px() == 1400

    # ...and a tiny or not-yet-laid-out view still gets a usable render.
    queue._view.resize(10, 10)
    assert queue._preview_target_px() >= 600


def test_a_drag_does_not_block_the_gui_thread(qtbot, queue):
    """The lag: an Agg draw of the volcano is ~110 ms, and it ran inline.

    The 27-entry legend alone is ~63 ms of that, and none of it gets cheaper
    by lowering the resolution -- the cost is text layout and marker geometry,
    not pixels. Agg releases the GIL, so the same draw on a worker costs the
    GUI thread only the figure copy.
    """
    import time

    import numpy as np
    from matplotlib.figure import Figure

    figure = Figure(figsize=(20, 20))
    axis = figure.add_subplot(111)
    rng = np.random.default_rng(0)
    for index in range(27):
        axis.scatter(rng.normal(size=45), rng.normal(size=45), label=f"c{index}")
    handles, _ = axis.get_legend_handles_labels()
    axis.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left",
                ncol=2, frameon=False)

    queue.add_figure(figure)
    live = queue.figure_for(0)

    worst = 0.0
    for step in range(30):
        for collection in live.axes[0].collections:
            collection.set_alpha(0.3 + 0.01 * step)
        start = time.perf_counter()
        queue.refresh_current_figure(preview=True)
        worst = max(worst, (time.perf_counter() - start) * 1000)
        qtbot.wait(1)

    # Inline this was ~260 ms per change. A frame is 16 ms.
    assert worst < 60, f"a single change blocked the GUI thread for {worst:.0f} ms"
    # And the 30 changes did not become 30 renders.
    assert queue._preview_seq < 30


def test_a_change_during_a_draw_is_not_lost(qtbot, queue):
    """Coalescing must not drop the last change.

    The user stops moving the control at some point, and that final position
    is the one they are looking at. If it arrived mid-draw and was merely
    discarded, the picture would settle showing something else.
    """
    from matplotlib.figure import Figure

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    queue.add_figure(figure)
    live = queue.figure_for(0)

    queue._preview_busy = True
    assert queue._render_preview_async(live) is True
    assert queue._preview_pending, "the change was dropped, not deferred"

    queue._on_preview_rendered(None)
    assert not queue._preview_pending, "the deferred change never ran"


def test_a_fresh_render_is_not_discarded_by_its_own_successor(qtbot, queue):
    """Ordering inside the completion handler.

    Starting the pending draw before painting bumps the sequence, so the
    payload that just arrived -- freshly drawn and perfectly good -- gets
    dropped as stale by the render it itself kicked off. A continuous drag
    would then show nothing at all until the user stopped moving.
    """
    from matplotlib.figure import Figure

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])
    queue.add_figure(figure)

    painted = []
    queue._paint_preview = lambda payload: painted.append(payload)
    queue._preview_busy = True
    queue._preview_pending = True
    queue._preview_seq = 7

    queue._on_preview_rendered((0, 7, None))
    assert painted, "the finished render was never painted"
    assert painted[0][1] == 7, "painted a stale token"


def test_cancel_puts_the_figure_back(qtbot, queue):
    """Live apply with no way out is a trap.

    The dialog this replaced restored the figure on Cancel and said exactly
    why: "the user drags a spin box to see what it does and there is no longer
    an 'as it was'". This one changes far more, so the trap is worse -- and
    the Close-only button box it shipped with had no way out at all.
    """
    from matplotlib.figure import Figure

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = Figure()
    axis = figure.add_subplot(111)
    axis.plot([0, 1, 2], [0, 5, 3], color="blue")
    axis.set_title("original")

    queue.add_figure(figure)
    live = queue.figure_for(0)
    dialog = FigureSettingsDialog(live, queue)
    qtbot.addWidget(dialog)

    # Mangle it the way a user exploring the controls would.
    live_axis = live.axes[0]
    live_axis.lines[0].set_color("red")
    live_axis.set_title("mangled")
    live_axis.set_yscale("log")

    dialog.reject()

    assert live.axes, "the restore left the figure with no axes at all"
    restored = live.axes[0]
    assert restored.get_title() == "original"
    assert restored.lines[0].get_color() == "blue"
    assert restored.get_yscale() == "linear"


def test_the_dialog_can_still_propagate_into_the_settings_panel(qtbot, queue):
    """A feature the swap to this dialog silently dropped.

    "Propagate settings" writes the values into the module's settings panel so
    the next run starts from them.
    """
    from matplotlib.figure import Figure

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    figure = Figure()
    figure.add_subplot(111).plot([0, 1], [0, 1])

    sent = []
    dialog = FigureSettingsDialog(
        figure, propagate_callback=lambda values: sent.append(values))
    qtbot.addWidget(dialog)
    assert dialog._propagate_btn.isEnabled()
    dialog._propagate_btn.click()
    assert sent, "propagate did not reach the callback"

    # ...and it is offered but disabled when there is nowhere to write.
    plain = FigureSettingsDialog(figure)
    qtbot.addWidget(plain)
    assert not plain._propagate_btn.isEnabled()


def test_a_umap_figure_still_gets_its_live_umap_controls(qtbot, queue):
    """Instruction 75, dropped when this dialog replaced the old one.

    Every Image UMAP setting, live against the figure -- offered only for a
    figure carrying the embedding it was drawn from, because without it
    "live" would mean re-running the reduction and every point would move.
    """
    import numpy as np
    from matplotlib.figure import Figure

    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(30, 2))
    figure = Figure(figsize=(4, 4))
    figure.subplots().scatter(embedding[:, 0], embedding[:, 1])
    figure._spacr_umap_payload = {
        "embedding": embedding,
        "labels": list(range(30)),
        "settings": {},
    }

    dialog = FigureSettingsDialog(figure, queue)
    qtbot.addWidget(dialog)
    tabs = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
    assert "Image UMAP" in tabs, f"no live UMAP controls; tabs were {tabs}"
    assert dialog.umap_values()

    # A figure without the embedding must not offer them.
    plain = Figure()
    plain.subplots().plot([0, 1], [0, 1])
    other = FigureSettingsDialog(plain, queue)
    qtbot.addWidget(other)
    assert "Image UMAP" not in [
        other.tabs.tabText(i) for i in range(other.tabs.count())]
