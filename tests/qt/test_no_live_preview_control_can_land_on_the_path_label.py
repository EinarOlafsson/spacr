"""Nothing on the live-preview panel may sit outside a layout.

Reported three times, most recently 2026-09-10: "in the live view in mask
generation i see a field in the top left corner of live view overlapping with
a path ... a black field which has a 3 in it".

WHY THE TOP LEFT, EVERY TIME. A ``QWidget`` parented to a widget but never
added to a layout keeps its default geometry, ``(0, 0, 100, 30)``. That is
the top-left corner, which is where the loaded-path label sits. Nothing is
misplacing these controls; they were never placed at all, and ``(0, 0)`` is
where Qt leaves what nobody positioned.

BOTH EARLIER FIXES MOVED ONE WIDGET. First the path label's own eliding,
then ``_fov_box`` and ``_channel_box`` into ``_offscreen_controls``. Each
made the reported symptom go away and left the class untouched: measured on
a headless build, the panel had **113 direct children and 95 of them in no
layout** -- 45 spin boxes, 21 double spin boxes, 17 toggles, 12 combo boxes.
The thing reading "3" was a spin box and there were 66 of those to choose
from, which is why identifying it was never going to be the fix.

THE CONTROLS ARE SUPPOSED TO BE HOMELESS. They belong to the panel so their
values outlive the settings dialog, and :class:`LiveSettingsDialog` lays
them out only while it is open. So the invariant is not "everything is in a
layout" -- it is "everything is in a layout OR in the one container that can
never be drawn".

``test_the_invariant_catches_a_planted_stray`` is the control, and it is not
decoration. A guard in this directory passed for months on a counterfactual
that had quietly stopped reproducing, because "the fault did not appear" and
"the fault cannot be staged any more" are the same green. Planting a stray
widget and requiring the check to catch it is how this file tells those
apart.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSpinBox, QWidget


@pytest.fixture(autouse=True)
def _qapp(qapp):
    """The panel builds pixmaps, which needs a QGuiApplication."""
    return qapp


@pytest.fixture
def panel(qtbot):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    widget = LivePreviewPanel()
    qtbot.addWidget(widget)
    return widget


def _laid_out(root: QWidget) -> set:
    """Every widget reachable through ``root``'s layout tree, by id."""
    found = set()
    stack = [root.layout()]
    while stack:
        layout = stack.pop()
        if layout is None:
            continue
        for index in range(layout.count()):
            item = layout.itemAt(index)
            child = item.widget()
            if child is not None:
                found.add(id(child))
            stack.append(item.layout())
    return found


def _strays(panel) -> list:
    """Direct children that are neither laid out nor the stow container.

    Direct children only: a widget inside some sub-container is positioned
    by that container's layout and cannot land on the panel's own corner.

    Windows are excluded, and that exclusion was found by the control below
    rather than reasoned out in advance. ``LiveSettingsDialog`` is parented
    to the panel and sits in no layout, so a check without this clause
    reports the open settings dialog as a widget drawn over the path label.
    It carries ``Qt::Window``; the window manager places it, and it never
    paints inside the panel.
    """
    laid = _laid_out(panel)
    container = panel._offscreen_controls
    return [w for w in panel.findChildren(
                QWidget, options=Qt.FindDirectChildrenOnly)
            if w is not container and id(w) not in laid
            and not w.isWindow()]


def _describe(strays) -> str:
    import collections
    counts = collections.Counter(type(w).__name__ for w in strays)
    return ", ".join(f"{n}x {c}" for c, n in sorted(counts.items()))


def test_every_child_is_laid_out_or_stowed(panel):
    """The invariant itself. It failed on 95 widgets before this existed."""
    strays = _strays(panel)
    assert not strays, (
        f"{len(strays)} widget(s) parented to the panel sit in no layout "
        f"({_describe(strays)}). Each one occupies (0, 0, 100, 30) -- the "
        "corner the loaded-path label is drawn in -- and is one stray "
        "show() from being painted over it. Move them into "
        "`_offscreen_controls`; `_stow_free_widgets` does it for anything "
        "built in the constructor.")


def test_the_invariant_catches_a_planted_stray(panel):
    """The control. If this cannot fail, the test above proves nothing."""
    assert not _strays(panel), "precondition: the panel starts clean"

    planted = QSpinBox(panel)
    planted.setValue(3)                 # the value in the report, for flavour
    try:
        caught = _strays(panel)
        assert len(caught) == 1 and caught[0] is planted, (
            "a spin box parented straight to the panel and added to no "
            "layout went unnoticed, so the check above is not measuring "
            "what put a field over the path")
        assert planted.geometry().topLeft().x() == 0
        assert planted.geometry().topLeft().y() == 0
    finally:
        planted.setParent(None)
        planted.deleteLater()


def test_the_sweep_is_idempotent(panel):
    """A second pass finds nothing, so the constructor's pass was complete."""
    assert panel._stow_free_widgets() == 0


def test_the_stowed_controls_still_belong_to_the_panel(panel):
    """Stowing must not orphan the controls or lose their values.

    ``_fov_box`` and ``_channel_box`` are the precedent: they were kept
    rather than deleted because ``apply_sample_to_combo`` fills one and
    ``selected_channel()`` reads the other. The same is true of everything
    else moved here -- the panel reads all of them when it builds a request.
    """
    container = panel._offscreen_controls
    for control in (panel._lo_pct, panel._hi_pct, panel._diameter,
                    panel._model_box, panel._object_box):
        assert control.parent() is container
        assert control.window() is panel.window()
    panel._lo_pct.setValue(3.25)
    assert panel._lo_pct.value() == pytest.approx(3.25)


def test_closing_the_settings_dialog_stows_rather_than_strands(panel, qtbot):
    """The dialog's own close path was the other half of the defect.

    ``closeEvent`` re-parented every control it had borrowed back onto the
    PANEL -- necessary, since Qt destroys a dialog's children with it, but
    it put each one back at (0, 0) with only ``hide()`` holding it off
    screen. That is the state the report describes, and it is reached by
    opening the live settings once and closing it.
    """
    from spacr.qt.widgets.live_preview import LiveSettingsDialog

    dialog = LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)
    borrowed = dialog._managed_widgets()
    assert borrowed, "the dialog borrowed nothing, so this proves nothing"
    dialog.close()

    strays = _strays(panel)
    assert not strays, (
        f"after one open/close of the live settings, {len(strays)} "
        f"control(s) ({_describe(strays)}) are back on the panel with no "
        "layout, at the corner the path label occupies")
    container = panel._offscreen_controls
    assert all(w.parent() is container for w in borrowed)


def test_the_dialog_gives_back_everything_it_borrowed(panel, qtbot):
    """Not just the widgets `_managed_widgets()` names.

    Qt destroys a dialog's children with it, so a panel control still
    parented under the dialog when it goes would go too. `_managed_widgets()`
    is the DECLARED list and it was incomplete: `_pathogen_channel`,
    `_organelle_channel` and `_model_zoo_btn` are laid out in the dialog and
    were not in it, so one open-and-close left them parented to a group box
    belonging to a closed dialog.

    NOTHING FAILED WHEN IT HAPPENED, which is why it survived review by eye:
    the panel still held Python references, so the widgets stayed alive and
    the next open re-parented them again. The damage is only visible if the
    dialog is actually destroyed first.

    The sweep is by identity now, so this test asks the question that matters
    -- is every control the panel owns back under the panel -- rather than
    re-listing the names, which is the thing that went stale.
    """
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.live_preview import LiveSettingsDialog

    owned = {id(v) for v in vars(panel).values() if isinstance(v, QWidget)}
    dialog = LiveSettingsDialog(panel)
    qtbot.addWidget(dialog)
    borrowed = [w for w in dialog.findChildren(QWidget) if id(w) in owned]
    assert len(borrowed) >= 12, (
        f"the dialog borrowed only {len(borrowed)} panel controls; this test "
        "is not exercising the case it was written for")
    dialog.close()

    stranded = [w for w in borrowed
                if w.parent() is not panel._offscreen_controls]
    names = {id(v): k for k, v in vars(panel).items()}
    assert not stranded, (
        "these panel controls were left under the closed dialog and would be "
        "destroyed with it: "
        + ", ".join(names.get(id(w), type(w).__name__) for w in stranded))
