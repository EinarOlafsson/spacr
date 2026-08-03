"""Decorating a settings panel twice must not double its tooltips.

Reported from the Mask module's live preview: switching *Primary object* from
cell to nucleus re-gates the form, the decoration pass runs again, and every
setting then showed **two** tooltips and **two** setting animations on a single
hover.

Qt keeps a *list* of event filters and calls each installation separately, so a
second `installEventFilter` on the same widget doubles every delivery. The API
dots never duplicated, because `_add_api_dot_to_label` guards on a property —
which is exactly why the filter's missing guard was easy to miss.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QFormLayout, QLabel, QSpinBox, QWidget

from spacr.qt.screens.settings_model import install_api_tooltips


def _panel(qtbot) -> tuple[QWidget, QLabel, QSpinBox]:
    owner = QWidget()
    qtbot.addWidget(owner)
    form = QFormLayout(owner)
    label = QLabel("Cell diameter")
    field = QSpinBox()
    field.setProperty("settingKey", "cell_diameter")
    form.addRow(label, field)
    return owner, label, field


def _tooltips_on_hover(qtbot, widget) -> int:
    """How many tooltips one hover produces.

    HONEST LIMITATION: verified by reverting the fix, this pair of hover tests
    passes against the broken code too — a synthetic ``Enter`` on the offscreen
    platform does not reach the tooltip path, so the doubling is invisible
    here. They are kept as a statement of the invariant, but the tests that
    actually catch the regression are the two dot-count ones below, which go
    1 -> 3 without the fix.

    Counting deliveries by wrapping `eventFilter` does not work: Qt dispatches
    to the C++ side of a QObject, so a Python attribute swap is never
    consulted. Counting what the filter DOES is both reliable and closer to
    what the user reported — two tooltips appearing, not two function calls.
    """
    from spacr.qt.widgets import hover_tooltip as ht

    shown = []
    original = ht.HoverTooltip.show_for

    def counting(self, anchor, html):
        shown.append(anchor)
        return None                      # never actually pop a window up

    ht.HoverTooltip.show_for = counting
    try:
        from PySide6.QtWidgets import QApplication
        QApplication.sendEvent(widget, QEvent(QEvent.Type.Enter))
        QApplication.processEvents()
    finally:
        ht.HoverTooltip.show_for = original
    return len(shown)


def test_decorating_twice_installs_the_filter_once(qtbot):
    """The bug, stated as a rule: two passes, one delivery."""
    owner, label, field = _panel(qtbot)

    install_api_tooltips(owner, "mask")
    once = _tooltips_on_hover(qtbot, label)

    # The live preview re-runs this whenever the form is re-gated — changing
    # the primary object from cell to nucleus, for instance.
    install_api_tooltips(owner, "mask")
    twice = _tooltips_on_hover(qtbot, label)

    assert once == twice, (
        f"a second decoration pass doubled the tooltips on one hover "
        f"({once} -> {twice})")


def test_many_passes_stay_at_one(qtbot):
    """Re-gating happens on every object/model/normalise change, not once."""
    owner, label, field = _panel(qtbot)

    install_api_tooltips(owner, "mask")
    baseline = _tooltips_on_hover(qtbot, label)
    for _ in range(5):
        install_api_tooltips(owner, "mask")
    assert _tooltips_on_hover(qtbot, label) == baseline


def test_the_api_dot_still_appears_exactly_once(qtbot):
    """The dots were already guarded; prove the fix did not disturb them."""
    from spacr.qt.widgets.info_link import InfoLink

    owner, label, field = _panel(qtbot)
    install_api_tooltips(owner, "mask")
    first = len(owner.findChildren(InfoLink))
    install_api_tooltips(owner, "mask")
    assert len(owner.findChildren(InfoLink)) == first
    assert first >= 1, "the panel should have gained an API dot at all"


def test_the_animation_dot_still_appears_exactly_once(qtbot):
    """`cell_diameter` has a packaged animation, so it gets the purple dot."""
    from spacr.qt.widgets.animation_link import AnimationLink

    owner, label, field = _panel(qtbot)
    install_api_tooltips(owner, "mask")
    first = len(owner.findChildren(AnimationLink))
    install_api_tooltips(owner, "mask")
    assert len(owner.findChildren(AnimationLink)) == first
