"""The tooltip contract across every registered settings module.

Examples on Mask are not enough: registered modules can add defaults and
special-purpose screens can build their rows by hand.  These guards inventory
the live registry and prove that the generic and hand-built paths converge on
the same label-only, sticky, single-surface tooltip.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QFormLayout, QLabel, QLineEdit, QWidget


def test_every_registered_displayed_setting_has_authored_help():
    """All live defaults have prose, rather than the generated name fallback."""
    from spacr.qt.app import APPS
    from spacr.qt.screens.settings_model import (
        _APP_HIDDEN_KEYS,
        get_tooltips,
        resolve_default_settings,
    )

    # Resolving first imports modules that contribute settings metadata
    # through register_defaults. Reading the tooltip table before this step
    # would make the result depend on whichever module another test imported.
    inventories = {
        app_key: resolve_default_settings(app_key)
        for app_key, _name, _description, _section in APPS
    }
    descriptions = get_tooltips()
    missing = []
    checked = 0
    for app_key, defaults in inventories.items():
        shown = set(defaults) - set(_APP_HIDDEN_KEYS.get(app_key, set()))
        checked += len(shown)
        for key in sorted(shown):
            if not str(descriptions.get(key, "")).strip():
                missing.append(f"{app_key}.{key}")

    assert checked > 800, (
        f"only {checked} setting occurrences were checked; the registry "
        "inventory is no longer exhaustive")
    assert not missing, (
        "registered settings with no authored tooltip:\n  "
        + "\n  ".join(missing))


def test_a_hand_built_setting_uses_the_shared_label_only_popup(qtbot):
    """The retargeting pass is a behaviour adapter, not only a text move."""
    from spacr.qt.screens.settings_model import retarget_field_tooltips
    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    field = QLineEdit()
    field.setToolTip("Select the measured score column used for this analysis.")
    form.addRow("Score column", field)
    label = form.labelForField(field)
    assert isinstance(label, QLabel)

    assert retarget_field_tooltips(host) == 1
    assert field.toolTip() == ""
    assert label.property("apiTooltipHtml")

    popup = HoverTooltip.instance()
    popup.cancel_hide()
    popup.hide()
    QApplication.sendEvent(field, QEvent(QEvent.Enter))
    QApplication.processEvents()
    assert not popup.isVisible(), "hovering the input field opened its help"

    QApplication.sendEvent(label, QEvent(QEvent.Enter))
    QApplication.processEvents()
    assert popup.isVisible(), "hovering the setting text opened no help"
    assert popup._anchor is label
    assert "measured score column" in popup.text_label().text()

    # The same popup owns this path, so the hand-built setting inherits the
    # rounded dark-grey frame and transparent inner containers instead of a
    # native black tooltip nested inside it.
    sheet = popup.styleSheet()
    assert "QFrame#HoverTooltip" in sheet
    assert "border-radius: 6px" in sheet
    assert "QWidget#HoverTooltipTextColumn" in sheet
    assert "background: transparent" in sheet

    QApplication.sendEvent(label, QEvent(QEvent.Leave))
    popup._anchor = None
    popup._pointer_is_on_me = lambda: False
    try:
        popup._maybe_hide()
        QApplication.processEvents()
        assert not popup.isVisible(), (
            "the tooltip stayed open after the pointer left its hover area")
    finally:
        del popup._pointer_is_on_me
        popup.cancel_hide()
        popup.hide()
