"""The setting's NAME is the hover target, on every screen."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QAbstractButton, QFormLayout, QLabel

import spacr.qt.app as app_module


@pytest.fixture(scope="module")
def window(qapp):
    win = app_module.MainWindow()
    win.resize(1400, 900)
    win.show()
    yield win
    win.close()


def _survey(screen):
    """Count where the help sits, on the rows a user can actually reach.

    Counting `_widgets` instead is misleading: touching it MATERIALISES lazy
    widgets that were never placed on any form, which is what made an early
    measurement read 1,551 rows on a screen showing 106.
    """
    on_field = []
    on_name = 0
    for form in screen.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label_item = form.itemAt(row, QFormLayout.LabelRole)
            field_item = form.itemAt(row, QFormLayout.FieldRole)
            if field_item is None or field_item.widget() is None:
                continue
            field = field_item.widget()
            if field.isHidden():
                continue
            label = label_item.widget() if label_item else None
            if label is not None and label.isHidden():
                continue
            if (field.toolTip() or "").strip():
                # A control with visible text of its own IS its own name.
                if isinstance(field, QAbstractButton) and \
                        (field.text() or "").strip():
                    continue
                on_field.append(field.property("settingKey")
                                or type(field).__name__)
            elif label is not None and (
                    (label.toolTip() or "").strip()
                    or any((c.toolTip() or "").strip()
                           for c in label.findChildren(QLabel))):
                on_name += 1
    return on_field, on_name


@pytest.mark.parametrize("key", ["mask", "measure", "regression"])
def test_the_help_is_on_the_name_not_the_field(window, qapp, key):
    """Asked for repeatedly, and measured through the path a user takes."""
    window._on_nav_selected(key)
    qapp.processEvents()
    screen = window._screens[key]

    on_field, on_name = _survey(screen)

    assert on_name > 20, f"{key}: only {on_name} names carry the help"
    assert on_field == [], (
        f"{key}: {len(on_field)} settings still pop from the field, "
        f"e.g. {on_field[:5]}")


def test_the_move_survives_the_language_pass(window, qapp):
    """The pass that re-tips runs on ARRIVAL, after the panel is built.

    This is what made every earlier attempt look fixed and then not be:
    `refresh_api_tooltips` walks every widget carrying a `settingKey` and
    re-applies the html, so a field with no display role was tipped again a
    moment later.
    """
    from spacr.qt.screens.settings_model import refresh_api_tooltips

    window._on_nav_selected("mask")
    qapp.processEvents()
    screen = window._screens["mask"]
    field = screen._settings_model._widgets["cell_channel"]
    assert (field.toolTip() or "") == ""

    refresh_api_tooltips(screen)
    qapp.processEvents()
    assert (field.toolTip() or "") == "", (
        "the language pass put the help back on the field")


def test_a_control_that_is_its_own_label_keeps_its_help():
    """Taking it off would leave that setting with no help anywhere."""
    from PySide6.QtWidgets import QCheckBox

    from spacr.qt.screens.settings_model import _is_a_settings_field

    box = QCheckBox("Resume")
    box.setProperty("settingKey", "resume")
    assert _is_a_settings_field(box) is False

    unnamed = QCheckBox("")
    unnamed.setProperty("settingKey", "resume")
    assert _is_a_settings_field(unnamed) is True
