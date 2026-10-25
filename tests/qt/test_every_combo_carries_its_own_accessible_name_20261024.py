"""Every visible combo box on a module screen carries an accessible name of its own.

Qt on Linux and macOS announces an unnamed combo box by its current item, so
``test_every_visible_control_has_an_accessible_name`` passed here while the
Windows sweep found the database browser's operator menu, the two Run
History filters and the Experiment Design role menus nameless: Windows
gives an unnamed combo no name at all. This reads ``accessibleName()``
itself, so it fails on every platform the way Windows does.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox  # noqa: E402

from spacr.qt.app import APPS  # noqa: E402

from .test_accessibility_pass import _open  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("app_key", [key for key, *_ in APPS])
def test_every_visible_combo_has_its_own_accessible_name(
        qtbot, qt_theme_applied, monkeypatch, app_key):
    screen = _open(qtbot, monkeypatch, app_key)
    nameless = [
        f"{widget.objectName() or '?'} in "
        f"{type(widget.parentWidget()).__name__}, first item "
        f"{widget.itemText(0)!r}"
        for widget in screen.findChildren(QComboBox)
        if widget.isVisibleTo(screen) and not widget.accessibleName().strip()
    ]
    assert not nameless, f"{app_key}: {nameless}"


def test_the_named_menus_say_what_they_choose(qtbot, qt_theme_applied,
                                              monkeypatch):
    from spacr.qt import i18n

    names = {}
    for app_key in ("db_browser", "run_history", "experiment_design"):
        screen = _open(qtbot, monkeypatch, app_key)
        names[app_key] = sorted({
            widget.accessibleName()
            for widget in screen.findChildren(QComboBox)
            if widget.isVisibleTo(screen)})
    assert i18n.tr("Comparison") in names["db_browser"]
    assert {i18n.tr("Module"), i18n.tr("Status")} <= set(names["run_history"])
    assert i18n.tr("Role") in names["experiment_design"]
