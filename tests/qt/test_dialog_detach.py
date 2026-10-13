"""Modal dialogs must not be attachable to the main window.

A ``QDialog`` with a parent is transient-for that parent, and ``exec()``
makes it modal. A window manager that implements **attached modal dialogs**
-- GNOME/Mutter does by default -- glues that combination to its parent: the
dialog is drawn centred on the parent, cannot be dragged elsewhere, and
pulling at it manipulates the parent. On a maximised main window the user
sees the app un-maximise itself when they try to move the settings.

The bug was reported, and diagnosed, by which dialogs misbehave: Preferences
and Annotate's settings do, and the UMAP search settings, the Mask live
preview settings and the crop live settings do not. The first two are the
ones opened with ``exec()``; the rest are opened with ``show()``. That is
the WM's rule exactly, and it has nothing to do with how any of them are
built -- so the fix is at the window type, not in any one dialog.

These assertions are on the window TYPE rather than on where the window
lands, because where it lands is the window manager's decision and there is
no window manager under ``QT_QPA_PLATFORM=offscreen``. What can be asserted
here is that spaCR stops asking to be attached; the user confirms the rest
on a real desktop.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                          # noqa: E402
from PySide6.QtWidgets import QDialog, QWidget         # noqa: E402


def _window_type(widget):
    return widget.windowFlags() & Qt.WindowType.WindowType_Mask


def test_a_plain_qdialog_really_is_the_attachable_kind(qtbot):
    """The control. Without this the assertions below prove nothing."""
    parent = QWidget()
    qtbot.addWidget(parent)
    plain = QDialog(parent)
    qtbot.addWidget(plain)
    assert _window_type(plain) == Qt.WindowType.Dialog


def test_preferences_is_not_advertised_as_a_dialog(qtbot):
    from spacr.qt.preferences import PreferencesDialog

    parent = QWidget()
    qtbot.addWidget(parent)
    dlg = PreferencesDialog(parent)
    qtbot.addWidget(dlg)

    assert _window_type(dlg) == Qt.WindowType.Window
    # The parent is KEPT on purpose: dropping it would detach the dialog too,
    # but it would also stop stacking above the app and could be lost behind
    # it, and Qt would no longer destroy it with its owner.
    assert dlg.parent() is parent


def test_annotate_settings_is_not_advertised_as_a_dialog(qtbot):
    from spacr.qt.screens.annotate import AnnotateSettings, _SettingsDialog

    parent = QWidget()
    qtbot.addWidget(parent)
    dlg = _SettingsDialog(AnnotateSettings(), parent)
    qtbot.addWidget(dlg)

    assert _window_type(dlg) == Qt.WindowType.Window
    assert dlg.parent() is parent


def test_the_exec_contract_is_untouched(qtbot):
    """Every caller reads the exec() result to decide whether to apply.

    Changing the window type must not change that, or the settings silently
    stop being saved -- which would be a far worse bug than the one fixed.
    """
    from spacr.qt.screens.annotate import AnnotateSettings, _SettingsDialog

    parent = QWidget()
    qtbot.addWidget(parent)
    dlg = _SettingsDialog(AnnotateSettings(), parent)
    qtbot.addWidget(dlg)

    dlg.accept()
    assert dlg.result() == QDialog.Accepted
    assert dlg.collect() is not None


def test_detach_is_idempotent_and_survives_a_parentless_dialog(qtbot):
    from spacr.qt.dialogs import detach_from_window_manager

    orphan = QDialog()
    qtbot.addWidget(orphan)
    detach_from_window_manager(orphan)
    detach_from_window_manager(orphan)
    assert _window_type(orphan) == Qt.WindowType.Window
    assert detach_from_window_manager(orphan) is orphan
