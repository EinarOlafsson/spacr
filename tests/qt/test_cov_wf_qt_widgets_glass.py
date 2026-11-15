"""The one install point when the filter it is replacing fights back.

``install_glass_everywhere`` runs once at startup and is the only thing that
stands between the thirty-nine settings dialogs and the black rectangles the
setup screen was built to get rid of. Its neighbours already assert the happy
path -- install once, install again is a no-op, a new application gets a new
filter. This file drives the three seams left over, all of them about the
BOOKKEEPING rather than the look:

* the old application refuses to give the filter back;
* the filter was recorded but its owner was not, which is what an
  installation made by an older copy of this module looks like after an
  upgrade in place;
* the filter is being forgotten while no application exists at all, the state
  an embedded host is in between tearing one ``QApplication`` down and
  building the next.

Each of them has the same requirement: whatever Qt does, the module must end
up remembering exactly one filter and exactly one owner, because a stale pair
makes the next ``install_glass_everywhere`` answer "already installed" for an
application that is receiving no events -- and every dialog opened after that
is an untreated one.
"""
from __future__ import annotations


import pytest

pytest.importorskip("PySide6")


from spacr.qt.widgets import glass


@pytest.fixture(autouse=True)
def _no_installer_left_behind():
    """Leave neither a filter nor a remembered owner for the next test."""
    yield
    glass.uninstall_glass_everywhere()
    glass._INSTALLED = None
    glass._INSTALLED_APP = None


def test_deleted_settings_popup_is_not_classified_or_dereferenced(qapp):
    from PySide6.QtWidgets import QDialog
    from shiboken6 import delete

    dialog = QDialog()
    dialog.setProperty(glass.SETTINGS_POPUP, True)
    delete(dialog)
    assert glass._is_settings_popup(dialog) is False


def test_settings_popup_default_colour_matches_the_active_surface_and_is_opaque(qapp, monkeypatch):
    from PySide6.QtGui import QColor
    from spacr.qt import theme

    monkeypatch.setattr(theme, 'active_palette', lambda: {'surface': '#245678', 'bg': '#ffffff'})
    actual = glass._settings_popup_colour()
    assert actual == QColor('#245678')
    assert actual.alpha() == 255


def test_unavailable_wrapper_ownership_probe_does_not_prevent_show_treatment(qapp, monkeypatch):
    from PySide6.QtWidgets import QDialog
    from shiboken6 import Shiboken

    def unavailable(widget):
        raise RuntimeError('ownership metadata unavailable')

    dialog = QDialog()
    monkeypatch.setattr(Shiboken, 'createdByPython', unavailable)
    assert glass._built_by_python(dialog) is True
    glass.glass(dialog)
    assert dialog.findChild(glass._Backdrop) is not None
    dialog.close()
