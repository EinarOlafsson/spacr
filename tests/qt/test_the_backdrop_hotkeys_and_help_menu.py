"""Ctrl+T stops the background, Ctrl+Shift+B blanks it, and Help is tidy."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction  # noqa: E402
from PySide6.QtWidgets import QMenu  # noqa: E402


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    import spacr.qt.app as A

    win = A.MainWindow()
    qtbot.addWidget(win)
    return win


def test_ctrl_t_stops_the_animation(window):
    action = window.findChild(QAction, "ToggleBackdrop")
    assert action is not None
    assert action.shortcut().toString() == "Ctrl+T"
    assert action.isCheckable()


def test_ctrl_shift_b_blanks_it(window):
    action = window.findChild(QAction, "BlankBackdrop")
    assert action is not None
    assert action.shortcut().toString() == "Ctrl+Shift+B"


def test_it_is_not_ctrl_b(window):
    """Ctrl+B is the app drawer and the only keyboard route to it. Taking it
    would remove a panel from keyboard users to gain a decoration toggle."""
    taken = {a.shortcut().toString() for a in window.actions()
             if a.shortcut().toString()}
    assert "Ctrl+B" in taken, "the app drawer lost its shortcut"
    blank = window.findChild(QAction, "BlankBackdrop")
    assert blank.shortcut().toString() != "Ctrl+B"


def test_blanking_stops_the_animation_before_hiding(window):
    """A hidden backdrop still rendering spends the threads and shows
    nothing for them."""
    import inspect

    body = inspect.getsource(type(window)._set_backdrop_blank)
    stop = body.index("setter(False)")
    hide = body.index("child.hide()")
    assert stop < hide


def test_the_two_toggles_agree(window):
    """Otherwise Ctrl+T appears to do nothing while blanked."""
    animate = window.findChild(QAction, "ToggleBackdrop")
    animate.setChecked(True)
    window._set_backdrop_blank(True)
    assert animate.isChecked() is False
    window._set_backdrop_blank(False)
    assert animate.isChecked() is True


def test_blanking_a_window_with_no_backdrop_is_fine(window):
    assert window._set_backdrop_blank(True) >= 0


def test_neither_toggle_can_break_the_menu(window, monkeypatch):
    """A decoration that will not stop must not break a menu action."""
    class _Angry:
        def set_animating(self, _on):
            raise RuntimeError("no")

        def hide(self):
            raise RuntimeError("no")

    monkeypatch.setattr(window, "findChildren", lambda _k: [_Angry()])
    assert window._set_backdrop_blank(True) == 0
    assert window._set_backdrop_animating(False) == 0


# --- the help menu ---------------------------------------------------------


def _help_actions(window):
    for menu in window.menuBar().findChildren(QMenu):
        if menu.title().replace("&", "") == "Help":
            return [a for a in menu.actions() if a.text()]
    return []


def test_the_web_suffix_is_gone(window):
    labels = [a.text() for a in _help_actions(window)]
    assert "Tutorial" in labels
    assert "Documentation" in labels
    assert not [t for t in labels if "(web)" in t]


def test_no_help_entry_wears_an_icon(window):
    """Both carried SP_MessageBoxInformation -- the blue circled i a dialog
    uses for a notice, which beside a menu label reads as a badge. The same
    one on both made it noise twice over."""
    wearing = [a.text() for a in _help_actions(window) if not a.icon().isNull()]
    assert wearing == []


def test_where_the_page_opens_is_still_said(window):
    """Dropping "(web)" from the label must not lose the fact."""
    tips = {a.text(): a.statusTip() for a in _help_actions(window)}
    assert "browser" in tips["Tutorial"]
    assert "browser" in tips["Documentation"]


def test_the_translations_moved_with_the_labels():
    """The catalog keys on the English string, so a rename here alone drops
    the row in nine languages."""
    from spacr.qt.i18n import _ROWS

    assert "Tutorial" in _ROWS
    assert "Documentation" in _ROWS
    assert "Tutorial (web)" not in _ROWS
