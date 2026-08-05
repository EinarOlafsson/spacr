"""The console / AI-chat splitter inside :class:`ConsolePanel`.

The user's ask was "make the AI chatbox larger like I can with the live
preview, with the console becoming smaller when the chatbox becomes bigger".
The AI chat is not a sibling of the console -- it lives *inside* it -- so the
fix was a vertical ``QSplitter`` between the console's scrolling output box
and the chat row beneath it, matching the live-preview / console splitter one
level up in ``AppScreen``.

Everything here is asserted on MEASURED geometry after a real
``QTest``-driven drag of the handle, not on the sizes we asked for: a
``setSizes`` call that Qt then ignores would pass a test written the other
way round and still leave the user unable to resize anything.

Four things have to hold, and each has a test:

* dragging the handle grows one pane and shrinks the other by the SAME
  number of pixels -- the panel does not grow, the split moves;
* the position survives closing and reopening the screen, and survives it at
  a *different* window size, which is why the stored value is
  ``QSplitter.saveState()`` and not a pixel pair;
* neither pane collapses to nothing on a normal drag, or even on a violent
  one;
* the DEFAULT split is exactly what a user saw before the splitter existed,
  at every window height -- a resize handle nobody asked for must not move
  the furniture of a user who never touches it.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import QByteArray, QPoint, Qt
from PySide6.QtTest import QTest


PANEL_W = 700
PANEL_H = 700

#: Height the chat box was pinned at by the old ``setMaximumHeight(120)``,
#: measured on the pre-splitter layout at every window height below.
LEGACY_CHAT_H = 120


def _panel(qtbot, persist_key: str = "", height: int = PANEL_H):
    """A shown, laid-out ConsolePanel whose geometry is real."""
    from spacr.qt.widgets.console_panel import ConsolePanel
    panel = ConsolePanel(active_app_label="Mask", persist_key=persist_key)
    qtbot.addWidget(panel)
    panel.resize(PANEL_W, height)
    panel.show()
    qtbot.waitExposed(panel)
    return panel


def _drag_handle(qtbot, panel, dy: int) -> None:
    """Drag the splitter handle ``dy`` pixels (negative = up = bigger chat).

    A real press / move / release on the ``QSplitterHandle``, so this
    exercises the same code path the user's mouse does rather than calling
    ``setSizes`` and asserting we called it.
    """
    handle = panel._split.handle(1)
    start = QPoint(4, 4)
    end = QPoint(4, 4 + dy)
    QTest.mousePress(handle, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(handle, end)
    QTest.mouseRelease(handle, Qt.LeftButton, Qt.NoModifier, end)
    qtbot.wait(1)


# ---------------------------------------------------------------------------
# The split exists, and it is the two panes the user named
# ---------------------------------------------------------------------------

def test_console_and_chat_are_the_two_halves_of_a_splitter(qtbot,
                                                           qt_theme_applied):
    """There is a handle to drag, between exactly the console and the chat."""
    from PySide6.QtWidgets import QSplitter
    panel = _panel(qtbot)
    assert isinstance(panel._split, QSplitter)
    assert panel._split.orientation() == Qt.Vertical
    assert panel._split.count() == 2
    assert panel._split.widget(0) is panel._console_box
    assert panel._split.widget(1) is panel._chat_row
    assert panel._input.parentWidget() is panel._chat_row


# ---------------------------------------------------------------------------
# Dragging trades height, pixel for pixel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dy", [-150, -40, 60])
def test_drag_grows_chat_and_shrinks_console_by_the_same_amount(
        qtbot, qt_theme_applied, dy):
    """Whatever the console loses, the chat box gains, and vice versa."""
    panel = _panel(qtbot)
    console_before = panel._console_box.height()
    chat_before = panel._chat_row.height()

    _drag_handle(qtbot, panel, dy)

    console_after = panel._console_box.height()
    chat_after = panel._chat_row.height()

    # Measured on the widgets themselves, not on `sizes()`.
    assert console_after - console_before == dy
    assert chat_after - chat_before == -dy
    # The panel did not grow to accommodate anyone: the two panes plus the
    # handle still add up to the panel.
    assert console_after + chat_after == console_before + chat_before


def test_dragging_up_is_what_the_user_asked_for(qtbot, qt_theme_applied):
    """Bigger AI chat box, smaller console. Stated in the user's words."""
    panel = _panel(qtbot)
    console_before = panel._console_box.height()
    chat_before = panel._chat_row.height()

    _drag_handle(qtbot, panel, -200)

    assert panel._chat_row.height() > chat_before
    assert panel._console_box.height() < console_before


def test_chat_box_can_exceed_the_old_hard_cap(qtbot, qt_theme_applied):
    """The 120px ceiling is gone -- it is what made this impossible.

    ``_ChatInput`` used to carry ``setMaximumHeight(120)``, so the row was
    pinned there no matter what the layout offered it. A splitter over a
    capped child would have produced a handle that moves and a chat box that
    does not.
    """
    panel = _panel(qtbot)
    _drag_handle(qtbot, panel, -260)
    assert panel._chat_row.height() > LEGACY_CHAT_H
    assert panel._input.height() > LEGACY_CHAT_H


# ---------------------------------------------------------------------------
# Neither pane collapses
# ---------------------------------------------------------------------------

def test_neither_pane_collapses_on_a_normal_drag(qtbot, qt_theme_applied):
    """A big drag either way leaves both panes visible and non-empty."""
    from spacr.qt.widgets.console_panel import (
        CHAT_MIN_HEIGHT, CONSOLE_MIN_HEIGHT,
    )
    panel = _panel(qtbot)

    _drag_handle(qtbot, panel, -400)          # chat as tall as it will go
    assert panel._console_box.height() >= CONSOLE_MIN_HEIGHT
    assert panel._chat_row.height() >= CHAT_MIN_HEIGHT
    assert panel._console_box.isVisible()
    assert panel._chat_row.isVisible()

    _drag_handle(qtbot, panel, 400)           # and back the other way
    assert panel._console_box.height() >= CONSOLE_MIN_HEIGHT
    assert panel._chat_row.height() >= CHAT_MIN_HEIGHT


def test_a_violent_drag_clamps_rather_than_collapsing(qtbot, qt_theme_applied):
    """Throwing the handle off the top of the widget still leaves a console."""
    from spacr.qt.widgets.console_panel import (
        CHAT_MIN_HEIGHT, CONSOLE_MIN_HEIGHT,
    )
    panel = _panel(qtbot)
    _drag_handle(qtbot, panel, -5000)
    assert panel._console_box.height() >= CONSOLE_MIN_HEIGHT
    _drag_handle(qtbot, panel, 5000)
    assert panel._chat_row.height() >= CHAT_MIN_HEIGHT


def test_children_are_not_collapsible(qtbot, qt_theme_applied):
    """The same guard the live-preview splitter uses."""
    panel = _panel(qtbot)
    assert panel._split.childrenCollapsible() is False


# ---------------------------------------------------------------------------
# The default is today's appearance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("height", [500, 700, 1000, 1400])
def test_default_split_is_unchanged_from_the_pre_splitter_layout(
        qtbot, qt_theme_applied, height):
    """A user who never drags sees exactly what they saw before.

    The old layout was ``QVBoxLayout(spacing=SPACING["sm"])`` with the
    console on stretch 1 and a chat input clamped to 120px, which measured
    120 for the chat and ``height - 120 - 8`` for the console at every window
    size tried. The splitter reproduces both numbers -- including the 8px gap,
    which is now the handle.
    """
    from spacr.qt.widgets.console_panel import DEFAULT_CHAT_HEIGHT
    panel = _panel(qtbot, height=height)
    assert DEFAULT_CHAT_HEIGHT == LEGACY_CHAT_H
    assert panel._chat_row.height() == LEGACY_CHAT_H
    assert panel._split.handleWidth() == 8
    assert panel._console_box.height() == height - LEGACY_CHAT_H - 8


def test_resizing_the_window_does_not_move_the_chat_box(qtbot,
                                                        qt_theme_applied):
    """Only the console absorbs a window resize.

    Both halves on stretch 1 -- the live preview's settings, where both halves
    are content -- would grow the chat box every time the window got taller.
    That is a visible change for a user who never touched the handle, so the
    chat box is stretch 0 instead.
    """
    panel = _panel(qtbot, height=600)
    chat_before = panel._chat_row.height()
    panel.resize(PANEL_W, 1100)
    qtbot.wait(1)
    assert panel._chat_row.height() == chat_before
    assert panel._console_box.height() == 1100 - chat_before - 8


def test_a_dragged_split_keeps_its_chat_height_across_a_resize(
        qtbot, qt_theme_applied):
    """The pixels the user gave the chat box stay with the chat box."""
    panel = _panel(qtbot, height=700)
    _drag_handle(qtbot, panel, -180)
    chat = panel._chat_row.height()
    panel.resize(PANEL_W, 1000)
    qtbot.wait(1)
    assert panel._chat_row.height() == chat


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_split_survives_closing_and_reopening_the_screen(qtbot,
                                                         qt_theme_applied):
    """Enlarge the chat box, close the screen, open it again: still enlarged."""
    first = _panel(qtbot, persist_key="mask")
    _drag_handle(qtbot, first, -200)
    console, chat = first._console_box.height(), first._chat_row.height()
    assert chat > LEGACY_CHAT_H          # the drag actually did something
    first.close()

    second = _panel(qtbot, persist_key="mask")
    assert second._chat_row.height() == chat
    assert second._console_box.height() == console


def test_split_is_restored_at_a_different_window_size(qtbot, qt_theme_applied):
    """Why the stored value is ``saveState()`` and not ``[572, 120]``.

    A pixel pair saved on one display is wrong on the next. Restoring the
    splitter's own state keeps the chat box at the height the user chose and
    gives the extra window height to the console, which is where it belongs.
    """
    first = _panel(qtbot, persist_key="mask", height=700)
    _drag_handle(qtbot, first, -200)
    chat = first._chat_row.height()
    first.close()

    bigger = _panel(qtbot, persist_key="mask", height=1000)
    assert bigger._chat_row.height() == chat
    assert bigger._console_box.height() == 1000 - chat - 8


def test_split_is_remembered_per_screen(qtbot, qt_theme_applied):
    """A tall chat box on Mask does not force one on Measure."""
    mask = _panel(qtbot, persist_key="mask")
    _drag_handle(qtbot, mask, -200)
    mask.close()

    measure = _panel(qtbot, persist_key="measure")
    assert measure._chat_row.height() == LEGACY_CHAT_H


def test_a_panel_without_a_persist_key_saves_nothing(qtbot, qt_theme_applied):
    """The bare panel a test or a dock builds must not write to settings."""
    from spacr.qt.widgets.console_panel import get_split_state
    panel = _panel(qtbot)
    assert panel._persist_key == ""
    _drag_handle(qtbot, panel, -150)
    assert get_split_state("") is None
    # And it still starts on the default next time.
    again = _panel(qtbot)
    assert again._chat_row.height() == LEGACY_CHAT_H


def test_set_split_sizes_persists_like_a_drag(qtbot, qt_theme_applied):
    """The programmatic route leaves the panel in the same place as a drag."""
    panel = _panel(qtbot, persist_key="umap")
    panel.set_split_sizes(300, 392)
    qtbot.wait(1)
    assert panel.split_sizes() == [300, 392]
    panel.close()

    reopened = _panel(qtbot, persist_key="umap")
    assert reopened.split_sizes() == [300, 392]


def test_state_helpers_ignore_an_empty_key(qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import get_split_state, set_split_state
    set_split_state("", QByteArray(b"nonsense"))     # must not raise
    assert get_split_state("") is None
    assert get_split_state("   ") is None


def test_unreadable_stored_state_falls_back_to_the_default(qtbot,
                                                           qt_theme_applied):
    """A garbage entry leaves the user on the default, not on a broken split.

    ``QSplitter.restoreState`` rejects a blob without its magic marker and
    returns False rather than raising, so the panel keeps the default sizes
    it was already given.
    """
    from spacr.qt.widgets.console_panel import set_split_state
    set_split_state("classify", QByteArray(b"not a splitter state"))
    panel = _panel(qtbot, persist_key="classify")
    assert panel._chat_row.height() == LEGACY_CHAT_H
    assert panel._split.childrenCollapsible() is False


def test_stored_state_of_the_wrong_type_is_ignored(qtbot, qt_theme_applied,
                                                   monkeypatch):
    """A settings backend handing back a str is 'no preference', not a crash."""
    from spacr.qt.widgets import console_panel as cp

    class _Fake:
        def value(self, _key, default=None):
            return "572,120"

    monkeypatch.setattr(cp, "_settings", lambda: _Fake())
    assert cp.get_split_state("mask") is None


def test_a_settings_backend_that_raises_does_not_break_the_console(
        qtbot, qt_theme_applied, monkeypatch):
    """Reading or writing preferences must never take the console with it."""
    from spacr.qt.widgets import console_panel as cp

    def _boom():
        raise RuntimeError("no settings here")

    monkeypatch.setattr(cp, "_settings", _boom)
    assert cp.get_split_state("mask") is None
    cp.set_split_state("mask", QByteArray(b"x"))      # swallowed
    panel = _panel(qtbot, persist_key="mask")         # still builds
    _drag_handle(qtbot, panel, -100)                  # and still drags
    assert panel._chat_row.height() > LEGACY_CHAT_H


# ---------------------------------------------------------------------------
# The restructure must not have cost the console anything
# ---------------------------------------------------------------------------

def test_zoom_still_reaches_both_the_console_and_the_chat(qtbot,
                                                          qt_theme_applied):
    """Font scale is applied through the holder, which moved inside the
    splitter. Entries added after the move must still pick it up."""
    from spacr.qt.widgets.console_panel import _StdoutBlock
    panel = _panel(qtbot)
    panel.append_stdout("hello from the pipeline\n")
    panel.set_console_font_pt(21)
    blocks = panel._holder.findChildren(_StdoutBlock)
    assert blocks
    assert all(b.font().pointSize() == 21 for b in blocks)


def test_the_splitter_does_not_paint_over_the_backdrop(qtbot,
                                                       qt_theme_applied):
    """Page opacity has to survive the new container.

    An untagged ``QWidget`` inherits the blanket ``QWidget { background-color:
    bg }`` rule and paints the WINDOW colour, which no opacity setting can
    reach -- exactly the black slab that used to span the console and the chat
    box. ``AppScreen`` sweeps every QSplitter, and the panel tags this one
    itself so a standalone ConsolePanel is right too.
    """
    from spacr.qt.theme import TRANSPARENT_PROPERTY
    panel = _panel(qtbot)
    assert panel._split.property(TRANSPARENT_PROPERTY) is True
    # The chat row it now parents is scaffolding too, and was covered by the
    # anonymous-QWidget half of the same sweep before the move.
    assert not panel._chat_row.objectName()


def test_appscreen_sweep_still_tags_the_console_splitter(qtbot,
                                                         qt_theme_applied):
    """The generic page-surface sweep reaches the new splitter by type.

    ``clear_container_surfaces`` tags every ``QSplitter`` regardless of name,
    which is what keeps the console splitter transparent inside a real
    AppScreen even though it carries an ``objectName``.
    """
    from PySide6.QtWidgets import QSplitter
    from spacr.qt.theme import TRANSPARENT_PROPERTY, clear_container_surfaces
    panel = _panel(qtbot)
    panel._split.setProperty(TRANSPARENT_PROPERTY, None)
    assert panel._split.property(TRANSPARENT_PROPERTY) is not True

    tagged = clear_container_surfaces(panel)

    assert tagged
    assert panel._split in panel.findChildren(QSplitter)
    assert panel._split.property(TRANSPARENT_PROPERTY) is True


def test_console_output_is_unaffected_by_the_split(qtbot, qt_theme_applied):
    """Appending still works, still coalesces into one block, still scrolls."""
    panel = _panel(qtbot)
    for i in range(50):
        panel.append_stdout(f"line {i}\n")
    from spacr.qt.widgets.console_panel import _StdoutBlock
    blocks = panel._holder.findChildren(_StdoutBlock)
    assert len(blocks) == 1
    assert "line 49" in blocks[0].text()
