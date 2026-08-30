"""The toggle re-styles itself when the application sheet moves under it.

It is not styled by the application stylesheet -- it carries its own, so that
Zoom can reach it through ``font_px`` -- which means every preference change
has to arrive through ``changeEvent`` or the label keeps the size and the
colour it was built with until the app restarts. That handler is also the one
place the widget is handed an event by somebody else, so it is written to
survive an event it cannot read.

The rest is the elision ladder: a caption wider than its slot is shortened,
one narrow enough is not, and a slot too narrow for even an ellipsis gets the
full text slightly clipped rather than a blank control.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from spacr.qt.widgets import ai_toggle_label as atl
from spacr.qt.widgets.ai_toggle_label import ELIDE_ABOVE_PX, AiToggleLabel

pytestmark = pytest.mark.qt


class _EventWhoseTypeCannotBeRead(QEvent):
    """A ``QEvent`` whose ``type()`` raises, as a broken sender's would."""

    def type(self):                          # noqa: A003 (Qt naming)
        raise RuntimeError("this event cannot say what it is")


def _press(button):
    return QMouseEvent(QEvent.MouseButtonPress, QPointF(4.0, 4.0),
                       QPointF(4.0, 4.0), button, button, Qt.NoModifier)


# -- the handler that has to survive whatever it is given ---------------------

def test_an_event_that_cannot_say_what_it_is_leaves_the_style_alone(
        qapp, monkeypatch):
    """An unreadable event restyles nothing, and a readable one still does.

    Both halves matter: swallowing the exception is only correct if the
    handler still works for the events it was written for.
    """
    toggle = AiToggleLabel(text="AI")
    before = toggle.styleSheet()
    monkeypatch.setattr(atl, "font_px", lambda name: 31)

    toggle.changeEvent(_EventWhoseTypeCannotBeRead(QEvent.StyleChange))

    assert toggle.styleSheet() == before
    assert "font-size: 31px" not in toggle.styleSheet()

    toggle.changeEvent(QEvent(QEvent.StyleChange))

    assert "font-size: 31px" in toggle.styleSheet()


@pytest.mark.parametrize("kind", [QEvent.PaletteChange,
                                  QEvent.ApplicationPaletteChange,
                                  QEvent.ApplicationFontChange])
def test_every_style_bearing_event_restyles_the_toggle(qapp, monkeypatch,
                                                       kind):
    """A theme or font change reaches this widget through one of four events."""
    toggle = AiToggleLabel(text="AI")
    monkeypatch.setattr(atl, "font_px", lambda name: 29)

    toggle.changeEvent(QEvent(kind))

    assert "font-size: 29px" in toggle.styleSheet()


def test_an_unrelated_event_does_not_restyle(qapp, monkeypatch):
    """A show or an enable is not a reason to rebuild the sheet."""
    toggle = AiToggleLabel(text="AI")
    before = toggle.styleSheet()
    monkeypatch.setattr(atl, "font_px", lambda name: 33)

    toggle.changeEvent(QEvent(QEvent.EnabledChange))

    assert toggle.styleSheet() == before


def test_a_restyle_that_changes_nothing_is_not_written_again(qapp):
    """The sheet is compared before it is set, or a StyleChange storms.

    ``setStyleSheet`` posts a ``StyleChange`` straight back at this widget,
    so an unconditional write would call in again on every delivery.
    """
    toggle = AiToggleLabel(text="AI")
    sheet = toggle.styleSheet()

    toggle.changeEvent(QEvent(QEvent.StyleChange))

    assert toggle.styleSheet() == sheet


def test_a_reentrant_restyle_is_refused(qapp, monkeypatch):
    """While a sheet is being applied, a second application is a no-op."""
    toggle = AiToggleLabel(text="AI")
    monkeypatch.setattr(atl, "font_px", lambda name: 27)
    toggle._restyling = True

    toggle._refresh_style()

    assert "font-size: 27px" not in toggle.styleSheet()

    toggle._restyling = False
    toggle._refresh_style()

    assert "font-size: 27px" in toggle.styleSheet()


# -- the colours the two states ink in ----------------------------------------

def test_the_on_state_inks_in_the_theme_invariant_accent(qapp):
    """ON is the accent in every theme; OFF is the theme's own foreground."""
    from spacr.qt.theme import active_palette

    palette = active_palette()
    toggle = AiToggleLabel(text="AI")

    assert f"color: {palette['fg']}" in toggle.styleSheet()

    toggle.setChecked(True)

    assert f"color: {palette['button_accent']}" in toggle.styleSheet()


# -- the check-box compatible API ---------------------------------------------

def test_setting_the_state_it_already_has_emits_nothing(qapp, qtbot):
    """``toggled`` is a change notification, not a state report."""
    toggle = AiToggleLabel(text="AI")
    seen = []
    toggle.toggled.connect(seen.append)

    toggle.setChecked(False)

    assert seen == []
    assert toggle.isChecked() is False

    toggle.setChecked(True)

    assert seen == [True]
    assert toggle.isChecked() is True

    toggle.setChecked(True)

    assert seen == [True]


def test_a_truthy_value_is_taken_as_on(qapp):
    """The API is QCheckBox-compatible, so it takes what a caller passes."""
    toggle = AiToggleLabel(text="AI")

    toggle.setChecked(1)

    assert toggle.isChecked() is True


def test_a_left_click_flips_the_switch_and_says_so(qapp):
    toggle = AiToggleLabel(text="AI")
    seen = []
    toggle.toggled.connect(seen.append)

    toggle.mousePressEvent(_press(Qt.LeftButton))
    toggle.mousePressEvent(_press(Qt.LeftButton))

    assert seen == [True, False]
    assert toggle.isChecked() is False


# -- the caption -------------------------------------------------------------

def test_the_default_toggle_is_the_ai_switch(qapp):
    """Built with no arguments it is still the AI switch it started as."""
    toggle = AiToggleLabel()

    assert toggle.text() == "AI"
    assert "AI" in toggle.toolTip()
    assert toggle.property("_spacr_i18n_text") == "AI"


def test_a_caller_supplied_tooltip_replaces_the_ai_flavoured_default(qapp):
    toggle = AiToggleLabel(text="Live", tooltip="Preview as you type.")

    assert toggle.toolTip() == "Preview as you type."
    assert toggle.property("_spacr_i18n_tooltip") == "Preview as you type."


def test_an_empty_caption_reports_whatever_the_label_holds(qapp):
    """With nothing stored, the logical text falls through to QLabel's own."""
    toggle = AiToggleLabel(text="")

    assert toggle.text() == ""
    assert toggle.displayed_text() == ""


def test_a_language_switch_moves_the_stored_caption_too(qapp):
    """``setText`` is where the new translation lands, so it must be kept."""
    toggle = AiToggleLabel(text="Live")

    toggle.setText("Beo")

    assert toggle.text() == "Beo"
    assert toggle.displayed_text() == "Beo"


def test_a_slot_too_narrow_for_an_ellipsis_keeps_the_whole_caption(qapp,
                                                                  qtbot):
    """A blank toggle says nothing; a clipped one still says which control it is.

    Zoom lands here: the enlarged font is measured against the width the
    layout granted the smaller one, one relayout behind.
    """
    toggle = AiToggleLabel(text="Hyperparameter search")
    qtbot.addWidget(toggle)
    toggle.show()
    qapp.processEvents()

    # Wide enough that the padding still leaves a measurable slot -- so the
    # elision is really attempted -- but far too narrow for the ellipsis it
    # comes back with.
    toggle.resize(22, 20)
    qapp.processEvents()
    assert toggle.contentsRect().width() > 0

    assert toggle.displayed_text() == "Hyperparameter search"


def test_a_toggle_with_no_width_yet_is_not_elided(qapp, qtbot):
    """With no width granted there is nothing to measure, so nothing is written.

    A width of nothing is not a width of everything: a toggle that is
    showing an elided caption when its slot collapses -- a hidden page, a
    splitter closed to the handle -- keeps what it was showing, rather than
    being repainted with the full caption it has no room for.
    """
    toggle = AiToggleLabel(text="Hyperparameter search")
    qtbot.addWidget(toggle)
    toggle.show()
    qapp.processEvents()
    toggle.resize(130, 20)
    qapp.processEvents()
    elided = toggle.displayed_text()
    assert elided.endswith("\u2026")

    toggle.resize(0, 0)
    toggle._apply_elision()

    assert toggle.displayed_text() == elided
    assert toggle.text() == "Hyperparameter search"


def test_a_caption_that_already_fits_is_not_rewritten(qapp, qtbot):
    """Re-eliding to the same string must not touch the label again."""
    toggle = AiToggleLabel(text="Live")
    qtbot.addWidget(toggle)
    toggle.show()
    qapp.processEvents()
    toggle.resize(200, 20)
    qapp.processEvents()

    toggle._apply_elision()
    toggle._apply_elision()

    assert toggle.displayed_text() == "Live"
    assert toggle.text() == "Live"


def test_a_short_caption_keeps_its_full_minimum_width(qapp):
    """Only a caption wider than the cap is held back."""
    short = AiToggleLabel(text="AI")

    assert short.minimumSizeHint().width() == short.sizeHint().width()
    assert short.minimumSizeHint().width() <= ELIDE_ABOVE_PX


def test_a_long_caption_is_capped_at_the_elide_width(qapp):
    long_one = AiToggleLabel(text="Hyperparameter search")

    assert long_one.sizeHint().width() > ELIDE_ABOVE_PX
    assert long_one.minimumSizeHint().width() == ELIDE_ABOVE_PX
    assert long_one.minimumSizeHint().height() > 0


def test_a_caption_set_while_an_elision_is_running_is_not_stored(qapp):
    """The shortened string must never become the toggle's logical caption.

    ``_apply_elision`` raises ``_eliding`` for the duration of the write it
    makes; anything that reaches ``setText`` in that window is the elided
    form, not a new caption. Storing it would truncate the remembered text
    one character per relayout until the toggle had forgotten its own name.
    """
    toggle = AiToggleLabel(text="Hyperparameter search")

    toggle._eliding = True
    try:
        toggle.setText("Hyperpa\u2026")
    finally:
        toggle._eliding = False

    assert toggle.text() == "Hyperparameter search"
    assert toggle.displayed_text() == "Hyperpa\u2026"
