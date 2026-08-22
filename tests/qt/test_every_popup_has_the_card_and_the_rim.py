"""The card and the travelling rim, on every popup in the program.

    "i want every settings pop up throughout the entire spacr program
     (preferences, hyperparamiters, settings, live settings, AI settings,
     figure settings, etc.) to have the same transparent background with the
     new rim ... do a sweep of the entire software"

THE SWEEP IS AN INSTALL, NOT A LIST. There are thirty-nine QDialog
subclasses in this package; a look applied by hand in each is a look the
fortieth will not have. `spacr.qt.widgets.glass` installs one application
event filter, so what is asserted here is that the filter reaches a dialog
NOBODY EDITED -- including one defined inside this test, which is the
closest thing to a dialog somebody adds next week.

    "the main settings i want controlls for the allignment lag, length, and
     mouse allignment"

Those three are the rest of the file. They are settings rather than
constants because the chase has now been reported on twice and changed
twice, which is the definition of a matter of taste.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QRectF            # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox,       # noqa: E402
                               QDialog, QLabel, QVBoxLayout)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_config():
    """The three rim keys saved and put back, and nothing else touched.

    NOT `importlib.reload`, which several older files in this directory
    use. Reloading swaps the module object while every other module goes
    on holding the old one, and anything that sandboxed its settings
    through that module quietly stops being sandboxed -- measured here as
    a neighbouring GitHub test seeing a stored token that a LATER test in
    its own file had written.

    `_settings()` builds a fresh QSettings per call, so there is nothing
    cached that a reload would have been clearing anyway. Saving the three
    values and putting them back is exact, and leaks in neither direction.
    """
    from spacr.qt import preferences

    before = (preferences.get_rim_length(), preferences.get_rim_lag(),
              preferences.get_rim_alignment())
    yield preferences
    preferences.set_rim_length(before[0])
    preferences.set_rim_lag(before[1])
    preferences.set_rim_alignment(before[2])


@pytest.fixture(autouse=True)
def nothing_is_left_ticking(app):
    """Deliver this file's own deleteLater calls before the next test.

    EVERY GLASSED DIALOG CARRIES AN AMBIENT WIDGET WITH A RUNNING TIMER.
    `deleteLater` only posts an event, and a headless test file may never
    spin a loop -- so without this the dialogs built here stay alive,
    keep ticking, and go on answering PaletteChange for the rest of the
    session. `test_space_theme` counts exactly that, and a test that fails
    because of a widget another file forgot is the least useful kind of
    failure there is.
    """
    yield
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture()
def glassed(app):
    """The filter installed, as it is at startup -- and taken off after.

    A FILTER LEFT ON DECIDES THE LOOK OF EVERY DIALOG EXAMINED AFTER IT.
    This file sorts early, so without the teardown every later dialog test
    in the same process would be looking at a card and a backdrop that its
    author never put there.
    """
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    apply_preferences_to_app(app)
    install_glass_everywhere(app)
    yield app
    uninstall_glass_everywhere(app)


def _popup(glassed, title="a settings popup"):
    dialog = QDialog()
    column = QVBoxLayout(dialog)
    column.addWidget(QLabel(title))
    dialog.resize(420, 300)
    dialog.show()
    for _ in range(8):
        glassed.processEvents()
    return dialog


def _card(dialog):
    from spacr.qt.widgets.setup_card import SetupCard

    cards = dialog.findChildren(SetupCard)
    return cards[0] if cards else None


# ---------------------------------------------------------------------------
# The sweep reaches a dialog nobody edited
# ---------------------------------------------------------------------------

def test_a_dialog_nobody_touched_gets_the_card(glassed):
    """The point of the filter: this dialog is defined in a test file and
    knows nothing about spaCR's look."""
    dialog = _popup(glassed)
    try:
        assert _card(dialog) is not None
    finally:
        dialog.deleteLater()


def test_the_card_is_behind_the_contents_not_in_the_layout(glassed):
    """A backdrop that took part in a layout would push the dialog's own
    contents around, and the contents are the point."""
    dialog = _popup(glassed)
    try:
        card = _card(dialog)
        assert dialog.layout().indexOf(card) == -1
        assert card.parentWidget() is dialog
    finally:
        dialog.deleteLater()


def test_the_card_is_sized_to_the_dialog_and_follows_a_resize(glassed):
    dialog = _popup(glassed)
    try:
        card = _card(dialog)
        dialog.resize(700, 520)
        for _ in range(6):
            glassed.processEvents()
        assert card.width() < dialog.width()
        assert card.width() > dialog.width() - 60, card.geometry()
        assert card.height() > dialog.height() - 60, card.geometry()
    finally:
        dialog.deleteLater()


def test_the_card_takes_no_mouse_events(glassed):
    """It sits over nothing and under everything; a backdrop that swallowed
    a click would make the dialog beneath it unusable."""
    from PySide6.QtCore import Qt

    dialog = _popup(glassed)
    try:
        assert _card(dialog).testAttribute(Qt.WA_TransparentForMouseEvents)
    finally:
        dialog.deleteLater()


def test_showing_a_dialog_twice_does_not_stack_two_cards(glassed):
    """A dialog opened, closed and opened again is the ordinary case for
    Preferences, and two cards would mean two rims and two timers."""
    from spacr.qt.widgets.setup_card import SetupCard

    dialog = _popup(glassed)
    try:
        dialog.hide()
        dialog.show()
        for _ in range(6):
            glassed.processEvents()
        assert len(dialog.findChildren(SetupCard)) == 1
    finally:
        dialog.deleteLater()


def test_a_dialog_can_say_no(glassed):
    """Nothing sets this today; it is there for the next thing somebody
    embeds that must own its own painting."""
    from spacr.qt.widgets.glass import NO_GLASS

    dialog = QDialog()
    dialog.setProperty(NO_GLASS, True)
    QVBoxLayout(dialog).addWidget(QLabel("mine"))
    dialog.show()
    try:
        for _ in range(6):
            glassed.processEvents()
        assert _card(dialog) is None
    finally:
        dialog.deleteLater()


def test_the_containers_are_cleared_but_the_controls_are_not(glassed):
    """A control you can see through is a control you cannot read, so the
    combo keeps its surface while the page behind it loses one."""
    dialog = QDialog()
    column = QVBoxLayout(dialog)
    combo = QComboBox()
    combo.addItem("a value")
    column.addWidget(combo)
    dialog.resize(400, 200)
    dialog.show()
    try:
        for _ in range(8):
            glassed.processEvents()
        assert not combo.property("transparentBg")
    finally:
        dialog.deleteLater()


def test_the_rim_has_room_to_be_seen(glassed):
    """A dialog's contents run to its edges, so without this the card's
    border is under a tab bar or a button and the light travels behind
    them."""
    from spacr.qt.widgets.glass import RIM_ROOM

    plain = QDialog()
    QVBoxLayout(plain).addWidget(QLabel("x"))
    before = plain.layout().getContentsMargins()

    dialog = _popup(glassed)
    try:
        after = dialog.layout().getContentsMargins()
        assert after[0] >= before[0] + RIM_ROOM
    finally:
        dialog.deleteLater()
        plain.deleteLater()


def test_the_real_preferences_dialog_gets_it_too(glassed):
    """The one the request names first, driven rather than assumed."""
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    dialog.resize(700, 620)
    dialog.show()
    try:
        for _ in range(8):
            glassed.processEvents()
        assert _card(dialog) is not None
    finally:
        dialog.deleteLater()


def test_the_backdrop_honours_the_ambient_preference(glassed, monkeypatch):
    """Somebody who turned the animated background off on the module
    screens has not asked for it back in every popup. The card and the rim
    still apply -- the look is the same one, just still."""
    from spacr.qt import preferences
    from spacr.qt.widgets.ambient import AmbientWidget

    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: False)
    dialog = _popup(glassed)
    try:
        assert _card(dialog) is not None, "the card goes on either way"
        assert not dialog.findChildren(AmbientWidget), (
            "an animation the user switched off came back in a popup")
    finally:
        dialog.deleteLater()


def test_the_backdrop_is_installed_when_it_is_wanted(glassed, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.widgets.ambient import AmbientWidget

    monkeypatch.setattr(preferences, "get_ambient_enabled", lambda: True)
    dialog = _popup(glassed)
    try:
        assert dialog.findChildren(AmbientWidget), (
            "the card has nothing to be translucent over")
    finally:
        dialog.deleteLater()


# ---------------------------------------------------------------------------
# The three settings
# ---------------------------------------------------------------------------

def test_the_length_round_trips(own_config):
    assert own_config.set_rim_length(320) == 320
    assert own_config.get_rim_length() == 320


def test_the_length_is_clamped_on_read_as_well_as_on_write(own_config):
    """A rim longer than its own perimeter is a border rather than a
    highlight, and the stored value can come from a hand-edited file."""
    low, high = own_config.RIM_LENGTH_RANGE
    own_config._settings().setValue(own_config._KEY_RIM_LENGTH, 99999)
    assert own_config.get_rim_length() == high
    own_config._settings().setValue(own_config._KEY_RIM_LENGTH, -5)
    assert own_config.get_rim_length() == low


def test_nonsense_falls_back_to_the_default(own_config):
    own_config._settings().setValue(own_config._KEY_RIM_LENGTH, "wide")
    assert own_config.get_rim_length() == own_config.DEFAULT_RIM_LENGTH
    own_config._settings().setValue(own_config._KEY_RIM_LAG, "slow")
    assert own_config.get_rim_lag() == own_config.DEFAULT_RIM_LAG


def test_the_lag_round_trips_and_is_bounded(own_config):
    low, high = own_config.RIM_LAG_RANGE
    assert own_config.set_rim_lag(0.25) == pytest.approx(0.25)
    assert own_config.get_rim_lag() == pytest.approx(0.25)
    assert own_config.set_rim_lag(9.0) == pytest.approx(high)
    assert own_config.set_rim_lag(0.0) == pytest.approx(low)


def test_the_alignment_round_trips(own_config):
    assert own_config.set_rim_alignment("head") == "head"
    assert own_config.get_rim_alignment() == "head"


def test_an_unknown_alignment_stores_the_default(own_config):
    assert own_config.set_rim_alignment("sideways") == \
        own_config.DEFAULT_RIM_ALIGNMENT
    assert own_config.get_rim_alignment() == own_config.DEFAULT_RIM_ALIGNMENT


def test_the_card_reads_all_three(app, own_config):
    from spacr.qt.widgets.setup_card import SetupCard

    own_config.set_rim_length(200)
    own_config.set_rim_lag(0.09)
    own_config.set_rim_alignment("head")
    card = SetupCard()
    card.resize(600, 420)
    try:
        assert card._arc == 200
        assert card.ease() == pytest.approx(0.09)
        assert card.alignment() == "head"
    finally:
        card.deleteLater()


def test_a_card_told_a_value_uses_it_over_the_stored_one(app, own_config):
    """A caller that wants a particular look for a particular card can
    still say so."""
    from spacr.qt.widgets.setup_card import SetupCard

    own_config.set_rim_alignment("centre")
    card = SetupCard(arc=123, lag=0.5, align="head")
    try:
        assert card._arc == 123
        assert card.ease() == pytest.approx(0.5)
        assert card.alignment() == "head"
    finally:
        card.deleteLater()


def test_changing_the_length_reaches_a_card_already_on_screen(app,
                                                              own_config):
    """A preference the user cannot see take effect is a preference they
    will set twice."""
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard()
    card.resize(600, 420)
    card.show()
    try:
        own_config.set_rim_length(150)
        assert own_config._tell_the_cards_the_rim_changed() >= 1
        assert card._arc == 150
    finally:
        card.deleteLater()


# ---------------------------------------------------------------------------
# Centred on the pointer
# ---------------------------------------------------------------------------

def test_the_middle_of_the_run_lands_on_the_pointer(app, own_config):
    """Asked for 2026-08-22: "allign the middle of the rim to where the
    mouse is". With the run trailing from its head the bright part sits to
    one side of the thing it is pointing at."""
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard(align="centre")
    card.resize(600, 420)
    try:
        span = card.accent_span(QRectF(card.rect()))
        card._at = 0.25
        middle = card.accent_start(span) + span / 2.0
        assert middle == pytest.approx(0.25, abs=1e-9)
    finally:
        card.deleteLater()


def test_trailing_still_ends_on_the_pointer(app, own_config):
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard(align="head")
    card.resize(600, 420)
    try:
        span = card.accent_span(QRectF(card.rect()))
        card._at = 0.25
        assert card.accent_start(span) + span == pytest.approx(0.25)
    finally:
        card.deleteLater()


def test_the_default_is_centred(own_config):
    """It is what was asked for, so it is what a fresh profile gets."""
    assert own_config.DEFAULT_RIM_ALIGNMENT == "centre"


def test_the_default_chase_is_slower_than_it_was(own_config):
    """"it should allign slower than now" -- 0.34 was the value being
    described."""
    assert own_config.DEFAULT_RIM_LAG < 0.34


def test_the_filter_can_be_taken_back_off(app):
    """An installer with no uninstaller changes every dialog for the rest
    of the process -- right for a running spaCR, wrong for anything that
    wants to see a dialog as its author wrote it."""
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    install_glass_everywhere(app)
    assert uninstall_glass_everywhere(app) is True
    assert uninstall_glass_everywhere(app) is False, "twice is not an error"
    try:
        dialog = QDialog()
        QVBoxLayout(dialog).addWidget(QLabel("after"))
        dialog.show()
        for _ in range(6):
            app.processEvents()
        assert _card(dialog) is None, "the filter was still acting"
    finally:
        dialog.deleteLater()


def test_installing_twice_installs_one_filter(app):
    """Two filters is every event in the application handled twice."""
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    uninstall_glass_everywhere(app)
    assert install_glass_everywhere(app) is True
    assert install_glass_everywhere(app) is False
    uninstall_glass_everywhere(app)
