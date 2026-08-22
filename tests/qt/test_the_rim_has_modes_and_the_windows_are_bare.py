"""The rim's three modes, the circuits, the bare rounded windows, and the
backdrop behind a settings panel.

    "id like some different modes to choose from, normal glow, rainbow and
     beat (it should pulsate)"

    "olso at every positive click like next clockwise circle and every
     negative lige close or preveous counter clockwise"

    "the preferences should also have the line (as well as all other
     settings windows they also dont need the x and minus at the top make
     the edges rounded on all.) and there should an option to controll the
     settings background theme"

WHAT IS ACTUALLY WORTH ASSERTING here is not that a preference round-trips
-- that is true of every preference in the file -- but that choosing a mode
CHANGES WHAT IS PAINTED, that a button's direction is read from what the
button IS rather than from a list of the buttons that existed the day this
was written, and that a window which drops its title bar is still on
screen afterwards.

That last one was a real defect and not a hypothetical: `setWindowFlags`
hides a widget that is visible, this runs from a filter that fires while a
dialog is being shown, and an `exec()` on a hidden modal window is an
application that has stopped responding for no visible reason.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt        # noqa: E402
from PySide6.QtGui import QColor                              # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox,       # noqa: E402
                               QDialog, QDialogButtonBox, QLabel,
                               QPushButton, QSlider, QVBoxLayout)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def own_config():
    """Every rim key this file writes, saved and put back.

    Saved and restored rather than reloaded: `preferences._settings()`
    builds a fresh QSettings per call, so there is no cache a reload would
    clear, and swapping the module object out from under everything that
    already imported it is how a neighbouring file ends up unsandboxed.
    """
    from spacr.qt import preferences

    keys = ("rim_length", "rim_lag", "rim_alignment", "rim_mode",
            "rim_period", "popup_backdrop")
    before = {k: getattr(preferences, f"get_{k}")() for k in keys}
    yield preferences
    for key, value in before.items():
        getattr(preferences, f"set_{key}")(value)


@pytest.fixture(autouse=True)
def nothing_is_left_ticking(app):
    """This file's own deleteLater calls, delivered before the next test."""
    yield
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture()
def glassed(app):
    """The filter installed as it is at startup, and taken off after."""
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    apply_preferences_to_app(app)
    install_glass_everywhere(app)
    yield app
    uninstall_glass_everywhere(app)


def _card(app, mode="glow"):
    from spacr.qt.widgets.setup_card import SetupCard

    card = SetupCard(mode=mode)
    card.resize(420, 300)
    return card


# ---------------------------------------------------------------------------
# The three modes
# ---------------------------------------------------------------------------
class TestTheModes:
    def test_the_offered_modes_are_the_three_that_were_asked_for(self):
        from spacr.qt.preferences import RIM_MODES

        assert set(RIM_MODES) == {"glow", "rainbow", "beat"}

    def test_glow_is_one_colour_all_the_way_along(self, app):
        """The original look, unchanged: a mode nobody chose should not
        have quietly become a different mode."""
        card = _card(app, "glow")
        accent = QColor("#4a9eff")
        hues = {card.ink_at(along / 12.0, accent).hue()
                for along in range(13)}
        assert len(hues) == 1

    def test_rainbow_walks_the_hue_along_the_light(self, app):
        card = _card(app, "rainbow")
        accent = QColor("#4a9eff")
        hues = [card.ink_at(along / 12.0, accent).hue()
                for along in range(13)]
        assert len(set(hues)) > 4, hues

    def test_beat_keeps_the_accent_and_pulses_instead(self, app):
        """A pulse is a change in STRENGTH, not in colour -- a beat that
        also changed hue would be rainbow with extra steps."""
        card = _card(app, "beat")
        accent = QColor("#4a9eff")
        hues = {card.ink_at(along / 12.0, accent).hue()
                for along in range(13)}
        assert len(hues) == 1

        seen = set()
        for _ in range(240):
            card._phase += card._timer.interval() / 1000.0
            seen.add(round(card.beat(), 2))
        assert max(seen) > min(seen) + 0.2, sorted(seen)[:5]

    def test_a_pulse_never_goes_out_and_never_overshoots(self, app):
        """Zero would be a rim that vanishes and reappears, which reads as
        a fault rather than as a beat."""
        card = _card(app, "beat")
        for _ in range(600):
            card._phase += 0.007
            assert 0.2 < card.beat() <= 1.0

    def test_only_the_moving_modes_ask_for_a_frame_every_frame(self, app):
        """The repaint-skip is worth keeping for the mode that does not
        move, and wrong for the two that do."""
        assert _card(app, "glow").animates() is False
        assert _card(app, "rainbow").animates() is True
        assert _card(app, "beat").animates() is True

    def test_the_cycle_length_is_what_sets_the_rhythm(self, own_config,
                                                      app):
        """Otherwise the control in Preferences is a slider that does
        nothing."""
        def swing(seconds):
            own_config.set_rim_period(seconds)
            card = _card(app, "beat")
            card._phase = 0.0
            seen = []
            for _ in range(30):
                card._phase += 0.01
                seen.append(card.beat())
            return max(seen) - min(seen)

        assert swing(0.6) > swing(6.0)

    def test_a_card_told_nothing_takes_the_stored_mode(self, own_config,
                                                       app):
        own_config.set_rim_mode("rainbow")
        assert _card(app, "").mode() == "rainbow"

    def test_an_unknown_mode_falls_back_rather_than_painting_nothing(
            self, own_config, app):
        own_config.set_rim_mode("disco")
        assert own_config.get_rim_mode() in ("glow", "rainbow", "beat")

    def test_changing_it_in_preferences_reaches_a_card_already_open(
            self, own_config, app):
        """A settings window open while the setting changes is the normal
        case, not the exotic one."""
        card = _card(app, "")
        own_config.set_rim_mode("beat")
        card.reread_the_preferences()
        assert card.mode() == "beat"


# ---------------------------------------------------------------------------
# A click sends the light round
# ---------------------------------------------------------------------------
class TestTheCircuits:
    def test_a_forward_button_sends_it_clockwise(self, app):
        card = _card(app)
        card.circuit(clockwise=True)
        assert card._laps > 0

    def test_a_backward_button_sends_it_the_other_way(self, app):
        card = _card(app)
        card.circuit(clockwise=False)
        assert card._laps < 0

    def test_a_lap_ends_exactly_where_it_started(self, app):
        """Floating error left over from thirty-odd frames, accumulated
        across ten slides, is the light drifting away from the pointer."""
        card = _card(app)
        card._aim_at_the_cursor = lambda: False
        card._at = card._towards = 0.25
        card.circuit(clockwise=True)
        for _ in range(400):
            card._tick()
            if not card.spinning:
                break
        assert card.spinning is False
        assert card.position == pytest.approx(0.25, abs=1e-6)

    def test_the_pointer_does_not_steer_it_mid_circuit(self, app):
        """A lap that could be dragged off course by a mouse move would
        not end where it started."""
        card = _card(app)
        card.circuit(clockwise=True)
        before = card._laps
        card.flow_towards(QPointF(5, 5))
        assert card._laps == before

    @pytest.mark.parametrize("text,expected", [
        ("Next ›", True), ("OK", True), ("Save", True), ("Apply", True),
        ("Run", True), ("Install", True),
        ("‹ Back", False), ("Cancel", False), ("Close", False),
        ("Previous", False), ("Reset", False),
    ])
    def test_a_buttons_direction_is_read_from_the_button(self, app, text,
                                                         expected):
        from spacr.qt.widgets.glass import button_direction

        assert button_direction(QPushButton(text)) is expected

    def test_a_button_that_is_neither_spins_nothing(self, app):
        """"Browse…" is not progress and not retreat, and a light that
        went round on it would mean nothing by the third click."""
        from spacr.qt.widgets.glass import button_direction

        assert button_direction(QPushButton("Browse…")) is None

    def test_the_role_outranks_the_words(self, app):
        """A translated dialog has no English in it at all, so the words
        are the fallback and the role is the answer."""
        from spacr.qt.widgets.glass import button_direction

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok = box.button(QDialogButtonBox.Ok)
        cancel = box.button(QDialogButtonBox.Cancel)
        ok.setText("Bestätigen")
        cancel.setText("Abbrechen")
        assert button_direction(ok) is True
        assert button_direction(cancel) is False

    def test_a_real_dialogs_buttons_are_all_wired(self, glassed):
        from spacr.qt.widgets.glass import glass, spin_on_every_button
        from spacr.qt.widgets.setup_card import SetupCard

        dialog = QDialog()
        column = QVBoxLayout(dialog)
        column.addWidget(QLabel("settings"))
        column.addWidget(QDialogButtonBox(QDialogButtonBox.Save
                                          | QDialogButtonBox.Cancel))
        dialog.show()
        for _ in range(6):
            glassed.processEvents()
        try:
            card = dialog.findChildren(SetupCard)[0]
            # Already wired by the filter; wiring again must not double up.
            assert spin_on_every_button(dialog, card) == 0
            save = dialog.findChild(QDialogButtonBox).button(
                QDialogButtonBox.Save)
            card._laps = 0.0
            save.click()
            assert card._laps > 0
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# The windows themselves
# ---------------------------------------------------------------------------
class TestTheBareWindow:
    def _popup(self, glassed):
        dialog = QDialog()
        column = QVBoxLayout(dialog)
        column.addWidget(QLabel("a settings popup"))
        dialog.resize(420, 300)
        dialog.show()
        for _ in range(8):
            glassed.processEvents()
        return dialog

    def test_there_is_no_title_bar(self, glassed):
        dialog = self._popup(glassed)
        try:
            assert bool(dialog.windowFlags() & Qt.FramelessWindowHint)
        finally:
            dialog.deleteLater()

    def test_the_corners_can_be_round_because_the_window_is_translucent(
            self, glassed):
        """Without this the square window fills the four corners with the
        theme's background and the card's rounded shape is invisible."""
        dialog = self._popup(glassed)
        try:
            assert dialog.testAttribute(Qt.WA_TranslucentBackground)
        finally:
            dialog.deleteLater()

    def test_a_dialog_that_loses_its_frame_is_still_on_screen(self,
                                                             glassed):
        """THE DEFECT THIS FILE EXISTS FOR. `setWindowFlags` hides a
        visible widget, and this runs while the dialog is being shown --
        so Preferences opened, went frameless, and disappeared."""
        dialog = self._popup(glassed)
        try:
            assert dialog.isHidden() is False
        finally:
            dialog.deleteLater()

    def test_a_dialog_glassed_before_it_is_shown_is_not_shown_early(self,
                                                                    app):
        """The restore must not turn into a window that opens itself."""
        from spacr.qt.widgets.glass import glass

        dialog = QDialog()
        QVBoxLayout(dialog).addWidget(QLabel("not yet"))
        try:
            glass(dialog)
            assert dialog.isHidden() is True
        finally:
            dialog.deleteLater()

    def test_it_can_still_be_moved(self, glassed):
        """The title bar was where a window was dragged from, so dropping
        it without a replacement would nail every settings panel to the
        spot it opened at."""
        from spacr.qt.widgets.glass import _DragByBackground

        dialog = self._popup(glassed)
        try:
            assert any(isinstance(child, _DragByBackground)
                       for child in dialog.children())
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# The backdrop behind a settings panel
# ---------------------------------------------------------------------------
class TestTheSettingsBackdrop:
    def test_off_is_offered(self, own_config):
        """A backdrop is a taste, and "none" has to be one of the tastes."""
        assert "off" in own_config.POPUP_BACKDROPS

    def test_every_offered_backdrop_survives_being_chosen(self, own_config,
                                                          glassed):
        """A name in the list that the installer cannot build is a setting
        that breaks every settings window in the program."""
        from spacr.qt.widgets.setup_card import SetupCard

        for name in own_config.POPUP_BACKDROPS:
            own_config.set_popup_backdrop(name)
            dialog = QDialog()
            QVBoxLayout(dialog).addWidget(QLabel(name))
            dialog.show()
            for _ in range(4):
                glassed.processEvents()
            try:
                assert dialog.findChildren(SetupCard), name
            finally:
                dialog.deleteLater()
                glassed.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_an_unknown_name_falls_back(self, own_config):
        own_config.set_popup_backdrop("lava lamp")
        assert own_config.get_popup_backdrop() in own_config.POPUP_BACKDROPS


# ---------------------------------------------------------------------------
# The controls in Preferences
# ---------------------------------------------------------------------------
class TestTheControls:
    @pytest.fixture()
    def dialog(self, app, own_config):
        from spacr.qt.preferences import PreferencesDialog

        panel = PreferencesDialog()
        for _ in range(4):
            app.processEvents()
        yield panel
        panel.deleteLater()

    @pytest.mark.parametrize("kind,name", [
        (QSlider, "RimLength"), (QSlider, "RimLag"), (QSlider, "RimPeriod"),
        (QComboBox, "RimAlignment"), (QComboBox, "RimMode"),
        (QComboBox, "PopupBackdrop"),
    ])
    def test_the_control_is_there(self, dialog, kind, name):
        assert dialog.findChild(kind, name) is not None

    def test_the_mode_box_offers_exactly_the_three_modes(self, dialog,
                                                         own_config):
        box = dialog.findChild(QComboBox, "RimMode")
        offered = [box.itemData(i) for i in range(box.count())]
        assert offered == list(own_config.RIM_MODES)

    def test_the_backdrop_box_offers_every_backdrop(self, dialog,
                                                    own_config):
        box = dialog.findChild(QComboBox, "PopupBackdrop")
        offered = [box.itemData(i) for i in range(box.count())]
        assert offered == list(own_config.POPUP_BACKDROPS)

    def test_each_box_opens_on_what_is_stored(self, app, own_config):
        """A panel that opens on the first item tells a user their choice
        was forgotten, and then saves that lie back over it."""
        from spacr.qt.preferences import PreferencesDialog

        own_config.set_rim_mode("beat")
        own_config.set_popup_backdrop("off")
        panel = PreferencesDialog()
        try:
            assert panel.findChild(QComboBox, "RimMode").currentData() \
                == "beat"
            assert panel.findChild(QComboBox, "PopupBackdrop").currentData() \
                == "off"
        finally:
            panel.deleteLater()

    def test_saving_writes_all_three_new_settings(self, app, own_config):
        from spacr.qt.preferences import PreferencesDialog

        own_config.set_rim_mode("glow")
        own_config.set_rim_period(2.4)
        own_config.set_popup_backdrop("aurora")
        panel = PreferencesDialog()
        try:
            mode = panel.findChild(QComboBox, "RimMode")
            mode.setCurrentIndex(mode.findData("rainbow"))
            panel.findChild(QComboBox, "PopupBackdrop").setCurrentIndex(
                panel.findChild(QComboBox, "PopupBackdrop").findData("off"))
            panel.findChild(QSlider, "RimPeriod").setValue(51)
            # Save is a closure on `accepted`; the panel is not a QDialog
            # subclass, so there is no `_save` to call.
            panel.findChild(QDialogButtonBox).accepted.emit()
        finally:
            panel.deleteLater()
        assert own_config.get_rim_mode() == "rainbow"
        assert own_config.get_popup_backdrop() == "off"
        assert own_config.get_rim_period() == pytest.approx(5.1)

    def test_the_cycle_slider_covers_the_range_the_setting_allows(self,
                                                                  dialog,
                                                                  own_config):
        """A slider narrower than the setting is a setting a user cannot
        reach; a wider one saves a value the getter then clamps away."""
        slider = dialog.findChild(QSlider, "RimPeriod")
        low, high = own_config.RIM_PERIOD_RANGE
        assert slider.minimum() / 10.0 == pytest.approx(low)
        assert slider.maximum() / 10.0 == pytest.approx(high)
