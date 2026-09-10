"""Five things reported on 2026-08-22, in the words they were reported in.

    "in the setup window i cant click the github sign in"
    "there is a box with square edges behind the box with rounded edges.
     you should remove the box in the background."
    "the settings windowns sould also have rounded edges."
    "the default for the setings windows should be no theme (but the user
     should be able to cahnge that)"
    "and the settings boxes should all also have the perimiter line that
     follows the mouse!"

THE SQUARE BOX WAS THREE THINGS AT ONCE:

* the card sat eight pixels inside the dialog, so the dialog's own
  background framed it -- a square band all the way round the rounded card,
  in the one place a square corner is most visible;
* `WA_TranslucentBackground` was set AFTER `setWindowFlags`, and
  `setWindowFlags` recreates the native window, so the translucency applied
  to a window that no longer existed;
* and the application stylesheet's `QDialog` rule paints a background that
  `WA_TranslucentBackground` does not stop.

THE GITHUB BUTTON WAS TWO:

* the setup screen builds its own card and lay its slides out inside it,
  and the glass filter gave it a SECOND one on top -- `childAt` over the
  button returned the card;
* and the button was DISABLED in two of its three states, including the
  common one where `gh` already reports you signed in. A disabled control
  is exactly what "can't click" looks like from the outside.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt                        # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog,        # noqa: E402
                               QDialogButtonBox, QLabel, QVBoxLayout)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def nothing_is_left_ticking(app):
    yield
    app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


@pytest.fixture()
def glassed(app):
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    apply_preferences_to_app(app)
    install_glass_everywhere(app)
    yield app
    uninstall_glass_everywhere(app)


def _popup(glassed, width=560, height=420):
    dialog = QDialog()
    column = QVBoxLayout(dialog)
    column.addWidget(QLabel("a settings popup"))
    column.addWidget(QDialogButtonBox(QDialogButtonBox.Save
                                      | QDialogButtonBox.Cancel))
    dialog.resize(width, height)
    dialog.show()
    for _ in range(10):
        glassed.processEvents()
    return dialog


def _card(dialog):
    from spacr.qt.widgets.setup_card import SetupCard

    cards = dialog.findChildren(SetupCard)
    return cards[0] if cards else None


# ---------------------------------------------------------------------------
# The square box behind the rounded one
# ---------------------------------------------------------------------------
class TestThereIsNoBoxBehindTheCard:
    def test_the_card_is_the_window(self, glassed):
        """THE BAND. Eight pixels of the dialog's own background all the
        way round the rounded card is the square box that was reported."""
        from spacr.qt.widgets.glass import INSET

        assert INSET == 0
        dialog = _popup(glassed)
        try:
            card = _card(dialog)
            assert card.geometry() == dialog.rect()
        finally:
            dialog.deleteLater()

    def test_the_dialog_paints_nothing_of_its_own(self, glassed):
        """`WA_TranslucentBackground` stops Qt filling the window with the
        palette's base. It does NOT stop the application stylesheet's
        `QDialog { background: ... }`, which is the other half of the same
        square."""
        dialog = _popup(glassed)
        try:
            assert dialog.testAttribute(Qt.WA_TranslucentBackground)
            assert "transparent" in (dialog.styleSheet() or "")
        finally:
            dialog.deleteLater()

    def test_the_translucency_is_asked_for_before_the_flags(self):
        """`setWindowFlags` RECREATES the native window, so a translucency
        asked for afterwards applies to a window that no longer exists --
        which on X11 means it does not apply at all."""
        import ast
        import inspect
        import textwrap

        from spacr.qt.widgets import glass

        # PARSED, not searched: the docstring explains the ORDER, so it
        # names both in the wrong one. The same mistake this test exists to
        # catch, made by the test.
        tree = ast.parse(textwrap.dedent(
            inspect.getsource(glass.make_frameless)))
        attribute = flags = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            if not isinstance(target, ast.Attribute):
                continue
            if target.attr == "setAttribute" and attribute is None:
                attribute = node.lineno
            if target.attr == "setWindowFlags" and flags is None:
                flags = node.lineno
        assert attribute is not None and flags is not None
        assert attribute < flags, (
            f"setAttribute is on line {attribute} and setWindowFlags on "
            f"{flags}; the flags recreate the window and would drop it")

    def test_a_dialog_with_its_own_stylesheet_keeps_it(self, glassed):
        """Appended rather than replaced, or a dialog that styled itself
        loses that styling the moment it opens."""
        from spacr.qt.widgets.glass import make_frameless

        dialog = QDialog()
        QVBoxLayout(dialog).addWidget(QLabel("mine"))
        dialog.setStyleSheet("QLabel { color: #ff0000; }")
        try:
            make_frameless(dialog)
            sheet = dialog.styleSheet()
            assert "#ff0000" in sheet
            assert "transparent" in sheet
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# Rounded, and lit
# ---------------------------------------------------------------------------
class TestEverySettingsWindow:
    def test_it_is_frameless_so_the_corners_can_be_round(self, glassed):
        dialog = _popup(glassed)
        try:
            assert bool(dialog.windowFlags() & Qt.FramelessWindowHint)
        finally:
            dialog.deleteLater()

    def test_it_carries_the_rim(self, glassed):
        dialog = _popup(glassed)
        try:
            assert _card(dialog) is not None
        finally:
            dialog.deleteLater()

    def test_the_rim_has_room_to_be_seen(self, glassed):
        """A dialog's contents run to its edges, so without this the light
        travels behind a tab bar and a button."""
        from spacr.qt.widgets.glass import RIM_ROOM

        dialog = _popup(glassed)
        try:
            margins = dialog.layout().contentsMargins()
            assert margins.left() >= RIM_ROOM
            assert margins.right() >= RIM_ROOM
        finally:
            dialog.deleteLater()

    def test_the_rim_follows_a_pointer(self, glassed):
        """The whole point of it: "the perimiter line that follows the
        mouse"."""
        from PySide6.QtCore import QPointF

        dialog = _popup(glassed)
        try:
            card = _card(dialog)
            card._aim_at_the_cursor = lambda: False
            card.flow_towards(QPointF(5, 5))
            for _ in range(200):
                card._tick()
            assert card.corner() == "topLeft"
            card.flow_towards(QPointF(dialog.width() - 5, dialog.height() - 5))
            for _ in range(200):
                card._tick()
            assert card.corner() == "bottomRight"
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# No theme unless asked for
# ---------------------------------------------------------------------------
class TestTheBackdropIsOffByDefault:
    def test_the_shipped_default_is_no_theme(self):
        from spacr.qt.preferences import DEFAULT_POPUP_BACKDROP

        assert DEFAULT_POPUP_BACKDROP == "off"

    def test_off_is_still_one_of_the_choices(self):
        from spacr.qt.preferences import POPUP_BACKDROPS

        assert "off" in POPUP_BACKDROPS
        assert len(POPUP_BACKDROPS) > 1, "there is nothing to change it to"

    def test_the_card_and_the_rim_stay_when_the_theme_is_off(self, glassed):
        """'off' drops the MOVEMENT, not the look. A settings window with
        no card would have no rim either."""
        from spacr.qt import preferences

        before = preferences.get_popup_backdrop()
        preferences.set_popup_backdrop("off")
        try:
            dialog = _popup(glassed)
            try:
                assert _card(dialog) is not None
            finally:
                dialog.deleteLater()
        finally:
            preferences.set_popup_backdrop(before)


# ---------------------------------------------------------------------------
# The setup screen keeps its own card, and its button does something
# ---------------------------------------------------------------------------
class TestTheSetupScreen:
    def test_it_is_not_given_a_second_card(self, glassed):
        """It builds one and lays its slides out INSIDE it. A second card
        on top of that is a sheet of glass over the controls -- `childAt`
        over the GitHub button returned the card, so the click never
        reached the button."""
        from spacr.qt.widgets.setup_card import SetupCard
        from spacr.qt.widgets.setup_slides import SetupSlides

        slides = SetupSlides()
        slides.resize(900, 700)
        slides.show()
        for _ in range(20):
            glassed.processEvents()
        try:
            assert len(slides.findChildren(SetupCard)) == 1
            assert not slides.property("spacrGlassed")
        finally:
            slides.close()
            slides.deleteLater()

    def test_it_has_no_title_bar_either(self, glassed):
        """LEAVING IT ALONE LEFT IT WITH ITS FRAME. Excluding it from the
        glass filter -- which is right, it builds its own card -- also took
        away the frameless treatment, so it kept a square window with a
        close and a minimise button around its rounded card. It goes
        frameless itself now."""
        from PySide6.QtCore import Qt
        from spacr.qt.widgets.setup_slides import SetupSlides

        slides = SetupSlides()
        slides.show()
        for _ in range(10):
            glassed.processEvents()
        try:
            assert bool(slides.windowFlags() & Qt.FramelessWindowHint)
            assert slides.testAttribute(Qt.WA_TranslucentBackground)
            assert "transparent" in (slides.styleSheet() or "")
        finally:
            slides.close()
            slides.deleteLater()

    def test_it_can_still_be_moved_without_one(self, glassed):
        """The title bar was the only way to drag it."""
        from spacr.qt.widgets.glass import _DragByBackground
        from spacr.qt.widgets.setup_slides import SetupSlides

        slides = SetupSlides()
        slides.show()
        for _ in range(10):
            glassed.processEvents()
        try:
            assert any(isinstance(child, _DragByBackground)
                       for child in slides.children())
        finally:
            slides.close()
            slides.deleteLater()

    def test_a_dialog_that_brought_its_own_card_is_left_alone(self, glassed):
        """Checked by LOOKING rather than by asking, so anything else that
        builds its own card is covered without having to remember to say
        so."""
        from spacr.qt.widgets.glass import wants_glass
        from spacr.qt.widgets.setup_card import SetupCard

        dialog = QDialog()
        QVBoxLayout(dialog).addWidget(QLabel("mine"))
        card = SetupCard(dialog)
        try:
            assert card is not None
            assert wants_glass(dialog) is False
        finally:
            dialog.deleteLater()

    def test_an_ordinary_dialog_still_wants_one(self, glassed):
        from spacr.qt.widgets.glass import wants_glass

        dialog = QDialog()
        QVBoxLayout(dialog).addWidget(QLabel("plain"))
        try:
            assert wants_glass(dialog) is True
        finally:
            dialog.deleteLater()


class TestTheGithubButtonAlwaysDoesSomething:
    # The "button" is the GitHub LOGO since 2026-08-23 -- "i want the
    # github button to also be a github logo just like the AI icons work".
    # The text push button beside it is gone, so what each state offers is
    # read off the mark's status and tooltip instead of off a caption.
    def _slides(self, app):
        from spacr.qt.widgets.setup_slides import SetupSlides

        made = SetupSlides()
        made.show()
        for _ in range(20):
            app.processEvents()
        return made

    def test_it_is_never_disabled(self, app, monkeypatch):
        """It was greyed in two of its three states, one of them the common
        one -- `gh` already signed in. A disabled control is what "i cant
        click the github sign in" looks like from the outside."""
        from spacr.qt.widgets import setup_slides

        slides = self._slides(app)
        try:
            for source in ("gh", "env", "token", None):
                monkeypatch.setattr(
                    "spacr.qt.ai.github_auth.auth_source", lambda s=source: s)
                slides._refresh_github()
                assert slides._gh_mark.isEnabled(), source
                assert slides._gh_mark.toolTip().strip(), source
                assert slides._gh_action in ("login", "install"), source
        finally:
            slides.close()
            slides.deleteLater()

    def test_already_signed_in_offers_to_sign_in_again(self, app,
                                                       monkeypatch):
        """A second account, or a token that expired while `auth_source`
        still finds a stale one. `gh auth login` handles both."""
        monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source",
                            lambda: "gh")
        slides = self._slides(app)
        try:
            slides._refresh_github()
            assert "again" in slides._gh_mark.toolTip().lower()
            assert slides._gh_mark.status == slides._gh_mark.READY
            assert slides._gh_action == "login"
        finally:
            slides.close()
            slides.deleteLater()

    def test_a_missing_cli_offers_to_install_it(self, app, monkeypatch):
        """"The CLI is not installed" beside a dead button tells the user
        what is wrong and gives them nothing to do about it."""
        import shutil

        monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source",
                            lambda: None)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        slides = self._slides(app)
        try:
            slides._refresh_github()
            assert "install" in slides._gh_mark.toolTip().lower()
            assert slides._gh_mark.status == slides._gh_mark.NOT_INSTALLED
            assert slides._gh_action == "install"
        finally:
            slides.close()
            slides.deleteLater()

    def test_the_install_button_opens_the_install_page(self, app,
                                                       monkeypatch):
        import shutil

        from PySide6.QtGui import QDesktopServices

        monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source",
                            lambda: None)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        opened = []
        monkeypatch.setattr(QDesktopServices, "openUrl",
                            staticmethod(lambda url: opened.append(url.toString())
                                         or True))
        slides = self._slides(app)
        try:
            slides._refresh_github()
            assert slides._sign_in_to_github() is True
            assert opened and "cli.github.com" in opened[0]
        finally:
            slides.close()
            slides.deleteLater()

    def test_a_logged_out_cli_offers_to_sign_in(self, app, monkeypatch):
        import shutil

        monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source",
                            lambda: None)
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/gh")
        slides = self._slides(app)
        try:
            slides._refresh_github()
            assert slides._gh_mark.status == slides._gh_mark.SIGNED_OUT
            assert "sign-in" in slides._gh_mark.toolTip().lower()
            assert slides._gh_action == "login"
        finally:
            slides.close()
            slides.deleteLater()

    def test_every_state_says_what_it_found(self, app, monkeypatch):
        """The status line is what makes the button's word make sense."""
        import shutil

        slides = self._slides(app)
        try:
            for source, which in (("gh", "/usr/bin/gh"), (None, None),
                                  (None, "/usr/bin/gh")):
                monkeypatch.setattr("spacr.qt.ai.github_auth.auth_source",
                                    lambda s=source: s)
                monkeypatch.setattr(shutil, "which", lambda name, w=which: w)
                slides._refresh_github()
                assert slides._gh_status.text().strip()
                assert slides._gh_mark.toolTip().strip()
        finally:
            slides.close()
            slides.deleteLater()
