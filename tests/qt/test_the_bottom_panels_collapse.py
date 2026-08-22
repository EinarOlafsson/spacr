"""Console, AI chat and System all fold, on every module that has them.

Instruction 228: "console, AI chat and system should all be colapsable to
make space for the container above if necessary, this should be tru of all
moduals that have the three", and then: "the colapsing of the console and AI
box should be controlled bt clicking on the Console text and colapsing of
System by clicking on System".

THE SPACE IS THE POINT, NOT THE FOLDING. A panel that collapses and leaves a
gap has cost the user a click and given them nothing.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, Qt   # noqa: E402
from PySide6.QtGui import QMouseEvent           # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QWidget  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _click(widget):
    """A real left click on ``widget``, through the event system.

    Not `folder.toggle()`: the thing worth testing is that the HEADING is
    wired to the fold, and calling the method directly proves nothing about
    the wiring the user actually touches.
    """
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
        event = QMouseEvent(kind, QPoint(4, 4), Qt.LeftButton,
                            Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(widget, event)


class TestClickingTheNameFolds:

    def test_a_click_folds_and_a_click_restores(self, app):
        from spacr.qt.widgets.foldable import make_foldable

        heading, body = QLabel("Console"), QWidget()
        folder = make_foldable(heading, body)
        _click(heading)
        assert folder.shut and body.isHidden()
        _click(heading)
        assert not folder.shut and not body.isHidden()

    def test_the_name_looks_clickable_before_it_is_clicked(self, app):
        from spacr.qt.widgets.foldable import make_foldable

        heading = QLabel("Console")
        make_foldable(heading, QWidget())
        assert heading.cursor().shape() == Qt.PointingHandCursor
        assert "fold" in heading.toolTip().lower()

    def test_a_folded_panel_leaves_a_strip_that_names_it(self, app):
        from spacr.qt.widgets.foldable import make_foldable

        from PySide6.QtWidgets import QVBoxLayout

        holder = QWidget()
        column = QVBoxLayout(holder)
        heading, body = QLabel("Console"), QWidget()
        column.addWidget(heading)
        column.addWidget(body)
        folder = make_foldable(heading, body)
        folder.toggle()
        assert "Console" in heading.text(), (
            "a folded panel the user cannot name is one they cannot restore")
        # The BODY goes and the heading stays -- compared against each other
        # rather than against isHidden() alone, which is True for anything
        # in a window nobody has shown.
        assert body.isHidden() and not heading.isHidden()

    def test_the_strip_says_which_way_it_will_go(self, app):
        from spacr.qt.widgets.foldable import (OPEN_MARK, SHUT_MARK,
                                               make_foldable)

        heading = QLabel("System")
        folder = make_foldable(heading, QWidget())
        assert OPEN_MARK in heading.text()
        folder.toggle()
        assert SHUT_MARK in heading.text()


class TestAFoldedConsoleStillSpeaks:
    """Silence from a folded panel is indistinguishable from silence from an
    empty one, and the first is the one that matters."""

    def test_an_alert_shows_on_the_strip(self, app):
        from spacr.qt.widgets.foldable import make_foldable

        heading = QLabel("Console")
        folder = make_foldable(heading, QWidget())
        folder.toggle()
        folder.alert("3 errors")
        assert "3 errors" in heading.text()

    def test_an_open_panel_needs_no_alert(self, app):
        from spacr.qt.widgets.foldable import make_foldable

        heading = QLabel("Console")
        folder = make_foldable(heading, QWidget())
        folder.alert("3 errors")
        assert "3 errors" not in heading.text()

    def test_unfolding_clears_it(self, app):
        """Arriving is seeing: an alert that survived would keep claiming
        there is something to look at after the user has looked."""
        from spacr.qt.widgets.foldable import make_foldable

        heading = QLabel("Console")
        folder = make_foldable(heading, QWidget())
        folder.toggle()
        folder.alert("3 errors")
        folder.toggle()
        assert "3 errors" not in heading.text()


class TestTheSpaceIsReleased:

    def test_a_folded_body_contributes_no_height(self, app):
        """Which is what lets the stretch above take the room without
        anybody computing it."""
        from spacr.qt.widgets.foldable import make_foldable
        from PySide6.QtWidgets import QVBoxLayout

        holder = QWidget()
        column = QVBoxLayout(holder)
        heading, body = QLabel("Console"), QWidget()
        body.setMinimumHeight(180)
        column.addWidget(heading)
        column.addWidget(body)
        folder = make_foldable(heading, body)

        tall = holder.sizeHint().height()
        folder.toggle()
        short = holder.sizeHint().height()
        assert short < tall - 100, (
            f"folding released {tall - short}px of the body's 180 -- a panel "
            f"that collapses and leaves a gap has given the user nothing")


class TestTheCard:

    def test_a_foldable_card_folds_by_its_title(self, app):
        from spacr.qt.widgets.card import Card

        card = Card(title="System", foldable=True)
        assert card.folder is not None
        _click(card.title_label)
        assert card.body.isHidden()

    def test_an_ordinary_card_is_untouched(self, app):
        """Most cards are the only thing in their slot, and folding one would
        leave a strip over empty space."""
        from spacr.qt.widgets.card import Card

        card = Card(title="Plain")
        assert card.folder is None
        assert card.title_label.text() == "Plain"

    def test_a_card_with_no_title_cannot_fold(self, app):
        from spacr.qt.widgets.card import Card

        card = Card(foldable=True)
        assert card.folder is None, (
            "there would be nothing to click, and nothing to restore it by")


class TestEveryModuleThatHasThem:
    """"all moduals that have the three" is the part that decays."""

    @pytest.mark.parametrize("app_key", ["mask", "measure", "regression"])
    def test_the_console_folds_on_each(self, app, app_key):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen(app_key)
        folder = getattr(screen, "_console_folder", None)
        assert folder is not None, (
            f"{app_key} has a console with no fold; two panels that fold and "
            f"one that does not is worse than none folding")
        _click(screen._console_header)
        assert screen._console.isHidden()
        # AND THE AI BOX WITH IT. The maintainer asked for "the colapsing of
        # the console and AI box ... by clicking on the Console text" -- one
        # name, both boxes, because the chat row is the console panel's own
        # second half and has no heading of its own to click.
        #
        # ASSERTED BY CONTAINMENT, not by isHidden() on the row: hiding a
        # parent leaves a child's own hidden FLAG false, so the row would
        # report itself visible while being nowhere on screen.
        assert screen._console.isAncestorOf(screen._console._chat_row)
        assert not screen._console._chat_row.isVisibleTo(screen)
        _click(screen._console_header)
        assert not screen._console.isHidden()

    @pytest.mark.parametrize("app_key", ["mask", "regression"])
    def test_the_system_card_folds_on_each(self, app, app_key):
        from spacr.qt.screens.app_screen import AppScreen

        screen = AppScreen(app_key)
        card = getattr(screen, "_usage_card", None)
        assert card is not None and card.folder is not None
        _click(card.title_label)
        assert card.body.isHidden()


class TestTheStateSurvivesARestart:
    """The last of the instruction's criteria. A restart is simulated by
    building a second Folder against the same key -- which is exactly what a
    relaunch does."""

    @pytest.fixture(autouse=True)
    def _own_config(self, tmp_path, monkeypatch):
        """A config dir of this test's own.

        WITHOUT THIS THE TEST FOLDS A PANEL ON THE USER'S NEXT LAUNCH. The
        preferences are real QSettings on a real path, and a test that writes
        to them is a test that edits the machine it runs on.
        """
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        import importlib

        from spacr.qt import preferences

        importlib.reload(preferences)
        yield
        importlib.reload(preferences)

    def test_a_folded_panel_comes_back_folded(self, app):
        from spacr.qt.widgets.foldable import make_foldable

        first = make_foldable(QLabel("Console"), QWidget(),
                              persist_key="mask/Console")
        first.toggle()
        assert first.shut

        second = make_foldable(QLabel("Console"), QWidget(),
                               persist_key="mask/Console")
        assert second.shut, "the fold did not survive the restart"

    def test_an_unfolded_one_comes_back_open(self, app):
        from spacr.qt.widgets.foldable import make_foldable

        first = make_foldable(QLabel("Console"), QWidget(),
                              persist_key="measure/Console")
        first.toggle()
        first.toggle()
        second = make_foldable(QLabel("Console"), QWidget(),
                               persist_key="measure/Console")
        assert not second.shut

    def test_it_is_per_module(self, app):
        """Folding the console on Mask must not fold it on Sequencing: the
        modules are used for different work and want different room."""
        from spacr.qt.widgets.foldable import make_foldable

        make_foldable(QLabel("Console"), QWidget(),
                      persist_key="mask/Console").toggle()
        other = make_foldable(QLabel("Console"), QWidget(),
                              persist_key="sequencing/Console")
        assert not other.shut

    def test_no_key_means_no_writing(self, app):
        """A bare panel in a test must not touch the preferences."""
        from spacr.qt.preferences import get_folded_panels
        from spacr.qt.widgets.foldable import make_foldable

        make_foldable(QLabel("Console"), QWidget()).toggle()
        assert get_folded_panels() == {}

    def test_an_open_panel_is_not_stored(self, app):
        """The default is open, so storing it would grow the dict by one
        entry for every panel ever touched and never shrink it."""
        from spacr.qt.preferences import get_folded_panels
        from spacr.qt.widgets.foldable import make_foldable

        folder = make_foldable(QLabel("System"), QWidget(),
                               persist_key="mask/System")
        folder.toggle()
        folder.toggle()
        assert "mask/System" not in get_folded_panels()
