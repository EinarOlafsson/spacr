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

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets import glass


class _StandInApplication(QObject):
    """An application-shaped object that records what is put on and taken off.

    A stand-in rather than the live ``QApplication``: a filter accidentally
    left on the real one would glass every dialog built by every test that
    runs after this file.
    """

    def __init__(self, refuse_removal: bool = False):
        super().__init__()
        self.installed = []
        self.removed = []
        self._refuse_removal = refuse_removal

    def installEventFilter(self, event_filter):     # noqa: N802 - Qt naming
        self.installed.append(event_filter)

    def removeEventFilter(self, event_filter):      # noqa: N802 - Qt naming
        if self._refuse_removal:
            raise RuntimeError("that filter is already gone")
        self.removed.append(event_filter)


@pytest.fixture(autouse=True)
def _no_installer_left_behind():
    """Leave neither a filter nor a remembered owner for the next test."""
    yield
    glass.uninstall_glass_everywhere()
    glass._INSTALLED = None
    glass._INSTALLED_APP = None


def test_an_old_filter_that_will_not_come_off_does_not_block_the_new_one(
        caplog):
    """A torn-down application must not be able to veto the next startup.

    An embedded host, or a test harness, hands spaCR a second application
    after the first is already half gone. Removing the filter from the dead
    one can raise; if that escaped, ``install_glass_everywhere`` would fall
    into its outer handler, answer False and clear both globals, and the new
    application would run with no filter at all -- every settings dialog
    opened in it painting its own black background over the card, with only a
    debug line to say why.
    """
    caplog.set_level(logging.DEBUG, logger="spacr.qt.glass")

    stubborn = _StandInApplication(refuse_removal=True)
    assert glass.install_glass_everywhere(stubborn) is True
    old_filter = glass._INSTALLED
    assert stubborn.installed == [old_filter], \
        "the first application should have been given a filter of its own"

    successor = _StandInApplication()
    assert glass.install_glass_everywhere(successor) is True, \
        "a refusal from the old application must not fail the new install"

    new_filter = glass._INSTALLED
    assert new_filter is not old_filter, \
        "the new application needs a filter of its own, not the dead one"
    assert successor.installed == [new_filter]
    assert glass._INSTALLED_APP is successor
    assert "the old glass filter would not come off" in caplog.text, \
        "the refusal is swallowed, so the debug line is the only record"


def test_an_install_with_no_remembered_owner_still_yields_to_a_new_one():
    """Upgrading in place must not strand the filter of the older module.

    Ownership tracking was added after the filter itself. A process that
    reloads this module, or an installation performed by a copy of it that
    predates ``_INSTALLED_APP``, leaves a filter recorded with no owner
    beside it. There is then nobody to remove it from -- and the code has to
    say so by skipping the removal rather than by raising ``AttributeError``
    on ``None``, which the outer handler would turn into "no filter
    installed" and leave the new application bare.
    """
    orphan = _StandInApplication()
    assert glass.install_glass_everywhere(orphan) is True
    orphaned_filter = glass._INSTALLED
    # Exactly the state an owner-less installation leaves behind.
    glass._INSTALLED_APP = None

    successor = _StandInApplication()
    assert glass.install_glass_everywhere(successor) is True
    assert orphan.removed == [], \
        "with no owner recorded there is no application to remove it from"
    assert glass._INSTALLED is not orphaned_filter, \
        "the successor must be given a filter of its own"
    assert successor.installed == [glass._INSTALLED]
    assert glass._INSTALLED_APP is successor

    # The very same call, this time with an owner remembered, DOES take the
    # old filter off -- so the empty list above is the missing owner and not
    # a stand-in that never records anything.
    successors_filter = glass._INSTALLED
    third = _StandInApplication()
    assert glass.install_glass_everywhere(third) is True
    assert successor.removed == [successors_filter], \
        "a remembered owner is the one the old filter comes off"
    assert third.installed == [glass._INSTALLED]


def test_forgetting_the_filter_needs_no_application_to_take_it_off(
        monkeypatch):
    """Shutdown must complete after the application is already gone.

    ``uninstall_glass_everywhere`` is called on the way down, and by tests
    that must not leak a filter into the next one. By then
    ``QApplication.instance()`` can already be ``None`` and no owner may have
    been recorded. If the function insisted on an application it would raise
    on ``None.removeEventFilter``; the globals would still be cleared by the
    ``finally``, but the caller would see an exception out of teardown. It
    has to report the filter forgotten instead.
    """
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    stray = _StandInApplication()
    glass._INSTALLED = glass._GlassInstaller(stray)
    glass._INSTALLED_APP = None

    assert glass.uninstall_glass_everywhere() is True, \
        "a filter was registered, so the call reports that it removed one"
    assert stray.removed == [], \
        "no owner and no instance means nothing to take the filter off"
    assert glass._INSTALLED is None
    assert glass._INSTALLED_APP is None
    assert glass.uninstall_glass_everywhere() is False, \
        "a second teardown has nothing left to forget"

    # And with an owner remembered the same call reaches it, so the empty
    # list above is the absent application rather than a removal this
    # stand-in never notices.
    owner = _StandInApplication()
    assert glass.install_glass_everywhere(owner) is True
    owners_filter = glass._INSTALLED
    assert glass.uninstall_glass_everywhere() is True
    assert owner.removed == [owners_filter]


def test_a_whole_messy_lifecycle_still_ends_on_one_filter_and_one_owner(
        monkeypatch, caplog):
    """The bookkeeping has to survive every seam in a single session.

    The three cases above are each driven on their own, from a clean pair of
    globals. A long-lived process does not get that: an embedded host that
    reloads spaCR, swaps applications, and tears the last one down hits the
    orphaned owner, the refusing owner and the absent application one after
    another, each starting from the state the previous one left. If any of
    them leaked -- an old filter kept while the owner was replaced, or an
    owner kept after the filter was forgotten -- the next
    ``install_glass_everywhere`` would see a matching pair, answer "already
    installed", and hand the live application nothing. Every settings dialog
    opened from then on paints its own black rectangle over the card, and
    nothing raises to say so.
    """
    caplog.set_level(logging.DEBUG, logger="spacr.qt.glass")

    first = _StandInApplication()
    assert glass.install_glass_everywhere(first) is True
    # An upgrade in place, or a module reload: the filter outlives the memory
    # of who owns it.
    glass._INSTALLED_APP = None

    # No owner recorded, so there is nobody to take the old filter off and the
    # successor is installed anyway.
    second = _StandInApplication(refuse_removal=True)
    assert glass.install_glass_everywhere(second) is True, \
        "an orphaned filter must not veto the next application"
    assert first.removed == [], \
        "with the owner forgotten there is no application to remove it from"
    assert second.installed == [glass._INSTALLED]

    # Now the owner IS remembered, and refuses to give the filter back.
    third = _StandInApplication()
    assert glass.install_glass_everywhere(third) is True, \
        "a refusal from the previous owner must not fail the new install"
    assert "the old glass filter would not come off" in caplog.text, \
        "the refusal is swallowed, so the debug line is the only record"
    thirds_filter = glass._INSTALLED
    assert third.installed == [thirds_filter]

    # A cooperating owner does record the removal -- so both empty lists in
    # this test are the seam under test and not a stand-in that never notices.
    fourth = _StandInApplication()
    assert glass.install_glass_everywhere(fourth) is True
    assert third.removed == [thirds_filter], \
        "a remembered, willing owner is the one the old filter comes off"

    # Shutdown, with the application already gone and its ownership forgotten.
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))
    glass._INSTALLED_APP = None
    assert glass.uninstall_glass_everywhere() is True, \
        "a filter was registered, so the call reports that it forgot one"
    assert fourth.removed == [], \
        "no owner and no instance leaves no application to take it off"
    assert glass._INSTALLED is None and glass._INSTALLED_APP is None, \
        "the pair must end empty, or the next install is a false no-op"

    # And the very next install still works from that emptied state.
    fifth = _StandInApplication()
    assert glass.install_glass_everywhere(fifth) is True, \
        "an emptied pair must leave the next application installable"
    assert fifth.installed == [glass._INSTALLED]
    assert glass.uninstall_glass_everywhere() is True
    assert fifth.removed == fifth.installed, \
        "a remembered owner still gets the filter taken off at shutdown"
