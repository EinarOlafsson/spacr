"""`retire_pyqtgraph_menus`: repairing an ownership hole before teardown.

pyqtgraph's PlotItem and ViewBox build context menus that Qt does not
own -- and a ViewBoxMenu embeds spin boxes and combo popups that Qt
promotes to top-level windows. Left alone they outlive the plot, which
is how a closed screen keeps a handful of invisible windows alive for
the rest of the session.

The helper reparents each parentless menu onto the closing widget so
there is a synchronous destruction boundary, then queues the whole tree
for deletion. Every uncovered line was one of the two reparenting steps
or its RuntimeError guard -- and a binding that refuses an ownership
transfer during its own close notification is exactly the case those
guards are for.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QVBoxLayout, QWidget

from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

pytestmark = pytest.mark.qt


@pytest.fixture()
def plot_owner(qtbot):
    """A widget holding a real pyqtgraph plot, as a screen does."""
    pg = pytest.importorskip("pyqtgraph")
    owner = QWidget()
    qtbot.addWidget(owner)
    layout = QVBoxLayout(owner)
    plot = pg.PlotWidget()
    layout.addWidget(plot)
    plot.plot([1, 2, 3], [1, 4, 9])
    return owner


def test_a_plot_owners_menus_are_retired(plot_owner):
    """PlotItem and ViewBox each contribute a menu root."""
    assert retire_pyqtgraph_menus(plot_owner) == 2


def test_retiring_twice_finds_nothing_the_second_time(plot_owner):
    """`id(menu) in menus` -- and the wrappers are kept alive to make
    that identity meaningful, because Shiboken can otherwise reuse an id."""
    assert retire_pyqtgraph_menus(plot_owner) == 2
    assert retire_pyqtgraph_menus(plot_owner) == 0


def test_an_owner_with_no_plots_does_no_work(qtbot):
    owner = QWidget()
    qtbot.addWidget(owner)
    assert retire_pyqtgraph_menus(owner) == 0


def test_an_owner_whose_c_half_has_gone_is_survived():
    """This runs from close handlers, where the owner may already be gone."""
    class _Dead:
        def findChildren(self, *_a, **_k):
            raise RuntimeError("Internal C++ object already deleted.")

    assert retire_pyqtgraph_menus(_Dead()) == 0


def test_an_owner_that_is_not_a_widget_at_all_is_survived():
    class _NotAWidget:
        pass

    assert retire_pyqtgraph_menus(_NotAWidget()) == 0


def test_without_pyqtgraph_imported_nothing_is_attempted(plot_owner,
                                                         monkeypatch):
    """Cleanup must not be the event that loads an optional plotting stack.

    A real pyqtgraph view can only exist after the package is imported,
    so an absent module means there is nothing of its kind to retire --
    and importing it here would make teardown pay for a dependency the
    session had chosen not to use.
    """
    import sys

    monkeypatch.setitem(sys.modules, "pyqtgraph", None)
    assert retire_pyqtgraph_menus(plot_owner) == 0


class TestTheOwnershipTransfersThatMayBeRefused:
    """Both `setParent` calls are guarded, and for the same reason.

    A binding may reject an ownership transfer while it is delivering its
    own close notification. Deletion is still queued afterwards either
    way, so refusing the transfer costs the synchronous boundary and
    nothing else -- but raising would abandon the rest of the tree.
    """

    def test_a_menu_that_refuses_reparenting_is_still_retired(
            self, plot_owner, monkeypatch):
        from PySide6.QtWidgets import QMenu

        refused = []
        real = QMenu.setParent

        def refuse(self, parent):
            refused.append(parent)
            raise RuntimeError("cannot transfer ownership during close")

        monkeypatch.setattr(QMenu, "setParent", refuse)
        assert retire_pyqtgraph_menus(plot_owner) == 2, (
            "a refused transfer stopped the retirement")
        assert refused, "the transfer was never attempted"

    def test_a_control_widget_that_refuses_reparenting_is_survived(
            self, plot_owner, monkeypatch):
        """The second transfer: pyqtgraph's generated control holders.

        Those are ordinary Python objects, not QObject children, so
        closing the menu reparents a few of their editors to None. They
        are put back under the menu while it still owns them -- and if
        that is refused, the loop carries on.
        """
        from PySide6.QtWidgets import QWidget as _QWidget

        real = _QWidget.setParent

        def refuse(self, parent):
            raise RuntimeError("cannot transfer ownership during close")

        monkeypatch.setattr(_QWidget, "setParent", refuse)
        assert retire_pyqtgraph_menus(plot_owner) == 2


class TestTheControlHoldersAfterTheMenuHasClosed:
    """pyqtgraph's `ctrl` holders are plain Python objects, not children.

    Closing a ViewBoxMenu reparents a few of their editors and popup
    views to None, and those are the widgets Qt promotes to top-level
    windows. They have to be put back under the menu while it still owns
    the holders, or deleting the menu leaves them behind as invisible
    windows for the rest of the session.

    Reproducing "already reparented to None" is the whole difficulty: a
    freshly built plot has not shown its menu, so nothing has been
    detached yet.
    """

    def test_a_parentless_control_editor_is_put_back_under_its_menu(
            self, plot_owner, monkeypatch):
        from PySide6.QtWidgets import QWidget as _QWidget

        adopted = []
        real_set_parent = _QWidget.setParent

        # Every widget reports itself parentless, which is the state the
        # menu's own close leaves its editors in.
        monkeypatch.setattr(_QWidget, "parentWidget", lambda self: None)
        monkeypatch.setattr(
            _QWidget, "setParent",
            lambda self, parent: adopted.append((type(self).__name__,
                                                 type(parent).__name__)))

        assert retire_pyqtgraph_menus(plot_owner) == 2
        assert adopted, "nothing was re-adopted before deletion"

    def test_a_control_editor_that_refuses_adoption_is_survived(
            self, plot_owner, monkeypatch):
        """And the guard beside it, with the same widgets parentless."""
        from PySide6.QtWidgets import QWidget as _QWidget

        attempts = []

        def refuse(self, parent):
            attempts.append(type(self).__name__)
            raise RuntimeError("cannot transfer ownership during close")

        monkeypatch.setattr(_QWidget, "parentWidget", lambda self: None)
        monkeypatch.setattr(_QWidget, "setParent", refuse)

        assert retire_pyqtgraph_menus(plot_owner) == 2, (
            "a refused adoption stopped the retirement")
        assert attempts, "adoption was never attempted"


def test_a_menu_that_already_has_a_parent_is_not_reparented(plot_owner,
                                                            monkeypatch):
    """The other side of the ownership repair: there is no hole to fill.

    A menu Qt already owns needs no transfer -- it will be destroyed with
    its parent. Reparenting it anyway would move it out from under an
    owner that was going to handle it, for no gain.
    """
    from PySide6.QtWidgets import QWidget as _QWidget

    transfers = []
    monkeypatch.setattr(_QWidget, "parentWidget", lambda self: plot_owner)
    monkeypatch.setattr(
        _QWidget, "setParent",
        lambda self, parent: transfers.append(type(self).__name__))

    assert retire_pyqtgraph_menus(plot_owner) == 2
    assert transfers == [], (
        "a menu that already had a parent was reparented anyway")
