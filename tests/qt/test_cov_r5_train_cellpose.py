"""Edges of the Cellpose workbench: its masthead, its tab signal, its factory.

The workbench is two module pages under one masthead, so it has to be
tolerant of the three things that can be missing at those seams:

* a module page that carries no masthead of its own -- there is nothing to
  hide, and the page still has to become a tab;
* a ``currentChanged`` that arrives with no previous tab to carry settings
  out of, which is what Qt sends when the strip is emptied and refilled;
* a masthead with no API link on it, and a host that offers only one of the
  two slots the factory wires -- neither may cost the caller the screen.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel                             # noqa: E402

from spacr.qt.screens import train_cellpose as screen_module     # noqa: E402
from spacr.qt.screens.train_cellpose import (                    # noqa: E402
    APPLY_KEY,
    TABS,
    TRAIN_KEY,
    CellposeWorkbenchScreen,
)

pytestmark = pytest.mark.qt


@pytest.fixture()
def workbench(qtbot):
    """The merged screen with both module pages live."""
    screen = CellposeWorkbenchScreen()
    qtbot.addWidget(screen)
    return screen


# ---------------------------------------------------------------------------
# The masthead of each page
# ---------------------------------------------------------------------------

def test_a_page_with_no_masthead_of_its_own_still_becomes_a_tab(qtbot,
                                                                monkeypatch):
    """Nothing to hide is not a reason to stop building the tab.

    Each half's own ``ModuleHeader`` is hidden because the workbench draws
    one masthead for both -- 30px of module title under 30px of page title,
    and the folded half has no registry row left to read a name out of. A
    page that never built one simply has nothing to hide, and dropping it
    from the strip over that would cost the user half the module.
    """
    class _Headerless(screen_module.AppScreen):
        """A module page whose masthead was never built."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._header = None

    monkeypatch.setattr(screen_module, "AppScreen", _Headerless)
    bare = CellposeWorkbenchScreen()
    qtbot.addWidget(bare)

    assert bare._tabs.count() == len(TABS)
    assert [page.app_key for page in bare._screens] == [TRAIN_KEY, APPLY_KEY]
    assert all(page._header is None for page in bare._screens)
    # The rest of the loop still ran: both pages are wired to re-emit.
    with qtbot.waitSignal(bare.error_explain_requested, timeout=1000) as blocked:
        bare._screens[1].error_explain_requested.emit("boom", APPLY_KEY)
    assert blocked.args == ["boom", APPLY_KEY]

    # And a page that DOES carry one has it hidden, which is what makes the
    # branch above a case rather than the only case.
    monkeypatch.undo()
    normal = CellposeWorkbenchScreen()
    qtbot.addWidget(normal)
    assert all(page._header is not None for page in normal._screens)
    assert all(page._header.isHidden() for page in normal._screens), (
        "a module page drew its own title under the workbench's")


# ---------------------------------------------------------------------------
# Tab changes
# ---------------------------------------------------------------------------

def test_a_tab_change_with_no_previous_tab_carries_nothing(workbench):
    """``currentChanged(-1)`` then ``(0)`` must not carry out of nowhere.

    Qt sends ``currentChanged(-1)`` whenever the strip is emptied and a
    fresh index when it refills, so the index the screen remembers can be
    ``-1`` by the time the next change arrives. ``self._screens[-1]`` is a
    real element in Python -- it is the OTHER tab -- so an unguarded carry
    here would copy the Apply half's knobs into the Apply half while the
    user was opening the Train half.

    Driven through the real signal (the strip is reached directly because
    the screen exposes no way to empty it, which is the point).
    """
    carried: list = []
    tabs = workbench._tabs
    train, apply_ = workbench.train_screen, workbench.apply_screen

    tabs.clear()
    assert workbench._current == -1

    # Watch only what happens after the strip is empty.
    workbench.carry = lambda source, target: carried.append((source, target))
    tabs.addTab(apply_, "Apply")

    assert carried == [], "the screen carried settings out of no tab"
    assert workbench._current == 0
    assert workbench._header.instruction_label.text() == TABS[0][2]

    # With a real previous tab the same signal does carry, so the silence
    # above is the missing tab and not a lost connection.
    tabs.addTab(train, "Train")
    tabs.setCurrentIndex(1)

    assert carried == [(workbench._screens[0], workbench._screens[1])]


def test_a_masthead_with_no_api_link_still_says_what_src_means(workbench):
    """The instruction line lands whether or not there is a link to repoint.

    One masthead serves two modules, so the help link follows the visible
    tab. The link is optional -- a header built without a description has
    none -- and the sentence under the title is not: it is the only thing
    that tells the user which folder layout the tab in front of them reads.
    """
    header = workbench._header
    assert header.api_help is not None
    header.api_help = None

    workbench._tabs.setCurrentIndex(1)

    assert header.instruction_label.text() == TABS[1][2]
    assert header.instruction_label.isVisible() or not workbench.isVisible()
    assert workbench.active_app_key() == APPLY_KEY

    # Give the link back and the same change repoints it, so the branch
    # above is the missing label rather than a dead follow.
    header.api_help = header.description_label
    workbench._tabs.setCurrentIndex(0)

    assert header.instruction_label.text() == TABS[0][2]
    assert header.api_help.property("moduleApiAppKey") == TRAIN_KEY


# ---------------------------------------------------------------------------
# The registry factory
# ---------------------------------------------------------------------------

class _Window:
    """A main window offering the two slots a module page is wired to."""

    def __init__(self, explain=True, remote=True) -> None:
        self.explained: list = []
        self.submitted: list = []
        if explain:
            self._on_explain_error = (
                lambda trace, key: self.explained.append((trace, key)))
        if remote:
            self._on_remote_submit_requested = (
                lambda key, settings: self.submitted.append((key, settings)))


def test_the_factory_builds_the_workbench_for_a_caller_with_no_window(qtbot):
    """``host=None`` is the registry's own call; it wires nothing.

    ``spacr.qt.app._call_screen_factory`` offers the host, and a preview or
    a test harness has none. The screen has to come back complete either
    way -- the two connections are how a window hears about a failure, not
    how the screen works.
    """
    window = _Window()

    hostless = screen_module.build_screen(TRAIN_KEY, None)
    qtbot.addWidget(hostless)

    assert isinstance(hostless, CellposeWorkbenchScreen)
    assert hostless.app_key == TRAIN_KEY
    hostless.error_explain_requested.emit("trace", TRAIN_KEY)
    hostless.remote_submit_requested.emit(TRAIN_KEY, {"src": "/data"})
    assert window.explained == [] and window.submitted == []

    # The same factory WITH a window makes both connections, so the two
    # empty lists above are the missing host and not a broken factory.
    hosted = screen_module.build_screen(TRAIN_KEY, window)
    qtbot.addWidget(hosted)
    hosted.error_explain_requested.emit("trace", TRAIN_KEY)
    hosted.remote_submit_requested.emit(TRAIN_KEY, {"src": "/data"})

    assert window.explained == [("trace", TRAIN_KEY)]
    assert window.submitted == [(TRAIN_KEY, {"src": "/data"})]


def test_a_window_offering_only_one_slot_gets_that_one_connected(qtbot):
    """Each connection is judged on its own, not as a pair.

    The two slots arrived at different times and a window that predates one
    of them still has the other. Skipping both because one is missing would
    take the AI explanation off a window that has the console for it.
    """
    window = _Window(explain=True, remote=False)
    assert not hasattr(window, "_on_remote_submit_requested")

    screen = screen_module.build_screen(TRAIN_KEY, window)
    qtbot.addWidget(screen)

    screen.error_explain_requested.emit("trace", TRAIN_KEY)
    screen.remote_submit_requested.emit(TRAIN_KEY, {"src": "/data"})

    assert window.explained == [("trace", TRAIN_KEY)]
    assert window.submitted == []

    # The mirror image: a window with only the remote slot gets that one.
    other = _Window(explain=False, remote=True)
    second = screen_module.build_screen(TRAIN_KEY, other)
    qtbot.addWidget(second)

    second.error_explain_requested.emit("trace", TRAIN_KEY)
    second.remote_submit_requested.emit(TRAIN_KEY, {"src": "/data"})

    assert other.explained == []
    assert other.submitted == [(TRAIN_KEY, {"src": "/data"})]


def test_the_workbench_tabs_are_labelled_and_ordered(workbench):
    """A guard rail for the loop the tests above walk into its edges."""
    assert [workbench._tabs.tabText(i) for i in range(len(TABS))] == [
        label for _key, label, _sentence in TABS]
    assert isinstance(workbench._header.instruction_label, QLabel)
