"""A module is explained in the strip at the bottom, never in a popup.

Asked for on 2026-09-03: "remove the popup window tooltip on the moduals. the
tooltip is shown at the botom of the screen. these should also have an API
link and a tutorial link and the list onw hovered should be shown at the
botom for 30 seconds. ... the hover over the doc should function the same."

MEASURED before the change, one 1440x900 MainWindow:

    Home tiles carrying a native tooltip     19  ->  0
    dock rows carrying a native tooltip      63  ->  0
    links in the strip                        0  ->  2  (API, Tutorial)
    the strip holds after the pointer leaves  no ->  30 s
    dock hover writes to the strip            no ->  yes

`AppTile` had said "NO TOOLTIP on the tile" in its own constructor since the
hint bar was introduced, and `i18n._refresh_module_help` was quietly putting
one back on the next language refresh -- which runs at startup. That is why
the popups were still there after they had been removed once.

THE HOLD IS THE FEATURE, not a nicety. The API and Tutorial words appear in a
strip at the bottom of the window; a strip cleared on Leave takes them away
the instant the pointer starts moving toward them, so neither could ever be
pressed. Instruction 371 made the same argument for the per-setting strip and
settled on ten seconds; thirty here because these two links open a browser.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.app import MainWindow
from spacr.qt.widgets.home import AppTile
from spacr.qt.widgets.module_hint_bar import ModuleHintBar


@pytest.fixture(scope="module")
def window(qapp, qt_theme_applied):
    win = MainWindow()
    win.resize(1440, 900)
    win.show()
    qapp.processEvents()
    yield win
    win.close()
    win.deleteLater()
    qapp.sendPostedEvents(win, QEvent.Type.DeferredDelete)


@pytest.fixture
def home(window):
    """Home, with its strip put back to the prompt after each test."""
    page = window._startup
    yield page
    page._hint_bar.release()


def _enter(widget):
    where = QPointF(widget.width() / 2, widget.height() / 2)
    QApplication.sendEvent(widget, QEnterEvent(
        where, where,
        QPointF(widget.mapToGlobal(QPoint(int(where.x()),
                                          int(where.y()))))))


# ---------------------------------------------------------------------------
# The popup is gone
# ---------------------------------------------------------------------------

def test_no_home_tile_carries_a_native_tooltip(home):
    """The popup drew a second copy of the strip's sentence over the grid."""
    tiles = home.findChildren(AppTile)
    assert tiles, "no tiles to check"
    with_tips = {t.text_label: t.toolTip() for t in tiles if t.toolTip()}
    assert not with_tips, f"these tiles still pop a tooltip: {with_tips}"


def test_no_dock_row_carries_a_native_tooltip(window):
    """Same for the dock: "the hover over the doc should function the same"."""
    rows = window._sidebar._items
    assert rows, "no dock rows to check"
    with_tips = {str(r.property("navKey")): r.toolTip()
                 for r in rows if r.toolTip()}
    assert not with_tips, f"these dock rows still pop a tooltip: {with_tips}"


def test_the_sentence_moved_rather_than_vanishing(window, home):
    """A screen reader must still hear what the popup used to say."""
    tiles = [t for t in home.findChildren(AppTile) if t.text_label]
    assert tiles
    bare = [t.text_label for t in tiles if not t.accessibleDescription()]
    assert not bare, f"these tiles say nothing to a screen reader: {bare}"

    rows = [r for r in window._sidebar._items
            if str(r.property("navKey")) != "__home__"]
    quiet = [str(r.property("navKey")) for r in rows
             if not r.accessibleName()]
    assert not quiet, f"these dock rows have no accessible name: {quiet}"


# ---------------------------------------------------------------------------
# The strip
# ---------------------------------------------------------------------------

def test_hovering_a_tile_writes_the_module_and_both_links(home):
    """Summary, API and Tutorial, all in the strip."""
    tile = next(t for t in home.findChildren(AppTile)
                if t.text_label == "Mask")
    _enter(tile)
    bar = home._hint_bar
    assert isinstance(bar, ModuleHintBar)
    assert bar.module_key == "mask", (
        f"the strip is explaining {bar.module_key!r}, not mask")
    text = bar.text()
    assert "API" in text and "Tutorial" in text, (
        f"the strip has no links: {text!r}")
    assert text.count("<a href=") == 2, f"expected two links: {text!r}"
    assert "einarolafsson.github.io/spacr/tutorials/#lesson=" in text


def test_the_strip_holds_after_the_pointer_leaves(home):
    """The reach the hold exists to make possible."""
    tile = next(t for t in home.findChildren(AppTile)
                if t.text_label == "Measure")
    _enter(tile)
    held = home._hint_bar.text()
    assert home._hint_bar.is_holding()

    QApplication.sendEvent(tile, QEvent(QEvent.Type.Leave))
    assert home._hint_bar.text() == held, (
        "the strip was cleared on Leave, so the links cannot be reached")
    assert home._hint_bar.module_key == "measure"


def test_the_hold_is_thirty_seconds():
    """The number the maintainer asked for, not a value to tune down."""
    assert ModuleHintBar.HOLD_MS == 30_000


def test_the_hold_restarts_on_the_next_module(home):
    """Reading across a row of tiles is not a race against the first clock."""
    tiles = {t.text_label: t for t in home.findChildren(AppTile)}
    _enter(tiles["Mask"])
    assert home._hint_bar.module_key == "mask"
    _enter(tiles["Measure"])
    assert home._hint_bar.module_key == "measure"
    assert home._hint_bar.is_holding()


def test_releasing_puts_the_prompt_back(home):
    tile = next(t for t in home.findChildren(AppTile)
                if t.text_label == "Mask")
    _enter(tile)
    home._hint_bar.release()
    assert home._hint_bar.module_key == ""
    assert not home._hint_bar.is_holding()
    assert "API" not in home._hint_bar.text()


# ---------------------------------------------------------------------------
# The dock writes to the same strip
# ---------------------------------------------------------------------------

def test_hovering_a_dock_row_explains_it_in_home_s_strip(window, home, qapp):
    """"the hover over the doc should function the same"."""
    window._stack.setCurrentWidget(home)
    qapp.processEvents()
    row = next(r for r in window._sidebar._items
               if str(r.property("navKey")) == "regression")
    home._hint_bar.release()
    _enter(row)
    qapp.processEvents()
    assert home._hint_bar.module_key == "regression", (
        f"the dock hover wrote {home._hint_bar.module_key!r}")
    assert "Tutorial" in home._hint_bar.text()
    assert home._hint_bar.is_holding()


def test_the_home_row_is_not_announced_as_a_module(window, home, qapp):
    """`__home__` has no documentation page and no lesson."""
    window._stack.setCurrentWidget(home)
    home._hint_bar.release()
    row = next(r for r in window._sidebar._items
               if str(r.property("navKey")) == "__home__")
    _enter(row)
    qapp.processEvents()
    assert home._hint_bar.module_key == "", (
        "Home was announced as a module")


def test_the_router_is_silent_on_a_page_that_cannot_show_help(window):
    """A screen with no strip is not a broken screen."""
    from PySide6.QtWidgets import QWidget

    blank = QWidget()
    window._stack.addWidget(blank)
    window._stack.setCurrentWidget(blank)
    try:
        window._show_module_hint("mask")        # must not raise
        # Silent means silent: the router must not BUILD a strip on a page
        # that has none. "It did not raise" would also be true of a router
        # that quietly grew a second help bar on every screen it could not
        # find one on, which is the failure worth naming here.
        assert blank.children() == [], (
            "the router put something on a page that cannot show help")
        assert window._stack.currentWidget() is blank, (
            "the router navigated away from the page it could not help")
    finally:
        window._stack.removeWidget(blank)
        blank.deleteLater()


# ---------------------------------------------------------------------------
# The links themselves
# ---------------------------------------------------------------------------

def test_every_registry_module_has_a_lesson_or_a_registered_gap():
    """Missing lessons stay explicit and do not acquire misleading links."""
    import json
    from pathlib import Path
    from spacr.qt.app import APPS
    from spacr.qt.tutorials import has_tutorial

    source = (Path(__file__).resolve().parents[2] /
              'docs/source/_extra/tutorials/module_navigation.js').read_text()
    navigation = json.loads(source.split('Object.freeze(', 1)[1].rsplit(');', 1)[0])
    gaps = navigation['missing_tutorials']
    registered = {row['app_key'] for row in gaps}
    assert registered == {'candida', 'host_pathogen', 'plasmodium', 'toxoplasma'}
    assert len(registered) == len(gaps)
    assert all(row['status'] == 'needs_tutorial' for row in gaps)
    missing = {key for key, *_ in APPS if not has_tutorial(key)}
    assert missing == registered


def test_a_module_with_no_lesson_simply_loses_the_word(qtbot):
    """A Tutorial link to an index of seventy-three lessons is worse than
    none, which is the rule the tooltip footer follows for Animation."""
    bar = ModuleHintBar()
    qtbot.addWidget(bar)
    bar.show_module("not-a-real-module", "Some summary.")
    assert "Tutorial" not in bar.text()


def test_a_dock_hover_reaches_a_module_screen_s_own_strip(window, qapp):
    """On a module screen the strip is the settings one, and it takes this.

    "the hover over the doc should function the same" is about the dock,
    not about Home, so it has to work wherever the dock is -- and the dock
    is on screen the whole time. A module screen already owns the bottom of
    its window; a second strip stacked under it would be two places to look.
    """
    window.open_module("mask")
    qapp.processEvents()
    screen = window._stack.currentWidget()
    if not hasattr(screen, "_hint_strip"):
        pytest.skip("this screen has no per-setting strip to write to")

    # DRIVEN THROUGH THE ROUTER, which is the path a dock hover takes. The
    # screen's own method needs the sentence handed to it -- `module_summary`
    # falls back to the registry's English description and a screen has no
    # registry -- so calling it directly would be testing a half of the
    # wiring that never runs on its own.
    window._show_module_hint("regression")
    qapp.processEvents()
    text = screen._hint_strip.text()
    assert "API" in text and "Tutorial" in text, (
        f"the module hint reached the strip without its links: {text!r}")
    assert "#lesson=" in text
    # And the module hold, not the shorter per-setting one.
    assert screen.MODULE_HINT_HOLD_MS == 30_000
    timer = getattr(screen, "_hint_hold_timer", None)
    assert timer is not None and timer.isActive()
    assert timer.interval() == 30_000, (
        f"the strip is holding for {timer.interval()} ms, not the module's "
        "thirty seconds")


# ---------------------------------------------------------------------------
# The strip itself: what it does NOT do, and the signal it exists to emit
# ---------------------------------------------------------------------------

def test_the_strip_does_not_pop_a_tooltip_over_itself(qtbot):
    """The one widget in the window that may never show a popup.

    Its whole reason for existing is to replace popups; a QLabel whose text
    is elided shows the full text in a tooltip by default, so hovering the
    strip would pop up the very box it was built to prevent -- over the
    links the reader is crossing the window to click.
    """
    from PySide6.QtWidgets import QLabel

    reached_qlabel = []

    class RecordedLabel(QLabel):
        def event(self, event):
            reached_qlabel.append(event.type())
            return super().event(event)

    class RecordedBar(ModuleHintBar, RecordedLabel):
        """Observe this bar's super call without patching every native QLabel."""

    bar = RecordedBar()
    qtbot.addWidget(bar)
    bar.show_module("mask", "Segment cells and nuclei.")
    reached_qlabel.clear()

    tooltip = QEvent(QEvent.Type.ToolTip)
    assert bar.event(tooltip) is True
    assert tooltip.isAccepted()
    assert QEvent.Type.ToolTip not in reached_qlabel, (
        "the tooltip request must stop at the strip"
    )

    bar.event(QEvent(QEvent.Type.User))
    assert QEvent.Type.User in reached_qlabel, (
        "only the tooltip is swallowed; everything else goes to QLabel"
    )


def test_following_a_link_says_which_module_it_belonged_to(qtbot):
    """`link_followed` carries the KEY, not just the href.

    The window opens API pages in one place and lessons in another, and the
    strip is the only thing that still knows which module the reader was
    pointing at when they pressed the word.
    """
    bar = ModuleHintBar()
    qtbot.addWidget(bar)
    bar.show_module("mask", "Segment cells and nuclei.")
    seen = []
    bar.link_followed.connect(lambda key, href: seen.append((key, href)))

    bar._on_link("https://example.invalid/api/mask")

    assert seen == [("mask", "https://example.invalid/api/mask")]


def test_a_strip_with_no_room_left_shows_the_whole_sentence_rather_than_none(
    qtbot,
):
    """Elision needs a positive width; a bar not yet laid out has none.

    Handing `elidedText` a negative width returns an empty string, so the
    strip would come up blank on the first paint of a window that has not
    been shown yet. The sentence unelided is the safe answer: the label
    clips it, and nothing is lost that was not going to be clipped anyway.
    """
    bar = ModuleHintBar()
    qtbot.addWidget(bar)
    bar.setFixedWidth(1)

    assert bar._fit("Segment cells and nuclei.") == "Segment cells and nuclei."


def test_a_strip_with_no_module_key_offers_no_api_link(qtbot):
    """The links belong to a module; with no module there is no link.

    A strip showing a sentence and an "API" word that goes to the index of
    every module would be a link that looks like an answer and is not one --
    the rule `tutorial_url` already follows for a module with no lesson.
    """
    bar = ModuleHintBar()
    qtbot.addWidget(bar)

    assert bar._api_url("") == ""

    bar.show_module("", "Some summary.")
    assert "API" not in bar.text()
    assert "Some summary." in bar.text()


def test_an_api_lookup_that_raises_costs_the_link_and_nothing_else(
    qtbot, monkeypatch,
):
    """The lookup imports a large module from a hover handler.

    An ImportError or a registry half-built during startup must cost the
    word "API" and leave the description on screen, not raise out of a
    paint.
    """
    import spacr.qt.screens.settings_model as settings_model
    from PySide6.QtCore import QTimer

    def explode(_key):
        raise RuntimeError("the settings registry is not built yet")

    bar = ModuleHintBar()
    qtbot.addWidget(bar)
    pending_help = []
    QTimer.singleShot(0, lambda: pending_help.append(
        settings_model.plain_tooltip("Input directory.", "mask", "src")))

    with monkeypatch.context() as lookup:
        lookup.setattr(settings_model, "api_docs_url", explode)
        assert bar._api_url("mask") == ""
        bar.show_module("mask", "Segment cells and nuclei.")
        assert "Segment cells and nuclei." in bar.text()
    qtbot.waitUntil(lambda: bool(pending_help))
    assert "Input directory." in pending_help[0]
