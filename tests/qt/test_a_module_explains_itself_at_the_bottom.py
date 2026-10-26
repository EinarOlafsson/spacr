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

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication

from spacr.qt.app import MainWindow
from spacr.qt.widgets.home import AppTile
from spacr.qt.widgets.module_hint_bar import ModuleHintBar


@pytest.mark.parametrize('gui', [2., .9, 1.])
def test_module_hint_height_uses_actual_two_line_metrics(qtbot, qapp, gui):
    import json
    import math
    from PySide6.QtCore import QRect, QSize, Qt
    from PySide6.QtGui import QFontInfo, QTextDocument
    from spacr.qt import app, gui_scale, preferences

    saved_gui, saved_font = preferences.get_gui_scale(), preferences.get_font_scale()
    live_gui = gui_scale.current_scale()
    old_sheet, old_font, old_palette = qapp.styleSheet(), qapp.font(), qapp.palette()
    bar = None
    try:
        app._load_bundled_fonts()
        app._use_open_sans(qapp)
        gui_scale.install_scaling_layer()
        preferences.set_gui_scale(gui)
        preferences.set_font_scale(2.)
        gui_scale.set_gui_scale_live(gui)
        preferences.apply_preferences_to_app()
        bar = ModuleHintBar()
        qtbot.addWidget(bar)
        gui_scale._resize_actual(bar, QSize(900, 600))
        bar.show()
        previous = [None]

        def settled():
            now = (bar.font().toString(), bar.property('size').toTuple(),
                   bar.property('minimumSize').toTuple(), bar.property('maximumSize').toTuple())
            same = now == previous[0]
            previous[0] = now
            return same

        qtbot.waitUntil(settled)
        assert QFontInfo(bar.font()).family() == 'Open Sans'
        metrics = bar.fontMetrics()
        measured = metrics.boundingRect(QRect(0, 0, 1 << 20, 0),
            int(Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft), 'Xg\nXg').height()
        insets = bar.rect().height() - bar.contentsRect().height()
        expected = max(1, measured) + max(10, insets)
        record = {'gui': gui, 'font_pixels': bar.font().pixelSize(),
                  'metric': measured, 'old_metric_plus_10': max(1, measured) + 10,
                  'vertical_insets': insets, 'expected_actual_height': expected,
                  'size': bar.property('size').toTuple(),
                  'minimum': bar.property('minimumSize').toTuple(),
                  'maximum': bar.property('maximumSize').toTuple(),
                  'logical': dict(getattr(bar, '_gs_geo', {})),
                  'applied': dict(getattr(bar, '_gs_geo_set', {})),
                  'contents': bar.contentsRect().getRect()}
        print('MODULE_HINT_HEIGHT', json.dumps(record), flush=True)
        assert record['minimum'][1] == record['maximum'][1] == record['size'][1] == expected, record
        assert record['logical']['minh'] == record['logical']['maxh'] == math.ceil(expected / gui)
        assert record['applied']['minh'] == record['applied']['maxh'] == expected
        for width in (900, 600):
            gui_scale._resize_actual(bar, QSize(width, expected))
            previous[0] = None
            qtbot.waitUntil(settled)
            bar.show_module('mask', 'A module summary with enough words to require width-aware shortening. ' * 4)
            assert '<br>' in bar.text() and 'API' in bar.text() and 'Tutorial' in bar.text()
            document = QTextDocument()
            document.setDocumentMargin(0)
            document.setDefaultFont(bar.font())
            document.setHtml(bar.text())
            assert document.size().height() <= bar.contentsRect().height(), record
            assert bar.property('size').height() == expected
            assert not bar.wordWrap()
            bar.release()
            assert bar.accessibleName() == bar._default and not bar.is_holding()
            document.setHtml(bar.text())
            assert document.size().width() <= bar.contentsRect().width()
            assert document.size().height() <= bar.contentsRect().height()
    finally:
        try:
            if bar is not None:
                bar.close()
        finally:
            preferences.set_gui_scale(saved_gui)
            preferences.set_font_scale(saved_font)
            gui_scale.set_gui_scale_live(live_gui)
            qapp.setStyleSheet(old_sheet)
            qapp.setFont(old_font)
            qapp.setPalette(old_palette)


@pytest.mark.parametrize('gui', [2., .9, 1.])
def test_held_module_hint_refits_actual_contents_without_restarting_hold(qtbot, qapp, gui):
    import json
    import math
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QFontInfo, QTextDocument
    from spacr.qt import app, gui_scale, preferences

    saved_gui, saved_font = preferences.get_gui_scale(), preferences.get_font_scale()
    live_gui = gui_scale.current_scale()
    old_sheet, old_font, old_palette = qapp.styleSheet(), qapp.font(), qapp.palette()
    bar = None
    try:
        app._load_bundled_fonts()
        app._use_open_sans(qapp)
        gui_scale.install_scaling_layer()
        preferences.set_gui_scale(gui)
        preferences.set_font_scale(2.)
        gui_scale.set_gui_scale_live(gui)
        preferences.apply_preferences_to_app()
        bar = ModuleHintBar()
        qtbot.addWidget(bar)
        gui_scale._resize_actual(bar, QSize(900, 600))
        bar.show()
        previous = [None]

        def settled():
            now = (bar.property('size').toTuple(), bar.font().toString(), bar.text())
            same = now == previous[0]
            previous[0] = now
            return same

        qtbot.waitUntil(settled)
        assert QFontInfo(bar.font()).family() == 'Open Sans'
        summary = 'Segment cells and nuclei while retaining the original image and every measured object.'
        stage = 'Preview'
        bar.show_module('mask', summary, stage)
        timer, timer_id = bar._timer, bar._timer.timerId()
        remaining = timer.remainingTime()
        height = bar.property('size').height()
        links = bar._links_html('mask')
        full = QTextDocument()
        full.setDocumentMargin(0)
        full.setDefaultFont(bar.font())
        full.setPlainText(summary + ' — ' + stage)
        insets = bar.rect().width() - bar.contentsRect().width()
        grow_width = math.ceil(full.idealWidth()) + insets + 2
        rows, failures = [], []
        for width in (900, 600, grow_width):
            gui_scale._resize_actual(bar, QSize(width, height))
            previous[0] = None
            qtbot.waitUntil(settled)
            document = QTextDocument()
            document.setDocumentMargin(0)
            document.setDefaultFont(bar.font())
            document.setHtml(bar.text())
            contents = bar.contentsRect()
            row = {'width': width, 'contents': contents.getRect(),
                   'document': document.size().toTuple(), 'html': bar.text(),
                   'timer_id': timer.timerId(), 'remaining': timer.remainingTime()}
            rows.append(row)
            if document.size().width() > contents.width() or document.size().height() > contents.height():
                failures.append(row)
            assert bar._timer is timer and timer.timerId() == timer_id and timer.isActive()
            assert timer.interval() == 30_000 and 0 < timer.remainingTime() <= remaining
            remaining = timer.remainingTime()
            assert bar.module_key == 'mask'
            assert bar.accessibleName() == bar.accessibleDescription() == summary
            assert stage in bar.text() and links in bar.text()
            assert bar.property('size').height() == height
        restored = document.toPlainText().split('\n')[0]
        print('HELD_MODULE_HINT_WIDTH', json.dumps({'gui': gui, 'rows': rows,
              'restored': restored, 'expected_original': summary + ' — ' + stage}), flush=True)
        assert not failures, failures
        assert restored == summary + ' — ' + stage
        seen = []
        bar.link_followed.connect(lambda key, href: seen.append((key, href)))
        bar._on_link('https://example.invalid/mask')
        assert seen == [('mask', 'https://example.invalid/mask')]
        bar.release()
        gui_scale._resize_actual(bar, QSize(600, height))
        previous[0] = None
        qtbot.waitUntil(settled)
        assert bar.accessibleName() == bar._default and not bar.is_holding()
        assert bar.module_key == ''
        document.setHtml(bar.text())
        print('RELEASED_MODULE_HINT_WIDTH', json.dumps({
            'gui': gui, 'document': document.size().toTuple(),
            'contents': bar.contentsRect().getRect(), 'fit_acceptance': True}), flush=True)
        assert document.size().width() <= bar.contentsRect().width()
        assert document.size().height() <= bar.contentsRect().height()
    finally:
        try:
            if bar is not None:
                bar.close()
        finally:
            preferences.set_gui_scale(saved_gui)
            preferences.set_font_scale(saved_font)
            gui_scale.set_gui_scale_live(live_gui)
            qapp.setStyleSheet(old_sheet)
            qapp.setFont(old_font)
            qapp.setPalette(old_palette)


@pytest.mark.parametrize('gui', [2., .9, 1.])
def test_default_module_prompt_refits_without_holding_or_reviving_modules(qtbot, qapp, gui):
    import json
    import math
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QFontInfo, QTextDocument
    from spacr.qt import app, gui_scale, preferences

    saved_gui, saved_font = preferences.get_gui_scale(), preferences.get_font_scale()
    live_gui = gui_scale.current_scale()
    old_sheet, old_font, old_palette = qapp.styleSheet(), qapp.font(), qapp.palette()
    bar = None
    try:
        app._load_bundled_fonts()
        app._use_open_sans(qapp)
        gui_scale.install_scaling_layer()
        preferences.set_gui_scale(gui)
        preferences.set_font_scale(2.)
        gui_scale.set_gui_scale_live(gui)
        preferences.apply_preferences_to_app()
        bar = ModuleHintBar()
        qtbot.addWidget(bar)
        gui_scale._resize_actual(bar, QSize(600, 600))
        bar.show()
        previous = [None]

        def settled():
            now = (bar.property('size').toTuple(), bar.font().toString(), bar.text())
            same = now == previous[0]
            previous[0] = now
            return same

        qtbot.waitUntil(settled)
        assert QFontInfo(bar.font()).family() == 'Open Sans'
        full = QTextDocument()
        full.setDocumentMargin(0)
        full.setDefaultFont(bar.font())
        full.setPlainText(bar._default)
        grow = max(900, math.ceil(full.idealWidth()) + bar.rect().width() - bar.contentsRect().width() + 2)
        height = bar.property('size').height()
        rows, failures = [], []
        for phase in ('initial', 'released'):
            if phase == 'released':
                bar.show_module('mask', 'Segment cells and nuclei.', 'Preview')
                bar.release()
            for width in (600, grow):
                gui_scale._resize_actual(bar, QSize(width, height))
                previous[0] = None
                qtbot.waitUntil(settled)
                document = QTextDocument()
                document.setDocumentMargin(0)
                document.setDefaultFont(bar.font())
                document.setHtml(bar.text())
                row = {'phase': phase, 'width': width, 'html': bar.text(),
                       'document': document.size().toTuple(), 'contents': bar.contentsRect().getRect()}
                rows.append(row)
                if (document.size().width() > bar.contentsRect().width()
                        or document.size().height() > bar.contentsRect().height()):
                    failures.append(row)
                assert bar.accessibleName() == bar._default
                assert not bar.is_holding() and bar.module_key == ''
                assert 'Preview' not in bar.text() and '<a ' not in bar.text()
                assert bar.property('size').height() == height
                if width == grow:
                    assert document.toPlainText() == bar._default
        print('DEFAULT_MODULE_HINT_WIDTH', json.dumps({'gui': gui, 'rows': rows}), flush=True)
        assert not failures, failures
    finally:
        try:
            if bar is not None:
                bar.close()
        finally:
            preferences.set_gui_scale(saved_gui)
            preferences.set_font_scale(saved_font)
            gui_scale.set_gui_scale_live(live_gui)
            qapp.setStyleSheet(old_sheet)
            qapp.setFont(old_font)
            qapp.setPalette(old_palette)


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
    window._stack.setCurrentWidget(page)
    page._tabs.setCurrentIndex(0)
    window.show()
    QApplication.processEvents()
    yield page
    page._hint_bar.release()



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

def test_hovering_a_tile_writes_the_module_and_both_links(home, immediate_hover_help):
    """Summary, API and Tutorial, all in the strip."""
    tile = next(t for t in home.findChildren(AppTile)
                if t.text_label == "Mask" and t.isVisible())
    immediate_hover_help(tile, lambda: home._hint_bar.module_key == tile.property("moduleAppKey"))
    bar = home._hint_bar
    assert isinstance(bar, ModuleHintBar)
    assert bar.module_key == "mask", (
        f"the strip is explaining {bar.module_key!r}, not mask")
    text = bar.text()
    assert "API" in text and "Tutorial" in text, (
        f"the strip has no links: {text!r}")
    assert text.count("<a href=") == 2, f"expected two links: {text!r}"
    assert "einarolafsson.github.io/spacr/tutorials/#lesson=" in text


def test_the_strip_holds_after_the_pointer_leaves(home, immediate_hover_help):
    """The reach the hold exists to make possible."""
    tile = next(t for t in home.findChildren(AppTile)
                if t.text_label == "Measure" and t.isVisible())
    immediate_hover_help(tile, lambda: home._hint_bar.module_key == tile.property("moduleAppKey"))
    held = home._hint_bar.text()
    assert home._hint_bar.is_holding()

    QApplication.sendEvent(tile, QEvent(QEvent.Type.Leave))
    assert home._hint_bar.text() == held, (
        "the strip was cleared on Leave, so the links cannot be reached")
    assert home._hint_bar.module_key == "measure"


def test_the_hold_is_thirty_seconds():
    """The number the maintainer asked for, not a value to tune down."""
    assert ModuleHintBar.HOLD_MS == 30_000


def test_the_hold_restarts_on_the_next_module(home, immediate_hover_help):
    """Reading across a row of tiles is not a race against the first clock."""
    tiles = {t.text_label: t for t in home.findChildren(AppTile) if t.isVisible()}
    immediate_hover_help(tiles["Mask"], lambda: home._hint_bar.module_key == "mask")
    assert home._hint_bar.module_key == "mask"
    immediate_hover_help(tiles["Measure"], lambda: home._hint_bar.module_key == "measure")
    assert home._hint_bar.module_key == "measure"
    assert home._hint_bar.is_holding()


def test_releasing_puts_the_prompt_back(home, immediate_hover_help):
    tile = next(t for t in home.findChildren(AppTile)
                if t.text_label == "Mask" and t.isVisible())
    immediate_hover_help(tile, lambda: home._hint_bar.module_key == tile.property("moduleAppKey"))
    home._hint_bar.release()
    assert home._hint_bar.module_key == ""
    assert not home._hint_bar.is_holding()
    assert "API" not in home._hint_bar.text()


# ---------------------------------------------------------------------------
# The dock writes to the same strip
# ---------------------------------------------------------------------------

def test_hovering_a_dock_row_explains_it_in_home_s_strip(window, home, qapp, immediate_hover_help):
    """"the hover over the doc should function the same"."""
    window._stack.setCurrentWidget(home)
    qapp.processEvents()
    row = next(r for r in window._sidebar._items
               if str(r.property("navKey")) == "regression")
    window.apply_dock_mode("locked")
    section = window._sidebar._section_of[row.key]
    if not window._sidebar.section_is_open(section):
        window._sidebar.toggle_section(section)
    qapp.processEvents()
    home._hint_bar.release()
    immediate_hover_help(row, lambda: home._hint_bar.module_key == "regression")
    qapp.processEvents()
    assert home._hint_bar.module_key == "regression", (
        f"the dock hover wrote {home._hint_bar.module_key!r}")
    assert "Tutorial" in home._hint_bar.text()
    assert home._hint_bar.is_holding()


def test_the_home_row_is_not_announced_as_a_module(window, home, qapp, immediate_hover_help):
    """`__home__` has no documentation page and no lesson."""
    window._stack.setCurrentWidget(home)
    home._hint_bar.release()
    row = next(r for r in window._sidebar._items
               if str(r.property("navKey")) == "__home__")
    window.apply_dock_mode("locked")
    qapp.processEvents()
    seen = []
    window._sidebar.module_hovered.connect(seen.append)
    try:
        immediate_hover_help(row, lambda: seen == ["__home__"])
    finally:
        window._sidebar.module_hovered.disconnect(seen.append)
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
              'docs/source/_extra/tutorials/module_navigation.js').read_text(encoding="utf-8")
    navigation = json.loads(source.split('Object.freeze(', 1)[1].rsplit(');', 1)[0])
    gaps = navigation['missing_tutorials']
    registered = {row['app_key'] for row in gaps}
    assert registered == set()
    assert all(has_tutorial(key) for key in ('host_pathogen', 'toxoplasma'))
    # 83_plasmodium and 84_candida were withdrawn in tutorial wave 3 (alpha species).
    assert not any(has_tutorial(key) for key in ('candida', 'plasmodium'))
    assert len(registered) == len(gaps)
    assert all(row['status'] == 'needs_tutorial' for row in gaps)
    from spacr.settings import _alpha_names

    # Alpha-gated apps (item 634) ship without lessons by rule. A module
    # whose lesson is archived has no link either; it shows Coming soon.
    from spacr.qt.tutorials import _coming_soon
    coming_soon = set(_coming_soon())
    assert not any(has_tutorial(key) for key in coming_soon)
    missing = {key for key, *_ in APPS if not has_tutorial(key)} - _alpha_names("apps")
    assert missing - coming_soon == registered


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
