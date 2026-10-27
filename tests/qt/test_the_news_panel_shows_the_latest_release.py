"""News reflects the newest release, and never at the cost of Home.

The report, 2026-09-24: "the news section in spacr dosnt seem to get
automatically updated with new releases. im on 1.5.1.0 and the news only
goes to 1.5.0.7. the news section should always automatically reflect the
latest spacr release news."

Two halves, and this file holds the running half. A wheel's bundled
``spacr/resources/release_notes.json`` cannot contain its own release note
-- the note is written when the GitHub release is published, which is after
that wheel is immutable on PyPI -- so the release workflow keeps the
resource current for the NEXT wheel and the running copy catches up by
itself.

What is pinned here is the whole bargain, both sides of it: the panel shows
a release newer than its bundle when the fetch succeeds, and it shows the
bundled list, unchanged, when the fetch fails, is rate-limited, is switched
off in Preferences, or answers with rubbish. And nothing about any of it
reaches the network from the GUI thread or before Home is drawn.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel  # noqa: E402

from spacr.qt.widgets.home import NewsPanel  # noqa: E402


NEWER = {
    "tag": "v9.9.9.9",
    "name": "spaCR 9.9.9.9",
    "published": "2099-01-01",
    "url": "https://github.com/EinarOlafsson/spacr/releases/tag/v9.9.9.9",
    "body": "**Full Changelog**: https://example.invalid/compare",
    "links": ["https://example.invalid/compare"],
    "prerelease": False,
}


@pytest.fixture
def news(qtbot):
    """A real News panel, built from the bundled resource."""
    panel = NewsPanel("1.5.1.0")
    qtbot.addWidget(panel)
    return panel


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """Point the news cache at a throwaway directory.

    These tests fake the release feed, so they lift the suite-wide
    ``SPACR_NO_NETWORK`` the way conftest asks a feature's own tests to.
    """
    from spacr import updater

    monkeypatch.delenv("SPACR_NO_NETWORK", raising=False)
    monkeypatch.setenv(updater.ENV_NEWS_CACHE, str(tmp_path / "news"))
    return updater.news_cache_path()


def _titles(panel) -> list:
    """Every label the panel draws, in order."""
    return [lbl.text() for lbl in panel.findChildren(QLabel)]


def test_news_title_words_fit_the_real_notes_viewport(qtbot, qapp, monkeypatch):
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
    from PySide6.QtGui import QFontInfo, QRegion, QTextDocument, QTextOption
    from spacr.qt import app, gui_scale, preferences, theme
    from spacr.qt.widgets import home

    calls = []
    original_fit = home._NewsTitleFit._fit

    def observed_fit(owner):
        calls.append(1)
        return original_fit(owner)

    monkeypatch.setattr(home._NewsTitleFit, '_fit', observed_fit)

    saved_gui, saved_font = preferences.get_gui_scale(), preferences.get_font_scale()
    live_gui = gui_scale.current_scale()
    sheet, font, palette = qapp.styleSheet(), qapp.font(), qapp.palette()
    panel = None
    monkeypatch.setattr(NewsPanel, '_ask_for_newer_releases', lambda self: None)
    try:
        app._load_bundled_fonts()
        app._use_open_sans(qapp)
        gui_scale.install_scaling_layer()
        preferences.set_gui_scale(2.)
        preferences.set_font_scale(2.)
        gui_scale.set_gui_scale_live(2.)
        preferences.apply_preferences_to_app()
        panel = NewsPanel('1.5.1.0')
        qtbot.addWidget(panel)
        scroll = panel._notes
        # The observed allocation, not a fake font or replacement release body.
        for key, value in (('minw', 566), ('maxw', 566), ('minh', 400), ('maxh', 400)):
            gui_scale._set_actual_dim(scroll, key, value)
        panel.show()
        qtbot.waitExposed(panel)
        panel._older.setChecked(True)
        qtbot.waitUntil(lambda: not panel._pending_releases)
        blocks = [panel._notes_column.itemAt(i).widget() for i in range(panel._notes_column.count())
                  if panel._notes_column.itemAt(i).widget() not in (None, panel._placeholder)]
        assert len(blocks) == len(panel._releases) == 17
        title = blocks[0].layout().itemAt(0).widget()
        qtbot.waitUntil(lambda: title.font().pixelSize() == gui_scale.scale_int(theme.font_px(12, 2.), 2.))
        previous = [None]

        def settled():
            if panel._news_title_fit._busy or panel._news_title_fit._timer.isActive():
                return False
            now = (scroll.viewport().property('size').toTuple(), scroll.widget().property('size').toTuple(),
                   title.property('geometry').getRect(), title.font().toString(),
                   scroll.horizontalScrollBar().value(), scroll.verticalScrollBar().value())
            same = now == previous[0]
            previous[0] = now
            return same

        qtbot.waitUntil(settled)
        scroll.ensureWidgetVisible(title, 0, 0)
        previous[0] = None
        qtbot.waitUntil(settled)
        sources = []
        for block, entry in zip(blocks, panel._releases):
            for index, role in enumerate(('title', 'date', 'body')):
                widget = block.layout().itemAt(index).widget()
                sources.append({'tag': entry['tag'], 'role': role,
                    'hint': widget.minimumSizeHint().toTuple(), 'actual_min': widget.property('minimumSize').toTuple(),
                    'font': widget.font().toString(), 'wrap': widget.wordWrap()})
        print('NEWS_WIDTH_SOURCE', {'viewport': scroll.viewport().property('size').toTuple(),
            'content': scroll.widget().property('size').toTuple(),
            'content_min_hint': scroll.widget().minimumSizeHint().toTuple(),
            'horizontal_policy': scroll.horizontalScrollBarPolicy().value,
            'horizontal_range': (scroll.horizontalScrollBar().minimum(), scroll.horizontalScrollBar().maximum()),
            'horizontal_value': scroll.horizontalScrollBar().value(),
            'largest_children': sorted(sources, key=lambda row: row['hint'][0], reverse=True)[:3]}, flush=True)
        originals = [(w, w.text(), w.accessibleName(), w.openExternalLinks(), w.font().toString())
                     for block in blocks for w in (block.layout().itemAt(0).widget(), block.layout().itemAt(1).widget())]
        body_html = [block.layout().itemAt(2).widget().text() for block in blocks]
        long_title = blocks[-1].layout().itemAt(0).widget()
        heights = []
        for width in (566, 900, 566):
            for key in ('minw', 'maxw'):
                gui_scale._set_actual_dim(scroll, key, width)
            previous[0] = None
            qtbot.waitUntil(settled)
            assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAsNeeded
            bar = scroll.horizontalScrollBar()
            if bar.maximum():
                assert bar.isVisible()
                bar.setValue(bar.maximum())
                assert bar.value() > 0
            for widget, html, name, external, selected_font in originals:
                previous[0] = None
                qtbot.waitUntil(settled)
                scroll.ensureWidgetVisible(widget, 0, 0)
                frozen = (widget.property('geometry').getRect(), widget.mapTo(panel, QPoint()).toTuple(),
                          widget.font().toString(), bar.value(), scroll.verticalScrollBar().value())
                qapp.processEvents()
                assert frozen == (widget.property('geometry').getRect(), widget.mapTo(panel, QPoint()).toTuple(),
                                  widget.font().toString(), bar.value(), scroll.verticalScrollBar().value())
                assert not panel._news_title_fit._timer.isActive()
                assert (widget.text(), widget.accessibleName(), widget.openExternalLinks(), widget.font().toString()) == (html, name, external, selected_font)
                assert QFontInfo(widget.font()).family() == 'Open Sans'
                if widget.wordWrap():
                    document = QTextDocument()
                    document.setDocumentMargin(0)
                    document.setDefaultFont(widget.font())
                    option = QTextOption()
                    option.setWrapMode(QTextOption.WordWrap)
                    option.setAlignment(widget.alignment())
                    document.setDefaultTextOption(option)
                    document.setHtml(widget.text())
                    document.setTextWidth(widget.contentsRect().width())
                    document.size()
                    assert widget.margin() == widget.frameWidth() == 0 and widget.indent() < 0
                    block = document.begin()
                    layout = block.layout()
                    assert layout.lineCount() > 0
                    origin = widget.contentsRect().topLeft()
                    dy = max(0., (widget.contentsRect().height()-document.size().height())/2)
                    rects = [layout.lineAt(i).naturalTextRect().translated(
                        document.documentLayout().blockBoundingRect(block).topLeft()).translated(origin.x(), origin.y()+dy).toAlignedRect()
                        for i in range(layout.lineCount())]
                else:
                    rects = [widget.style().itemTextRect(widget.fontMetrics(), widget.contentsRect(),
                        int(widget.alignment() | Qt.TextSingleLine), widget.isEnabled(), widget.text())]
                for rect in rects:
                    assert widget.contentsRect().contains(rect), (width, name, rect)
                    assert QRegion(rect).subtracted(widget.visibleRegion()).isEmpty(), (width, name, rect, widget.visibleRegion())
                    assert panel.rect().contains(rect.translated(widget.mapTo(panel, QPoint()))), (width, name, rect)
            assert [block.layout().itemAt(2).widget().text() for block in blocks] == body_html
            heights.append(long_title.property('minimumSize').height())
        assert heights[0] == heights[2] and heights[0] > heights[1], heights
        before = len(calls)
        for _ in range(8):
            qapp.processEvents()
        assert len(calls) == before
        old_headlines = list(panel._headline_labels)
        panel._fill()
        title = panel._notes_column.itemAt(1).widget().layout().itemAt(0).widget()
        previous[0] = None
        qtbot.waitUntil(settled)
        assert len(panel._headline_labels) == 34 and not set(old_headlines).intersection(panel._headline_labels)
        assert all(w.property('maximumSize').width() <= scroll.viewport().property('size').width()
                   for w in panel._headline_labels if w.wordWrap())
        timer = panel._news_title_fit._timer
        timer.start(0)
        before = len(calls)
        panel.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not shiboken6.isValid(timer)
        qapp.processEvents()
        assert len(calls) == before
        panel = None
    finally:
        try:
            if panel is not None:
                panel.close()
        finally:
            preferences.set_gui_scale(saved_gui)
            preferences.set_font_scale(saved_font)
            gui_scale.set_gui_scale_live(live_gui)
            qapp.setStyleSheet(sheet)
            qapp.setFont(font)
            qapp.setPalette(palette)


def test_news_headings_follow_normal_mainwindow_live_scale_rebuilds(qtbot, qapp, monkeypatch):
    """Measure normal Home rebuilds, not a surrogate font or retained old panel."""
    from math import ceil
    import shiboken6
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, QPointF, QSize, Qt
    from PySide6.QtGui import QFontInfo, QRegion, QTextDocument, QTextOption
    from PySide6.QtWidgets import QScrollArea
    from spacr.qt import app, gui_scale, preferences, theme
    from spacr.qt.widgets import home
    from spacr.qt.widgets.collapsible_splitter import CollapsibleSplitter

    calls = []
    original_fit = home._NewsTitleFit._fit

    def observed_fit(owner):
        calls.append(owner)
        return original_fit(owner)

    monkeypatch.setattr(home._NewsTitleFit, '_fit', observed_fit)
    monkeypatch.setattr(home.HomePage, '_install_ambient', lambda self: None)
    monkeypatch.setattr(home.HomePage, '_read_the_journal', lambda self: None)
    monkeypatch.setattr(home.NewsPanel, '_ask_for_newer_releases', lambda self: None)
    saved_gui, saved_font = preferences.get_gui_scale(), preferences.get_font_scale()
    live_gui = gui_scale.current_scale()
    old_sheet, old_font, old_palette = qapp.styleSheet(), qapp.font(), qapp.palette()
    window = None
    retired = []
    source_baseline = None
    measurements = []
    try:
        app._load_bundled_fonts()
        app._use_open_sans(qapp)
        gui_scale.install_scaling_layer()
        preferences.set_gui_scale(.9)
        preferences.set_font_scale(.9)
        gui_scale.set_gui_scale_live(.9)
        preferences.apply_preferences_to_app()
        window = app.MainWindow()
        qtbot.addWidget(window)
        gui_scale._resize_actual(window, QSize(1200, 640))
        window.show()
        qtbot.waitExposed(window)
        dock_preferences = (preferences.get_dock_mode(), preferences.get_dock_width())
        assert window._dock_edge.isVisible() and not window._dock_edge.is_collapsed()
        qtbot.mouseClick(window._dock_edge, Qt.LeftButton)
        assert window._dock_edge.isVisible() and window._dock_edge.is_collapsed()
        assert (preferences.get_dock_mode(), preferences.get_dock_width()) == dock_preferences

        def snapshot(page, panel, headings, require_idle=True):
            fit_types = (home._NewsTitleFit, home._HomeBodyFit, home._HeroTextFit, home._AsideFloorFit)
            for owner in page.findChildren(QObject):
                if require_idle and isinstance(owner, fit_types) and (owner._busy or owner._timer.isActive()):
                    return None
                if require_idle and isinstance(owner, CollapsibleSplitter) and any(getattr(owner, name, False)
                        for name in ('_rebalance_queued', '_limits_queued', '_refit_queued')):
                    return None
            widgets = [panel, panel._notes, panel._notes.viewport(), panel._notes.widget(), *headings]
            result = []
            for widget in widgets:
                result.append((tuple(shiboken6.getCppPointer(widget)), widget.property('size').toTuple(),
                    widget.mapTo(window, QPoint()).toTuple(), widget.contentsRect().getRect(),
                    widget.property('minimumSize').toTuple(), widget.property('maximumSize').toTuple(),
                    widget.font().toString()))
            result.append((panel._notes.horizontalScrollBar().value(), panel._notes.verticalScrollBar().value()))
            return tuple(result)

        def wait_idle(page, panel, headings):
            previous = [None]

            def stable():
                current = snapshot(page, panel, headings)
                same = current is not None and current == previous[0]
                previous[0] = current
                return same

            qtbot.waitUntil(stable)

        def document_for(label):
            assert label.margin() == label.frameWidth() == 0 and label.indent() < 0
            document = QTextDocument()
            document.setDocumentMargin(0)
            document.setDefaultFont(label.font())
            option = QTextOption()
            option.setWrapMode(QTextOption.WordWrap)
            option.setAlignment(label.alignment())
            document.setDefaultTextOption(option)
            document.setHtml(label.text())
            document.setTextWidth(label.contentsRect().width())
            document.size()
            return document

        for step, scale in enumerate((.9, 2., .9, 2.)):
            if step:
                old_page = window._startup
                old_panel = old_page._news
                old_timer = old_panel._news_title_fit._timer
                retired.append((old_page, old_panel, old_timer))
                gui_scale.change_scales(window, gui=scale, font=scale, ask=False)
                assert window._startup is not old_page and window._startup._news is not old_panel
                qtbot.waitUntil(lambda: not shiboken6.isValid(old_timer))
            page, panel = window._startup, window._startup._news
            assert window._dock_edge.isVisible() and window._dock_edge.is_collapsed()
            assert (preferences.get_dock_mode(), preferences.get_dock_width()) == dock_preferences
            panel._older.setChecked(True)
            qtbot.waitUntil(lambda: not panel._pending_releases)
            blocks = [panel._notes_column.itemAt(i).widget() for i in range(panel._notes_column.count())
                if panel._notes_column.itemAt(i).widget() not in (None, panel._placeholder)]
            assert len(blocks) == len(panel._releases) == 17
            headings = [block.layout().itemAt(index).widget() for block in blocks for index in (0, 1)]
            assert len(headings) == 34
            wait_idle(page, panel, headings)
            actual_sources = []
            expected_fonts = (gui_scale.scale_int(theme.font_px(12, scale), scale),
                              gui_scale.scale_int(theme.font_px(10, scale), scale))
            metrics = []
            for block, entry in zip(blocks, panel._releases):
                widgets = [block.layout().itemAt(index).widget() for index in range(3)]
                actual_sources.append((entry['tag'], tuple((widget.text(), widget.accessibleName(),
                    widget.accessibleDescription(), widget.openExternalLinks(), widget.textInteractionFlags().value)
                    for widget in widgets)))
                for index, label in enumerate(widgets[:2]):
                    assert QFontInfo(label.font()).family() == 'Open Sans'
                    assert label.font().pixelSize() == expected_fonts[index]
                title = widgets[0]
                viewport_width = panel._notes.viewport().property('size').width()
                document = document_for(title)
                expected_height = ceil(document.size().height()
                    + title.property('size').height()-title.contentsRect().height())
                actual_min = title.property('minimumSize')
                actual_max = title.property('maximumSize')
                assert actual_min.width() == actual_max.width() == viewport_width
                assert title._gs_geo_set['minw'] == title._gs_geo_set['maxw'] == viewport_width
                assert actual_min.height() == title._gs_geo_set['minh'] == expected_height
                metrics.append((entry['tag'], viewport_width, title.font().toString(), expected_height))
            if source_baseline is None:
                source_baseline = actual_sources
            assert actual_sources == source_baseline
            measurements.append(metrics)

            for label in (blocks[-1].layout().itemAt(0).widget(), blocks[0].layout().itemAt(1).widget()):
                wait_idle(page, panel, headings)
                ancestor = label.parentWidget()
                while ancestor is not None and ancestor is not window:
                    if isinstance(ancestor, QScrollArea):
                        ancestor.ensureWidgetVisible(label, 0, 0)
                    ancestor = ancestor.parentWidget()
                frozen = snapshot(page, panel, headings, require_idle=False)

                def unchanged_then_idle():
                    assert snapshot(page, panel, headings, require_idle=False) == frozen
                    return snapshot(page, panel, headings) is not None

                qapp.processEvents()
                qtbot.waitUntil(unchanged_then_idle)
                if label.wordWrap():
                    document = document_for(label)
                    area = label.contentsRect()
                    origin = QPointF(area.x(), area.y()+max(0., (area.height()-document.size().height())/2))
                    rects = []
                    block = document.begin()
                    while block.isValid():
                        offset = document.documentLayout().blockBoundingRect(block).topLeft()+origin
                        layout = block.layout()
                        for index in range(layout.lineCount()):
                            line = layout.lineAt(index)
                            rects.append(line.naturalTextRect().translated(offset).toAlignedRect())
                            rects.extend(run.boundingRect().translated(offset).toAlignedRect() for run in line.glyphRuns())
                        block = block.next()
                else:
                    rects = [label.style().itemTextRect(label.fontMetrics(), label.contentsRect(),
                        int(label.alignment() | Qt.TextSingleLine), label.isEnabled(), label.text())]
                assert rects
                for rect in rects:
                    assert label.contentsRect().contains(rect), (step, label.text(), rect)
                    assert QRegion(rect).subtracted(label.visibleRegion()).isEmpty(), (step, label.text(), rect)
                    assert window.rect().contains(rect.translated(label.mapTo(window, QPoint())))
            wait_idle(page, panel, headings)
            count = len(calls)
            frozen = snapshot(page, panel, headings)
            for _ in range(8):
                qapp.processEvents()
            assert len(calls) == count and snapshot(page, panel, headings) == frozen
            print('NEWS_NORMAL_LIVE_SCALE', {'step': step, 'scale': scale,
                'background_only_isolation': ['ambient', 'journal', 'news-refresh'],
                'viewport': panel._notes.viewport().property('size').toTuple(), 'title_metrics': metrics}, flush=True)

        for before, after in ((measurements[0], measurements[2]), (measurements[1], measurements[3])):
            for old, new in zip(before, after):
                assert old[0] == new[0]
                if old[1:3] == new[1:3]:
                    assert old[3] == new[3]
        assert any(small[3] < large[3] for small, large in zip(measurements[2], measurements[3]))
        assert all(not shiboken6.isValid(timer) for _, _, timer in retired)
        timer = window._startup._news._news_title_fit._timer
        assert window.close()
        timer.start(0)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not shiboken6.isValid(timer)
        count = len(calls)
        qapp.processEvents()
        assert len(calls) == count
        window = None
    finally:
        try:
            if window is not None:
                window.close()
                window.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        finally:
            preferences.set_gui_scale(saved_gui)
            preferences.set_font_scale(saved_font)
            gui_scale.set_gui_scale_live(live_gui)
            qapp.setStyleSheet(old_sheet)
            qapp.setFont(old_font)
            qapp.setPalette(old_palette)


# -- the panel ---------------------------------------------------------------

def _following_release(version: tuple) -> tuple:
    """The release numbered after ``version`` in spaCR's scheme, where each
    of the last three places counts 0-9 and carries (1.5.0.9 -> 1.5.1.0)."""
    major, *places = version
    count = int("".join(str(place) for place in places)) + 1
    digits = str(count).zfill(len(places))
    if len(digits) > len(places):
        return (major + 1,) + (0,) * len(places)
    return (major,) + tuple(int(digit) for digit in digits)


def test_following_release_counts_like_the_tags():
    assert _following_release((1, 5, 1, 0)) == (1, 5, 1, 1)
    assert _following_release((1, 5, 0, 9)) == (1, 5, 1, 0)
    assert _following_release((1, 4, 9, 9)) == (1, 5, 0, 0)
    assert _following_release((1, 9, 9, 9)) == (2, 0, 0, 0)


def test_the_bundle_is_current_through_the_running_version(news):
    """The resource itself must be rebuilt, not only refreshable.

    The stale file is half the defect: 1.5.1.0 shipped with a bundle that
    stopped at 1.5.0.7. The bundle may lag the version in ``setup.py`` by
    exactly one release and no more: the note for version X is written when
    X is published, after the commit that is tagged X has passed CI, so that
    commit can only carry the release before it (see
    ``tools/build_release_notes.py``). A bundle two or more releases behind
    fails here, which is the reminder to run ``tools/build_release_notes.py``
    -- and the sign that the release workflow's ``release-notes`` job did
    not run.
    """
    from spacr import __version__

    releases = news.read_releases()
    assert releases, "no bundled release notes at all"
    newest = NewsPanel._newest_first(releases[0])[1]
    running = tuple(int(part) for part in __version__.split(".")
                    if part.isdigit())
    running = running + (0,) * (4 - len(running))
    assert newest >= running or _following_release(newest) == running, (
        f"the bundle stops at {releases[0]['tag']} on a {__version__} "
        f"build; run tools/build_release_notes.py")


def test_a_newer_release_than_the_bundle_is_shown_first(news):
    """The fetched release wins the top of the list, links and all."""
    bundled = news.releases
    news.apply_releases([NEWER])
    assert news.releases[0]["tag"] == NEWER["tag"]
    assert news.releases[1:] == bundled, (
        "merging a newer release reordered or dropped the bundled ones")
    labels = _titles(news)
    assert any(NEWER["name"] in text for text in labels)
    assert any(NEWER["url"] in text for text in labels), (
        "the new release's title is not a link to its own page")


def test_a_failed_or_empty_fetch_changes_nothing(news):
    """Offline, rate-limited and 'nothing newer' are one thing here."""
    before = news.releases
    drawn = _titles(news)
    for answer in ([], None, "rate limited", [{"not": "a release"}]):
        news.apply_releases(answer)
        assert news.releases == before, answer
    assert _titles(news) == drawn, "the panel was redrawn for nothing"


def test_the_fetched_copy_of_a_bundled_release_wins(news):
    """A note edited on GitHub after the release replaces the bundled one."""
    bundled = news.releases[0]
    edited = dict(bundled, body="A correction written after the release.")
    news.apply_releases([edited])
    assert news.releases[0]["body"] == edited["body"]
    assert len(news.releases) == len(bundled and news.releases)
    assert any("A correction written after the release." in text
               for text in _titles(news))


def test_the_merge_orders_by_date_then_by_version():
    """Two releases published the same day still land in version order."""
    same_day = [
        {"tag": "v1.5.0.5", "published": "2026-09-10"},
        {"tag": "v1.5.0.6", "published": "2026-09-10"},
        {"tag": "v1.5.0.4", "published": "2026-09-02"},
    ]
    merged = NewsPanel.merge_releases(same_day, [])
    assert [r["tag"] for r in merged] == ["v1.5.0.6", "v1.5.0.5", "v1.5.0.4"]


# -- and never before the page is drawn --------------------------------------

def test_building_the_panel_asks_for_nothing(monkeypatch, qtbot):
    """Construction reads the wheel. It does not open a socket.

    A panel that fetched in ``__init__`` would put a DNS lookup on the path
    to Home's first paint, which is the objection the bundled resource
    exists to answer.
    """
    import urllib.request

    def _refuse(*args, **kwargs):
        raise AssertionError("the News panel opened a socket while building")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    asked = []
    panel = NewsPanel("1.5.1.0")
    qtbot.addWidget(panel)
    panel.refresh_requested.connect(lambda: asked.append(True))
    assert panel.releases, "the bundled notes did not load"
    assert not asked


def test_the_refresh_is_asked_for_after_the_panel_is_shown(news, qtbot):
    """Once, on a later event-loop turn than the show, and only once."""
    asked = []
    news.refresh_requested.connect(lambda: asked.append(True))
    news.show()
    assert not asked, "the request was emitted inside showEvent"
    qtbot.waitUntil(lambda: bool(asked), timeout=2000)
    news.hide()
    news.show()
    qtbot.wait(50)
    assert asked == [True], "a second show asked again"


def test_home_relays_the_request_and_the_answer(qtbot):
    """The page carries the ask out and the release list back in."""
    from spacr.qt.app import make_home_page

    page = make_home_page()
    qtbot.addWidget(page)
    relayed = []
    page.news_refresh_requested.connect(lambda: relayed.append(True))
    page.news_panel.refresh_requested.emit()
    assert relayed
    page.apply_release_news([NEWER])
    assert page.news_panel.releases[0]["tag"] == NEWER["tag"]


# -- the window's side of it -------------------------------------------------

class _FakeWorker:
    """Stands in for ``_UpdateWorker`` so no thread is actually started."""

    started = []

    def __init__(self, operation, fn, parent=None):
        self.operation = operation
        self.fn = fn
        self.succeeded = SimpleNamespace(connect=lambda slot: None)
        self.failed = SimpleNamespace(connect=lambda slot: None)
        self.finished = SimpleNamespace(connect=lambda slot: None)

    def isRunning(self):                                         # noqa: N802
        return False

    def deleteLater(self):                                       # noqa: N802
        pass

    def start(self):
        _FakeWorker.started.append(self)


class _FakeWindow(SimpleNamespace):
    """Just the attributes ``_refresh_news`` reads off ``self``."""

    def __init__(self):
        super().__init__(_closing=False)

    def _on_news_ready(self, releases):
        self.delivered = releases

    def _on_news_failed(self, operation, details):
        self.failed = details


@pytest.fixture
def fake_worker(monkeypatch):
    """Replace the updater thread wrapper and hand back its log."""
    from spacr.qt import app as app_mod

    _FakeWorker.started = []
    monkeypatch.setattr(app_mod, "_UpdateWorker", _FakeWorker)
    return _FakeWorker.started


def test_the_refresh_runs_off_the_gui_thread(monkeypatch, fake_worker):
    """The window hands the fetch to a worker; it never calls it itself."""
    from spacr import updater
    from spacr.qt import app as app_mod
    from spacr.qt import preferences as prefs

    monkeypatch.delenv("SPACR_NO_NETWORK", raising=False)
    monkeypatch.setattr(prefs, "get_refresh_news", lambda: True)
    app_mod.MainWindow._refresh_news(_FakeWindow())
    assert len(fake_worker) == 1
    worker = fake_worker[0]
    assert worker.operation == "news"
    assert worker.fn is updater.fetch_release_notes, (
        "the fetch is not the callable the worker runs")


def test_the_preference_switches_the_refresh_off(monkeypatch, fake_worker):
    """Off means no worker, no request, and the bundled list untouched."""
    from spacr.qt import app as app_mod
    from spacr.qt import preferences as prefs

    monkeypatch.setattr(prefs, "get_refresh_news", lambda: False)
    app_mod.MainWindow._refresh_news(_FakeWindow())
    assert not fake_worker


@pytest.mark.parametrize("flag, starts", [
    ("1", False), ("yes", False), (" TRUE ", False),
    ("", True), ("0", True), ("false", True), ("no", True), ("OFF", True)])
def test_the_no_network_switch_starts_no_news_fetch(monkeypatch, fake_worker, flag, starts):
    """SPACR_NO_NETWORK forbids the fetch thread, read as crash reports read it.

    A lookup that cannot resolve has no deadline, so a fetch started anyway
    holds a closing window for the updater wait and parks its thread.
    """
    from spacr import crashreport, logging_util
    from spacr.qt import app as app_mod
    from spacr.qt import preferences as prefs

    monkeypatch.setattr(prefs, "get_refresh_news", lambda: True)
    monkeypatch.setattr(logging_util, "_in_safe_mode", lambda: False)
    monkeypatch.setenv("SPACR_NO_NETWORK", flag)
    app_mod.MainWindow._refresh_news(_FakeWindow())
    assert len(fake_worker) == (1 if starts else 0)
    assert crashreport.network_allowed() is starts


def test_the_preference_is_on_by_default_and_remembered(monkeypatch):
    """A fresh install refreshes; a reader who says no is obeyed."""
    from spacr.qt import preferences as prefs

    assert prefs.DEFAULT_REFRESH_NEWS is True
    assert prefs.get_refresh_news() is True
    try:
        prefs.set_refresh_news(False)
        assert prefs.get_refresh_news() is False
    finally:
        prefs.set_refresh_news(True)


# -- the fetch itself --------------------------------------------------------

def test_a_rate_limited_answer_is_empty_and_silent(monkeypatch, cache):
    """403 is the expected failure for an unauthenticated caller."""
    import urllib.error
    import urllib.request

    from spacr import updater

    def _rate_limited(*args, **kwargs):
        raise urllib.error.HTTPError(
            updater.GITHUB_RELEASES_API, 403, "rate limit exceeded", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", _rate_limited)
    assert updater.fetch_release_notes(timeout=0.1) == []
    assert cache.is_file(), (
        "a rate-limited attempt must still be stamped, or every launch "
        "behind this address asks again")


def test_an_offline_answer_is_empty_and_silent(monkeypatch, cache):
    """Anything at all may be raised; nothing at all escapes."""
    import urllib.request

    from spacr import updater

    def _offline(*args, **kwargs):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", _offline)
    assert updater.fetch_release_notes(timeout=0.1) == []


def test_a_successful_fetch_is_shaped_like_the_bundle(monkeypatch, cache):
    """Fetched and bundled records are interchangeable, by construction."""
    import urllib.request

    from spacr import updater

    payload = [
        {"tag_name": "v9.9.9.9", "name": "spaCR 9.9.9.9",
         "published_at": "2099-01-01T10:00:00Z",
         "html_url": "https://example.invalid/tag/v9.9.9.9",
         "body": "See https://example.invalid/compare", "prerelease": False},
        {"tag_name": "v9.9.9.8", "draft": True, "body": ""},
    ]
    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda *a, **k: _Response(json.dumps(payload).encode()))
    entries = updater.fetch_release_notes(timeout=0.1)
    assert [e["tag"] for e in entries] == ["v9.9.9.9"], (
        "a draft is not a release and must never be listed")
    assert set(entries[0]) == set(NEWER)
    assert entries[0]["published"] == "2099-01-01"
    assert entries[0]["links"] == ["https://example.invalid/compare"]


def test_the_cache_is_honoured_for_a_day(monkeypatch, cache):
    """A cached answer is used, and no request is made at all."""
    import urllib.request

    from spacr import updater

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"fetched": time.time(), "releases": [NEWER]}), encoding="utf-8")

    def _refuse(*args, **kwargs):
        raise AssertionError("a cached answer still went to the network")

    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    assert updater.fetch_release_notes(timeout=0.1) == [NEWER]


def test_a_cache_older_than_a_day_is_asked_again(monkeypatch, cache):
    """At most once a day is a floor as well as a ceiling."""
    import urllib.request

    from spacr import updater

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {"fetched": time.time() - updater.NEWS_MAX_AGE_S - 60,
         "releases": [NEWER]}), encoding="utf-8")
    asked = []

    def _answer(*args, **kwargs):
        asked.append(True)
        return _Response(b"[]")

    monkeypatch.setattr(urllib.request, "urlopen", _answer)
    assert updater.fetch_release_notes(timeout=0.1) == []
    assert asked


class _Response:
    """The two methods ``urlopen``'s context manager has to provide."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def test_older_releases_follow_the_first_paint(qtbot, monkeypatch):
    """Only the newest releases are sized before Home paints; none is lost.

    Sizing every wrapped release body cost about 0.4 s of Home's first paint
    on an Intel Mac. The rest are appended one event-loop turn after the
    panel paints, in order, and an explicit refill still draws them all.
    """
    monkeypatch.setattr(NewsPanel, '_ask_for_newer_releases', lambda self: None)
    panel = NewsPanel('1.5.1.0')
    qtbot.addWidget(panel)

    def drawn():
        return [panel._notes_column.itemAt(i).widget()
                for i in range(panel._notes_column.count())
                if panel._notes_column.itemAt(i).widget() not in (None, panel._placeholder)]

    total = len(panel._releases)
    assert total > NewsPanel.FIRST_RELEASES
    assert len(drawn()) == NewsPanel.FIRST_RELEASES
    panel.show()
    qtbot.waitExposed(panel)
    qtbot.waitUntil(lambda: len(drawn()) == total)
    names = [block.layout().itemAt(0).widget().accessibleName() for block in drawn()]
    assert names == [str(r.get('name') or r.get('tag') or 'spaCR') for r in panel._releases]
    assert panel._notes_column.itemAt(panel._notes_column.count() - 1).spacerItem()

    panel._fill()
    assert len(drawn()) == total and not panel._pending_releases


def test_long_notes_show_the_newest_release_and_scroll_inside_the_card(qtbot, monkeypatch):
    """1.5.1.5's notes wrap to ~21,000 px; the card stays capped and scrolls.

    Only the newest release is shown; the older ones wait behind the "Show
    older releases" switch, which sits under the notes box rather than in
    the scrolled column, so it is never scrolled away.
    """
    long_body = "\n".join(f"- line {index}: " + "a long release note sentence " * 6
                          for index in range(400))
    releases = [{"tag": f"v1.5.1.{n}", "name": f"spaCR 1.5.1.{n}", "published": f"2026-10-{20 - n:02d}",
                 "url": "", "body": long_body if n == 5 else f"notes {n}"} for n in (5, 4, 3)]
    monkeypatch.setattr(NewsPanel, "read_releases", staticmethod(lambda: [dict(r) for r in releases]))
    monkeypatch.setattr(NewsPanel, "_ask_for_newer_releases", lambda self: None)
    panel = NewsPanel("1.5.1.5")
    qtbot.addWidget(panel)
    panel.resize(420, 600)
    panel.show()
    qtbot.waitExposed(panel)
    qtbot.waitUntil(lambda: not panel._pending_releases)

    view = panel.notes_view
    blocks = [panel._notes_column.itemAt(i).widget() for i in range(panel._notes_column.count())
              if panel._notes_column.itemAt(i).widget() not in (None, panel._placeholder)]
    assert len(blocks) == 3
    assert blocks[0].isVisible() and not any(block.isVisible() for block in blocks[1:])
    assert panel._older.isVisible() and not view.widget().isAncestorOf(panel._older)
    assert panel._older.text() == "Show older releases"

    from spacr.qt.preferences import scaled_px
    assert view.height() <= scaled_px(NewsPanel.NOTES_H_MAX)
    assert view.widget().height() > 5 * view.height(), "the long notes scroll inside the card"
    assert view.verticalScrollBar().maximum() > 0
    assert panel.height() < 2 * scaled_px(NewsPanel.NOTES_H_MAX)

    panel._older.setChecked(True)
    assert all(block.isVisible() for block in blocks)
    panel._older.setChecked(False)
    assert not any(block.isVisible() for block in blocks[1:])
