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
    """Point the news cache at a throwaway directory."""
    from spacr import updater

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
