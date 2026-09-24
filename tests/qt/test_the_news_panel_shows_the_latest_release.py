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


# -- the panel ---------------------------------------------------------------

def test_the_bundle_is_current_through_the_running_version(news):
    """The resource itself must be rebuilt, not only refreshable.

    The stale file is half the defect: 1.5.1.0 shipped with a bundle that
    stopped at 1.5.0.7. A build whose newest bundled release is older than
    the version in ``setup.py`` fails here, which is the reminder to run
    ``tools/build_release_notes.py`` -- and, from now on, the sign that the
    release workflow's ``release-notes`` job did not run.
    """
    from spacr import __version__

    releases = news.read_releases()
    assert releases, "no bundled release notes at all"
    newest = NewsPanel._newest_first(releases[0])[1]
    running = tuple(int(part) for part in __version__.split(".")
                    if part.isdigit())
    running = running + (0,) * (4 - len(running))
    assert newest >= running, (
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
