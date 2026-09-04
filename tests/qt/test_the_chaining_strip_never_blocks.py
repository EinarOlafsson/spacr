"""The chaining strip resolves the registry off the GUI thread.

THE FREEZE, 2026-09-04. `install_chaining` builds a `ChainingBar` during
screen construction and calls `refresh()` synchronously. That went:

    ChainingBar._refresh
      -> spacr.chaining.resolve_settings(roots=self.search_roots())
        -> chained_inputs -> _registry_for(root)
          -> os.path.isfile('<root>/artifacts.db')      for EVERY root
          -> _artifacts.open_registry -> sqlite3.connect on the same path

and `search_roots()` is the folders the user last worked in, read from
QSettings. One of the maintainer's was an ``autofs`` mount whose share was
asleep, and a single `os.path.isfile` on it had not returned after TWENTY
SECONDS. The whole interface was frozen on every module open. It was
reported as "opening map barcodes crashes spacr" and left no traceback,
because a stalled event loop is not a crash.

Found by a fan-out over `spacr/qt/`; the agent that verified it reproduced
the blocking stat on the main thread with the maintainer's own remembered
path seeded into QSettings.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

#: Longer than any human would call responsive, shorter than the twenty
#: seconds actually measured. A test that waited the real duration would be
#: a test nobody runs.
SLOW_S = 8.0


@pytest.fixture
def slow_registry(monkeypatch):
    """Make every registry probe take :data:`SLOW_S`, as a sleeping mount does."""
    import spacr.chaining as chaining

    def crawl(*_args, **_kwargs):
        time.sleep(SLOW_S)
        raise AssertionError("the GUI thread waited for the registry")

    monkeypatch.setattr(chaining, "resolve_settings", crawl)
    monkeypatch.setattr(chaining, "staleness_notes", crawl)
    return crawl


def _bar(qtbot, screen):
    """A strip parented to ``screen``.

    NOT registered with qtbot: it is a CHILD of a widget that already is, so
    the parent deletes it at teardown and a second close would be a close of
    a deleted object.
    """
    from spacr.qt.chaining import ChainingBar

    return ChainingBar(screen)


class _Screen:
    """The little a `ChainingBar` asks of the screen it sits on."""

    app_key = "map_barcodes"

    def __init__(self):
        self.applied = {}

    def collect(self):
        return {"src": "/somewhere"}

    def apply_settings_dict(self, values):
        self.applied.update(values)


def test_refresh_returns_before_the_registry_answers(qtbot, slow_registry,
                                                     qt_theme_applied):
    """The property the freeze violated: construction does not wait."""
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    screen = _Screen()
    bar = _bar(qtbot, host)
    bar._screen = screen
    bar.app_key = "map_barcodes"

    started = time.monotonic()
    bar.refresh()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"refresh() took {elapsed:.1f}s -- it is resolving the registry on "
        "the GUI thread again, which is the freeze")


def test_a_second_refresh_does_not_queue_another_resolution(
        qtbot, slow_registry, qt_theme_applied):
    """Every keystroke refreshes the strip; they must not each start a job.

    Coalesced rather than queued, because they all ask the same question of
    the same registry.
    """
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    bar = _bar(qtbot, host)
    bar._screen = _Screen()
    bar.app_key = "map_barcodes"

    bar.refresh()
    assert bar._resolving is True
    for _ in range(20):
        bar.refresh()
    assert bar._resolve_again is not None, (
        "a refresh during an in-flight one should be remembered, not queued")


def test_an_unreachable_root_is_skipped_rather_than_stat_ed(
        qtbot, monkeypatch, qt_theme_applied):
    """Belt and braces: `search_roots` never stats a root itself.

    `path_probe.isdir` answers False for a root it has not probed, which is
    the pessimistic direction and the right one -- skipping a root costs one
    refresh, and stating a sleeping mount costs the application.
    """
    from PySide6.QtWidgets import QWidget

    from spacr.qt import chaining as qt_chaining
    from spacr.qt import path_probe

    monkeypatch.setattr(qt_chaining._ports, "upstream_modules",
                        lambda _key: ())
    import spacr.qt.prefs as prefs
    monkeypatch.setattr(prefs, "get_last_source",
                        lambda _key: "/nas/asleep")
    monkeypatch.setattr(prefs, "get_recent_sources",
                        lambda _key, limit=4: ())

    def never(_path):
        time.sleep(SLOW_S)
        return True

    monkeypatch.setattr(path_probe.os.path, "isdir", never)
    path_probe.forget()

    host = QWidget()
    qtbot.addWidget(host)
    bar = _bar(qtbot, host)
    bar.app_key = "map_barcodes"

    started = time.monotonic()
    roots = bar.search_roots()
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"search_roots() took {elapsed:.1f}s -- it is stat-ing roots inline")
    assert "/nas/asleep" not in roots, (
        "an unprobed root was handed to the registry resolver")


def test_the_strip_still_paints_once_the_worker_lands(qtbot, qt_theme_applied):
    """Off the GUI thread is only correct if the answer still arrives."""
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    bar = _bar(qtbot, host)
    bar._screen = _Screen()
    bar.app_key = "map_barcodes"

    bar.refresh()
    qtbot.waitUntil(lambda: bar._resolving is False, timeout=15000)
