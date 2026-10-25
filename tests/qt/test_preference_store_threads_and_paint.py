"""The preference store: one per thread, none opened in a paint, Qt's type untouched.

The released 1.5.1.5 crashed now and again on macOS with a null function
pointer inside ``Sbk_QSettings_Init`` (2026-10-20), twice in about eight
launches, reached from a setup card's paint and from the jobs registry.
On macOS the store had no file to watch, so it was opened afresh for every
read, a paint included, and ``setValue``/``remove``/``clear`` were replaced
on the Shiboken ``QSettings`` type itself. Now the type is left alone, each
thread keeps one store and syncs it once a second, and a paint opens none.
"""
from __future__ import annotations

import inspect
import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt import preferences as prefs  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def macos_like(monkeypatch):
    """A store with no file to watch, as on macOS, and a counter of openings."""
    count = [0]

    class Counted(QSettings):
        def __init__(self, *args):
            count[0] += 1
            super().__init__(*args)

    monkeypatch.setattr(prefs, "QSettings", Counted)
    monkeypatch.setattr(prefs, "_STORE", threading.local())
    monkeypatch.setattr(prefs, "_REMEMBERED", {})
    monkeypatch.setattr(prefs, "_store_file_state", lambda store: None)
    now = [1000.0]
    monkeypatch.setattr(prefs, "_store_clock", lambda: now[0])
    return count, now


def test_the_shiboken_settings_type_is_never_patched():
    """The binding's own type, not the stand-in ``tests/conftest.py`` installs.

    The conftest replaces ``QSettings`` with a subclass that moves
    ``setPath`` into the test's folder, so ``QSettings.__dict__`` here is that
    subclass's and holds none of these methods.
    """
    shiboken_type = next(cls for cls in QSettings.__mro__
                         if cls.__module__ == "PySide6.QtCore"
                         and cls.__name__ == "QSettings")
    for name in ("setValue", "remove", "clear", "value", "sync"):
        method = shiboken_type.__dict__[name]
        assert not inspect.isfunction(method), f"QSettings.{name} was replaced"
        assert not getattr(method, "_spacr_counts_writes", False)


def test_a_write_through_the_store_is_counted_without_the_type(macos_like):
    store = prefs._real_store()
    before = prefs._STORE_GENERATION[0]
    store.setValue("tests/threads_and_paint", 1)
    assert prefs._STORE_GENERATION[0] != before
    after = prefs._STORE_GENERATION[0]
    store.remove("tests/threads_and_paint")
    assert prefs._STORE_GENERATION[0] != after
    plain = QSettings("spacr-tests", "unrelated")
    unrelated = prefs._STORE_GENERATION[0]
    plain.setValue("x", 1)
    assert prefs._STORE_GENERATION[0] == unrelated


def test_each_thread_opens_one_store_and_keeps_it(macos_like):
    openings, now = macos_like
    gui = prefs._real_store()
    for _ in range(50):
        assert prefs._real_store() is gui
        now[0] += 0.25
    assert openings[0] == 1
    seen = []

    def worker():
        stores = {id(prefs._real_store()) for _ in range(50)}
        seen.append((stores, prefs._real_store() is gui))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert openings[0] == 1 + len(threads)
    assert all(len(stores) == 1 and not shared for stores, shared in seen)
    assert prefs._real_store() is gui


def test_painting_a_setup_card_opens_no_store_across_many_seconds(qtbot, macos_like):
    from spacr.qt.widgets.setup_card import SetupCard

    openings, now = macos_like
    card = SetupCard()
    qtbot.addWidget(card)
    card.resize(360, 240)
    card.show()
    card.grab()
    before = openings[0]
    for _ in range(30):
        now[0] += 0.4
        card.grab()
    assert openings[0] == before <= 1


def test_the_rim_length_is_remembered_between_frames(macos_like, monkeypatch):
    prefs._rim_length_fraction()
    reads = [0]
    real = prefs._settings

    def counted():
        reads[0] += 1
        return real()

    monkeypatch.setattr(prefs, "_SETTINGS", counted)
    monkeypatch.setattr(prefs, "_settings", counted)
    for _ in range(10):
        prefs._rim_length_fraction()
    assert reads[0] == 0
