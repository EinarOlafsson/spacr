"""The GUI thread must never wait for the model catalogue.

WHAT THIS IS ABOUT. ``spacr.settings.downloaded_zoo_models`` is called while
a settings panel is built -- inside ``MainWindow._on_nav_selected`` -- and it
asks ``model_zoo.catalogue(remote=True)``, which asked
``model_zoo.shared_catalogue`` for the community rows, which called
``urllib.request.urlopen`` with a thirty-second timeout.

Measured 2026-09-05 on the maintainer's machine with the catalogue host made
non-routable (``https://10.255.255.1/…`` -- the shape of a down VPN or a
captive portal, where the connect neither completes nor is refused): opening
the Mask module took **32.2 seconds**, every millisecond of it a GUI thread
parked in a socket. GNOME asks a window whether it is alive after five
seconds, so what the user saw, and reported, was spaCR's "force quit" dialog.
After the fix the same open is 2.2 s.

The tests below are black-box: they do not care HOW the answer is reached,
only that nothing on the GUI thread waits for a socket.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from spacr import model_zoo, settings


def _settle_background_fetches() -> None:
    """Let any daemon refresh finish, so the recording is complete.

    Without this the assertions could pass because a thread had not started
    yet rather than because it was the right thread.
    """
    for thread in list(threading.enumerate()):
        if thread.name == "spacr-model-catalogue":
            thread.join(timeout=5.0)


@pytest.fixture
def urlopen_recorder(monkeypatch):
    """Replace ``urlopen`` with something that records its caller's thread.

    It raises rather than returning a payload: a caller that is allowed to
    fetch is not the subject here, and an exception keeps the fake from
    having to imitate a JSON response.
    """
    threads = []

    def fake_urlopen(*_args, **_kwargs):
        threads.append(threading.current_thread())
        raise OSError("no network in this test")

    # A refresh started by an EARLIER test may still be in flight, and the
    # "one fetch at a time" flag would then make this test's call a no-op --
    # which fails as "nothing fetched at all" and only when the files run in
    # that order. Settle and clear before recording anything.
    _settle_background_fetches()
    model_zoo._SHARED_CATALOGUE_FETCHING.clear()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(model_zoo._SHARED_CATALOGUE_CACHE, "fetched_at", 0.0)
    monkeypatch.setitem(model_zoo._SHARED_CATALOGUE_CACHE, "entries", ())
    yield threads
    _settle_background_fetches()


def test_the_shared_catalogue_is_never_fetched_on_the_gui_thread(
        qapp, urlopen_recorder):
    """`shared_catalogue` off the GUI thread's clock, whatever it was asked."""
    # No `block` argument -- which is what every caller on the module-open
    # path had, and what the next one to be written will have.
    model_zoo.shared_catalogue(force=True)
    _settle_background_fetches()

    assert urlopen_recorder, (
        "nothing fetched at all -- the fetch must still happen, just not here")
    assert threading.main_thread() not in urlopen_recorder, (
        "the catalogue was fetched on the GUI thread; an unreachable host "
        "then freezes the window for the full timeout")


def test_building_a_settings_panel_makes_no_synchronous_request(
        qapp, urlopen_recorder):
    """The call the module-open path actually makes."""
    settings.downloaded_zoo_models()
    _settle_background_fetches()

    assert threading.main_thread() not in urlopen_recorder, (
        "opening a module fetched the model catalogue on the GUI thread")


def test_a_worker_thread_may_still_wait_for_it(qapp, urlopen_recorder):
    """The fix must not turn the fetch off, only move it somewhere it is
    allowed to take as long as it takes."""
    caller = []

    def job():
        caller.append(threading.current_thread())
        model_zoo.shared_catalogue(block=True, force=True)

    worker = threading.Thread(target=job, name="test-worker")
    worker.start()
    worker.join(timeout=5.0)

    assert caller and caller[0] in urlopen_recorder, (
        "a caller that is allowed to wait must do the fetch itself, in its "
        "own thread, rather than hand it to a third one and return stale")
