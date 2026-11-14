"""A resolution worker that raises must not kill the chaining strip.

`JobRunner` never calls `done` for a worker that raised, so `_resolving`
used to stay True for the session and every later refresh only set
`_resolve_again`.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


def _bar(qtbot, monkeypatch, tmp_path, resolve):
    from PySide6.QtWidgets import QWidget

    import spacr.chaining as chaining
    from spacr.qt.chaining import ChainingBar

    monkeypatch.setattr(chaining, "resolve_settings", resolve)
    monkeypatch.setattr(chaining, "staleness_notes", lambda *_a, **_k: ())
    host = QWidget()
    host.app_key = "measure"
    qtbot.addWidget(host)
    pins = chaining.PinStore(str(tmp_path / "pins.json"))
    return ChainingBar(host, pins=pins)


def _idle(bar):
    return (not bar._resolving and not bar._resolver.is_busy()
            and bar._resolver.active_jobs() == 0)


def test_a_resolver_error_clears_the_flag_and_shows_a_message(
        qtbot, monkeypatch, tmp_path):
    def corrupt(*_a, **_k):
        raise RuntimeError("registry is corrupt")

    bar = _bar(qtbot, monkeypatch, tmp_path, corrupt)
    bar.refresh()
    qtbot.waitUntil(lambda: _idle(bar), timeout=5000)
    assert bar._resolving is False
    assert not bar._stale.isHidden()
    assert "registry is corrupt" in bar._stale.text()


def test_a_later_refresh_resolves_again_after_an_error(
        qtbot, monkeypatch, tmp_path):
    import spacr.chaining as chaining

    calls = []

    def flaky(module, settings, **_kwargs):
        calls.append(module)
        if len(calls) == 1:
            raise RuntimeError("registry is corrupt")
        return chaining.Resolution(module=module, settings=dict(settings))

    bar = _bar(qtbot, monkeypatch, tmp_path, flaky)
    bar.refresh()
    qtbot.waitUntil(lambda: _idle(bar) and len(calls) == 1, timeout=5000)
    bar.refresh()
    qtbot.waitUntil(lambda: _idle(bar) and len(calls) == 2, timeout=5000)
    assert bar._resolve_again is None


def test_a_refresh_held_during_a_failing_resolution_still_runs(
        qtbot, monkeypatch, tmp_path):
    import threading

    import spacr.chaining as chaining

    gate = threading.Event()
    calls = []

    def flaky(module, settings, **_kwargs):
        calls.append(module)
        if len(calls) == 1:
            gate.wait(5)
            raise RuntimeError("registry is corrupt")
        return chaining.Resolution(module=module, settings=dict(settings))

    bar = _bar(qtbot, monkeypatch, tmp_path, flaky)
    bar.refresh()
    bar.refresh()
    assert bar._resolve_again is not None
    gate.set()
    qtbot.waitUntil(lambda: _idle(bar) and len(calls) == 2, timeout=5000)
    assert bar._resolve_again is None
