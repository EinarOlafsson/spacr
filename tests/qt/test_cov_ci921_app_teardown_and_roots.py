"""Main-window teardown helper and stylesheet roots for a bare window.

Covers ``_tear_down_before_finalization`` (it asks the backdrop to retire its
shading threads and only logs when that fails) and
``MainWindow.stylesheet_roots`` returning the window itself when it has no
central widget or no page stack yet.
"""
import logging

import pytest

pytest.importorskip("PySide6")

from spacr.qt import app
from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


def test_teardown_asks_the_backdrop_to_retire_its_threads(monkeypatch):
    calls = []
    monkeypatch.setattr(ambient, "_retire_shading_threads",
                        lambda *a, **k: calls.append((a, k)) or 0)
    app._tear_down_before_finalization()
    assert calls == [((), {})]


def test_teardown_logs_and_does_not_raise_when_retiring_fails(
        monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("driver gone")

    monkeypatch.setattr(ambient, "_retire_shading_threads", boom)
    with caplog.at_level(logging.DEBUG, logger=app.LOG.name):
        app._tear_down_before_finalization()
    assert any("could not stop the backdrop shading threads"
               in r.getMessage() for r in caplog.records)


class _Bare:
    def __init__(self, central, **attrs):
        self._central = central
        self.__dict__.update(attrs)

    def centralWidget(self):
        return self._central


def test_a_window_without_a_central_widget_is_its_own_style_root():
    bare = _Bare(None, _stack=object())
    assert app.MainWindow.stylesheet_roots(bare) == [bare]


def test_a_window_without_a_page_stack_is_its_own_style_root():
    bare = _Bare(object())
    assert app.MainWindow.stylesheet_roots(bare) == [bare]
