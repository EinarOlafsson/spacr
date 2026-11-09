"""The held field's process-local frame cap, without constructing GPU widgets.

``_set_capture_held_fps`` lets a capture profile raise the held field's
frame cap before the application starts. Its callers moved out of this
repository with the tutorials (item 756), so these tests keep its contract:
the default cap, live and later widgets following a request, other caps
staying authoritative, and invalid requests leaving the state alone.
"""
from types import SimpleNamespace

import pytest

from spacr.qt.widgets import ambient as a


class Field:
    """The rate methods of AmbientWidget on a stand-in with no GPU state."""

    _rate = a.AmbientWidget._rate
    _apply_rate = a.AmbientWidget._apply_rate

    def __init__(self, fps=60, held=True, run=False):
        self._fps, self._hold_paced, self._run_paced = fps, held, run
        self._exact_fps = None
        self.timer_rates, self.producer_rates = [], []
        self._timer = SimpleNamespace(setInterval=self.timer_rates.append)
        self._producer_box = [SimpleNamespace(set_fps=self.producer_rates.append)]


@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(a, '_CAPTURE_HELD_FPS', None)
    monkeypatch.setattr(a, '_live_ambient_widgets', lambda: [])


def test_the_default_held_cap_is_twenty():
    assert Field()._rate() == 20


def test_a_request_reaches_live_and_later_fields_and_none_restores(monkeypatch):
    old = Field()
    monkeypatch.setattr(a, '_live_ambient_widgets', lambda: [old])
    a._set_capture_held_fps(30)
    assert old._rate() == 30
    assert old.timer_rates == [33]
    assert old.producer_rates == [30]
    assert Field()._rate() == 30
    a._set_capture_held_fps(None)
    assert old._rate() == Field()._rate() == 20
    assert old.producer_rates == [30, 20]


@pytest.mark.parametrize('fps,held,run', [(15, True, False), (60, False, False), (60, True, True)])
def test_other_caps_remain_authoritative(fps, held, run):
    a._set_capture_held_fps(30)
    field = Field(fps, held, run)
    expected = min(fps, a._RUN_FPS) if run else fps
    if held:
        expected = min(expected, 30)
    assert field._rate() == expected
    field._apply_rate()
    assert field.producer_rates == [expected]


@pytest.mark.parametrize('bad', [True, 30.0, '30', 0, -1, a.MAX_FPS + 1])
def test_an_invalid_request_does_not_change_state(bad):
    with pytest.raises(ValueError):
        a._set_capture_held_fps(bad)
    assert a._CAPTURE_HELD_FPS is None
