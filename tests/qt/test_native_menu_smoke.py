"""Local state-machine guards; genuine Cocoa acceptance belongs to native CI."""
from types import SimpleNamespace

import pytest

from spacr.qt.startup_benchmark import _DistributionSmokeController as Controller


def test_cocoa_probe_cannot_be_passed_by_an_offscreen_qt_platform(monkeypatch):
    import sys

    monkeypatch.setattr(sys, 'platform', 'darwin')
    fake = SimpleNamespace(app=SimpleNamespace(platformName=lambda: 'offscreen'))
    with pytest.raises(RuntimeError, match='requires macOS Cocoa'):
        Controller._start_native_menu_check(fake)


def test_cocoa_probe_requires_the_real_native_menu_bar(monkeypatch):
    import sys

    monkeypatch.setattr(sys, 'platform', 'darwin')
    fake = SimpleNamespace(app=SimpleNamespace(platformName=lambda: 'cocoa'),
        window=SimpleNamespace(menuBar=lambda: SimpleNamespace(isNativeMenuBar=lambda: False)))
    with pytest.raises(RuntimeError, match='not using its native menu bar'):
        Controller._start_native_menu_check(fake)


@pytest.mark.parametrize('missing', [None, 'preferences_opened', 'preferences_closed', 'quit_dispatched'])
def test_only_a_complete_native_action_sequence_can_accept_application_quit(missing):
    menu = dict(preferences_opened=True, preferences_closed=True,
                quit_dispatched=True, quit_observed=False)
    if missing:
        menu[missing] = False
    events = []
    fake = SimpleNamespace(phase='native-menu-quitting', record={'native_menu': menu},
        timer=SimpleNamespace(stop=lambda: events.append('timer stopped')),
        _write=lambda: events.append('receipt written'))
    Controller._quitting(fake)
    assert fake.record['status'] == ('failed' if missing else 'passed')
    assert menu['quit_observed'] is (missing is None)
    assert ('timer stopped' in events) is (missing is None)
    assert events[-1] == 'receipt written'


def test_a_native_action_that_does_not_open_preferences_is_rejected():
    calls = []
    fake = SimpleNamespace(phase='native-menu-opening', _native_menu=123,
        _native_actions={'preferences': 2},
        _cocoa_message=lambda *args: calls.append(args),
        _pipeline_failed=lambda reason: calls.append(reason))
    Controller._invoke_native_preferences(fake)
    assert calls[0][1] == 'performActionForItemAtIndex:'
    assert 'did not open the verified dialog' in calls[1]
