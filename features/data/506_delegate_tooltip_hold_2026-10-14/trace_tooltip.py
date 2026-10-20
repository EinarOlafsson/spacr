import pytest

@pytest.fixture(autouse=True)
def trace_policy(monkeypatch):
    from spacr.qt import tooltip_policy as policy
    from PySide6.QtWidgets import QToolTip
    for name in ('_on_enter', '_show_now', '_replay', 'hide_now', '_start_the_linger'):
        original = getattr(policy._TooltipFilter, name)
        def traced(self, *args, _name=name, _original=original, **kwargs):
            print('TRACE', _name, 'ARGS', args, 'TEXT', self._text, 'WIDGET', self._widget, 'VISIBLE', QToolTip.isVisible(), 'TIMER', self._show_timer.remainingTime(), flush=True)
            answer = _original(self, *args, **kwargs)
            print('TRACE END', _name, 'TEXT', self._text, 'WIDGET', self._widget, 'VISIBLE', QToolTip.isVisible(), 'TIMER', self._show_timer.remainingTime(), flush=True)
            return answer
        monkeypatch.setattr(policy._TooltipFilter, name, traced)
