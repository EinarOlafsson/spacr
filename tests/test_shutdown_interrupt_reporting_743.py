"""A user interrupt remains visible without becoming an automatic bug report."""
import io

import pytest

from spacr import crashreport as cr
from spacr.logging_util import _ErrorStream
from tests.test_700_structured_error_dispatch import configured as configured
from tests.test_700_structured_error_dispatch import joined


@pytest.mark.parametrize('exception', [KeyboardInterrupt(), KeyboardInterrupt('requested stop')])
def test_actual_keyboard_interrupt_is_not_explained_or_reported(configured, exception):
    relay, outcomes, _ready = configured
    assert cr._dispatch_error(exception=exception, origin='direct') is None
    assert relay.calls == outcomes == []
    assert cr.queued_summaries() == []


def test_atexit_keyboard_interrupt_traceback_stays_on_stderr_without_reporting(configured):
    relay, outcomes, _ready = configured
    text = ('Exception ignored in atexit callback: <function drain at 0x123>\n'
            'Traceback (most recent call last):\n'
            '  File "/private/source/bridge.py", line 1047, in _drain_parked_threads_at_exit\n'
            '    still_running = wait_for_parked_threads()\n'
            '  File "/private/source/bridge.py", line 1039, in wait_for_parked_threads\n'
            '    thread.wait(int(remaining * 1000))\n'
            'KeyboardInterrupt: \n')
    output = io.StringIO()
    stream = _ErrorStream(output)
    for line in text.splitlines(keepends=True):
        stream.write(line)
    stream.flush()
    assert output.getvalue() == text
    assert relay.calls == outcomes == []
    assert cr.queued_summaries() == []


@pytest.mark.parametrize('exception,text', [
    (ValueError('KeyboardInterrupt is mentioned in this failure'), ''),
    (None, 'ERROR: failed to process a message mentioning KeyboardInterrupt'),
    (None, 'Traceback (most recent call last):\n'
           '  File "worker.py", line 3, in work\n'
           'ValueError: KeyboardInterrupt is mentioned in this failure\n'),
])
def test_ordinary_errors_mentioning_keyboard_interrupt_are_still_reported(configured, exception, text):
    relay, outcomes, _ready = configured
    joined(cr._dispatch_error(text, exception=exception, origin='terminal'))
    assert len(relay.calls_to('/diagnose')) == len(relay.calls_to('/report')) == 1
    assert len(outcomes) == 1
