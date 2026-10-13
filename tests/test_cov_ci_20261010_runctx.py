"""Deferred overload retry drain when the retry re-raises the original error."""
from __future__ import annotations

from spacr.runctx import _DeferredOverloadRetries


def test_retry_that_raises_the_original_error_keeps_its_cause_untouched():
    original = MemoryError('cannot allocate memory')
    marker = ValueError('earlier cause')
    original.__cause__ = marker
    queue = _DeferredOverloadRetries()

    def again():
        raise original

    assert queue.defer('unit-1', original, again) is True
    outcomes = list(queue.drain())
    assert len(outcomes) == 1
    identity, result, error = outcomes[0]
    assert identity == 'unit-1'
    assert result is None
    assert error is original
    assert error.__cause__ is marker
    assert list(queue.drain()) == []
