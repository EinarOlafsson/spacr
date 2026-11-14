"""Leaving a cancellation scope never raises, even after the token is gone.

``installed_token`` restores the thread's previous token on exit. When there
was no previous token it deletes the attribute instead -- and the attribute
may already be gone, because worker code inside the block can clear the
thread-local itself. A failure there would replace the pipeline's real
outcome, success or cancellation, with an AttributeError from the cleanup.
"""
from __future__ import annotations

from spacr.cancellation import (
    CancellationToken,
    current_token,
    installed_token,
)
import spacr.cancellation as C


def test_a_scope_that_lost_its_token_still_closes_cleanly():
    """Clearing the thread-local inside the block must not break the exit."""
    token = CancellationToken()
    assert current_token() is None, "the test thread starts with no token"

    with installed_token(token) as installed:
        assert installed is token
        assert current_token() is token
        # Whatever ran inside the scope already cleaned the thread-local up.
        delattr(C._LOCAL, "token")

    assert current_token() is None


def test_a_nested_scope_restores_the_outer_token():
    """The non-deleting branch is the one that keeps an outer run cancellable."""
    outer, inner = CancellationToken(), CancellationToken()
    with installed_token(outer):
        with installed_token(inner):
            assert current_token() is inner
        assert current_token() is outer
    assert current_token() is None
