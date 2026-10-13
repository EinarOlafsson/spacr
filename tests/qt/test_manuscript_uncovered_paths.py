"""The exporter when provider discovery itself misbehaves.

Both cases here are about the same promise: ``generate_sections`` always
returns usable Methods and Results. A provider list that cannot be built,
and a provider that stops being configured between being offered and being
used, are answers the user reads — never tracebacks.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.methods_export import (build_digest, render_methods,  # noqa: E402
                                  render_results)
from spacr.qt.ai import manuscript                               # noqa: E402
from spacr.qt.ai.providers import ChatProvider                   # noqa: E402

pytestmark = pytest.mark.qt


class _StubProvider(ChatProvider):
    """A configured provider that would answer if it were ever asked."""

    name = "stub"
    label = "Stub provider"
    cli_name = "stub"
    install_hint = "nothing to install"
    login_command = "nothing to log in to"

    def __init__(self):
        super().__init__()
        self.asked = 0

    def is_installed(self) -> bool:
        return True

    def stream_chat(self, messages, system="", model=None):
        self.asked += 1
        yield "## Methods\n\nnothing here.\n"


@pytest.fixture
def digest():
    return build_digest(
        title="a small pilot",
        settings={"on_error": "stop", "random_seed": 11},
        extra={"n_genes": 42},
    )


def test_a_provider_list_that_cannot_be_built_still_names_the_next_step(
        monkeypatch):
    """The catalog is what turns "none configured" into instructions.

    Without it the message has to stand on its own, and it still must say
    that the sections were written from the digest rather than by a model.
    """
    monkeypatch.setattr(manuscript, "configured_providers", lambda: [])

    def _refuse():
        raise RuntimeError("the provider catalog could not be imported")

    monkeypatch.setattr(manuscript, "list_providers", _refuse)

    state = manuscript.availability()

    assert bool(state) is False
    assert state.providers == ()
    assert "No AI provider is configured" in state.message
    assert "written by spaCR directly from the run digest" in state.message
    # No catalog means no per-provider bullets, and no half-written one.
    assert "•" not in state.message


def test_a_provider_that_stops_being_configured_is_reported_not_raised(
        digest):
    """It logs out between being offered and being used; the draft survives.

    ``availability`` and the lookup that follows it are two separate reads
    of the same list, so the provider named by the first can be gone by the
    second. The user gets the deterministic sections and a sentence saying
    what happened.
    """
    stub = _StubProvider()
    answers = [[stub], []]

    def _configured():
        return answers.pop(0) if answers else []

    original = manuscript.configured_providers
    manuscript.configured_providers = _configured
    try:
        draft = manuscript.generate_sections(digest)
    finally:
        manuscript.configured_providers = original

    assert draft.ok is False
    assert draft.source == "digest"
    assert draft.provider == ""
    assert draft.problems == [
        "The AI provider disappeared between being offered and being used."]
    assert stub.asked == 0, "a vanished provider was still asked for prose"
    assert draft.methods == render_methods(digest)
    assert draft.results == render_results(digest)
    assert draft.text().startswith(manuscript.METHODS_HEADING)
