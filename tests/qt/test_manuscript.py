"""The AI half of the methods/results exporter.

Two behaviours carry the whole feature and both are asserted here against a
stub provider, because the point is not what a model says — it is what the
system does with what a model says:

* a draft that carries a number the run digest does not have is **refused**,
  and the sections handed back are spaCR's own. That is what makes "every
  number in the output comes from the digest" a property rather than a hope;
* with no provider configured the exporter still returns both sections and a
  message naming exactly what is missing and what to type. No traceback.

The stub provider is a real :class:`spacr.qt.ai.providers.ChatProvider`
subclass, so the plumbing under test is the console's own.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.methods_export import build_digest, render_methods    # noqa: E402
from spacr.qt.ai import manuscript                               # noqa: E402
from spacr.qt.ai.manuscript import (ManuscriptDraft, availability,  # noqa: E402
                                    generate_sections,
                                    split_sections)
from spacr.qt.ai.providers import ChatProvider                   # noqa: E402

pytestmark = pytest.mark.qt

#: A number nothing in spaCR would produce, planted in the digest.
PLANTED = 48.3179


class _StubProvider(ChatProvider):
    """A provider that returns a canned reply and records what it was sent."""

    name = "stub"
    label = "Stub provider"
    cli_name = "stub"
    install_hint = "nothing to install"
    login_command = "nothing to log in to"

    def __init__(self, reply: str = "", raises: Exception = None):
        super().__init__()
        self._reply = reply
        self._raises = raises
        self.seen_system = ""
        self.seen_messages = None

    def is_installed(self) -> bool:
        return True

    def stream_chat(self, messages, system="", model=None):
        self.seen_system = system
        self.seen_messages = messages
        if self._raises is not None:
            raise self._raises
        for chunk in self._reply.splitlines(keepends=True):
            yield chunk


@pytest.fixture
def digest():
    return build_digest(
        title="the pilot screen",
        settings={"illumination_correction": True, "on_error": "stop",
                  "random_seed": 7714},
        extra={"max_effect": PLANTED, "n_genes": 517},
    )


def _clean_reply(digest):
    """A model reply that only ever quotes the digest."""
    caveats = "\n".join(f"- {c}" for c in digest["caveats"])
    return (f"## Methods\n\n"
            f"The screen was analysed with spaCR.\n{caveats}\n\n"
            f"## Results\n\n"
            f"The largest effect was {PLANTED} across 517 genes.\n")


# ---------------------------------------------------------------------------
# The enforcement
# ---------------------------------------------------------------------------

def test_a_clean_draft_is_accepted_and_returned(digest):
    provider = _StubProvider(_clean_reply(digest))

    draft = generate_sections(digest, provider=provider)

    assert draft.ok is True
    assert draft.source == "model"
    assert draft.provider == "stub"
    assert str(PLANTED) in draft.results
    assert draft.problems == []
    assert draft.methods_check.ok and draft.results_check.ok


def test_the_model_cannot_introduce_a_number_the_digest_does_not_have(digest):
    """The core assertion. An invented figure does not become a draft."""
    reply = ("## Methods\n\n" +
             "\n".join(f"- {c}" for c in digest["caveats"]) +
             "\n\n## Results\n\n"
             "Sequencing recovered 9999 barcodes across 42 plates.\n")
    provider = _StubProvider(reply)

    draft = generate_sections(digest, provider=provider)

    assert draft.ok is False
    assert draft.source == "digest"
    assert "9999" in draft.results_check.unsupported
    assert "42" in draft.results_check.unsupported
    assert "9999" not in draft.results, (
        "the invented figure must not reach the returned section")
    assert "9999" in draft.rejected, (
        "but a human must be able to see what the model said")
    assert any("9999" in problem for problem in draft.problems)
    assert draft.methods == render_methods(digest), (
        "the refusal falls back to the sections spaCR writes from the digest")


def test_the_refused_draft_still_hands_back_usable_sections(digest):
    from spacr.methods_export import render_results

    provider = _StubProvider(
        "## Methods\n\nAnything.\n\n## Results\n\nWe found 8888 hits.\n")

    draft = generate_sections(digest, provider=provider)

    assert draft.methods == render_methods(digest)
    assert draft.results == render_results(digest)
    assert draft.text().startswith("## Methods")


def test_a_methods_section_that_drops_a_caveat_is_refused(digest):
    reply = ("## Methods\n\nspaCR was used.\n\n"
             "## Results\n\nNothing to report.\n")
    provider = _StubProvider(reply)

    draft = generate_sections(digest, provider=provider)

    assert draft.ok is False
    assert draft.methods_check.missing_caveats
    assert any("caveat" in problem for problem in draft.problems)


def test_the_model_is_sent_the_digest_and_nothing_else(digest):
    import json

    provider = _StubProvider(_clean_reply(digest))

    generate_sections(digest, provider=provider)

    assert len(provider.seen_messages) == 1
    body = provider.seen_messages[0]["content"]
    payload = body.split("```json", 1)[1].rsplit("```", 1)[0]
    assert json.loads(payload) == json.loads(json.dumps(digest)), (
        "the model's only input is the digest")
    assert "EVERY NUMBER" in provider.seen_system


# ---------------------------------------------------------------------------
# Splitting the reply
# ---------------------------------------------------------------------------

def test_the_two_sections_are_split_on_their_headings():
    methods, results = split_sections(
        "## Methods\n\nfirst\n\n## Results\n\nsecond\n")

    assert methods == "first"
    assert results == "second"


def test_a_preamble_and_a_different_heading_level_are_tolerated():
    methods, results = split_sections(
        "Sure, here you go!\n\n# Methods\nfirst\n\n# Results\nsecond")

    assert methods == "first"
    assert results == "second"


def test_a_reply_with_only_one_section_gives_the_other_as_empty():
    assert split_sections("## Methods\n\nonly this") == ("only this", "")
    assert split_sections("## Results\n\nonly this") == ("", "only this")
    assert split_sections("no headings at all") == ("", "")


def test_a_reply_with_no_headings_is_refused_rather_than_guessed(digest):
    provider = _StubProvider("Here is your paper, roughly 5000 words.")

    draft = generate_sections(digest, provider=provider)

    assert draft.ok is False
    assert any("heading" in problem for problem in draft.problems)
    assert draft.rejected.startswith("Here is your paper")


# ---------------------------------------------------------------------------
# Degrading without a provider
# ---------------------------------------------------------------------------

def test_no_provider_configured_is_a_message_not_a_traceback(digest,
                                                             monkeypatch):
    monkeypatch.setattr(manuscript, "configured_providers", lambda: [])

    draft = generate_sections(digest)

    assert draft.ok is False
    assert draft.source == "digest"
    assert draft.methods and draft.results, (
        "a user with no AI still gets their methods section")
    assert len(draft.problems) == 1
    message = draft.problems[0]
    assert "No AI provider is configured" in message
    for name in ("claude", "codex", "gemini"):
        assert name in message, f"the fix must name {name}"
    assert "install" in message or "PATH" in message


def test_the_availability_message_says_what_to_type(monkeypatch):
    monkeypatch.setattr(manuscript, "configured_providers", lambda: [])

    state = availability()

    assert state.ok is False
    assert bool(state) is False
    assert state.providers == ()
    assert "npm install" in state.message or "curl" in state.message
    assert "login" in state.message or "setup-token" in state.message


def test_availability_reports_a_configured_provider(monkeypatch):
    provider = _StubProvider("")
    monkeypatch.setattr(manuscript, "configured_providers", lambda: [provider])

    state = availability()

    assert state.ok is True
    assert state.providers == ("stub",)
    assert "Stub provider" in state.message


def test_a_provider_that_raises_degrades_instead_of_propagating(digest):
    provider = _StubProvider(raises=RuntimeError("the CLI exploded"))

    draft = generate_sections(digest, provider=provider)

    assert draft.ok is False
    assert draft.source == "digest"
    assert any("the CLI exploded" in problem for problem in draft.problems)
    assert draft.methods, "the sections are still delivered"


def test_an_empty_reply_degrades_with_a_reason(digest):
    draft = generate_sections(digest, provider=_StubProvider(""))

    assert draft.ok is False
    assert any("returned nothing" in problem for problem in draft.problems)


def test_provider_discovery_failing_is_also_a_message(monkeypatch):
    def _boom():
        raise OSError("PATH is on fire")

    monkeypatch.setattr(manuscript, "configured_providers", _boom)

    state = availability()

    assert state.ok is False
    assert "PATH is on fire" in state.message


# ---------------------------------------------------------------------------
# Streaming and shape
# ---------------------------------------------------------------------------

def test_chunks_are_streamed_to_the_callback(digest):
    seen = []
    provider = _StubProvider(_clean_reply(digest))

    generate_sections(digest, provider=provider, stream=seen.append)

    assert seen, "the caller must be able to show the draft as it arrives"
    assert "".join(seen).strip().startswith("## Methods")


def test_a_failing_stream_callback_does_not_lose_the_generation(digest):
    def _boom(_chunk):
        raise RuntimeError("the widget went away")

    draft = generate_sections(digest, provider=_StubProvider(
        _clean_reply(digest)), stream=_boom)

    assert draft.ok is True, "a UI that fails to paint must not lose the draft"


def test_the_draft_is_json_serializable(digest):
    import json

    draft = generate_sections(digest, provider=_StubProvider(
        _clean_reply(digest)))

    payload = json.dumps(draft.to_dict())
    assert "methods_check" in payload


def test_an_empty_draft_renders_nothing_rather_than_stray_headings():
    assert ManuscriptDraft().text() == ""
    assert ManuscriptDraft(methods="a").text() == "## Methods\n\na"
