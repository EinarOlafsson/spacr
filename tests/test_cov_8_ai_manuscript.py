"""Generating a draft with no provider named, and a provider that pauses.

Two things the exporter has to get right when the caller does not hand it a
provider. It has to resolve one from the same availability report the UI
showed, so the draft is written by the provider the user was told about
rather than whichever happened to be first in the list. And it has to
tolerate the empty chunks a streaming CLI emits between tokens: an empty
string is a keep-alive, not a piece of the answer, and forwarding it to the
live view makes the draft look like it stalled.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.methods_export import build_digest            # noqa: E402
from spacr.qt.ai import manuscript                       # noqa: E402
from spacr.qt.ai.manuscript import generate_sections     # noqa: E402
from spacr.qt.ai.providers import ChatProvider           # noqa: E402

pytestmark = pytest.mark.qt

_MAX_EFFECT = 31.5006
_N_GENES = 412


class _ChunkedProvider(ChatProvider):
    """Streams a canned reply, with empty keep-alives between the pieces."""

    name = "chunked"
    label = "Chunked provider"
    cli_name = "chunked"
    install_hint = "nothing to install"
    login_command = "nothing to log in to"

    def __init__(self, pieces):
        super().__init__()
        self._pieces = list(pieces)

    def is_installed(self) -> bool:
        return True

    def stream_chat(self, messages, system="", model=None):
        for piece in self._pieces:
            yield piece


class _OtherProvider(_ChunkedProvider):
    """A second configured provider, so the choice is a real choice."""

    name = "other"
    label = "Other provider"
    cli_name = "other"


@pytest.fixture
def digest():
    return build_digest(
        title="the chunked screen",
        settings={"illumination_correction": True, "on_error": "stop",
                  "random_seed": 5150},
        extra={"max_effect": _MAX_EFFECT, "n_genes": _N_GENES},
    )


def _pieces(digest):
    """A clean reply split the way a streaming CLI delivers it."""
    caveats = "\n".join(f"- {c}" for c in digest["caveats"])
    return ["## Methods\n\n",
            "",
            f"The screen was analysed with spaCR.\n{caveats}\n\n",
            "",
            "## Results\n\n",
            "",
            f"The largest effect was {_MAX_EFFECT} across {_N_GENES} genes.\n"]


def test_with_no_provider_named_the_available_one_writes_the_draft(
        digest, monkeypatch):
    """The draft is credited to the provider availability actually offered."""
    chosen = _ChunkedProvider(_pieces(digest))
    monkeypatch.setattr(manuscript, "configured_providers",
                        lambda: [chosen, _OtherProvider([])])

    draft = generate_sections(digest)

    assert draft.provider == "chunked"
    assert draft.source == "model"
    assert draft.ok is True
    assert str(_MAX_EFFECT) in draft.results


def test_an_empty_keep_alive_chunk_never_reaches_the_live_view(
        digest, monkeypatch):
    """Blank chunks are dropped, so the streamed text is the reply itself."""
    chosen = _ChunkedProvider(_pieces(digest))
    monkeypatch.setattr(manuscript, "configured_providers", lambda: [chosen])
    seen = []

    draft = generate_sections(digest, stream=seen.append)

    assert "" not in seen
    assert seen == [piece for piece in _pieces(digest) if piece]
    assert draft.ok is True
    assert str(_MAX_EFFECT) in "".join(seen)
