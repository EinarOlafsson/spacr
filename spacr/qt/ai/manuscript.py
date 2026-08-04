"""Generate the Methods and Results sections through the AI console's providers.

:mod:`spacr.methods_export` builds the run digest, writes the prompt, and
checks a draft's numbers against the digest. This module is the half that
talks to a model, and it deliberately does NOT open a second client: it uses
the same :class:`spacr.qt.ai.providers.ChatProvider` objects the AI console
already streams through, so a user who authenticated once has authenticated
for this too, and a provider added there is available here the same day.

Two behaviours are the whole point of the module:

**A draft that invents a number is not returned as a draft.** The model's
output goes straight into :func:`spacr.methods_export.check_draft`. If it
carries a figure that is not in the digest — or if the Methods section drops
one of the caveats the run recorded — the draft is marked ``ok=False``, the
offending numbers are named in :attr:`ManuscriptDraft.problems`, the model's
text is preserved in :attr:`ManuscriptDraft.rejected` so a human can look at
it, and the sections handed back are the deterministic ones from the digest.
That is what makes "every number in the output comes from the digest" a
property of the system rather than a hope about the model.

**No key configured is an answer, not a traceback.** :func:`availability`
says exactly what is missing and what to type to fix it, and
:func:`generate_sections` returns a complete draft anyway — the deterministic
renderers need no model at all. A user with no CLI installed still gets their
methods section; it is just written by spaCR instead of by a model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...methods_export import (Verification, check_draft, render_methods,
                               render_prompt, render_results)
from .providers import ChatProvider, configured_providers, list_providers

__all__ = [
    "Availability",
    "ManuscriptDraft",
    "availability",
    "generate_sections",
    "split_sections",
]

LOG = logging.getLogger("spacr.qt.ai.manuscript")

#: The headings the model is told to return, and the ones :func:`split_sections`
#: looks for.
METHODS_HEADING = "## Methods"
RESULTS_HEADING = "## Results"


@dataclass(frozen=True)
class Availability:
    """Whether a model can be reached, and what to do when it cannot.

    :param ok: at least one provider is installed and logged in.
    :param providers: the names of the ones that are.
    :param message: one paragraph for the user. When ``ok`` is False it names
        every provider, whether its CLI is installed, and the command to
        install or log in — because "no AI configured" with no next step is
        the same as a traceback for anyone who wanted to use it.
    """

    ok: bool
    providers: Tuple[str, ...] = ()
    message: str = ""

    def __bool__(self) -> bool:
        """True when a model can be reached."""
        return self.ok


def availability() -> Availability:
    """Report which AI providers are usable, and how to fix it if none are.

    Never raises and never touches the network: :meth:`ChatProvider.is_installed`
    is a ``PATH`` lookup and :meth:`is_logged_in` is best-effort.
    """
    try:
        ready = configured_providers()
    except Exception as exc:                          # noqa: BLE001
        LOG.info("provider discovery failed", exc_info=True)
        return Availability(
            False, (),
            f"The AI providers could not be inspected ({exc}). The sections "
            f"below were written by spaCR from the run digest.")
    if ready:
        names = tuple(provider.name for provider in ready)
        return Availability(
            True, names,
            f"Using {ready[0].label}. Available: {', '.join(names)}.")

    lines = ["No AI provider is configured, so the sections below were "
             "written by spaCR directly from the run digest — every number "
             "in them is still from the run. To have a model write the prose "
             "instead, set up one of:"]
    try:
        candidates: Sequence[ChatProvider] = list_providers()
    except Exception:                                 # pragma: no cover
        candidates = ()
    for provider in candidates:
        if provider.is_installed():
            lines.append(
                f"  • {provider.label}: the {provider.cli_name} CLI is "
                f"installed but not logged in — run `{provider.login_command}`.")
        else:
            lines.append(
                f"  • {provider.label}: the {provider.cli_name} CLI is not on "
                f"PATH — install it with `{provider.install_hint}`, then run "
                f"`{provider.login_command}`.")
    return Availability(False, (), "\n".join(lines))


def split_sections(text: str) -> Tuple[str, str]:
    """Split a model's reply into ``(methods, results)``.

    Tolerant of the usual drift — a preamble before the first heading, a
    different heading level, a stray sign-off — because a reply that is
    otherwise correct must not be discarded over a hash mark. Anything before
    the Methods heading is dropped; anything after the Results heading is
    kept as part of Results.

    :param text: the model's whole reply.
    :returns: the two sections, each without its heading. Either may be
        ``""`` when the model did not produce it.
    """
    body = str(text or "")
    lower = body.lower()
    methods_at = lower.find("## methods")
    if methods_at < 0:
        methods_at = lower.find("# methods")
    results_at = lower.find("## results")
    if results_at < 0:
        results_at = lower.find("# results")

    if methods_at < 0 and results_at < 0:
        return "", ""
    if methods_at < 0:
        return "", _after_heading(body[results_at:])
    if results_at < 0 or results_at < methods_at:
        return _after_heading(body[methods_at:]), ""
    return (_after_heading(body[methods_at:results_at]),
            _after_heading(body[results_at:]))


def _after_heading(chunk: str) -> str:
    """Drop the first line (the heading) and trim."""
    _heading, _, rest = chunk.partition("\n")
    return rest.strip()


@dataclass
class ManuscriptDraft:
    """The two sections, plus everything about how they were arrived at.

    :param methods: the Methods section to use.
    :param results: the Results section to use.
    :param ok: the returned sections came from a model AND passed the number
        check. ``False`` means the deterministic renderer wrote them.
    :param source: ``"model"`` or ``"digest"``.
    :param provider: which provider was used, when one was.
    :param methods_check: the verification of the model's Methods section.
    :param results_check: the same for Results.
    :param problems: sentences for the user: what was missing, what was
        invented, what was rejected.
    :param rejected: the model's text, kept when it was refused so a human
        can see what it said.
    """

    methods: str = ""
    results: str = ""
    ok: bool = False
    source: str = "digest"
    provider: str = ""
    methods_check: Optional[Verification] = None
    results_check: Optional[Verification] = None
    problems: List[str] = field(default_factory=list)
    rejected: str = ""

    def text(self) -> str:
        """Both sections, ready to paste. No trailing newline."""
        parts = []
        if self.methods:
            parts.append(f"{METHODS_HEADING}\n\n{self.methods}")
        if self.results:
            parts.append(f"{RESULTS_HEADING}\n\n{self.results}")
        return "\n\n".join(parts).rstrip("\n")

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serializable copy."""
        return {
            "methods": self.methods, "results": self.results, "ok": self.ok,
            "source": self.source, "provider": self.provider,
            "problems": list(self.problems), "rejected": self.rejected,
            "methods_check": (self.methods_check.to_dict()
                              if self.methods_check else None),
            "results_check": (self.results_check.to_dict()
                              if self.results_check else None),
        }


def _fallback(digest: Mapping[str, Any], problems: Sequence[str],
              **kwargs: Any) -> ManuscriptDraft:
    """A complete draft written from the digest alone."""
    return ManuscriptDraft(
        methods=render_methods(digest), results=render_results(digest),
        ok=False, source="digest", problems=list(problems), **kwargs)


def generate_sections(digest: Mapping[str, Any], *,
                      provider: Optional[ChatProvider] = None,
                      model: Optional[str] = None,
                      stream=None) -> ManuscriptDraft:
    """Ask a model for the two sections; refuse a draft that invents a number.

    :param digest: the run digest — see :func:`spacr.methods_export.build_digest`.
        It is the model's ONLY input; raw data never reaches it.
    :param provider: the provider to use. Defaults to the first configured
        one; with none configured the deterministic renderer answers and the
        draft says so.
    :param model: optional model override passed to the provider.
    :param stream: optional callable invoked with each chunk as it arrives,
        for a live view. Exceptions from it are ignored — a UI that fails to
        paint must not lose the generation.
    :returns: a :class:`ManuscriptDraft`. **Always** carries usable sections:
        the model's when they passed, spaCR's own when they did not.
    """
    if provider is None:
        state = availability()
        if not state.ok:
            return _fallback(digest, [state.message])
        provider = next((p for p in configured_providers()
                         if p.name == state.providers[0]), None)
        if provider is None:                          # pragma: no cover - race
            return _fallback(digest, ["The AI provider disappeared between "
                                      "being offered and being used."])

    system, user = render_prompt(digest)
    chunks: List[str] = []
    try:
        for chunk in provider.stream_chat([{"role": "user", "content": user}],
                                          system=system, model=model):
            if not chunk:
                continue
            chunks.append(chunk)
            if stream is not None:
                try:
                    stream(chunk)
                except Exception:                     # noqa: BLE001
                    LOG.debug("stream callback failed", exc_info=True)
    except Exception as exc:                          # noqa: BLE001
        LOG.info("manuscript generation failed", exc_info=True)
        return _fallback(
            digest,
            [f"The {getattr(provider, 'label', 'AI')} provider failed: {exc}. "
             f"The sections below were written by spaCR from the run digest."],
            provider=getattr(provider, "name", ""))

    reply = "".join(chunks).strip()
    if not reply:
        return _fallback(
            digest,
            [f"{getattr(provider, 'label', 'The provider')} returned nothing. "
             f"The sections below were written by spaCR from the run digest."],
            provider=getattr(provider, "name", ""))

    methods, results = split_sections(reply)
    if not methods and not results:
        return _fallback(
            digest,
            ["The reply carried neither a '## Methods' nor a '## Results' "
             "heading, so it could not be split into sections. The sections "
             "below were written by spaCR from the run digest."],
            provider=getattr(provider, "name", ""), rejected=reply)

    methods_check, results_check = check_draft(methods, results, digest)
    if methods_check.ok and results_check.ok:
        return ManuscriptDraft(
            methods=methods, results=results, ok=True, source="model",
            provider=getattr(provider, "name", ""),
            methods_check=methods_check, results_check=results_check)

    problems = [
        "The generated draft was rejected because it does not match the run "
        "digest, so the sections below were written by spaCR instead. What "
        "was wrong:"]
    for label, verdict in (("Methods", methods_check),
                           ("Results", results_check)):
        if not verdict.ok:
            problems.append(f"  • {label}: {verdict.problem()}")
    draft = _fallback(digest, problems,
                      provider=getattr(provider, "name", ""), rejected=reply)
    draft.methods_check = methods_check
    draft.results_check = results_check
    return draft
