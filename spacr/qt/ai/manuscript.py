"""Generate the Methods and Results sections through the AI console's providers.

:mod:`spacr.methods_export` builds the run digest, writes the prompt and checks
generated sections. This module uses the :class:`spacr.qt.ai.providers.ChatProvider`
interface shared with the AI console. Automatic selection considers configured vendor
CLI providers and excludes the hosted relay; callers can also pass a provider
explicitly. Configuration checks do not establish that authentication or a model request
will succeed.

The module has two behaviours:

**Generated sections are checked before acceptance.** The split reply is passed to
:func:`spacr.methods_export.check_draft`, which checks recognized number tokens against
the digest and looks for the recorded caveat phrases in Methods. If either verdict
fails, ``ok=False``, :attr:`ManuscriptDraft.problems` describes the failures,
:attr:`ManuscriptDraft.rejected` retains the reply and the deterministic renderers
supply the returned sections. These checks do not establish scientific accuracy or
verify every possible numerical claim; the fallback sections are not passed through this
check.

**No configured vendor provider triggers a deterministic fallback.**
:func:`availability` reports candidates or setup guidance, and :func:`generate_sections`
uses the digest renderers when discovery finds no candidate. Caught streaming errors,
empty replies and rejected drafts also use this fallback. Discovery outside the guarded
calls, prompt construction, verification and fallback rendering can still raise
exceptions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ...methods_export import (Verification, check_draft, render_methods,
                               render_prompt, render_results)
from .providers import (HOSTED_PROVIDER_NAME, ChatProvider,
                        configured_providers, list_providers)

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
    """Recorded vendor-provider configuration status and setup guidance.

    :param ok: at least one non-hosted provider passed the configuration checks used by
        :func:`availability`; this does not confirm authentication or model access.
    :param providers: the names of the non-hosted providers that passed those checks.
    :param message: status or setup guidance for the user. It can contain several lines;
        discovery failures can produce an error message instead of a complete provider
        list.
    """

    ok: bool
    providers: Tuple[str, ...] = ()
    message: str = ""

    def __bool__(self) -> bool:
        """Return the recorded ``ok`` flag without checking model access."""
        return self.ok


def availability() -> Availability:
    """Report configured vendor AI candidates or available setup guidance.

    Current vendor checks inspect local CLI availability rather than making a model
    request. Initial configured-provider discovery failures are converted into an
    unavailable status, and a failed provider-list lookup yields no setup rows. Later
    provider-property and installation checks are not guarded and can raise; a
    configured candidate is not proof of a successful login.
    """
    try:
        ready = [provider for provider in configured_providers()
                 if provider.name != HOSTED_PROVIDER_NAME]
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
    except Exception:
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

    Search case-insensitively for ``## methods`` and ``## results``, falling back to ``#
    methods`` and ``# results`` respectively. Text before the selected Methods heading
    is discarded; text after the selected Results heading remains in Results. If Results
    precedes Methods, return only Methods; if only Results is found, return only
    Results. The selected heading lines are removed, but other text such as a sign-off
    remains.

    :param text: the model's whole reply.
    :returns: the two sections without their selected heading lines. A section can be
        ``""`` when its heading is absent, its body is empty or the heading order
        excludes it.
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
    """Request two model-written sections and use digest renderers when generation or the
    draft checks fail.

    :param digest: the run digest — see :func:`spacr.methods_export.build_digest`. It is
        serialized into the user prompt; this function does not remove raw or sensitive
        values supplied in the mapping.
    :param provider: the provider to use. When omitted, select the first configured
        non-hosted candidate from :func:`availability`; when none is found, use the
        deterministic renderers. An explicitly supplied provider bypasses discovery.
    :param model: optional model override passed to the provider.
    :param stream: optional callable invoked with each chunk as it arrives,
        for a live view. Exceptions from it are ignored — a UI that fails to
        paint must not lose the generation.
    :returns: a :class:`ManuscriptDraft` containing accepted model sections or
        deterministic fallback sections. Errors in discovery outside the guarded calls,
        prompt construction, verification or fallback rendering can propagate instead of
        returning a draft.
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
