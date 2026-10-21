"""
AI Console for the spacr Qt GUI.

The AI console shares a streaming `ChatProvider` interface across the spaCR hosted relay
and the Anthropic Claude, OpenAI Codex and Google Gemini command-line tools. Vendor
tools use their own authentication; this package does not retrieve API keys from the OS
keyring. Provider configuration checks select candidates and do not guarantee a working
authenticated session.

Public entry points::

    list_providers()           -> [ChatProvider, ...]
    get_provider(name)         -> ChatProvider | None
    configured_providers()     -> [ChatProvider, ...] (hosted when allowed, then configured CLIs)
    default_system_prompt()    -> str (spacr-aware assistant persona)
    error_explainer_prompt()   -> str (system prompt for the "Explain error" flow)
    availability()             -> Availability (configured vendor candidates
                                  and setup guidance)
    generate_sections(digest)  -> ManuscriptDraft (generated sections checked
                                  against the digest, or deterministic
                                  fallback sections)
"""
from __future__ import annotations

from . import settings
from .manuscript import Availability, ManuscriptDraft, availability, generate_sections
from .providers import (
    ChatProvider,
    configured_providers,
    get_provider,
    list_providers,
)
from .prompts import default_system_prompt, error_explainer_prompt

__all__ = [
    "Availability",
    "ChatProvider",
    "ManuscriptDraft",
    "availability",
    "configured_providers",
    "default_system_prompt",
    "error_explainer_prompt",
    "generate_sections",
    "get_provider",
    "list_providers",
    "settings",
]
