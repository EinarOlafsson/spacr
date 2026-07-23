"""
AI Console for the spacr Qt GUI.

Three providers (Anthropic Claude, OpenAI, Google Gemini) behind a
common streaming `ChatProvider` interface. API keys are pulled from
env vars first (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` /
`GOOGLE_API_KEY`), then from the OS keyring under service name
`spacr-qt-ai`.

Public entry points:
    list_providers()           -> [ChatProvider, ...]
    get_provider(name)         -> ChatProvider | None
    configured_providers()     -> [ChatProvider, ...] (only ones with a key)
    default_system_prompt()    -> str (spacr-aware assistant persona)
    error_explainer_prompt()   -> str (system prompt for the "Explain error" flow)
"""
from __future__ import annotations

from . import settings
from .providers import (
    ChatProvider,
    configured_providers,
    get_provider,
    list_providers,
)
from .prompts import default_system_prompt, error_explainer_prompt

__all__ = [
    "ChatProvider",
    "configured_providers",
    "default_system_prompt",
    "error_explainer_prompt",
    "get_provider",
    "list_providers",
    "settings",
]
