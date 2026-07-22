"""
API key store — LEGACY.

The AI Console now shells out to the vendor coding-agent CLIs
(claude / codex / gemini), each of which authenticates against the
user's chat subscription instead of a metered API key. Nothing in
the Qt GUI uses API keys anymore.

The module is kept as a thin stub so any external code that imported
`spacr.qt.ai.keys` doesn't break — every function returns "not
configured".
"""
from __future__ import annotations

from typing import Optional


SERVICE_NAME = "spacr-qt-ai"


def get_key(provider: str) -> Optional[str]:
    """Always None — the CLI-based providers don't take API keys."""
    return None


def set_key(provider: str, key: str) -> bool:
    """No-op — CLI-based providers do not store API keys. Always ``False``."""
    return False


def delete_key(provider: str) -> None:
    """No-op — CLI-based providers do not store API keys."""
    return None


def source_of(provider: str) -> str:
    """Return a compat label indicating the CLI-login model."""
    return "n/a (uses vendor CLI login)"
