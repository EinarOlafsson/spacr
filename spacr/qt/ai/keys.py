"""
API key store — env var wins, keyring falls back.

`keyring` is already a spacr dependency (used elsewhere) so no new
requirement. Values are stored under a single service name so the OS
credential manager groups them together.
"""
from __future__ import annotations

import os
from typing import Optional

SERVICE_NAME = "spacr-qt-ai"


def get_key(provider: str) -> Optional[str]:
    """Return the API key for a provider ('anthropic'/'openai'/'google'),
    consulting env vars first, then the keyring."""
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
    }.get(provider)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var].strip() or None
    try:
        import keyring
        stored = keyring.get_password(SERVICE_NAME, provider)
        if stored:
            return stored.strip() or None
    except Exception:
        pass
    return None


def set_key(provider: str, key: str) -> bool:
    """Persist a key to the OS keyring; returns True on success."""
    try:
        import keyring
        keyring.set_password(SERVICE_NAME, provider, key.strip())
        return True
    except Exception:
        return False


def delete_key(provider: str) -> None:
    try:
        import keyring
        keyring.delete_password(SERVICE_NAME, provider)
    except Exception:
        pass


def source_of(provider: str) -> str:
    """Return a short human-readable label for where the key was found."""
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai":    "OPENAI_API_KEY",
        "google":    "GOOGLE_API_KEY",
    }.get(provider)
    if env_var and os.environ.get(env_var):
        return f"env {env_var}"
    try:
        import keyring
        if keyring.get_password(SERVICE_NAME, provider):
            return f"keyring ({SERVICE_NAME})"
    except Exception:
        pass
    return "not configured"
