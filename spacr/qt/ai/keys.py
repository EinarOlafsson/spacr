"""
API key store — LEGACY.

The AI console uses vendor command-line tools and a hosted relay rather than retrieving
keys through this module. Vendor authentication and account usage follow each tool's own
configuration; this stub does not establish how a vendor account is billed.

This module remains as a compatibility stub for code importing `spacr.qt.ai.keys`.
``get_key`` returns ``None``, ``set_key`` returns ``False``, ``delete_key`` returns
``None``, and ``source_of`` returns a descriptive compatibility label.
"""
from __future__ import annotations

from typing import Optional


SERVICE_NAME = "spacr-qt-ai"


def get_key(provider: str) -> Optional[str]:
    """Return ``None``; this compatibility stub does not retrieve a provider key.

    :param provider: AI provider name; accepted for API compatibility and
        ignored.
    """
    return None


def set_key(provider: str, key: str) -> bool:
    """Ignore the supplied key and return ``False``.

    :param provider: AI provider name; accepted for API compatibility and
        ignored.
    :param key: API key; accepted for API compatibility, ignored and never
        stored.
    """
    return False


def delete_key(provider: str) -> None:
    """Return ``None`` without changing a credential store.

    :param provider: AI provider name; accepted for API compatibility and
        ignored.
    """
    return None


def source_of(provider: str) -> str:
    """Return a compat label indicating the CLI-login model.

    :param provider: AI provider name; accepted for API compatibility and
        ignored, since every provider gets the same label.
    """
    return "n/a (uses vendor CLI login)"
