"""Persist AI Console preferences through Qt application settings.

The module stores provider response speed, the optional system-prompt
override, error-reporting preferences, and console-context sharing. Values
persist across application sessions through :class:`PySide6.QtCore.QSettings`.
"""
from __future__ import annotations

from typing import Dict, List

from PySide6.QtCore import QSettings

from .prompts import default_system_prompt

_SETTINGS_ORG = "spacr"
_SETTINGS_APP = "qt"

_KEY_SPEED = "ai/response_speed"
_KEY_PROMPT = "ai/system_prompt"
_KEY_AUTO_ISSUE = "ai/auto_file_issues"
_KEY_ROUTE_ERRORS = "ai/route_errors_through_ai"
_KEY_CONSOLE_AWARE = "ai/console_aware"

VALID_SPEEDS = ("fast", "balanced", "deep")
DEFAULT_SPEED = "balanced"

#: Per-provider CLI argument mapping. Each level maps to the extra
#: argv fragments to append when invoking that provider's CLI.
#: Empty tuple = provider uses its own default at that level.
SPEED_MAP: Dict[str, Dict[str, tuple]] = {
    # Claude Code CLI supports --model to pick between Haiku (fast) /
    # Sonnet (balanced) / Opus (deep). Newer builds also honour
    # ``--reasoning-effort low|medium|high`` — safest is model.
    "claude": {
        "fast":     ("--model", "haiku"),
        "balanced": ("--model", "sonnet"),
        "deep":     ("--model", "opus"),
    },
    # Codex CLI: model picks fast (o4-mini) / balanced (o1-preview) /
    # deep (o1). The exact model IDs may drift; provider falls back
    # to CLI default if the flag is unrecognised.
    "codex": {
        "fast":     ("--model", "gpt-5-mini"),
        "balanced": ("--model", "gpt-5"),
        "deep":     ("--model", "gpt-5-pro"),
    },
    # Gemini CLI: model picks flash (fast) / pro (balanced) / pro
    # thinking (deep, via same model with --thinking flag).
    "gemini": {
        "fast":     ("--model", "gemini-2.5-flash"),
        "balanced": ("--model", "gemini-2.5-pro"),
        "deep":     ("--model", "gemini-2.5-pro"),
    },
}


def _settings() -> QSettings:
    """Open spaCR's ``QSettings``.

    :returns: the settings store.
    """
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


# ---------------------------------------------------------------------------
# Response speed
# ---------------------------------------------------------------------------

def get_response_speed() -> str:
    """Return the validated response-speed preference.

    Returns
    -------
    {"fast", "balanced", "deep"}
        Stored speed, or :data:`DEFAULT_SPEED` if the stored value is absent
        or invalid.
    """
    raw = str(_settings().value(_KEY_SPEED, DEFAULT_SPEED))
    return raw if raw in VALID_SPEEDS else DEFAULT_SPEED


def set_response_speed(speed: str) -> None:
    """Store the response-speed preference.

    Parameters
    ----------
    speed : {"fast", "balanced", "deep"}
        Provider-independent speed label.

    Raises
    ------
    ValueError
        If ``speed`` is not supported.
    """
    if speed not in VALID_SPEEDS:
        raise ValueError(f"unknown speed: {speed!r}. "
                          f"Choose from {VALID_SPEEDS}.")
    _settings().setValue(_KEY_SPEED, speed)


def provider_args(provider_name: str) -> List[str]:
    """Return command-line arguments for the selected provider and speed.

    Parameters
    ----------
    provider_name : str
        Provider identifier used as a key in :data:`SPEED_MAP`.

    Returns
    -------
    list of str
        Additional command-line arguments. Unknown providers or unmapped
        speed levels return an empty list.
    """
    speed = get_response_speed()
    return list(SPEED_MAP.get(provider_name, {}).get(speed, ()))


# ---------------------------------------------------------------------------
# System prompt override
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    """Return the stored system-prompt override or the spaCR default.

    Returns
    -------
    str
        Non-empty stored override, otherwise the value returned by
        :func:`default_system_prompt`.
    """
    raw = _settings().value(_KEY_PROMPT, None)
    if raw is None or not str(raw).strip():
        return default_system_prompt()
    return str(raw)


def set_system_prompt(text: str) -> None:
    """Store a system-prompt override.

    Parameters
    ----------
    text : str
        Prompt text. Surrounding whitespace is removed; an empty value clears
        the override.
    """
    text = (text or "").strip()
    if not text:
        reset_system_prompt()
        return
    _settings().setValue(_KEY_PROMPT, text)


def reset_system_prompt() -> None:
    """Remove the stored system-prompt override."""
    _settings().remove(_KEY_PROMPT)


def is_system_prompt_overridden() -> bool:
    """Return whether a non-empty system-prompt override is stored."""
    raw = _settings().value(_KEY_PROMPT, None)
    return raw is not None and bool(str(raw).strip())


# ---------------------------------------------------------------------------
# Auto-file GitHub issue on error (opt-in)
# ---------------------------------------------------------------------------

def get_auto_file_issues() -> bool:
    """Return whether error explanations may offer GitHub issue filing.

    When enabled, the error-explanation interface can open a pre-filled issue
    in the browser. The issue is not submitted automatically.
    """
    raw = _settings().value(_KEY_AUTO_ISSUE, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def set_auto_file_issues(enabled: bool) -> None:
    """Store the GitHub issue-filing preference."""
    _settings().setValue(_KEY_AUTO_ISSUE, bool(enabled))


def get_route_errors_through_ai() -> bool:
    """Return whether pipeline errors are routed to the AI Console.

    The preference is effective only when an AI provider is configured. It
    defaults to ``True``.
    """
    raw = _settings().value(_KEY_ROUTE_ERRORS, True)   # default ON
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def set_route_errors_through_ai(enabled: bool) -> None:
    """Store the pipeline-error routing preference."""
    _settings().setValue(_KEY_ROUTE_ERRORS, bool(enabled))


def get_console_aware() -> bool:
    """Return whether new console output is attached to AI questions.

    The preference defaults to ``True``. The console panel reports the amount
    of context attached to each message and applies its own output-length and
    traceback retention rules.
    """
    raw = _settings().value(_KEY_CONSOLE_AWARE, True)   # default ON
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def set_console_aware(enabled: bool) -> None:
    """Store the console-context sharing preference."""
    _settings().setValue(_KEY_CONSOLE_AWARE, bool(enabled))
