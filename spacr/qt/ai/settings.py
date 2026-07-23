"""
Persisted AI Console settings — response speed + system prompt override.

All values are stored via ``QSettings`` so they survive across app
launches. Two knobs:

* ``response_speed`` — a per-provider quality dial. Three levels map
  to provider-specific CLI flags (see :data:`SPEED_MAP`). Faster =
  cheaper + snappier, slower = more thorough. Same three labels work
  for every provider so users never have to think about vendor knobs.
* ``system_prompt`` — the spaCR-aware persona string. Users can edit
  it in the Settings tab of the Providers dialog to change how the
  assistant frames answers (e.g. shorter, or with a different
  emphasis). :func:`reset_system_prompt` restores the default.

Public API::

    from spacr.qt.ai import settings as ai_settings

    ai_settings.get_response_speed()          -> "fast" | "balanced" | "deep"
    ai_settings.set_response_speed("deep")

    ai_settings.get_system_prompt()           -> str
    ai_settings.set_system_prompt(text)
    ai_settings.reset_system_prompt()
    ai_settings.is_system_prompt_overridden() -> bool

    ai_settings.provider_args(provider_name)  -> list[str]
        # Returns extra argv fragments the provider should append to
        # its CLI invocation to honour the current speed setting.
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
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


# ---------------------------------------------------------------------------
# Response speed
# ---------------------------------------------------------------------------

def get_response_speed() -> str:
    """Return the persisted response-speed label.

    :returns: one of ``"fast"``, ``"balanced"``, ``"deep"``. Falls
        back to :data:`DEFAULT_SPEED` if unset or invalid.
    """
    raw = str(_settings().value(_KEY_SPEED, DEFAULT_SPEED))
    return raw if raw in VALID_SPEEDS else DEFAULT_SPEED


def set_response_speed(speed: str) -> None:
    """Persist a new response-speed label.

    :param speed: one of ``"fast"``, ``"balanced"``, ``"deep"``.
    :raises ValueError: when ``speed`` is not a known label.
    """
    if speed not in VALID_SPEEDS:
        raise ValueError(f"unknown speed: {speed!r}. "
                          f"Choose from {VALID_SPEEDS}.")
    _settings().setValue(_KEY_SPEED, speed)


def provider_args(provider_name: str) -> List[str]:
    """Return the extra CLI argv fragments a provider should append
    to honour the current speed setting.

    :param provider_name: matches :attr:`ChatProvider.name` — currently
        one of ``"claude"``, ``"codex"``, ``"gemini"``.
    :returns: list of argv fragments (may be empty).
    """
    speed = get_response_speed()
    return list(SPEED_MAP.get(provider_name, {}).get(speed, ()))


# ---------------------------------------------------------------------------
# System prompt override
# ---------------------------------------------------------------------------

def get_system_prompt() -> str:
    """Return the current system prompt.

    If the user has overridden it via :func:`set_system_prompt` that
    text is returned. Otherwise :func:`default_system_prompt`.
    """
    raw = _settings().value(_KEY_PROMPT, None)
    if raw is None or not str(raw).strip():
        return default_system_prompt()
    return str(raw)


def set_system_prompt(text: str) -> None:
    """Persist a user-authored system prompt override.

    :param text: full prompt text; will be trimmed of surrounding
        whitespace before saving. Passing an empty string clears the
        override (equivalent to :func:`reset_system_prompt`).
    """
    text = (text or "").strip()
    if not text:
        reset_system_prompt()
        return
    _settings().setValue(_KEY_PROMPT, text)


def reset_system_prompt() -> None:
    """Remove any user-authored override; subsequent
    :func:`get_system_prompt` returns :func:`default_system_prompt`."""
    _settings().remove(_KEY_PROMPT)


def is_system_prompt_overridden() -> bool:
    """Return ``True`` iff the user has set a custom system prompt."""
    raw = _settings().value(_KEY_PROMPT, None)
    return raw is not None and bool(str(raw).strip())


# ---------------------------------------------------------------------------
# Auto-file GitHub issue on error (opt-in)
# ---------------------------------------------------------------------------

def get_auto_file_issues() -> bool:
    """Return ``True`` iff the user has opted in to auto-issue reporting.

    When True, the AI Console's "Explain error" flow shows an extra
    "File as GitHub issue" button that opens a pre-filled issue URL
    in the user's browser (they still click Submit themselves).
    """
    raw = _settings().value(_KEY_AUTO_ISSUE, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def set_auto_file_issues(enabled: bool) -> None:
    """Persist the auto-issue toggle."""
    _settings().setValue(_KEY_AUTO_ISSUE, bool(enabled))
