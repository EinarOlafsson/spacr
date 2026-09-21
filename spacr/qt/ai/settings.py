"""Persist AI Console preferences through Qt application settings.

The module stores provider response speed, the optional system-prompt
override, error-reporting preferences, the reports filed automatically from
this profile, and console-context sharing. Values persist across application
sessions through :class:`PySide6.QtCore.QSettings`.
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
    "claude": {
        "fast":     ("--model", "haiku"),
        "balanced": ("--model", "sonnet"),
        "deep":     ("--model", "opus"),
    },
    "codex": {
        "fast":     ("--model", "gpt-5-mini"),
        "balanced": ("--model", "gpt-5"),
        "deep":     ("--model", "gpt-5-pro"),
    },
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



#: Whether a profile that never chose reports failed runs as GitHub issues.
DEFAULT_AUTO_FILE_ISSUES = True


def get_auto_file_issues() -> bool:
    """Return whether a failed run is reported as a GitHub issue.

    This is the switch for reporting as a whole. How a report is sent is
    :func:`spacr.qt.preferences.get_issue_prompt_mode`: with 'always' it is
    filed automatically, with 'ask' a "File as issue" button opens it in a
    preview first, and with 'never' nothing is filed.

    :returns: the stored choice, or :data:`DEFAULT_AUTO_FILE_ISSUES` for a
        profile that never made one; reporting is on by default. A stored
        False, from the installer's consent
        page or from this switch, is kept.
    """
    raw = _settings().value(_KEY_AUTO_ISSUE, DEFAULT_AUTO_FILE_ISSUES)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def set_auto_file_issues(enabled: bool) -> None:
    """Store the GitHub issue-filing preference."""
    _settings().setValue(_KEY_AUTO_ISSUE, bool(enabled))


_KEY_AUTO_FILED = "ai/auto_filed_reports"

#: How many automatically filed fingerprints are remembered.
AUTO_FILED_LIMIT = 200


def _auto_filed() -> Dict[str, str]:
    """The fingerprints this profile has filed automatically, with their URLs.

    :returns: ``{fingerprint: issue url}``. Empty when nothing is stored or
        the stored value cannot be read.
    """
    import json

    raw = _settings().value(_KEY_AUTO_FILED, "")
    try:
        found = json.loads(str(raw or "{}"))
    except (TypeError, ValueError):
        return {}
    if not isinstance(found, dict):
        return {}
    return {str(k): str(v) for k, v in found.items()}


def auto_filed_url(fingerprint: str) -> str:
    """Where this profile already filed a report with this fingerprint.

    :param fingerprint: a traceback fingerprint from
        :func:`spacr.qt.ai.issue_report.fingerprint_of`.
    :returns: the issue URL, or ``""`` when this profile has not filed it.
    """
    return _auto_filed().get(str(fingerprint or ""), "")


def remember_auto_filed(fingerprint: str, url: str) -> None:
    """Record that a report with this fingerprint was filed automatically.

    The same crash is then never filed twice from this profile, however
    many times the run is repeated. The oldest entries are dropped beyond
    :data:`AUTO_FILED_LIMIT`.

    :param fingerprint: the report's fingerprint.
    :param url: the issue it was filed as, or commented on.
    """
    import json

    if not fingerprint:
        return
    filed = _auto_filed()
    filed.pop(str(fingerprint), None)
    filed[str(fingerprint)] = str(url or "")
    while len(filed) > AUTO_FILED_LIMIT:
        filed.pop(next(iter(filed)))
    _settings().setValue(_KEY_AUTO_FILED, json.dumps(filed))


def get_route_errors_through_ai() -> bool:
    """Return whether pipeline errors are routed to the AI Console.

    The preference is effective only when an AI provider is configured. It
    defaults to ``True``.
    """
    raw = _settings().value(_KEY_ROUTE_ERRORS, True)
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
    raw = _settings().value(_KEY_CONSOLE_AWARE, True)
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in ("true", "1", "yes")


def set_console_aware(enabled: bool) -> None:
    """Store the console-context sharing preference."""
    _settings().setValue(_KEY_CONSOLE_AWARE, bool(enabled))
