"""The questions the installer used to ask, asked on the first run instead.

Instruction 221.

THE INSTALLER IS THE WRONG PLACE, and this supersedes the earlier request
for installer options. An installer runs once, often unattended, sometimes
by an administrator who is not the user, and it asks its questions before
the person has seen a single screen of what they are configuring. Every
answer is a guess. The first RUN is the first moment the questions mean
anything.

EVERY QUESTION HAS A WORKING DEFAULT, so the screen can be dismissed without
answering any of it and nothing is left unset. A setup screen that must be
completed is a modal dialog wearing a nicer coat.

IT REOPENS AFTER AN UPDATE AND KEEPS THE ANSWERS. The trigger is "has this
been answered for THIS VERSION", not "has this ever been answered" -- an
update that adds a setting has a question the user has never seen. What it
must never do is reset what they already chose: a setup screen that
reappears with everything back at its default is a release that silently
un-configures the tool.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

#: Where the version that last answered these questions is remembered.
_KEY_ANSWERED_VERSION = "onboarding/setup_answered_version"


def _settings():
    from .preferences import _settings as store

    return store()


def answered_version() -> str:
    """The spaCR version that last completed the setup screen, or ``""``."""
    return str(_settings().value(_KEY_ANSWERED_VERSION, "") or "")


def mark_answered(version: str) -> None:
    """Record that this version's setup has been seen."""
    _settings().setValue(_KEY_ANSWERED_VERSION, str(version))


def current_version() -> str:
    """The running spaCR version, or ``"unknown"``."""
    try:
        from .. import __version__

        return str(__version__)
    except Exception:                                        # noqa: BLE001
        return "unknown"


def should_open(version: Optional[str] = None) -> bool:
    """Should the setup screen open now?

    :param version: the running version. Defaults to :func:`current_version`.

    True on a profile that has never answered, and again after an UPDATE.
    False otherwise, so a user who dismissed it is not asked again until
    something changes.
    """
    running = str(version or current_version())
    return answered_version() != running


#: The questions, in the order the request listed them, each as
#: ``(key, label, getter, setter, choices)``.
#:
#: `choices` is ``[(value, label)]`` for a chooser, or ``None`` for a
#: toggle. THE ACCESSORS ARE THE PREFERENCE MODULE'S OWN, not a copy: this
#: screen writes the same store every other panel reads, so an answer given
#: here and an answer given in Preferences are the same answer.
def questions() -> List[Tuple[str, str, Callable, Callable, Any]]:
    """Build the question list against the live preference module."""
    from . import preferences as prefs

    def choices_of(names):
        return [(n, str(n).replace("_", " ")) for n in names]

    out: List[Tuple[str, str, Callable, Callable, Any]] = [
        ("language", "Language", prefs.get_language, prefs.set_language,
         choices_of(getattr(prefs, "VALID_LANGUAGES", ("en",)))),
        ("theme", "Theme", prefs.get_theme_choice, prefs.set_theme_choice,
         list(prefs.theme_choices())),
        ("colour_blind", "Colour-blind mode", prefs.get_color_blind_mode,
         prefs.set_color_blind_mode, choices_of(prefs.VALID_CB_MODES)),
        ("spacr_mode", "spaCR mode", prefs.get_spacr_mode,
         prefs.set_spacr_mode,
         choices_of(getattr(prefs, "VALID_SPACR_MODES", ("balanced",)))),
        ("hash_inputs", "Reproducibility hash", prefs.get_hash_inputs,
         prefs.set_hash_inputs, None),
        ("issue_prompt", "One-click issue filing",
         prefs.get_issue_prompt_mode, prefs.set_issue_prompt_mode,
         choices_of(prefs.ISSUE_PROMPT_MODES)),
        ("ai_default", "AI assistant on at launch",
         prefs.get_ai_on_by_default, prefs.set_ai_on_by_default, None),
    ]
    return [q for q in out if q[4] is None or q[4]]


def apply(answers: Dict[str, Any]) -> List[str]:
    """Write the answers through the preference module. Returns what failed.

    ONE SETTING'S REFUSAL MUST NOT LOSE THE OTHERS. Each is written on its
    own, so a value the preference module rejects is reported and the rest
    are still saved -- a setup screen that discards six good answers because
    the seventh was bad has cost the user the whole screen.
    """
    trouble: List[str] = []
    for key, _label, _get, setter, _choices in questions():
        if key not in answers:
            continue
        try:
            setter(answers[key])
        except Exception as exc:                             # noqa: BLE001
            trouble.append(f"{key}: {exc}")
    return trouble


def current() -> Dict[str, Any]:
    """What the answers are now, for the screen to open on."""
    out: Dict[str, Any] = {}
    for key, _label, getter, _set, _choices in questions():
        try:
            out[key] = getter()
        except Exception:                                    # noqa: BLE001
            continue
    return out
