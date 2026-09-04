"""Manage first-run setup questions and their saved answers.

Every question has a usable default, so setup can be dismissed without
leaving the application unconfigured. The screen is offered once per spaCR
version, allowing new questions to appear after an update while preserving
answers saved for existing settings.
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


#: The flag a launcher passes, and the variable a server sets, to go
#: straight to the application. Both spellings, because the flag is typed by
#: hand and the variable is set in a job script.
SKIP_FLAGS = ("--no-setup", "--skip-setup", "--headless-setup")
SKIP_ENV = "SPACR_NO_SETUP"


def skipped_on_purpose(environ=None) -> bool:
    """Has this launch asked not to be shown the setup screen?

    THE SCREEN IS MODAL AND IT IS NOW THE FIRST THING A LAUNCH DRAWS, which
    is right at a desk and wrong on a server: a batch job that inherits a
    stale profile would sit on an invisible modal dialog until it was
    killed, and the only symptom would be a run that never starts.

    So a launch can say no, and one already has when it runs under the
    offscreen or minimal platform plugin -- nobody is there to answer a
    question drawn into a buffer nothing displays.
    """
    import os

    environ = os.environ if environ is None else environ
    said = str(environ.get(SKIP_ENV, "")).strip().lower()
    if said in ("1", "true", "yes", "on"):
        return True
    if said in ("0", "false", "no", "off"):
        return False
    return str(environ.get("QT_QPA_PLATFORM", "")).strip().lower() in (
        "offscreen", "minimal", "vnc")


def take_the_setup_flags(argv):
    """(remaining argv, asked to skip). Consumes the flags it recognises.

    They are consumed rather than ignored because `launch` reads the first
    argument as the module to open into, and an unconsumed `--no-setup`
    would be looked up as a module name and quietly open nothing.
    """
    kept, asked = [], False
    for word in list(argv or []):
        if str(word).strip().lower() in SKIP_FLAGS:
            asked = True
        else:
            kept.append(word)
    return kept, asked


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
        """Names paired with their readable spellings, for a picker."""
        return [(n, str(n).replace("_", " ")) for n in names]

    out: List[Tuple[str, str, Callable, Callable, Any]] = [
        # THE NATIVE NAME, not the code. A user choosing their own language
        # is the one person who cannot be expected to recognise its ISO
        # code, and `getattr(prefs, "VALID_LANGUAGES", ("en",))` -- which is
        # what this read before -- found no such attribute and offered
        # English alone, on a screen whose first question is the language.
        ("language", "Language", prefs.get_language, prefs.set_language,
         _language_choices()),
        # FLIPPED, because `theme_choices()` is (caption, value) while every
        # other list here is (value, caption). Normalised at the source
        # rather than special-cased in the screen: a screen that knows which
        # of its questions is back to front is a screen that gets it wrong
        # the next time a question is added.
        ("theme", "Theme", prefs.get_theme_choice, prefs.set_theme_choice,
         [(value, caption) for caption, value in prefs.theme_choices()]),
        ("colour_blind", "Colour-blind mode", prefs.get_color_blind_mode,
         prefs.set_color_blind_mode, choices_of(prefs.VALID_CB_MODES)),
        # THE LEVELS, NOT THE POSTURES. This offered `SPACR_MODES`, which is
        # the OLD three-value resource posture -- Extra Performance,
        # Performance, Balanced -- while Preferences offers the five
        # `PERFORMANCE_LEVELS`. So Laptop and Workstation existed, were
        # settable in Preferences, and could not be chosen on the screen
        # whose whole job is choosing them once.
        #
        # They are not interchangeable. `spacr_mode_for_level` folds five
        # levels onto three postures (laptop -> extra_performance,
        # workstation -> balanced), so writing through `set_spacr_mode`
        # cannot express either end of the scale: picking Balanced here and
        # Workstation in Preferences produced the same posture and two
        # different answers to "what did I choose".
        #
        # The level is the setting a user picks; the posture is what the
        # cleanup code reads. `set_performance_level` writes both, in that
        # order, which is why it is the one to call.
        #
        # (The previous defect here was the same shape one layer down: a
        # `getattr(prefs, "VALID_SPACR_MODES")` that found nothing and fell
        # back to a one-item default, so the screen offered Balanced alone.
        # Named directly ever since, so a rename breaks the import instead
        # of silently shortening the list.)
        ("spacr_mode", "spaCR mode", prefs.get_performance_level,
         prefs.set_performance_level,
         [(level, prefs.PERFORMANCE_LABELS.get(
             level, str(level).replace("_", " ")))
          for level in prefs.PERFORMANCE_LEVELS]),
        ("hash_inputs", "Reproducibility hash", prefs.get_hash_inputs,
         prefs.set_hash_inputs, None),
        ("issue_prompt", "One-click issue filing",
         prefs.get_issue_prompt_mode, prefs.set_issue_prompt_mode,
         choices_of(prefs.ISSUE_PROMPT_MODES)),
        ("ai_default", "AI assistant on at launch",
         prefs.get_ai_on_by_default, prefs.set_ai_on_by_default, None),
        ("share_logs", "Include recent logs in a report",
         prefs.get_share_diagnostic_logs, prefs.set_share_diagnostic_logs,
         None),
        ("ai_provider", "AI provider", prefs.get_preferred_provider,
         prefs.set_preferred_provider, _provider_choices()),
    ]
    return [q for q in out if q[4] is None or q[4]]


def _language_choices():
    """``[(code, native name)]`` for every language spaCR is translated to.

    IN THEIR OWN SCRIPT, because the reader of this list is by definition
    somebody who may not read the current one.
    """
    try:
        from .i18n import LANGUAGES

        return [(one.code, one.native_name) for one in LANGUAGES]
    except Exception:                                        # noqa: BLE001
        return [("en", "English")]


def _provider_choices():
    """The installed providers, or ``[]``.

    AN EMPTY LIST REMOVES THE QUESTION, which `questions()` does at the
    bottom. Asking somebody to choose between providers none of which are
    installed is asking them to answer a question with no true answers --
    and this screen's rule is that every question has a working default.
    """
    try:
        from .ai.providers import list_providers

        names = [str(getattr(p, "name", "") or "") for p in list_providers()]
    except Exception:                                        # noqa: BLE001
        return []
    found = [(n, n.replace("_", " ")) for n in names if n]
    # "whatever is available" first, and it IS the default: a machine with
    # two CLIs today may have one tomorrow, and a pinned name that is gone
    # is worse than no preference.
    return [("", "whatever is available")] + found if found else []


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
