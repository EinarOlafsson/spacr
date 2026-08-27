"""When the usual way of finding something fails, ask for it.

THESE ARE BACKUPS, not the normal route. The ordinary resolution runs first
every time; this is reached only once it has already failed. A run that
works today must not gain a dialog, and a user who never hits the failure
must never see one.

A DIALOG IS BETTER THAN AN ERROR HERE because the information is one the
user has and the program does not. The current behaviour on each of these
is to stop with a message naming the thing that is missing -- which means
the program already knows precisely what to ask for. Asking is strictly
more useful than reporting, and costs one dialog instead of one aborted run.

What every prompt owes, and what this module enforces so a caller cannot
forget one:

* SAY WHAT WAS TRIED FIRST, so a typo in a setting is distinguishable from
  a genuinely missing folder;
* VALIDATE BEFORE ACCEPTING -- a chosen folder with nothing in it is the
  same failure one step later;
* ASK ONCE PER RUN, not once per image, per well or per plate;
* WRITE IT BACK and say so, because a setting that changes without being
  announced is worse than one that does not;
* BE REFUSABLE: cancel means the run stops with the error it would have
  given anyway;
* NEVER APPEAR HEADLESS. In a script, a test or a batch run there is nobody
  to answer, so the fallback resolves to the original error rather than
  blocking on a dialog nobody can see. This is the one that would hang a
  pipeline overnight, and it is checked before anything else.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, Optional, Tuple

#: Answers already given, keyed by what was asked for. One run, one question.
_ANSWERED: Dict[str, str] = {}


def forget() -> None:
    """Drop every remembered answer. For tests, and for a new run."""
    _ANSWERED.clear()


def remembered(key: str) -> Optional[str]:
    """What was already answered for ``key``, if anything."""
    return _ANSWERED.get(key)


def somebody_is_there() -> bool:
    """Whether there is a person who could answer a dialog.

    False under pytest, with no display, or before a QApplication exists.
    Checked FIRST and on its own, because getting it wrong does not show a
    dialog to nobody -- it BLOCKS, and a blocked batch run looks like a hang.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if os.environ.get("SPACR_NO_PROMPTS"):
        return False
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:                                    # noqa: BLE001
        return False
    app = QApplication.instance()
    if app is None:
        return False
    # An offscreen platform is a test or a render farm, not a person.
    if os.environ.get("QT_QPA_PLATFORM", "").startswith("offscreen"):
        return False
    return True


def ask_for_a_folder(
    key: str,
    *,
    tried: str,
    what: str,
    validate: Optional[Callable[[str], Optional[str]]] = None,
    parent=None,
    chooser: Optional[Callable[..., str]] = None,
) -> Tuple[Optional[str], str]:
    """Ask for a folder after the usual resolution has failed.

    :param key: what is being asked for. The answer is remembered under it,
        so the second image of a run does not ask again.
    :param tried: what was tried and did not work, said before the chooser
        so the user knows why they are being asked.
    :param what: a short name for the thing wanted, for the dialog title.
    :param validate: given a chosen path, returns None when it is usable or
        a sentence saying why it is not. The dialog stays open on a refusal.
    :param chooser: injected for tests. Defaults to a real folder dialog.
    :returns: ``(path, why)``. `path` is None when nobody answered, when the
        user cancelled, or when there is nobody there -- and `why` always
        says which, because those are three different situations.
    """
    already = _ANSWERED.get(key)
    if already:
        return already, f"{what}: using {already}, chosen earlier in this run"

    if not somebody_is_there():
        return None, (f"{what}: {tried} -- and there is nobody to ask, so "
                      f"this run stops here rather than waiting for an "
                      f"answer that cannot come.")

    if chooser is None:                                  # pragma: no cover
        from PySide6.QtWidgets import QFileDialog

        def chooser(title, start=""):
            return QFileDialog.getExistingDirectory(parent, title, start)

    while True:
        chosen = chooser(f"{what} — {tried}")
        if not chosen:
            return None, (f"{what}: cancelled, so this run stops with the "
                          f"error it would have given anyway.")
        complaint = validate(chosen) if validate else None
        if complaint is None:
            _ANSWERED[key] = chosen
            return chosen, f"{what}: using {chosen}, chosen just now"
        # Rejected IN the dialog rather than accepted and failed afterwards.
        tried = complaint


def a_folder_holding(*suffixes: str) -> Callable[[str], Optional[str]]:
    """A validator: the folder must hold at least one file with a suffix.

    A chosen folder with nothing in it is the same failure one step later,
    which is what makes validating before accepting worth the code.
    """
    wanted = tuple(s.lower() for s in suffixes)

    def check(path: str) -> Optional[str]:
        if not os.path.isdir(path):
            return f"{path} is not a folder"
        try:
            names = os.listdir(path)
        except OSError as error:
            return f"{path} cannot be read ({error.strerror})"
        if not wanted:
            return None if names else f"{path} is empty"
        if any(n.lower().endswith(wanted) for n in names):
            return None
        return (f"{path} holds no {' or '.join(wanted)} file. "
                f"Choose the folder that does.")

    return check
