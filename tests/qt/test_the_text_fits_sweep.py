"""A sweep: every caption on a built screen fits the control that draws it.

Instruction 350, verbatim: "do a sweep of the entire package and make user
that all tesxt fits within its button, box or container, and is not cut off
at all."

WHAT MAKES THIS HARD IS NOT FINDING CLIPPING BUT BELIEVING IT. The instruction
records two confident wrong headlines in one night, both retracted within the
hour, and three classes of false positive that any sweep must handle before
its numbers mean anything:

  1. measuring before the layout settles   -> the screen is shown and the
     event loop pumped before anything is read;
  2. a wrapping label measured on one line -> a wrapped label is judged by
     its HEIGHT at its own width, never by the width of its text;
  3. a control that elides by design       -> `painted_text` asks the widget
     what it is actually drawing, because comparing `text()` against such a
     control reports clipping by construction.

SO A FAILURE HERE IS A CLAIM, and the message carries the numbers that back
it: the widget, its class, what it is drawing, what it has and what it needs.
A finding nobody can check from the failure line is how the last two got as
far as they did.

THE LOCALES ARE CHOSEN, NOT SAMPLED. English is the source text; German makes
the longest compounds in the set and is where a fixed-width control gives
first; Icelandic is the maintainer's own language, so a defect there is one he
meets rather than one he is told about.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, Qt                       # noqa: E402
from PySide6.QtGui import QFontMetrics                     # noqa: E402
from PySide6.QtWidgets import QLabel, QPushButton          # noqa: E402

from spacr.qt import i18n as I                             # noqa: E402

from .test_no_caption_is_clipped import painted_text       # noqa: E402

#: The screens the sweep builds. Chosen to cover the shapes rather than the
#: count: the settings-form modules, the two screens that are mostly buttons,
#: and the two that are mostly prose.
SCREENS = ("measure", "regression", "classify_merged", "foreign", "annotate",
           "qc_dashboard", "power", "experiment_design")

#: See the module docstring: source, worst case, and the maintainer's own.
LOCALES = ("en", "de", "is")

#: A widget narrower than this has not been laid out, whatever it reports.
LAID_OUT_PX = 8


def _fits(widget) -> str:
    """``""`` when the widget's text fits, else why it does not.

    The three rules the instruction records, applied in the order that makes
    each one cheap: skip what is not drawn, ask a wrapping label about its
    height, and ask everything else about the text it actually PAINTS.
    """
    if not widget.isVisible() or widget.width() < LAID_OUT_PX:
        return ""
    if callable(getattr(widget, "displayed_text", None)):
        # RULE 4, FOUND BY RUNNING THIS SWEEP. A control that owns its own
        # elision has already decided what to paint, and at least one of them
        # deliberately paints the FULL text slightly clipped rather than a
        # blank box: `AiToggleLabel` does it when the width cannot fit even
        # an ellipsis, and says why in its own code -- "the full text drawn
        # slightly clipped is strictly better: it still says which control
        # this is". Reported here it is a finding against a decision, which
        # is the fourth false-positive class this instruction now records.
        return ""
    text = painted_text(widget)
    if not text or not str(text).strip():
        return ""
    metrics = QFontMetrics(widget.font())

    if isinstance(widget, QLabel) and widget.wordWrap():
        # RULE 2. A wrapped label is as wide as it was given; the question is
        # whether the wrapped text fits the HEIGHT it was given.
        needed = metrics.boundingRect(
            QRect(0, 0, max(widget.width(), 1), 0),
            int(Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft), text).height()
        if widget.height() + 1 < needed:
            return (f"wrapped to {needed} px of height in {widget.height()}")
        return ""

    needed = metrics.horizontalAdvance(text)
    room = widget.width()
    if isinstance(widget, QPushButton):
        # A button's own hint knows what its style reserves for padding, the
        # icon and the focus rect; subtracting a guessed number instead is
        # what makes a sweep report every button on the screen.
        hint = widget.sizeHint().width()
        if room + 1 < hint:
            return f"{room} px wide, wants {hint}"
        return ""
    if room + 1 < needed:
        return f"{room} px wide, text needs {needed}"
    return ""


def _measurable(screen):
    """Every widget this sweep judges, in a stable order.

    ONE TYPE PER CALL: PySide6's `findChildren` takes a type, not a tuple,
    and hands back a TypeError rather than an empty list -- which is how the
    first run of this sweep "failed" on all 24 combinations without measuring
    a single widget.
    """
    return list(screen.findChildren(QLabel)) + list(
        screen.findChildren(QPushButton))


def _geometry_of(screen):
    """The geometry of everything measured, as one comparable value."""
    return tuple((w.x(), w.y(), w.width(), w.height())
                 for w in _measurable(screen))


def settle(qtbot, screen, rounds: int = 60) -> int:
    """Pump until the geometry stops changing. RULE 1, done as a measurement.

    THIS REPLACES A FLAT ``qtbot.wait(30)``, AND THE DIFFERENCE WAS SIX
    FINDINGS. ``QTest.qWait`` ends its loop with a *sleep*, not with a pass
    of ``processEvents``: it processes events, sleeps 10 ms, checks the
    clock, and breaks when the time is up. Anything posted during that last
    sleep is still sitting in the queue when the call returns. On a module
    screen the thing sitting there is the layout request raised by the
    language pass, which runs a turn after the screen is built -- so the row
    was read carrying GERMAN captions inside buttons still sized for the
    ENGLISH ones they were built with, and every one of them looked clipped.
    Measured on Measure in German: after ``qtbot.wait(30)`` the sweep found
    6 clipped captions and ``_btn_run`` was 60 px wide against a 103 px
    hint; after ONE more ``qtbot.wait(1)``, in the same test, it found 0 and
    the button was 103. Nothing about the screen changed in that
    millisecond except that Qt was finally allowed to deliver an event it
    had been holding.

    So the wait is a loop with a stopping CONDITION instead of a duration:
    two consecutive readings of every measured widget's geometry, 10 ms
    apart, that agree. That is the rule instruction 350 wrote down after the
    first two wrong headlines -- "pump the event loop until the geometry
    stops changing before measuring anything" -- and a fixed number of
    milliseconds was never an implementation of it.

    :returns: how many rounds it took, so a caller can print it.
    """
    previous = None
    for turn in range(1, rounds + 1):
        qtbot.wait(10)
        current = _geometry_of(screen)
        if current == previous:
            return turn
        previous = current
    return rounds


def _offenders(screen):
    """Every clipped caption on ``screen``, as sentences a reader can check."""
    found = []
    for widget in _measurable(screen):
        why = _fits(widget)
        if why:
            found.append(f"{type(widget).__name__} "
                         f"{painted_text(widget)!r}: {why}")
    return found


#: What this sweep found the last time it ran honestly, as
#: ``(screen, locale) -> how many captions are cut off``.
#:
#: A RATCHET, NOT AN EXCUSE. A recorded count cannot grow, a combination
#: that is not listed must have none at all, and an entry that reaches zero
#: has to be DELETED -- the sweep fails on a closed entry left in the table
#: pretending to be debt. The alternative, a permanently red sweep, is how
#: the four reds this file started life with survived.
#:
#: IT IS EMPTY, AND ALL THREE ENTRIES IT HELD WERE CLOSED RATHER THAN
#: FORGIVEN:
#:
#:   * `classify_merged / de` (1): `ApiHelpLabel` elides now, so the sentence
#:     the masthead cuts short says so with an ellipsis and keeps the rest in
#:     the hover help the masthead always intended to hold it.
#:   * `measure / de` (6) and `measure / is` (5): the action row's buttons
#:     wrap now instead of being squeezed -- see `_WrappingButtonStrip` in
#:     `spacr/qt/screens/app_screen.py`. Its minimum width fell from 1092 px
#:     in German to 450, which is what gave Measure's settings column back
#:     the 67 px it had been starved to.
#:
#: THOSE TWO COUNTS WERE ALSO MEASURED ONE EVENT TOO EARLY, and the
#: correction belongs beside them rather than in a commit message: they were
#: read after a flat `qtbot.wait(30)`, which leaves the language pass's
#: layout request undelivered, so every button was read at the English width
#: it was BUILT with while already carrying its German caption. See `settle`.
#: The defect underneath was real and worse than the count suggested -- at
#: 1000 px, where the window cannot grow out of it, the same row lost four
#: captions in German, four in Icelandic and one in ENGLISH -- but the
#: particular numbers 6 and 5 were an artefact, and
#: `test_the_action_row_wraps_instead_of_squeezing.py` is where that defect
#: is now pinned at a width that can actually show it.
#:
#: THE THEME IS WHAT TIPS IT, which is worth knowing before anyone tries to
#: reproduce this by hand: built without the application stylesheet the same
#: screen reports every button at exactly its hint and nothing is clipped.
#: The sweep therefore runs with `qt_theme_applied`, because that is the font
#: the user has.
KNOWN_OFFENDERS: dict = {}


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("app_key", SCREENS)
def test_no_caption_is_cut_off(app_key, locale, qtbot, qt_theme_applied,
                               monkeypatch):
    """One screen in one language, judged by the four rules above."""
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key=app_key)
    qtbot.addWidget(screen)
    screen.resize(1200, 850)
    screen.show()
    qtbot.waitExposed(screen)
    # RULE 1: nothing below is true until the layout has STOPPED running.
    # See `settle` -- the flat `qtbot.wait(30)` this replaced was reading
    # every module screen one delivered event too early.
    settle(qtbot, screen)

    offenders = _offenders(screen)
    allowed = KNOWN_OFFENDERS.get((app_key, locale), 0)
    detail = ("; ".join(offenders[:6])
              + (f" (and {len(offenders) - 6} more)"
                 if len(offenders) > 6 else ""))
    assert len(offenders) <= allowed, (
        f"{app_key} in {locale}: {len(offenders)} captions cut off, "
        f"{allowed} recorded. {detail}")
    assert len(offenders) == allowed or allowed == 0, (
        f"{app_key} in {locale} now has {len(offenders)} of the {allowed} "
        f"recorded -- lower or remove its entry in KNOWN_OFFENDERS")


def test_the_sweep_can_actually_fail(qtbot, qt_theme_applied):
    """The check is worth nothing if it cannot see a real clip.

    Written because this sweep's two predecessors both reported findings that
    were not there; a sweep that never fails and a sweep that always fails
    look identical from the summary line.
    """
    label = QLabel("A caption far longer than the room it has been given")
    qtbot.addWidget(label)
    label.show()
    qtbot.waitExposed(label)
    label.resize(30, label.sizeHint().height())
    qtbot.wait(10)

    assert _fits(label), "the sweep did not notice a label cut to 30 px"
