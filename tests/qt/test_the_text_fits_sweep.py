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

AND SO IS THE FONT SCALE, which is the axis this sweep spent its first night
without. Instruction 04 records that the tests once measured a font scale no
user has, and every number instruction 350 recorded before this file grew a
`scale` parameter was at 100 %. A LOCALE MAKES A STRING 20 % LONGER; THE SCALE
MAKES EVERY STRING ON THE SCREEN TWICE AS LONG AT ONCE, which is a different
kind of pressure -- it is the only one of the four causes that moves the
glyphs and the boxes independently, so it is the only one that can find a
control sized in hard-coded pixels. It found two of them, and neither was
visible in any locale at 100 %.

EVERY SCREEN IS BUILT AS A PAGE IN A CONTAINER THAT WILL NOT GROW, which is
the other half of what was missing -- see `screen_in_a_container`. A screen
shown as a top-level widget answers "does this fit?" by growing until it
does, and instruction 350's WATCH section has asked since the day it was
filed for the CONTAINER to be checked too.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect, Qt                       # noqa: E402
from PySide6.QtGui import QFontMetrics                     # noqa: E402
from PySide6.QtWidgets import (QLabel, QPushButton,        # noqa: E402
                               QStackedWidget)

from spacr.qt import i18n as I                             # noqa: E402
from spacr.qt.preferences import FONT_SCALE_MAX            # noqa: E402

from .test_no_caption_is_clipped import painted_text       # noqa: E402

#: The screens the sweep builds.
#:
#: It began as eight, "chosen to cover the shapes rather than the count": the
#: settings-form modules, the two that are mostly buttons, and the two that
#: are mostly prose. That sampling was the right call while the method was
#: being proven, and instruction 350 recorded what it left owing -- "What
#: remains is coverage -- more screens, more locales, and the dialogs".
#:
#: EVERY DECLARED APP IS NOW IN IT. The other sixteen were not skipped for a
#: reason; they had simply never been tried. All sixteen build through
#: `AppScreen` exactly as the original eight do, and all sixteen came up clean
#: on the first run -- 96 combinations, zero clipped captions -- so this is a
#: coverage debt repaid rather than a set of fixes. 24 screens x 3 locales x
#: 2 scales = 144 cases, measured at 22 s for the new 96.
SCREENS = ("measure", "regression", "classify_merged", "foreign", "annotate",
           "qc_dashboard", "power", "experiment_design",
           "data_manager", "pipeline_graph", "profiler", "lineage",
           "layer_viewer", "graph_builder", "run_compare", "tabulate",
           "investigate_hit", "trellis", "gate_editor", "feature_explorer",
           "outliers", "dose_response", "control_chart", "project_browser")

#: See the module docstring: source, worst case, and the maintainer's own.
LOCALES = ("en", "de", "is")

#: THE DIMENSION THIS SWEEP WAS MISSING, and instruction 350 said so from the
#: day it was filed: "a font scale other than 1.0 -- instruction 04 records
#: that the tests measured a font scale no user has, so a check that runs at
#: one scale proves nothing about the others". Every number this item
#: recorded before today was at 1.0.
#:
#: THE TWO ENDS, NOT A LADDER. `FONT_SCALE_MAX` rather than a hand-picked
#: 1.5, because "the largest one preferences offers" is what the file's own
#: HOW IT WILL BE CHECKED section asks for and it is the only value that
#: cannot go stale when the slider's range moves. Everything 1.5 finds, 2.0
#: finds too and further out: the one caption 1.5 clipped ("100%" in a 40 px
#: box, 46 px of text) is the same box that loses 22 px at 2.0.
SCALES = (1.0, FONT_SCALE_MAX)

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


@pytest.fixture
def at_font_scale(qapp, qt_theme_applied):
    """Put the font-scale PREFERENCE and the STYLESHEET on the same scale.

    THE REAL MECHANISM, NOT A STYLESHEET HACK, and the distinction decides
    whether the measurement describes anything a user can reach. The
    preference is one number with two consumers: the application stylesheet,
    built by `theme.stylesheet(font_scale=...)`, sets every FONT SIZE from it;
    `preferences.scaled_px` scales every WIDGET SIZE set from Python by the
    same number. `preferences.apply_preferences_to_app` writes the preference
    and then rebuilds the sheet from it, and this does exactly that pair --
    which is why it also catches the bug it found: a control sized in
    hard-coded pixels stays put while the glyphs inside it grow.

    Setting only the sheet would move the glyphs and not the boxes, and would
    "find" clipping in every control the package sizes correctly. Setting only
    the preference would move the boxes and not the glyphs and would find
    none. Both, or the number means nothing.

    The scale has to be in place BEFORE the screen is built, because
    `scaled_px` is read at construction; callers apply it first and then
    build.

    Restores the shared application's 100 % sheet on the way out --
    `conftest._restore_font_scale` puts the preference back, and leaving the
    sheet at 200 % would move every later test that measures a pixel.
    """
    from spacr.qt import preferences
    from spacr.qt.theme import set_widget_qss_context, stylesheet

    def _apply(scale: float) -> float:
        preferences.set_font_scale(scale)
        # What `apply_preferences_to_app` records before it composes the
        # sheet, so a late widget-QSS block agrees with the global one about
        # the scale rather than emitting a 100 % copy over a 200 % sheet.
        set_widget_qss_context(qapp, preferences.resolve_effective_theme(),
                               scale, preferences.get_pane_opacity())
        qapp.setStyleSheet(stylesheet(font_scale=scale))
        qapp.processEvents()
        return scale

    yield _apply
    qapp.setStyleSheet(stylesheet())
    qapp.processEvents()


def screen_in_a_container(qtbot, app_key, width=1200, height=850):
    """Build ``app_key`` the way the application holds it: as a PAGE.

    AND THIS IS THE CONTAINER CHECK, which instruction 350 has asked for
    since the day it was filed -- "a control that grows can push its
    neighbour off the screen. Assert the CONTAINER still fits too, or the
    sweep trades one clipped string for another" -- and which had never been
    run at a font scale other than 1.0.

    It is run by BUILDING the screen somewhere that cannot grow rather than
    by adding a second assertion, because a top-level widget answers the
    question by escaping it: `QLayout::activate` raises a WINDOW's minimum to
    its layout's total minimum, so a screen asked for 1200 px whose contents
    need 1529 simply becomes 1529 px wide and reports nothing clipped. That
    is not how a screen lives. It is a page in a `QStackedWidget` inside one
    window, and a page gets whatever the window has -- less than its own
    minimum included. Measured on Measure in English at 200 %: as a top-level
    widget it came up 1529 px when asked for 1200.

    So the container is a `QStackedWidget` with an EXPLICIT 1x1 minimum,
    which is what stops the container growing out of the problem in its turn.
    Anything that no longer fits is then squeezed, and a squeezed caption is
    what `_fits` reports.
    """
    from spacr.qt.screens.app_screen import AppScreen

    host = QStackedWidget()
    qtbot.addWidget(host)
    screen = AppScreen(app_key=app_key)
    host.addWidget(screen)
    host.setCurrentWidget(screen)
    host.setMinimumSize(1, 1)
    host.resize(width, height)
    host.show()
    qtbot.waitExposed(host)
    # RULE 1: nothing measured below is true until the layout has STOPPED
    # running. See `settle` -- the flat `qtbot.wait(30)` this replaced was
    # reading every module screen one delivered event too early.
    settle(qtbot, screen)
    return host, screen


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
#: THE SCALE AXIS ADDED IT NOTHING EITHER, and that is the result rather
#: than the absence of one. The one family the largest scale did find was
#: FIXED before it was recorded, because it was a defect and not a layout
#: decision: `UsageBar` set `setFixedWidth(48)` on its caption column and
#: `setFixedWidth(40)` on its percent readout, so at 200 % the glyphs
#: doubled and the boxes did not --
#:
#:     'VRAM'  66 px of text in 48 px of column
#:     'RAM'   52                48
#:     '100%'  62                40         (and 46 at 150 %)
#:
#: on the System card of every module screen, in every locale, which is why
#: it read as five offenders per screen and not as one. Both widths now take
#: the larger of `scaled_px` and the widest string the control can ever hold.
KNOWN_OFFENDERS: dict = {}


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("app_key", SCREENS)
def test_no_caption_is_cut_off(app_key, locale, scale, qtbot, at_font_scale,
                               monkeypatch):
    """One screen, in one language, at one font scale, by the rules above.

    THE SCALE IS APPLIED BEFORE THE SCREEN IS BUILT, which is not a detail:
    `scaled_px` is read at construction, so a screen built at 100 % and then
    handed a 200 % stylesheet is a shape no user has -- 200 % glyphs in 100 %
    boxes -- and would report clipping the application does not have.
    """
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    at_font_scale(scale)
    _host, screen = screen_in_a_container(qtbot, app_key)

    offenders = _offenders(screen)
    allowed = KNOWN_OFFENDERS.get((app_key, locale, scale), 0)
    detail = ("; ".join(offenders[:6])
              + (f" (and {len(offenders) - 6} more)"
                 if len(offenders) > 6 else ""))
    assert len(offenders) <= allowed, (
        f"{app_key} in {locale} at {scale:g}x: {len(offenders)} captions cut "
        f"off, {allowed} recorded. {detail}")
    assert len(offenders) == allowed or allowed == 0, (
        f"{app_key} in {locale} at {scale:g}x now has {len(offenders)} of "
        f"the {allowed} recorded -- lower or remove its entry in "
        f"KNOWN_OFFENDERS")


@pytest.mark.parametrize("scale,reported",
                         [(1.0, False), (FONT_SCALE_MAX, True)])
def test_the_scale_axis_can_actually_fail(scale, reported, qtbot,
                                          at_font_scale):
    """The new axis has teeth: the SAME widget passes at 1.0 and fails at 2.0.

    Written because a sweep that never fails and a sweep that always fails
    look identical from the summary line, and instruction 350 has four
    confident wrong headlines on the record -- one of them a run that
    "failed" twenty-five combinations without measuring a single widget.
    Adding a dimension that cannot report anything would be the fifth.

    This is the pre-fix `UsageBar` caption column exactly: a hard-coded 48 px
    box holding "VRAM". It fits at 100 %, where the number was chosen, and it
    does not at 200 %, where the glyphs doubled and the box did not -- so a
    green run of this file at 2.0 is evidence that the boxes moved with the
    glyphs, rather than evidence that nothing was looked at.
    """
    at_font_scale(scale)
    label = QLabel("VRAM")
    qtbot.addWidget(label)
    label.setFixedWidth(48)                     # the constant, as it was
    label.show()
    qtbot.waitExposed(label)
    qtbot.wait(10)

    assert bool(_fits(label)) is reported, (
        f"a 48 px box holding {label.text()!r} at {scale:g}x: "
        f"{_fits(label) or 'reported as fitting'}, and "
        f"{'a clip' if reported else 'no clip'} was expected")


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
