"""380: the guard that has to exist BEFORE the build's three passes become one.

380 located the build's 146 ms precisely -- "13 passes, 23,454 widget visits,
and THREE of those passes are 22,750 of them" -- and then declined to fix it,
for a reason worth repeating:

    "NOT ATTEMPTED, and the reason is the risk rather than the work: getting
    it wrong leaves a widget in English in one of nine locales, which is
    invisible on this machine and to this suite. It is verifiable headlessly
    -- capture every translated widget's text after a build, apply the
    change, compare the sets -- and that verification is the first half of
    the job."

This file is that first half. It captures every caption a built screen
carries, in every non-English locale, and asserts three things that a
deduplicating pass could plausibly break:

  * the SET of rendered captions is unchanged -- nothing regressed to English
  * a caption that arrives LATE still gets translated, which is the whole
    reason `_LateCaptionTranslator` exists
  * switching language afterwards still re-renders everything, because the
    obvious way to deduplicate a pass is a marker, and the obvious way to get
    a marker wrong is to forget that a language change invalidates it

The point is that these tests must pass identically before and after the
change. A run of this file on the unmodified tree is the baseline; the same
run after is the evidence.
"""

import os
from contextlib import contextmanager

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

#: Every locale spaCR ships a catalog for. All nine, not a sample: the defect
#: this guards against is invisible in eight of them and present in the ninth.
NON_ENGLISH = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")

#: One screen with a large settings surface. `mask` is the module 380
#: instrumented, so the counts in that finding are comparable to these.
APP_KEY = "mask"


def _screen(qtbot, app_key: str = APP_KEY):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    qtbot.wait(1)
    return screen


def _captions(root) -> dict:
    """Every visible string the translation pass is responsible for.

    Keyed by a path that survives a rebuild -- object name and class, not
    ``id()``, which is reused as soon as a widget dies. Widgets with neither
    a name nor any text are skipped: they carry nothing to regress.
    """
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QWidget

    found = {}
    widgets = [root] + list(root.findChildren(QWidget))
    for index, widget in enumerate(widgets):
        try:
            text = ""
            if isinstance(widget, (QLabel, QAbstractButton, QGroupBox)):
                text = str(widget.text() or "")
            tip = str(widget.toolTip() or "")
        except RuntimeError:
            continue
        if not text and not tip:
            continue
        key = (index, type(widget).__name__, str(widget.objectName() or ""))
        found[key] = (text, tip)
    return found


@contextmanager
def _language(code):
    """Set the UI language for the duration of a `with` block.

    Through the environment variable, because that is the one lever
    :func:`spacr.qt.i18n.current_language` consults before it reaches for
    preferences -- so a test does not have to write the user's settings file
    to ask what a German build looks like.
    """
    import os as _os

    from spacr.qt import i18n

    before = _os.environ.get(i18n.ENV_LANGUAGE)
    _os.environ[i18n.ENV_LANGUAGE] = code
    try:
        assert i18n.current_language() == code
        yield
    finally:
        if before is None:
            _os.environ.pop(i18n.ENV_LANGUAGE, None)
        else:
            _os.environ[i18n.ENV_LANGUAGE] = before


@pytest.mark.parametrize("code", NON_ENGLISH)
def test_a_full_pass_over_a_built_screen_changes_nothing(qtbot, code):
    """THE guard. If the build translated everything, another pass is a no-op.

    This is the invariant to hold while the build's three near-root passes
    become one, and it is stated this way rather than as "every caption
    equals its translation" for a reason that cost an hour to learn: there
    is no single function a test can call to say what a caption SHOULD read.
    The pass renders a settings label from the setting catalog by key
    (`cell_channel` -> "Voie cellulaire") while `tr` on the same visible
    English gives a word-by-word composite ("Cellule Canal"). Re-deriving
    the expected text means reimplementing the pass, and a reimplementation
    agreeing with itself proves nothing.

    So the question is asked of the pass instead: run it again, in full, over
    the finished screen, and see whether it moves anything. A widget the
    build skipped is a widget this pass would fix, and fixing it is a diff.
    A widget the build handled is one this pass leaves alone, because the
    pass is idempotent -- which the fr case below pins separately.

    That makes this test sensitive to exactly the defect a deduplicating
    marker would introduce, in all nine locales, without needing to know
    what any individual caption ought to say.
    """
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QWidget

    from spacr.qt.i18n import retranslate_widget_tree

    def snapshot(root):
        out = {}
        for index, widget in enumerate([root] + list(root.findChildren(QWidget))):
            try:
                text = (str(widget.text() or "")
                        if isinstance(widget, (QLabel, QAbstractButton, QGroupBox))
                        else "")
                out[index] = (text, str(widget.toolTip() or ""),
                              str(widget.windowTitle() or ""))
            except RuntimeError:
                continue
        return out

    with _language(code):
        screen = _screen(qtbot)
        qtbot.wait(5)
        before = snapshot(screen)
        retranslate_widget_tree(screen, code)
        qtbot.wait(1)
        after = snapshot(screen)

    moved = {k: (before[k], after[k]) for k in before
             if k in after and before[k] != after[k]}
    assert moved == {}, (
        f"{len(moved)} widget(s) in {code} were changed by a pass that ran "
        f"AFTER the build finished, so the build had not translated them; "
        f"first three: {list(moved.items())[:3]}")


@pytest.mark.parametrize("code", NON_ENGLISH)
def test_the_caption_set_is_stable_across_two_identical_builds(qtbot, code):
    """Two builds of one screen must render the same strings.

    This is the "compare the sets" check in its strongest available form:
    if a deduplicating marker leaked between builds -- a class attribute, a
    module-level generation counter that never resets -- the SECOND build
    would skip work the first one did, and only this comparison would see it.
    """
    with _language(code):
        first = _captions(_screen(qtbot))
        second = _captions(_screen(qtbot))

    assert {t for t, _ in first.values()} == {t for t, _ in second.values()}
    assert {t for _, t in first.values()} == {t for _, t in second.values()}


def test_a_caption_that_arrives_late_is_still_translated(qtbot):
    """`_LateCaptionTranslator` exists for exactly this, so it must survive.

    A widget parented into a screen AFTER the build has never met the pass.
    A deduplication that marks a host as done and then refuses to walk it
    again would strand every such widget in English -- and this is the case
    that is easiest to break and hardest to notice, because the strip only
    appears once a module is actually opened.
    """
    from PySide6.QtWidgets import QLabel

    from spacr.qt.i18n import tr

    code = "de"
    source = "Settings"
    with _language(code):
        screen = _screen(qtbot)
        late = QLabel(source)
        late.setProperty("settingKey", "nucleus_channel")
        late.setParent(screen)
        # Two turns: one for ChildAdded to be delivered, one for the
        # deferred pass that `_LateCaptionTranslator` schedules.
        qtbot.wait(10)
        rendered = str(late.text() or "")

    expected = tr(source, code)
    if expected != source:              # only assert where a catalog exists
        assert rendered == expected, (
            f"a late caption stayed in English: {rendered!r} for {source!r}")


def test_switching_language_re_renders_a_screen_that_was_already_translated(
        qtbot):
    """A marker that forgets the language is the obvious way to get this wrong.

    Build in German, switch to Swedish, and the captions must move. A
    per-widget "already translated" flag that is not keyed to the language
    passes every other test in this file and fails this one.
    """
    from spacr.qt.i18n import retranslate_widget_tree

    with _language("de"):
        screen = _screen(qtbot)
        german = _captions(screen)

    with _language("sv"):
        retranslate_widget_tree(screen, "sv")
        qtbot.wait(1)
        swedish = _captions(screen)

    german_text = {t for t, _ in german.values() if t}
    swedish_text = {t for t, _ in swedish.values() if t}
    assert german_text != swedish_text, (
        "the screen rendered identically in German and Swedish, so the "
        "language switch did not reach it")


def test_translating_the_same_tree_twice_changes_nothing(qtbot):
    """Idempotence, which is what makes deduplication safe to attempt.

    If a second pass over an already-translated tree were NOT a no-op, then
    removing redundant passes would change what the user sees, and no
    amount of care in the marker would make it safe.
    """
    from spacr.qt.i18n import retranslate_widget_tree

    with _language("fr"):
        screen = _screen(qtbot)
        once = _captions(screen)
        retranslate_widget_tree(screen, "fr")
        qtbot.wait(1)
        twice = _captions(screen)

    assert once == twice


def test_the_guard_above_actually_fails_when_a_caption_is_stranded(qtbot,
                                                                   monkeypatch):
    """Proof, not assertion: plant the defect and watch the guard catch it.

    A guard written for a change that has not been made yet is worth exactly
    as much as the evidence that it can fail. So this builds a screen with
    the translation pass suppressed -- the state a deduplicating marker would
    produce if it marked a widget done without doing it -- and asserts that
    the post-build pass then MOVES widgets, which is the failure
    :func:`test_a_full_pass_over_a_built_screen_changes_nothing` reports.

    If this test ever stops finding movement, the guard above has stopped
    being able to see anything and is passing for the wrong reason.
    """
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QWidget

    from spacr.qt import i18n

    real = i18n.retranslate_widget_tree

    def snapshot(root):
        out = {}
        for index, widget in enumerate([root] + list(root.findChildren(QWidget))):
            try:
                text = (str(widget.text() or "")
                        if isinstance(widget, (QLabel, QAbstractButton, QGroupBox))
                        else "")
                out[index] = (text, str(widget.toolTip() or ""))
            except RuntimeError:
                continue
        return out

    code = "de"
    with _language(code):
        # Build with the pass suppressed everywhere it is reached by name.
        monkeypatch.setattr(i18n, "retranslate_widget_tree",
                            lambda *a, **k: None)
        screen = _screen(qtbot)
        qtbot.wait(5)
        monkeypatch.undo()
        before = snapshot(screen)
        real(screen, code)
        qtbot.wait(1)
        after = snapshot(screen)

    moved = {k for k in before if k in after and before[k] != after[k]}
    assert moved, (
        "a screen built with the translation pass suppressed was left "
        "unchanged by a full pass afterwards, so the guard above cannot "
        "detect a stranded caption and is passing for the wrong reason")
