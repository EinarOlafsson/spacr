"""A validated regex survives a confirmation editor that fails while it is open.

When a dropped folder's filenames already match, ``_open_regex_editor`` is a
REVIEW step (``confirming=True``): dismissing it keeps the detected pattern,
and so must a dialog that fails. ``test_cov_dnd_handlers.py`` holds the half
where the editor cannot even be imported. This file holds the half where it
is built and then raises.

WHY A TEST OF ITS OWN. That arm was covered only incidentally, by
``test_dnd_dropzone.py``: a drop that validates opens the real editor, and the
suite's autouse ``_no_unguarded_modals`` guard makes its ``exec()`` raise. That
file is the one whose xdist worker segfaulted in CI, and each crash took the
line with it: ``dnd_handlers.py`` moved between 43 and 44 uncovered statements
with no source change. Here the failure is the test's own, in a file that
does not crash.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

PATTERN = "(?P<plateID>.*)_(?P<wellID>[A-P]\\d+)"


@pytest.fixture
def failing_editor(monkeypatch):
    """An editor that is built, then fails as it opens; pushes and logs kept."""
    import spacr.qt.regex_editor as editor_mod
    from spacr.qt import dnd_handlers as dh

    class _Editor:
        regex = None

        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            raise RuntimeError("the editor's event loop could not start")

    monkeypatch.setattr(editor_mod, "RegexEditorDialog", _Editor)
    pushed, logged = [], []
    monkeypatch.setattr(dh, "_push_regex_to_screen",
                        lambda pattern, screen: pushed.append(pattern))
    monkeypatch.setattr(dh, "_log", lambda screen, text: logged.append(text))
    return dh, pushed, logged


def test_the_detected_pattern_is_kept_when_the_review_editor_fails(
        failing_editor):
    dh, pushed, logged = failing_editor

    dh._open_regex_editor(["P1_A01.tif"], PATTERN, object(),
                          confirming=True, fallback=PATTERN)

    assert pushed == [PATTERN], "the screen was left with no regex at all"
    assert any("regex editor failed" in text for text in logged)


def test_an_editor_opened_to_fix_a_bad_match_keeps_nothing_when_it_fails(
        failing_editor):
    """The control: without a validated pattern there is nothing to keep."""
    dh, pushed, logged = failing_editor

    dh._open_regex_editor(["P1_A01.tif"], "", object())

    assert pushed == []
    assert any("regex editor failed" in text for text in logged)
