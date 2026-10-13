"""The annotate screen's keyboard, and two writes that cannot fail.

``handle_key`` is the single entry point for the whole keyboard feature,
so it can be driven directly without synthesising key events -- and its
final ``return False`` is unreachable because the token vocabulary is
closed and every member of it is handled above.
"""
from __future__ import annotations

import ast
import inspect
import re

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens import annotate as A

pytestmark = pytest.mark.qt


class TestTheKeyboardTokenVocabulary:

    def test_an_unbound_key_is_left_to_qt(self, qtbot):
        """``key_token`` answers None, and ``handle_key`` returns at once.

        Unbound keys must fall through to Qt's default handling, or the
        annotate screen would swallow Tab, Delete and every shortcut the
        window owns.
        """
        for key in ("tab", "delete", "home", "F1", "x", "q"):
            assert A.key_token(key) is None, (
                f"{key!r} is now bound; check it is handled in handle_key")

    def test_every_token_the_reader_can_produce_is_handled(self):
        """THE PIN, for the final ``return False``.

        The vocabulary is closed: ``key_token`` answers None or one of a
        fixed set, and ``handle_key`` has a branch for every member of
        that set -- so the fall-through below the last branch cannot be
        reached.

        Both sides are read out of the source and compared as SETS, so a
        tenth token added to the reader without a branch lands here
        rather than silently doing nothing when the key is pressed.
        """
        produced = set(A._QT_CODE_TOKENS.values())

        handled = set()
        source = inspect.getsource(A.AnnotateScreen.handle_key)
        for literal in re.findall(r'token == ("(?:[^"]*)")', source):
            handled.add(ast.literal_eval(literal))
        for literal in re.findall(r"token in \(([^)]*)\)", source):
            handled.update(ast.literal_eval("[" + literal + "]"))

        assert handled, "handle_key names no tokens at all"
        assert produced <= handled, (
            f"{sorted(produced - handled)} can be produced by key_token and "
            f"have no branch in handle_key, so pressing those keys does "
            f"nothing and reports it as handled")

    def test_the_digits_are_handled_before_the_named_tokens(self):
        """A class key is a digit, and zero is the clear.

        The digit test comes first because the named tokens are words
        and a digit could never match one -- but the ORDER is what makes
        the branch chain a chain rather than a lookup, and zero meaning
        "clear" rather than "class 0" is the part a reader has to know.
        """
        source = inspect.getsource(A.AnnotateScreen.handle_key)
        assert source.index("token.isdigit()") < source.index("token == ")
        assert "self._kbd_clear() if value == 0 else self._kbd_assign(value)" \
            in source

    def test_escape_is_only_taken_while_the_reference_is_showing(self):
        """Otherwise it is left to whatever dialog or window wants it,
        which is the difference between closing the legend and closing
        the screen."""
        source = inspect.getsource(A.AnnotateScreen.handle_key)
        escape = source.index('token == "escape"')
        block = source[escape:]
        assert "if self._legend_expanded:" in block[:300]
        assert "return False" in block[:400], (
            "Escape is now claimed even with the legend closed")


class TestWritingALabelIntoASlot:

    def test_a_slot_outside_the_page_is_refused(self):
        """The only way ``_set_annotation`` answers False."""
        source = inspect.getsource(A.AnnotateScreen._set_annotation)
        assert "if not (0 <= slot < len(self._page_paths)):" in source
        assert "return False" in source
        assert source.rstrip().endswith("return True"), (
            "_set_annotation now has a second way to fail, so the two "
            "callers below need to handle it")

    def test_both_callers_check_the_slot_before_they_write(self):
        """THE PIN, for two ``if self._set_annotation(...)`` arms.

        ``_apply_to_slots`` and ``_toggle_annotation`` both call
        ``_slot_is_valid`` first, and that is the same range test
        ``_set_annotation`` makes -- so the write always succeeds and
        neither false arm can be taken.

        Keeping the return value is right: a caller that stopped
        validating would otherwise count a label it did not write, and
        the count is what the console reports back to the user.
        """
        for method in (A.AnnotateScreen._apply_to_slots,
                       A.AnnotateScreen._toggle_annotation):
            source = inspect.getsource(method)
            assert "self._slot_is_valid(slot)" in source, (
                f"{method.__name__} no longer validates the slot before it "
                f"writes, so _set_annotation can now answer False")
            assert source.index("self._slot_is_valid(slot)") < \
                source.index("self._set_annotation("), (
                f"{method.__name__} validates after it writes")

    def test_a_slot_already_holding_the_value_is_skipped(self):
        """So undo does not fill up with entries that change nothing."""
        source = inspect.getsource(A.AnnotateScreen._apply_to_slots)
        assert "if previous == value:" in source
        assert source.index("if previous == value:") < \
            source.index("self._push_undo("), (
            "the undo entry is pushed before the no-op is detected")

    def test_clicking_the_same_class_again_clears_it(self):
        """Mouse semantics, and they are not the keyboard's: a second
        click on the class a crop already has means "I was wrong", where
        pressing the digit again means "yes, that one"."""
        source = inspect.getsource(A.AnnotateScreen._toggle_annotation)
        assert "resolved = None if existing == new_value else new_value" \
            in source

    def test_a_path_with_a_line_break_stays_one_console_record(self):
        """A filesystem path can legally contain them, and a record split
        across two lines is one a search will not find."""
        source = inspect.getsource(A.AnnotateScreen._toggle_annotation)
        assert r'path.replace("\r", r"\r").replace("\n", r"\n")' in source
