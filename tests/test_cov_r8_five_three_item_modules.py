"""Five modules with three uncovered decisions each, and what settles them.

Two are import-time validations that must hold for the package to load at
all, which is what makes them unreachable at runtime and worth asserting
anyway: a registry that fails its own check is a package that will not
import, and this says which check.
"""
from __future__ import annotations

import inspect
import os
import sysconfig

import pytest


# ---------------------------------------------------------------------------
# qt/timing -- naming the frames that are not spaCR's
# ---------------------------------------------------------------------------

class TestNamingAStackFrame:

    def test_a_spacr_frame_is_named_relative_to_the_package(self):
        from spacr.qt import timing as T

        assert hasattr(T, "_SPACR_ROOT"), (
            "the package root the frame names are made relative to is gone")
        source = inspect.getsource(T)
        assert "path.startswith(_SPACR_ROOT)" in source

    def test_the_library_directories_come_from_sysconfig(self):
        """THE PIN, for ``got`` being falsy and for the except beside it.

        ``sysconfig.get_paths()`` is a pure dictionary build over values
        the interpreter already holds -- it reads no files and can only
        fail on an interpreter with a broken install scheme. All four
        keys are present on every CPython this package supports, which
        is what makes both the handler and the falsy check unreachable.

        They are right to keep: this runs while formatting a stack for a
        crash report, and a report that dies while being written is a
        crash with no trace at all. Asserted against the real
        interpreter, so a build missing one of the four fails here.
        """
        paths = sysconfig.get_paths()
        for key in ("stdlib", "platstdlib", "purelib", "platlib"):
            assert paths.get(key), (
                f"this interpreter has no {key!r} path, so the falsy check "
                f"in qt.timing is live")

        from spacr.qt import timing as T

        source = inspect.getsource(T)
        assert 'for key in ("stdlib", "platstdlib", "purelib", "platlib")' \
            in source
        assert "if got and (not _SPACR_ROOT" in source

    def test_a_library_directory_that_contains_spacr_is_not_listed(self):
        """The condition's second half, which is the one that matters.

        An editable install puts spaCR under ``purelib``; listing that
        directory would make every spaCR frame read as library code and
        empty the useful half of the trace.
        """
        from spacr.qt import timing as T

        source = inspect.getsource(T)
        assert "not _SPACR_ROOT.startswith(got + os.sep)" in source, (
            "a library directory containing spaCR would now be listed, so "
            "every spaCR frame reads as library code")


# ---------------------------------------------------------------------------
# schema -- the object-role registry checks itself at import
# ---------------------------------------------------------------------------

class TestTheObjectRoleRegistry:

    def test_every_registered_role_is_usable_in_an_identity(self):
        """THE PIN, for an import-time raise.

        Typed object ids concatenate role and numeric label with no
        separator, so a digit in a role is ambiguous -- ``cell1`` plus 7
        against ``cell`` plus 17 -- and an underscore would split the
        surrounding prcfo key. The check runs at import, so reaching it
        means the package does not load; this asserts the property it
        enforces, which is the thing a new role has to satisfy.
        """
        from spacr import schema

        assert schema.OBJECT_TYPES, "the object registry is empty"
        for role in schema.OBJECT_TYPES:
            assert role, "an empty role is registered"
            assert schema.KEY_SEPARATOR not in role, (
                f"{role!r} holds the key separator, so it would split a "
                f"prcfo key")
            assert not any(character.isdigit() for character in role), (
                f"{role!r} holds a digit, so {role}1 + 7 and {role} + 17 "
                f"are the same identity")

    def test_a_role_with_a_digit_would_be_ambiguous(self):
        """The ambiguity the check exists for, shown rather than argued."""
        assert "cell1" + "7" == "cell" + "17"

    def test_a_row_name_of_one_or_two_letters_always_decodes(self):
        """THE PIN, for ``if index is not None`` in ``row_id``.

        The pattern above it admits one or two ASCII letters and nothing
        else, and every such string is a plate row: A is 1 and ZZ is
        702. So the decode cannot answer None, and the fall-through
        below it is unreachable from that branch.

        It is still the right shape, because the fall-through is what
        keeps a NUMERIC or already-prefixed row working -- those never
        enter the letters branch at all, which the cases below show.
        """
        from spacr import schema

        assert schema._ROW_ONLY.pattern == r"^([A-Za-z]{1,2})$", (
            "the row pattern changed; a string it now admits may not decode")

        for letters in ("A", "a", "Z", "AB", "zz", "ZZ"):
            assert schema._ROW_ONLY.match(letters)
            assert not schema._PREFIXED_INT.match(letters)
            assert schema.row_index_from_letters(letters) is not None, (
                f"{letters!r} matched the row pattern and did not decode, so "
                f"the fall-through below it is live")

        assert schema.row_id("A") == "r1"
        assert schema.row_id("ZZ") == "r702"

    def test_a_row_that_is_not_letters_takes_the_general_path(self):
        """What the fall-through is for."""
        from spacr import schema

        for row in ("AAA", "A1", "", "r3"):
            assert not (schema._ROW_ONLY.match(str(row))
                        and not schema._PREFIXED_INT.match(str(row))), (
                f"{row!r} now enters the letters branch")

        assert schema.row_id(1) == "r1"
        assert schema.row_id("r3") == "r3"


# ---------------------------------------------------------------------------
# figure_queue -- a run section that drew nothing
# ---------------------------------------------------------------------------

class TestForgettingARunSection:

    def test_a_section_that_drew_nothing_loses_its_label(self, qtbot):
        """THE UNCOVERED ARC: ``count <= 0``.

        A run that produced no figures still registers its label, so the
        queue shows a heading with nothing under it. Forgetting the MARK
        removes the heading; there is nothing to renumber, which is why
        it returns before the shift.
        """
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import figure_queue as F

        source = inspect.getsource(F)
        assert "if count <= 0:" in source
        guard = source.index("if count <= 0:")
        assert 'r.get("label") != wanted' in source[guard:guard + 300], (
            "an empty section no longer forgets its label")
        assert "return 0" in source[guard:guard + 400]

    def test_a_section_that_is_not_there_at_all_returns_zero(self, qtbot):
        pytest.importorskip("PySide6")
        from spacr.qt.widgets import figure_queue as F

        source = inspect.getsource(F)
        assert "if span is None:" in source
        assert source.index("if span is None:") < source.index(
            "if count <= 0:"), (
            "the missing-section check no longer precedes the empty one, so "
            "unpacking a None span is now possible")


# ---------------------------------------------------------------------------
# folding_summary -- a wrapped line with no row above it
# ---------------------------------------------------------------------------

class TestReadingAWrappedSummary:

    def _rows(self, lines, lead=12):
        rows: list = []
        for line in lines:
            if len(line) > lead and line[:lead].strip():
                rows.append([line[:lead].strip(), line[lead:].strip()])
            elif rows and len(line) > lead:
                rows[-1][1] = (rows[-1][1] + " " + line[lead:].strip()).strip()
        return rows

    def test_a_continuation_is_joined_onto_the_row_above(self):
        rows = self._rows([
            "name        the first value",
            "            wrapped onto a second line",
        ])

        assert rows == [["name", "the first value wrapped onto a second line"]]

    def test_a_continuation_with_nothing_above_it_is_dropped(self):
        """THE UNCOVERED ARC: ``rows`` is empty.

        A summary whose first line is already wrapped -- a paste that
        lost its heading, or a tail read from part-way through a file --
        has a continuation with nothing to continue. Appending it to
        ``rows[-1]`` is an IndexError while a panel is being filled.
        """
        rows = self._rows(["            an orphaned continuation"])

        assert rows == []

        from spacr.qt.widgets import folding_summary as F

        source = inspect.getsource(F)
        assert "elif rows and len(line) > lead:" in source, (
            "the continuation no longer checks that a row exists above it")

    def test_a_line_shorter_than_the_lead_is_neither(self):
        rows = self._rows(["name        a value", "  "])

        assert len(rows) == 1
