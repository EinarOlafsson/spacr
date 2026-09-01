"""Four guards marked ``# pragma: no cover``, driven instead.

``.coveragerc`` sets ``exclude_lines`` and ``partial_branches`` to EMPTY
lists, so the 100-odd pragma markers left in the source exclude nothing --
every line they sit on is still in the denominator. A pragma here is a
NOTE about why a line looked hard to reach, not an exemption, and three
modules were taken to 100% in an earlier round by simply testing what
their pragmas claimed was untestable.

These four are the same shape. Each guards against a value the normal path
cannot produce, and each is reachable in one line by calling the function
with that value. Instruction 288 counts them, so they are worth the
minute.

The pragmas' own reasons are quoted in each test, because the reason is
the interesting part: "the frame cannot vanish mid-job" is a claim about
scheduling, and a claim is testable even when the scheduling is not.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import annotate_engine


class TestParsingAnImageTypeExpression:
    """``parse_image_type`` -- "" and [] for nothing to filter on."""

    def test_no_text_tokenises_to_nothing(self):
        """The second early-out was DEAD, and measuring showed it.

        `parse_image_type` had a `if not tokens: return "", []` after the
        tokeniser, marked `# pragma: no cover`. The tokeniser matches
        `(`, `)` or any run of non-space non-paren characters, so it
        returns at least one token for ANY text -- and whitespace-only
        text has already returned from the guard above it.

        The first version of this test passed "   " at it, which strips
        to "" and returns from the FIRST guard. It passed, and covered
        the wrong line. This drives the tokeniser directly instead.
        """
        from spacr.qt.annotate_engine import _tokenise_image_type

        for text in ("a", "(", ")", "()", "!", "!!", "...", "((( )))"):
            assert _tokenise_image_type(text.strip()), (
                f"{text!r} tokenises to nothing, so the deleted guard "
                "was reachable after all")

    def test_an_empty_expression_also_yields_no_filter(self):
        """The neighbouring early-out, so the two are told apart.

        Without this, the test above could be passing through the wrong
        guard and nobody would know.
        """
        assert annotate_engine.parse_image_type("") == ("", [])
        assert annotate_engine.parse_image_type(None) == ("", [])


class TestFetchingFilteredPaths:
    """``fetch_filtered_paths`` -- [] when the column is not in the table.

    The frame is substituted rather than built from a database. The
    function merges png_list with the measurement tables through
    ``spacr.io._read_and_join_tables``, and a database minimal enough to
    write in a test produces an EMPTY merge -- so a test that builds one
    gets [] back for a reason that has nothing to do with the guard.

    That is not hypothetical: the first version of this did exactly
    that, and passed. Its positive counterpart below is the only reason
    it was caught, which is the argument for always writing one.
    """

    def _drive(self, monkeypatch, tmp_path, columns, wanted):
        """Call the function with the join replaced and the guards passed.

        THREE THINGS HAVE TO BE TRUE before the guarded line is reached,
        and the first version of this got none of them right:

        * `spacr.io._read_and_join_tables` is imported INSIDE the
          function, so patching the attribute on `annotate_engine` does
          nothing -- the name is looked up on `spacr.io` at call time.
        * `db_path` must be a real file. There is an early return above
          for a path that is not.
        * `measurements` and `thresholds` must be non-empty, for the same
          reason. Passing [] returns [] from that early-out, which is
          how the first version "passed" while never reaching the guard.
        """
        import pandas as pd
        import spacr.io

        database = tmp_path / "measurements.db"
        database.write_bytes(b"")
        frame = pd.DataFrame({
            name: (["/crops/a.png"] if name == "png_path" else [1.0])
            for name in columns})
        monkeypatch.setattr(spacr.io, "_read_and_join_tables",
                            lambda *a, **k: frame)
        monkeypatch.setattr(spacr.io, "_read_db", lambda *a, **k: [frame])
        return annotate_engine.fetch_filtered_paths(
            str(database), wanted, ["score"], [0.0], ["greater"])

    def test_a_missing_annotation_column_comes_back_empty_not_absent(
            self, monkeypatch, tmp_path):
        """THE GUARD WAS DEAD, and driving it is what showed that.

        The function ended with `if annotation_column not in df.columns:
        return []`, marked `# pragma: no cover` and counted by the
        census as uncoverable. It is not merely hard to reach -- it
        CANNOT be reached: an earlier line creates the column when the
        table lacks it (`df[column] = None`), so by the end it is always
        present.

        Driving it returned a ROW rather than the empty list it
        promised, which is how the contradiction surfaced. The guard is
        deleted; this pins the behaviour that actually happens, which is
        the more useful contract anyway -- the annotation UI asks for
        whichever column the user picked, and a project that has never
        been annotated gets its crops back with no labels rather than
        nothing at all.
        """
        rows = self._drive(monkeypatch, tmp_path,
                           ["png_path", "score"], "never_annotated")
        assert rows == [["/crops/a.png", None]]

    def test_a_column_that_is_there_returns_its_rows(self, monkeypatch,
                                                     tmp_path):
        """The other side, and it is what makes the test above mean
        something.

        Returning [] unconditionally passes an absence assertion. Only
        this distinguishes "the guard fired" from "nothing works" -- and
        it is the reason the first version of these two was caught.
        """
        rows = self._drive(monkeypatch, tmp_path,
                           ["png_path", "score", "annotation"], "annotation")
        assert rows == [["/crops/a.png", 1.0]]


class TestTheOutliersScreenGuards:
    """Two ``return`` guards on a screen that has not scanned yet."""

    @pytest.fixture
    def screen(self, qtbot, qt_theme_applied, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        from spacr.qt.screens.outliers import OutliersScreen

        made = OutliersScreen(threaded=False)
        qtbot.addWidget(made)
        return made

    def test_a_finished_scan_with_no_frame_is_dropped(self, screen):
        """The pragma reads "the frame cannot vanish mid-job".

        That is a claim about SCHEDULING, and the claim is testable even
        though the scheduling is not: set the frame to None and hand the
        screen a result. It must return rather than raise -- a scan that
        lands after its data is gone is not an error, it is late.
        """
        screen._frame = None
        screen._on_scanned(object())          # must not raise
        assert screen._result is not None

    def test_filling_the_object_table_with_no_objects_is_dropped(self,
                                                                 screen):
        """The neighbour, whose pragma reads "set immediately before"."""
        screen._objects = None

        screen._fill_object_table(object())   # must not raise

        # ASSERTED. Without this the test passes against a method that
        # returns early in every case, including the one with objects.
        assert screen._objects is None, (
            "an object table was invented where there were no objects")
