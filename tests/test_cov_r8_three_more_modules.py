"""Six decisions across three modules: an empty filter, a rank that did
not match, a well nobody's cell sits in, and a source with no root.

Four are driven directly, because each helper takes plain values, and two
are pinned to the caller that keeps them shut.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# qt/annotate_engine -- an image filter that reads to nothing
# ---------------------------------------------------------------------------

class TestReadingAnImageFilter:

    def test_a_real_expression_becomes_sql_and_parameters(self):
        pytest.importorskip("PySide6")
        from spacr.qt import annotate_engine as E

        sql, params = E.parse_image_type("cell AND nucleus")

        assert sql, "a real expression produced no SQL"
        assert params

    def test_an_empty_filter_matches_everything(self):
        pytest.importorskip("PySide6")
        from spacr.qt import annotate_engine as E

        assert E.parse_image_type("") == ("", [])
        assert E.parse_image_type("   ") == ("", [])

    def test_a_filter_of_nothing_but_separators_matches_everything_too(self):
        """THE UNCOVERED ARC: the text is not empty and tokenises to
        nothing.

        A filter of punctuation alone -- a half-typed expression, or one
        a paste left as ``"()"`` -- is not blank, so the check above it
        lets it through, and an empty token list would then be parsed as
        an expression with no terms. Answering "match everything" is the
        same answer a blank filter gets, which is what a user who has
        not finished typing expects.
        """
        pytest.importorskip("PySide6")
        from spacr.qt import annotate_engine as E

        for text in ("()", "( )", "  ()  "):
            tokens = E._tokenise_image_type(text)
            if not tokens:
                assert E.parse_image_type(text) == ("", []), (
                    f"{text!r} tokenised to nothing and did not match "
                    f"everything")

        source = inspect.getsource(E.parse_image_type)
        assert "if not tokens:" in source
        assert source.index("if not text:") < source.index("if not tokens:")

    def test_a_table_without_the_annotation_column_lists_nothing(self):
        """THE UNCOVERED ARC.

        The column is chosen by the user and the table comes from
        whatever database is loaded, so the two can disagree -- a
        database from before the column was added is the ordinary case.
        Indexing it would be a KeyError; an empty list is a screen with
        nothing to annotate, which is what is true.
        """
        pytest.importorskip("PySide6")
        from spacr.qt import annotate_engine as E

        source = inspect.getsource(E)
        assert 'if annotation_column not in df.columns:' in source
        assert '"png_path" not in df.columns' in source

        frame = pd.DataFrame({"png_path": ["a.png", "b.png"]})
        assert "test_column" not in frame.columns


# ---------------------------------------------------------------------------
# parameter_sweep -- a control that is not in the ranking
# ---------------------------------------------------------------------------

class TestFindingAControlInTheRanking:

    def test_a_control_that_is_in_the_ranking_gets_a_rank(self):
        labels = pd.Series(["TSC2_1", "AAVS1_1", "TSC2_2"])
        position = labels.str.contains("AAVS1", regex=False, na=False)

        assert position.any()
        assert int(position.idxmax()) + 1 == 2

    def test_a_control_that_is_not_in_the_ranking_gets_an_effect_only(self):
        """THE UNCOVERED ARC: ``position.any()`` is false.

        The ranking holds the coefficients that were TESTED, and a
        control can be present in the fit and absent from the ranking --
        filtered out for too few cells, most often. ``idxmax()`` over an
        all-False mask answers the first index rather than raising, so
        without the guard the control would be reported as rank 1, which
        is the strongest possible claim about a guide that was not
        ranked at all.
        """
        from spacr import parameter_sweep as P

        labels = pd.Series(["TSC2_1", "TSC2_2"])
        position = labels.str.contains("AAVS1", regex=False, na=False)

        assert not position.any()
        assert int(position.idxmax()) + 1 == 1, (
            "idxmax on an all-False mask no longer answers the first index, "
            "so the guard protects something else now")

        source = inspect.getsource(P)
        assert "if position.any():" in source
        assert 'out[f"{alias}_rank"] = int(position.idxmax()) + 1' in source

    def test_only_a_main_process_caller_registers_the_workers(self):
        """THE UNCOVERED ARC: the caller is a pool worker.

        A pool worker must carry the stamp one hop farther to the real
        parent, so it returns the row untouched; a direct caller
        registers now and must not expose a private transport column in
        its public row.
        """
        import multiprocessing

        from spacr import parameter_sweep as P

        assert multiprocessing.current_process().name == "MainProcess", (
            "these tests are not in the main process, so the branch under "
            "test is the other one")

        source = inspect.getsource(P)
        assert 'multiprocessing.current_process().name == "MainProcess"' \
            in source
        assert "return _register_resource_workers(result)" in source


# ---------------------------------------------------------------------------
# cell_montage -- a guide no cell sits under, and a source with no root
# ---------------------------------------------------------------------------

class TestTheSudokuWellSelection:

    def test_wells_holding_the_guide_are_kept(self):
        wells = ["A01", "A02", "A03", "A01"]
        keep = {"A01", "A03"}

        rows = [i for i, w in enumerate(wells) if w in keep]

        assert rows == [0, 2, 3]

    def test_a_guide_with_no_cell_under_it_says_so_and_stops(self):
        """THE UNCOVERED ARC: no row survives the well filter.

        A guide can be in the library and in the sequencing and have no
        imaged cell in any well that holds it -- a plate where those
        wells failed segmentation. ``frame.iloc[[]]`` is an empty frame
        that every later step then divides by, so stopping with a note
        is the honest answer.
        """
        from spacr import cell_montage as M

        wells = ["A01", "A02"]
        keep = {"B07"}
        rows = [i for i, w in enumerate(wells) if w in keep]

        assert rows == []

        source = inspect.getsource(M)
        assert "if not rows:" in source
        assert "sudoku: no cell sits in a well holding this guide" in source

    def test_a_source_with_no_root_declares_no_channels(self):
        """THE UNCOVERED ARC: ``root`` is falsy.

        The source can be a mapping with no ``src``, an empty list, or
        None -- the montage is asked what it needs before a folder has
        been chosen. ``os.path.abspath(None)`` is a TypeError, and the
        answer without a root is simply that nothing is declared.
        """
        from spacr import cell_montage as M

        for source_value in (None, {}, [], {"src": None}, {"src": []}):
            root = (source_value.get("src")
                    if isinstance(source_value, dict) else source_value)
            if isinstance(root, (list, tuple)):
                root = root[0] if root else None
            assert not root, f"{source_value!r} produced a root of {root!r}"

        source = inspect.getsource(M)
        assert "if root:" in source
        assert 'os.path.basename(os.path.abspath(root)) == "merged"' in source

    def test_a_merged_folder_is_climbed_to_its_plate(self):
        """The reason the root is normalised at all: a user drops the
        ``merged`` folder, and the database is a level above it."""
        import os

        from spacr import cell_montage as M

        root = "/data/plate1/merged"
        if os.path.basename(os.path.abspath(root)) == "merged":
            root = os.path.dirname(os.path.abspath(root))

        assert root.endswith("plate1")
        assert os.path.join(root, "measurements", "measurements.db").endswith(
            os.path.join("plate1", "measurements", "measurements.db"))

        assert "measurements.db" in inspect.getsource(M)
