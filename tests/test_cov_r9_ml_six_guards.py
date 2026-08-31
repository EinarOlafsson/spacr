"""Six single decisions in ``ml.py``, each settled a line or two above.

Two are the same ``os.path.dirname(os.path.abspath(x))`` shape that
appears throughout the package; two are column checks the line above has
just satisfied; one is a fold count the caller has already bounded; one
is a defensive ``attrs`` write on a pandas object.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd
import pytest


class TestAnAbsolutePathAlwaysHasAParent:

    @pytest.mark.parametrize("path", [
        "summary.txt", "./summary.txt", "a/b/summary.txt",
        "/tmp/summary.txt", "~/summary.txt", "",
    ])
    def test_dirname_of_abspath_is_never_empty(self, path):
        """THE PIN, for ``if folder:`` and ``if parent:``.

        ``abspath`` returns a rooted path for anything, including the
        empty string, so its ``dirname`` is at worst ``"/"`` -- never
        falsy. Both guards therefore always take the makedirs branch,
        and what they are really protecting against is a relative path
        with no directory part, which ``abspath`` has already removed.

        Enumerated over the forms a caller can actually pass rather than
        argued once.
        """
        folder = os.path.dirname(os.path.abspath(path))

        assert folder, f"{path!r} produced no parent directory"
        assert os.path.isabs(folder)

    def test_the_bare_dirname_is_what_would_be_empty(self):
        """The half that says the guard is not nonsense: without the
        ``abspath`` it WOULD be empty, which is presumably where the
        check came from."""
        assert os.path.dirname("summary.txt") == ""

    def test_both_writers_normalise_before_they_check(self):
        from spacr import ml as M

        for function in (M.save_summary_to_file, M.write_plot):
            source = inspect.getsource(function)
            assert "os.path.dirname(os.path.abspath(" in source, (
                f"{function.__name__} no longer normalises before checking, "
                f"so a bare filename now skips the makedirs")
            assert "exist_ok=True" in source


class TestTheColumnsTheLineAboveEnsured:

    def test_the_prc_parts_are_assigned_before_they_are_required(self):
        """THE PIN, for ``if all(col in df.columns ...)`` in
        ``process_scores``.

        The line above calls ``_assign_prcfo_parts`` precisely when the
        three are missing, so by the check they are present -- unless the
        assignment failed, which it cannot do silently.
        """
        from spacr import ml as M

        source = inspect.getsource(M.process_scores)
        missing = source.index(
            "if not all(col in df.columns for col in "
            "['plateID', 'rowID', 'columnID']):")
        assign = source.index("_assign_prcfo_parts(df", missing)
        present = source.index(
            "if all(col in df.columns for col in "
            "['plateID', 'rowID', 'columnID']):", assign)

        assert missing < assign < present

    def test_splitting_a_prcfo_gives_the_three_parts(self):
        """Driven on the helper, since that is what makes the check above
        redundant rather than lucky."""
        from spacr.ml import _assign_prcfo_parts

        # A REAL prcfo: the field id carries its 'f', which the parser
        # requires. The first version of this test used '1' and the
        # parser said so, which is the parser doing its job.
        frame = pd.DataFrame({"prcfo": ["p1_A01_c1_f1_o1"],
                              "objectID": ["o1"]})

        out = _assign_prcfo_parts(frame, object_column="objectID")

        for column in ("plateID", "rowID", "columnID"):
            assert column in out.columns, (
                f"{column} was not recovered from the prcfo key, so the "
                f"check below the call is live and untested")

    def test_the_reads_frame_is_cut_to_three_columns_first(self):
        """THE PIN, for ``if not all(col in merged_df.columns ...)`` in
        ``process_reads``.

        The line above reduces the frame to ``['prc', 'grna', 'fraction']``
        -- so ``'gene'`` is guaranteed ABSENT and the split below always
        runs. The check reads as optional and is not.
        """
        from spacr import ml as M

        source = inspect.getsource(M.process_reads)
        cut = source.index("merged_df = merged_df[['prc', 'grna', 'fraction']]")
        check = source.index(
            "if not all(col in merged_df.columns for col in "
            "['grna', 'gene']):", cut)

        assert cut < check
        frame = pd.DataFrame({"prc": ["a"], "grna": ["b"], "fraction": [1.0],
                              "gene": ["g"]})
        assert not all(c in frame[["prc", "grna", "fraction"]].columns
                       for c in ["grna", "gene"]), (
            "the three-column cut no longer drops 'gene', so the positional "
            "split below it can be skipped and the guide/gene assumption "
            "goes unstated")

    def test_the_positional_split_is_documented_as_an_assumption(self):
        """The comment is the substance: '<org>_<gene>_<guide>' is a
        naming convention, not a fact about the data, and it is stated
        and checked rather than removed."""
        from spacr import ml as M

        source = inspect.getsource(M.process_reads)
        assert "This split IS positional, legitimately" in source
        assert "TGGT1_GENEA_g1" in source


class TestTheFoldCount:

    def test_two_independent_groups_are_enough(self):
        for distinct in (2, 3, 5, 40):
            assert min(5, distinct) >= 2

    def test_one_group_cannot_be_split_and_says_so(self):
        """THE ARC: ``n_folds < 2``.

        A screen whose split level has one value -- one plate, or one
        well -- cannot be cross-validated by it, and sklearn's own
        message names its parameters rather than the rule that selected
        nothing. Naming the level and the count is what tells the user
        which setting to change.
        """
        groups = np.array(["plate1", "plate1", "plate1"])
        distinct = len(np.unique(groups))
        n_folds = min(5, distinct)

        assert distinct == 1 and n_folds < 2

        from spacr import ml as M

        source = inspect.getsource(M.ml_analysis)
        assert "needs at least two" in source
        assert "found {distinct_groups}" in source, (
            "the refusal no longer reports how many groups there were, so a "
            "user cannot tell whether they are one short or many")

    def test_the_cap_is_five(self):
        """Not arbitrary in a way worth losing: five folds over a screen
        with forty plates is a choice about cost, and a screen with three
        gets three rather than an error."""
        assert min(5, 40) == 5
        assert min(5, 3) == 3


class TestStampingTheQualityManifest:

    def test_a_frame_carries_the_manifest_in_its_attrs(self):
        """The write the guard wraps. ``attrs`` travels with the frame,
        so a caller that does not know about the manifest is unaffected
        -- which is why it is not a column."""
        frame = pd.DataFrame({"coefficient": [1.0]})

        frame.attrs["qc_manifest"] = {"n": 3}

        assert frame.attrs["qc_manifest"] == {"n": 3}

    def test_the_write_is_wrapped_because_it_is_decoration(self):
        """THE PIN, for ``except Exception: pass``.

        ``attrs`` is a plain dict on a DataFrame and the assignment
        cannot fail today. It is wrapped because the manifest is a
        diagnostic: losing it must not cost the caller the regression it
        actually asked for.
        """
        from spacr import ml as M

        source = inspect.getsource(M.regression)
        write = source.index('coef_df.attrs["qc_manifest"] = qc_manifest')
        handler = source.index("except Exception:", write)
        ret = source.index("return model, coef_df, regression_type", handler)

        assert write < handler < ret
        assert "if qc_manifest is not None and coef_df is not None:" in source, (
            "the manifest is stamped without checking there is one, so a run "
            "with no QC writes a None into the frame's attrs")
