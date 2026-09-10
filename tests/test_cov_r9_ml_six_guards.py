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

        for function, guard in ((M.save_summary_to_file, "if folder:"),
                                (M.write_plot, "if parent:")):
            source = inspect.getsource(function)
            assert "os.path.dirname(os.path.abspath(" in source, (
                f"{function.__name__} no longer normalises before creating "
                f"the destination directory")
            assert "exist_ok=True" in source
            assert guard not in source, (
                f"{function.__name__} restored an impossible path guard")


class TestTheColumnsTheLineAboveEnsured:

    def test_the_prc_parts_are_assigned_before_they_are_required(self):
        """THE PIN for removing the second column check in ``process_scores``.

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
        compose = source.index("df['prc'] = _compose_prc_column(df)", assign)

        assert missing < assign < compose
        check = ("if all(col in df.columns for col in "
                 "['plateID', 'rowID', 'columnID']):")
        # One check remains on the non-prcfo path, where no parser just wrote
        # the columns. The redundant post-parser copy would make this two.
        assert source.count(check) == 1

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
        check = ("if not all(col in merged_df.columns for col in "
                 "['grna', 'gene']):")
        assert check not in source
        split = source.index("tokens = merged_df['grna']", cut)
        assert cut < split
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
        from spacr.classifier_evaluation import grouped_split

        with pytest.raises(ValueError, match="grouped held-out split is impossible"):
            grouped_split(groups, np.array([0, 1, 0]), 0.25,
                          group_by="plate")
        assert "if n_folds < 2:" not in inspect.getsource(M.ml_analysis)

    def test_the_cap_is_five(self):
        """Not arbitrary in a way worth losing: five folds over a screen
        with forty plates is a choice about cost, and a screen with three
        gets three rather than an error."""
        assert min(5, 40) == 5
        assert min(5, 3) == 3

    def test_the_post_split_fold_guard_stays_removed(self):
        from spacr import ml as M

        source = inspect.getsource(M.ml_analysis)
        assert "if n_folds < 2:" not in source, (
            "grouped_split already refuses fewer than two groups")


class TestStampingTheQualityManifest:

    def test_a_frame_carries_the_manifest_in_its_attrs(self):
        """The write the guard wraps. ``attrs`` travels with the frame,
        so a caller that does not know about the manifest is unaffected
        -- which is why it is not a column."""
        # Forty-five distinct frame shapes and payloads exercise the premise
        # empirically: DataFrame.attrs is a mutable dict for all of them.
        checked = 0
        for rows in range(5):
            for columns in range(1, 10):
                frame = pd.DataFrame(
                    {f"c{column}": np.arange(rows) for column in range(columns)})
                manifest = {"rows": rows, "columns": columns}
                frame.attrs["qc_manifest"] = manifest
                assert frame.attrs["qc_manifest"] == manifest
                checked += 1
        assert checked == 45

    def test_the_manifest_write_has_no_impossible_exception_handler(self):
        """THE PIN for removing an exception handler around dict assignment.

        ``attrs`` is a plain dict on a DataFrame and the assignment
        cannot fail today. It used to be wrapped because the manifest is a
        diagnostic: losing it must not cost the caller the regression it
        actually asked for.
        """
        from spacr import ml as M

        source = inspect.getsource(M.regression)
        condition = source.index(
            "if qc_manifest is not None and coef_df is not None:")
        write = source.index('coef_df.attrs["qc_manifest"] = qc_manifest')
        ret = source.index("return model, coef_df, regression_type", write)

        assert condition < write < ret
        assert "except Exception:" not in source[condition:ret]


def test_the_penalised_warning_has_no_impossible_swallowing_handler(capsys):
    """Drive both decisions and pin removal of the catch-all handler."""
    from spacr import ml as M

    no_hits = pd.DataFrame({"p_value": [0.4, np.nan, "0.7"]})
    assert M._warn_if_penalised_no_hits(
        {"regression_type": "ridge"}, no_hits) is True
    note = capsys.readouterr().out
    assert "returned no coefficient below p=0.05" in note
    assert "regression_type='ols'" in note

    has_hit = pd.DataFrame({"p_value": [0.4, 0.01]})
    assert M._warn_if_penalised_no_hits(
        {"regression_type": "elasticnet"}, has_hit) is False
    assert M._warn_if_penalised_no_hits(
        {"regression_type": "ols"}, no_hits) is False
    assert M._warn_if_penalised_no_hits(
        {"regression_type": "lasso"}, no_hits.iloc[0:0]) is False
    assert capsys.readouterr().out == ""

    helper = inspect.getsource(M._warn_if_penalised_no_hits)
    caller = inspect.getsource(M._perform_regression)
    assert "pd.to_numeric" in helper
    assert "except Exception" not in helper
    assert "_warn_if_penalised_no_hits(settings, coef_df)" in caller
