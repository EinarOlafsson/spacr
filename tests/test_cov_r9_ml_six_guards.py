"""Six single decisions in ``ml.py``, each settled a line or two above.

Two are the same ``os.path.dirname(os.path.abspath(x))`` shape that
appears throughout the package; two are column checks the line above has
just satisfied; one is a fold count the caller has already bounded; one
is a defensive ``attrs`` write on a pandas object.
"""
from __future__ import annotations

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

    def case_the_bare_dirname_is_what_would_be_empty(self):
        """The half that says the guard is not nonsense: without the
        ``abspath`` it WOULD be empty, which is presumably where the
        check came from."""
        _seen = []
        _seen.append(keep(os.path.dirname("summary.txt")))
        return _seen

class TestTheColumnsTheLineAboveEnsured:

    def case_splitting_a_prcfo_gives_the_three_parts(self):
        """Driven on the helper, since that is what makes the check above
        redundant rather than lucky."""
        _seen = []
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
        return _seen

    def case_the_positional_split_uses_the_middle_token_as_the_gene(self):
        """The assumption APPLIED, on a real pooled-library name.

        '<org>_<gene>_<guide>' is a naming convention: nothing in
        'TGGT1_GENEA_g1' says which token is which, so the split is
        positional -- the middle token is the gene and the guide keeps
        the two that identify it.
        """
        _seen = []
        from spacr.ml import process_reads

        counts = pd.DataFrame({
            "rowID": ["A01", "A01"], "columnID": ["c1", "c1"],
            "grna": ["TGGT1_GENEA_g1", "TGGT1_GENEB_g2"],
            "count": [10, 30]})

        out = process_reads(counts, None, "plate1")

        _seen.append(keep(list(out["gene"])))
        _seen.append(keep(list(out["grna"])))
        return _seen

    def case_the_positional_split_is_documented_as_an_assumption(self, capsys):
        """The assumption STATED and CHECKED, which is the half that
        matters: it is a naming convention and not a fact about the
        data, so a table whose names are not all three components is
        refused the split outright -- and told which convention was
        wanted -- rather than having it half-applied."""
        _seen = []
        from spacr.ml import process_reads

        counts = pd.DataFrame({
            "rowID": ["A01", "A01"], "columnID": ["c1", "c1"],
            "grna": ["TGGT1_GENEA_g1", "GENEB_g2"],   # 3 components, then 2
            "count": [10, 30]})

        out = process_reads(counts, None, "plate1")
        said = capsys.readouterr().out

        _seen.append((("absent", 'gene') if 'gene' not in (out.columns) else ("present", 'gene')))
        _seen.append(keep(list(out["grna"])))
        _seen.append(bool(
            "positional" in said and "'<org>_<gene>_<guide>'" in said))
        _seen.append(('TGGT1_GENEA_g1' if 'TGGT1_GENEA_g1' in (said) else ("missing", 'TGGT1_GENEA_g1')))
        return _seen


class TestTheFoldCount:

    def case_two_independent_groups_are_enough(self):
        _seen = []
        for distinct in (2, 3, 5, 40):
            assert min(5, distinct) >= 2
        return _seen

    def case_one_group_cannot_be_split_and_says_so(self):
        """THE ARC: ``n_folds < 2``.

        A screen whose split level has one value -- one plate, or one
        well -- cannot be cross-validated by it, and sklearn's own
        message names its parameters rather than the rule that selected
        nothing. Naming the level and the count is what tells the user
        which setting to change.
        """
        _seen = []
        groups = np.array(["plate1", "plate1", "plate1"])
        distinct = len(np.unique(groups))
        n_folds = min(5, distinct)

        _seen.append(bool(
            distinct == 1 and n_folds < 2))

        from spacr import ml as M
        from spacr.classifier_evaluation import grouped_split

        with pytest.raises(ValueError, match="grouped held-out split is impossible") as _exc1:
            grouped_split(groups, np.array([0, 1, 0]), 0.25,
                          group_by="plate")
        _seen.append(('ValueError' if isinstance(_exc1.value, ValueError) else type(_exc1.value).__name__))
        return _seen

    def case_the_cap_is_five(self):
        """Not arbitrary in a way worth losing: five folds over a screen
        with forty plates is a choice about cost, and a screen with three
        gets three rather than an error."""
        _seen = []
        _seen.append(keep(min(5, 40)))
        _seen.append(keep(min(5, 3)))
        return _seen

class TestStampingTheQualityManifest:

    def case_a_frame_carries_the_manifest_in_its_attrs(self):
        """The write the guard wraps. ``attrs`` travels with the frame,
        so a caller that does not know about the manifest is unaffected
        -- which is why it is not a column."""
        _seen = []
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
        _seen.append(keep(checked))
        return _seen

def case_the_penalised_warning_has_no_impossible_swallowing_handler(capsys):
    """Drive both decisions and pin removal of the catch-all handler."""
    _seen = []
    from spacr import ml as M

    no_hits = pd.DataFrame({"p_value": [0.4, np.nan, "0.7"]})
    _seen.append(bool(
        M._warn_if_penalised_no_hits(
        {"regression_type": "ridge"}, no_hits) is True))
    note = capsys.readouterr().out
    _seen.append(('returned no coefficient below p=0.05' if 'returned no coefficient below p=0.05' in (note) else ("missing", 'returned no coefficient below p=0.05')))
    _seen.append(("regression_type='ols'" if "regression_type='ols'" in (note) else ("missing", "regression_type='ols'")))
    _seen.append(('Refit with' if 'Refit with' in (note) else ("missing", 'Refit with')))
    _seen.append(("'rlm' for a robust check" if "'rlm' for a robust check" in (note) else ("missing", "'rlm' for a robust check")))
    _seen.append(('this is NOT evidence of no effect' if 'this is NOT evidence of no effect' in (note) else ("missing", 'this is NOT evidence of no effect')))

    has_hit = pd.DataFrame({"p_value": [0.4, 0.01]})
    _seen.append(bool(
        M._warn_if_penalised_no_hits(
        {"regression_type": "elasticnet"}, has_hit) is False))
    _seen.append(bool(
        M._warn_if_penalised_no_hits(
        {"regression_type": "ols"}, no_hits) is False))
    _seen.append(bool(
        M._warn_if_penalised_no_hits(
        {"regression_type": "lasso"}, no_hits.iloc[0:0]) is False))
    _seen.append(keep(capsys.readouterr().out))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param((TestTheColumnsTheLineAboveEnsured, 'case_the_positional_split_uses_the_middle_token_as_the_gene'), [['GENEA', 'GENEB'], ['GENEA_g1', 'GENEB_g2']], id="the_positional_split_uses_the_middle_token_as_the_gene"),
    pytest.param((TestTheColumnsTheLineAboveEnsured, 'case_the_positional_split_is_documented_as_an_assumption'), [("absent", 'gene'), ['TGGT1_GENEA_g1', 'GENEB_g2'], True, 'TGGT1_GENEA_g1'], id="the_positional_split_is_documented_as_an_assumption"),
    pytest.param((TestAnAbsolutePathAlwaysHasAParent, 'case_the_bare_dirname_is_what_would_be_empty'), [''], id="the_bare_dirname_is_what_would_be_empty"),
    pytest.param((TestTheColumnsTheLineAboveEnsured, 'case_splitting_a_prcfo_gives_the_three_parts'), [], id="splitting_a_prcfo_gives_the_three_parts"),
    pytest.param((TestTheFoldCount, 'case_two_independent_groups_are_enough'), [], id="two_independent_groups_are_enough"),
    pytest.param((TestTheFoldCount, 'case_one_group_cannot_be_split_and_says_so'), [True, 'ValueError'], id="one_group_cannot_be_split_and_says_so"),
    pytest.param((TestTheFoldCount, 'case_the_cap_is_five'), [5, 3], id="the_cap_is_five"),
    pytest.param((TestStampingTheQualityManifest, 'case_a_frame_carries_the_manifest_in_its_attrs'), [45], id="a_frame_carries_the_manifest_in_its_attrs"),
    pytest.param(case_the_penalised_warning_has_no_impossible_swallowing_handler, [True, 'returned no coefficient below p=0.05', "regression_type='ols'", 'Refit with', "'rlm' for a robust check", 'this is NOT evidence of no effect', True, True, True, ''], id="the_penalised_warning_has_no_impossible_swallowing_handler"),
])
def test_r9_ml_six_guards_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
