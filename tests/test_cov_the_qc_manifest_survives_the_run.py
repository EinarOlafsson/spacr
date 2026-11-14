"""The QC manifest must reach the caller, not die inside the annotation.

`regression` puts the QC manifest on `coef_df.attrs["qc_manifest"]`, with a
comment saying `.attrs` "survives the frame being passed around".
`_perform_regression` reads it back and lifts it into `output['qc']`,
`output['qc_verdict']` and `output['qc_verdict_level']` -- instruction
115's promise that a run carries its QC verdict out.

Between the two, `_annotate_level_coefficients` merges the gRNA and gene
counts onto the frame, and `DataFrame.merge` does NOT propagate `.attrs`.
`copy` and `concat` do, which is what makes the loss easy to miss. The
manifest was therefore dropped on every run: `output` had no 'qc' key at
all.
"""

import pandas as pd
import pytest

from spacr.ml import _annotate_level_coefficients


def case_pandas_merge_really_does_drop_attrs():
    """The premise, pinned against pandas itself.

    If a future pandas starts carrying `.attrs` through merges, this fails
    and the workaround below can go.
    """
    _seen = []
    left = pd.DataFrame({"k": [1], "v": [2]})
    left.attrs["qc_manifest"] = {"verdict": "pass"}
    right = pd.DataFrame({"k": [1], "w": [3]})

    _seen.append(keep(left.merge(right, on="k").attrs))
    # ...while these two do carry it, which is the asymmetry that hides it.
    _seen.append(keep(left.copy().attrs))
    _seen.append(keep(pd.concat([left]).attrs))
    return _seen


def case_the_manifest_survives_the_level_annotation():
    """The fix: what `regression` wrote is still there for the reader."""
    _seen = []
    # `feature` is the input; `grna`/`gene` are derived from it by the
    # bracketed-identifier map, so they must not be supplied.
    coef = pd.DataFrame({
        "feature": ["grna[g1]", "gene[GENE1]"],
        "coefficient": [0.5, 0.25],
    })
    manifest = {"verdict": "warn", "verdict_level": "gene",
                "checks": ["dispersion"]}
    coef.attrs["qc_manifest"] = manifest

    n_grna = pd.DataFrame({"grna": ["g1"], "grna_count": [7]})
    n_gene = pd.DataFrame({"gene": ["GENE1"], "gene_count": [3]})

    out = _annotate_level_coefficients(coef, n_grna, n_gene)

    _seen.append(bool(
        out.attrs.get("qc_manifest") == manifest))
    # And the annotation still did its job, so this is not passing by
    # returning the frame untouched.
    _seen.append(keep(out.loc[out["grna"] == "g1", "grna_count"].iloc[0]))
    _seen.append(keep(out.loc[out["gene"] == "GENE1", "gene_count"].iloc[0]))
    return _seen


def case_a_frame_with_no_manifest_gains_no_empty_one():
    """Absent must stay absent: a key holding None reads as "QC ran and
    concluded nothing", which is not what happened."""
    _seen = []
    coef = pd.DataFrame({"feature": ["grna[g1]"], "coefficient": [0.5]})
    n_grna = pd.DataFrame({"grna": ["g1"], "grna_count": [1]})
    n_gene = pd.DataFrame({"gene": ["GENE1"], "gene_count": [1]})

    out = _annotate_level_coefficients(coef, n_grna, n_gene)

    _seen.append((("absent", 'qc_manifest') if 'qc_manifest' not in (out.attrs) else ("present", 'qc_manifest')))
    return _seen


from tests.observed_case import keep, observe  # noqa: E402

@pytest.mark.parametrize(("case", "expected"), [
    pytest.param(case_the_manifest_survives_the_level_annotation, [True, 7, 3], id="the_manifest_survives_the_level_annotation"),
    pytest.param(case_a_frame_with_no_manifest_gains_no_empty_one, [("absent", 'qc_manifest')], id="a_frame_with_no_manifest_gains_no_empty_one"),
    pytest.param(case_pandas_merge_really_does_drop_attrs, [{}, {'qc_manifest': {'verdict': 'pass'}}, {'qc_manifest': {'verdict': 'pass'}}], id="pandas_merge_really_does_drop_attrs"),
])
def test_the_qc_manifest_survives_the_run_contracts(request, case, expected):
    """One contract per row: the case records what it checked, the row states it."""
    observed = observe(case, request)
    assert observed == expected
