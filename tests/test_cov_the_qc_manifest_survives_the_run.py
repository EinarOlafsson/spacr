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

from spacr.ml import _annotate_level_coefficients


def test_pandas_merge_really_does_drop_attrs():
    """The premise, pinned against pandas itself.

    If a future pandas starts carrying `.attrs` through merges, this fails
    and the workaround below can go.
    """
    left = pd.DataFrame({"k": [1], "v": [2]})
    left.attrs["qc_manifest"] = {"verdict": "pass"}
    right = pd.DataFrame({"k": [1], "w": [3]})

    assert left.merge(right, on="k").attrs == {}
    # ...while these two do carry it, which is the asymmetry that hides it.
    assert left.copy().attrs == {"qc_manifest": {"verdict": "pass"}}
    assert pd.concat([left]).attrs == {"qc_manifest": {"verdict": "pass"}}


def test_the_manifest_survives_the_level_annotation():
    """The fix: what `regression` wrote is still there for the reader."""
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

    assert out.attrs.get("qc_manifest") == manifest
    # And the annotation still did its job, so this is not passing by
    # returning the frame untouched.
    assert out.loc[out["grna"] == "g1", "grna_count"].iloc[0] == 7
    assert out.loc[out["gene"] == "GENE1", "gene_count"].iloc[0] == 3


def test_a_frame_with_no_manifest_gains_no_empty_one():
    """Absent must stay absent: a key holding None reads as "QC ran and
    concluded nothing", which is not what happened."""
    coef = pd.DataFrame({"feature": ["grna[g1]"], "coefficient": [0.5]})
    n_grna = pd.DataFrame({"grna": ["g1"], "grna_count": [1]})
    n_gene = pd.DataFrame({"gene": ["GENE1"], "gene_count": [1]})

    out = _annotate_level_coefficients(coef, n_grna, n_gene)

    assert "qc_manifest" not in out.attrs
