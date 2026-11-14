"""Single-guide hits and guide-support columns from :mod:`spacr.guide_concordance`.

A gene carried by one guide has a gene-level p-value identical to that guide's,
which is not independent evidence.  ``flag_single_guide_hits`` lists those
hits and ``annotate_results`` joins the guide-support columns onto every row
of a coefficient table.  Both must also answer the degenerate tables a real run
produces (no guide terms, no ``feature`` column, ``None``) with an empty answer
rather than an exception.  One case table per unit, each row a subtest.
"""
from __future__ import annotations

import pandas as pd

from spacr import guide_concordance

SUPPORT_COLUMNS = ["n_guides", "n_guides_significant", "n_same_direction",
                   "concordance", "single_guide", "gene_p", "gene_coefficient",
                   "best_guide_p"]


def _coefficients():
    """225160 has three agreeing guides, 244480 has one."""
    return pd.DataFrame({
        "feature": [
            "fraction:grna[TGGT1_225160_1]",
            "fraction:grna[TGGT1_225160_2]",
            "fraction:grna[TGGT1_225160_3]",
            "gene_fraction:gene[T.225160]",
            "fraction:grna[TGGT1_244480_2]",
            "gene_fraction:gene[T.244480]",
        ],
        "coefficient": [0.4, 0.5, 0.6, 0.5, -0.9, -0.9],
        "p_value": [0.51, 0.14, 0.27, 4.6e-08, 1.6e-12, 1.6e-12],
    })


def _gene_only():
    return pd.DataFrame({"feature": ["gene_fraction:gene[T.225160]"],
                         "coefficient": [0.5], "p_value": [1e-6]})


def _two_single_guide_hits():
    """Two single-guide hits whose rank order differs from their table order."""
    frame = _coefficients()
    extra = pd.DataFrame({
        "feature": ["fraction:grna[TGGT1_300000_1]", "gene_fraction:gene[T.300000]"],
        "coefficient": [0.7, 0.7], "p_value": [1e-3, 1e-14]})
    return pd.concat([frame, extra], ignore_index=True)


def _miss_alpha():
    frame = _coefficients()
    frame.loc[frame["feature"] == "gene_fraction:gene[T.244480]", "p_value"] = 0.4
    return frame


def test_flag_single_guide_hits(subtests):
    """Only single-guide genes past alpha are listed, ranked by gene p."""
    cases = [
        ("one-guide gene flagged, concordant gene not", _coefficients(), {},
         {"gene": ["244480"], "n_guides": [1], "n_guides_significant": [1],
          "n_same_direction": [1], "concordance": [1.0], "single_guide": [True],
          "gene_p": [1.6e-12], "gene_coefficient": [-0.9],
          "best_guide_p": [1.6e-12]}),
        ("two hits ranked by gene p", _two_single_guide_hits(), {},
         {"gene": ["300000", "244480"]}),
        ("a single-guide gene missing alpha is not a hit", _miss_alpha(), {}, {}),
        ("alpha below every gene p", _coefficients(), {"alpha": 1e-13}, {}),
        ("a fit with no guide terms", _gene_only(), {}, {}),
        ("a table with no feature column", pd.DataFrame({"value": [1.0, 2.0]}),
         {}, {}),
        ("None", None, {}, {}),
    ]
    for case_id, frame, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            flagged = guide_concordance.flag_single_guide_hits(frame, **kwargs)
            assert list(flagged.columns) == SUPPORT_COLUMNS
            assert flagged.index.name == "gene"
            got = flagged.reset_index().to_dict("list")
            if not expected:
                assert flagged.empty
            for column, values in expected.items():
                assert got[column] == values


def test_annotate_results(subtests):
    """Support columns land on every row of a copy; nothing to join leaves a copy."""
    mixed = pd.concat([_coefficients(), pd.DataFrame({
        "feature": ["fraction:grna[TGGT1_300000_1]", "fraction:grna[TGGT1_300000_2]",
                    "fraction:grna[TGGT1_300000_3]", "unrelated_term"],
        "coefficient": [0.5, -0.2, 0.1, 3.0],
        "p_value": [0.01, 0.6, 0.02, 0.5]})], ignore_index=True)

    with subtests.test(msg="a real fit gains four support columns per row"):
        frame = _coefficients()
        annotated = guide_concordance.annotate_results(frame)
        assert annotated is not frame
        assert list(annotated.columns) == [
            "feature", "coefficient", "p_value", "n_guides",
            "n_guides_significant", "concordance", "single_guide"]
        assert annotated["n_guides"].tolist() == [3, 3, 3, 3, 1, 1]
        assert annotated["n_guides_significant"].tolist() == [0, 0, 0, 0, 1, 1]
        assert annotated["concordance"].tolist() == [1.0] * 6
        assert annotated["single_guide"].tolist() == [False] * 4 + [True] * 2
        pd.testing.assert_frame_equal(annotated[frame.columns], frame)
    with subtests.test(msg="disagreeing guides and an unmatched row"):
        annotated = guide_concordance.annotate_results(mixed, alpha=0.05)
        tail = annotated.iloc[6:]
        assert tail["n_guides"].tolist()[:3] == [3.0, 3.0, 3.0]
        assert tail["n_guides_significant"].tolist()[:3] == [2.0, 2.0, 2.0]
        assert tail["concordance"].tolist()[:3] == [2 / 3] * 3
        assert tail.iloc[3].drop(["feature", "coefficient", "p_value"]).isna().all()

    for case_id, frame in [("gene-only fit", _gene_only()),
                           ("no feature column", pd.DataFrame({"value": [1.0]}))]:
        with subtests.test(msg=case_id):
            annotated = guide_concordance.annotate_results(frame)
            assert annotated is not frame
            pd.testing.assert_frame_equal(annotated, frame)
    with subtests.test(msg="None passes through"):
        assert guide_concordance.annotate_results(None) is None
