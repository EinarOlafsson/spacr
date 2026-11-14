"""Table-driven tests of guide evidence, the hit-list build, and the crop join.

A hit list must drop unusable evidence rather than count it as support: a
guide with no readable coefficient is not a guide that agrees, a row naming no
gene belongs to no gene, and a table with no ``feature`` column carries no
guide-level evidence at all.  The annotation join must name the genes it
recognises without changing the row count or overwriting a column the user
supplied.  ``hit_investigation._read_cells`` joins predictions to measured
cells through ``png_list`` whatever the prediction file calls its crop-path
column.  One general test per unit, each walking a case table with
``subtests`` and asserting exact values.
"""
from __future__ import annotations

import math
import sqlite3

import pandas as pd
import pytest

from spacr import hit_investigation as hi
from spacr.hits import build_hit_list, grna_agreement


def _guides(features, coefficients, grna=None):
    frame = {"feature": features, "coefficient": coefficients}
    if grna is not None:
        frame["grna"] = grna
    return pd.DataFrame(frame)


def test_guide_agreement_counts_only_readable_guides_of_named_genes(subtests):
    """Unreadable, infinite, zero and gene-less rows never inflate agreement."""
    one_to_four = [f"fraction:grna[233460_{i}]" for i in range(1, 5)]
    cases = [
        ("no_table", {"233460": 0.5}, None, {"233460": (0, 0, [])}),
        ("empty_table", {"233460": 0.5}, pd.DataFrame(),
         {"233460": (0, 0, [])}),
        ("no_feature_column", {"233460": 0.5},
         pd.DataFrame({"coefficient": [0.4, -0.2]}),
         {"233460": (0, 0, [])}),
        ("intercept_row_skipped", {"233460": 0.5},
         _guides(["Intercept", "fraction:grna[233460_1]"], [1.2, 0.4],
                 ["Intercept", "233460_1"]),
         {"233460": (1, 1, ["233460_1"])}),
        ("name_from_term_when_grna_missing_or_nan", {"233460": 0.5},
         _guides(["fraction:grna[T.233460_1]", "fraction:grna[233460_2]"],
                 [0.4, 0.9], [None, float("nan")]),
         {"233460": (2, 2, ["233460_1", "233460_2"])}),
        ("name_from_term_without_grna_column", {"233460": 0.5},
         _guides(["fraction:grna[233460_2]", "fraction:grna[233460_1]"],
                 [0.9, 0.4]),
         {"233460": (2, 2, ["233460_1", "233460_2"])}),
        ("text_none_nan_coefficients_are_not_guides", {"233460": 0.5},
         _guides(one_to_four, ["not a number", None, float("nan"), 0.4],
                 [f"233460_{i}" for i in range(1, 5)]),
         {"233460": (1, 1, ["233460_4"])}),
        ("infinite_coefficient_dropped", {"233460": 0.5},
         _guides(one_to_four[:2], [math.inf, 0.4], ["233460_1", "233460_2"]),
         {"233460": (1, 1, ["233460_2"])}),
        ("zero_counts_but_never_agrees", {"233460": 0.5},
         _guides(one_to_four[:2], [0.0, 0.3], ["233460_1", "233460_2"]),
         {"233460": (1, 2, ["233460_2"])}),
        ("negative_effect_agrees_with_negative_guides", {"233460": -0.5},
         _guides(one_to_four[:3], [-0.1, 0.2, -0.3],
                 ["233460_1", "233460_2", "233460_3"]),
         {"233460": (2, 3, ["233460_1", "233460_3"])}),
        ("gene_without_finite_effect_agrees_with_nothing",
         {"233460": float("nan")},
         _guides(one_to_four[:2], [0.1, 0.2], ["233460_1", "233460_2"]),
         {"233460": (0, 2, [])}),
        ("guides_of_an_unlisted_gene_are_still_counted", {},
         _guides(["fraction:grna[999999_1]"], [0.1], ["999999_1"]),
         {"999999": (0, 1, [])}),
    ]
    for case_id, effects, frame, expected in cases:
        with subtests.test(msg=case_id):
            assert grna_agreement(effects, frame) == expected


def _gene_table(genes):
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{gene}]" for gene in genes],
        "coefficient": [0.4, -0.3, 0.2][: len(genes)],
        "p_value": [0.001, 0.02, 0.3][: len(genes)],
    })


NO_GUIDE_NOTE = ("No per-gRNA coefficient table was found, so guide agreement "
                 "could not be computed for any gene.")


def test_build_hit_list_refuses_non_tables_and_joins_annotation_once(
        subtests, tmp_path):
    """The bundled join names known genes, keeps one row each, never overrides."""
    with subtests.test(msg="results_without_feature_column_refused"):
        with pytest.raises(ValueError) as caught:
            build_hit_list({"all": pd.DataFrame({"value": [1.0, 2.0]})})
        assert str(caught.value) == (
            "no gene-level coefficients in (frames): expected "
            "results_gene.csv or a results.csv carrying gene terms.")

    genes = ["233460", "254470", "227280"]
    with subtests.test(msg="bundled_annotation_names_genes_one_row_each"):
        hits = build_hit_list({"gene": _gene_table(genes)}, toxoplasma=True)
        assert [hit.gene for hit in hits] == genes
        assert hits.gene("233460").annotation["gene_name"] == "SAG1"
        assert hits.gene("254470").annotation["gene_name"] == "MYR1"
        assert hits.gene("227280").annotation["hyperlopit"] == \
            "dense granules"
        bundled = [n for n in hits.notes
                   if n.startswith("Bundled Toxoplasma annotation joined by "
                                   "gene number: ")]
        assert len(bundled) == 1
        assert bundled[0].endswith(
            f" {len(hits.gene('233460').annotation)} column(s).")

    with subtests.test(msg="user_column_wins_over_bundle"):
        path = tmp_path / "my_names.csv"
        pd.DataFrame({"Gene ID": ["TGME49_233460"],
                      "gene_name": ["my own name for SAG1"]}).to_csv(
            path, index=False)
        hits = build_hit_list({"gene": _gene_table(["233460"])},
                              metadata_files=[str(path)], toxoplasma=True)
        assert hits.gene("233460").annotation["gene_name"] == \
            "my own name for SAG1"

    with subtests.test(msg="annotation_off_joins_nothing"):
        hits = build_hit_list({"gene": _gene_table(["233460"])})
        assert hits.gene("233460").annotation == {}
        assert hits.notes == (NO_GUIDE_NOTE,)


def _png_list_db(tmp_path, rows):
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as connection:
        pd.DataFrame(rows).to_sql("png_list", connection, index=False)
    return database


def test_read_cells_joins_predictions_through_png_list(subtests, tmp_path,
                                                       monkeypatch):
    """Any crop-path spelling joins; png_list's recorded path is the one kept."""
    from spacr import io

    cells = pd.DataFrame({"prcfo": ["a", "b"], "cell_area": [1.0, 2.0]})
    monkeypatch.setattr(io, "_read_and_join_tables",
                        lambda *a, **k: cells.copy())
    database = str(_png_list_db(tmp_path, [
        {"prcfo": "a", "png_path": "/old/crops/a.png"},
        {"prcfo": "b", "png_path": "/old/crops/b.png"},
    ]))
    expected = pd.DataFrame({"prcfo": ["a", "b"], "cell_area": [1.0, 2.0],
                             "score": [0.25, 0.75],
                             "png_path": ["/old/crops/a.png",
                                          "/old/crops/b.png"]})
    cases = [
        ("spaCR_png_path_spelling", "png_path",
         {"png_path": ["/moved/a.png", "/moved/b.png"]}),
        ("default_path_spelling", "path",
         {"path": ["/elsewhere/a.png", "/elsewhere/b.png"]}),
        ("bare_basenames", "crop", {"crop": ["a.png", "b.png"]}),
    ]
    for case_id, path_column, columns in cases:
        with subtests.test(msg=case_id):
            predictions = tmp_path / f"{case_id}.csv"
            pd.DataFrame({**columns, "score": [0.25, 0.75]}).to_csv(
                predictions, index=False)
            out = hi._read_cells(database, str(predictions), "score",
                                 path_column)
            pd.testing.assert_frame_equal(
                out[list(expected.columns)].reset_index(drop=True), expected)

    refusals = [
        ("no_score_column", {"png_path": ["/x/a.png"], "other": [1.0]},
         "png_path", "prediction file has no 'score' column"),
        ("no_key_column", {"elsewhere": ["/x/a.png"], "score": [1.0]},
         "png_path",
         "predictions need prcfo or the configured crop-path column"),
        ("crops_not_in_png_list", {"png_path": ["/x/zz.png"], "score": [1.0]},
         "png_path",
         f"none of the 1 crops named in 'png_path' is recorded in png_list "
         f"of {database}; choose the database whose crops the predictions "
         f"scored"),
    ]
    for case_id, columns, path_column, message in refusals:
        with subtests.test(msg=case_id):
            predictions = tmp_path / f"{case_id}.csv"
            pd.DataFrame(columns).to_csv(predictions, index=False)
            with pytest.raises(hi.HitAttributionError) as caught:
                hi._read_cells(database, str(predictions), "score",
                               path_column)
            assert str(caught.value) == message
