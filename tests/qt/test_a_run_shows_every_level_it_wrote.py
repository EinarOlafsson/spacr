"""A run's gene rows reach the panel even when they are in a sibling file.

The permutation path tested genes, corrected them as their own family, wrote
them to results_gene.csv, and left results.csv holding guides alone. The
panel loads results.csv and stops, so a reader who asked for genes was shown
nothing while the rows sat on disk.

The writer is fixed. This is the other half: a folder written before that
does not rewrite itself, so the reader has to cope.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.regression_results import read_run_tables   # noqa: E402


def _guides(n=3):
    return pd.DataFrame({
        "feature": [f"fraction:grna[g{i}]" for i in range(n)],
        "grna": [f"g{i}" for i in range(n)],
        "coefficient": [0.1 * i for i in range(n)],
        "p_value": [0.01] * n,
        "q_value": [0.05] * n,
        "wells_with_guide": [4] * n,
    })


def _genes(n=2):
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[G{i}]" for i in range(n)],
        "gene": [f"G{i}" for i in range(n)],
        "level": ["gene"] * n,
        "coefficient": [0.2 * i for i in range(n)],
        "p_value": [0.02] * n,
        "q_value": [0.06] * n,
        "wells_with_gene": [9] * n,
    })


def _run(tmp_path, primary, gene=None, grna=None):
    folder = tmp_path / "guide_permutation_1"
    folder.mkdir()
    paths = [folder / "results.csv"]
    primary.to_csv(paths[0], index=False)
    if gene is not None:
        p = folder / "results_gene.csv"
        gene.to_csv(p, index=False)
        paths.append(p)
    if grna is not None:
        p = folder / "results_grna.csv"
        grna.to_csv(p, index=False)
        paths.append(p)
    return [str(p) for p in paths]


def test_a_gene_sibling_is_merged(tmp_path):
    tables = _run(tmp_path, _guides(3), gene=_genes(2))
    frame, found, merged = read_run_tables(tables)
    assert len(frame) == 5
    assert found.endswith("results.csv")
    assert len(merged) == 1 and merged[0].endswith("results_gene.csv")


def test_the_merged_rows_carry_their_level(tmp_path):
    tables = _run(tmp_path, _guides(3), gene=_genes(2))
    frame, _, _ = read_run_tables(tables)
    assert sorted(frame["level"].dropna().unique()) == ["gene", "grna"]


def test_a_table_that_already_holds_both_is_untouched(tmp_path):
    """The fitted path writes both levels; merging again would double them."""
    both = pd.concat([_guides(3).assign(level="grna"), _genes(2)],
                     ignore_index=True, sort=False)
    tables = _run(tmp_path, both, gene=_genes(2))
    frame, _, merged = read_run_tables(tables)
    assert len(frame) == 5, "the gene rows were counted twice"
    assert merged == []


def test_a_guides_only_run_is_unchanged(tmp_path):
    tables = _run(tmp_path, _guides(3))
    frame, _, merged = read_run_tables(tables)
    assert len(frame) == 3
    assert merged == []


def test_an_empty_sibling_adds_nothing(tmp_path):
    """A run whose gene pass failed writes the file empty, deliberately."""
    tables = _run(tmp_path, _guides(3), gene=_genes(0))
    frame, _, merged = read_run_tables(tables)
    assert len(frame) == 3
    assert merged == []


def test_a_sibling_from_another_run_is_not_pulled_in(tmp_path):
    """Two runs under one parent must not have their tables mixed."""
    first = _run(tmp_path, _guides(3))
    other = tmp_path / "guide_permutation_2"
    other.mkdir()
    _genes(2).to_csv(other / "results_gene.csv", index=False)
    frame, _, merged = read_run_tables(first + [str(other / "results_gene.csv")])
    assert len(frame) == 3, "a different run's genes were merged in"
    assert merged == []


def test_the_real_permutation_folder_shows_both(tmp_path):
    """The shape the maintainer hit: guides in results.csv, genes beside it."""
    tables = _run(tmp_path, _guides(1380 // 460), gene=_genes(2),
                  grna=_guides(1380 // 460))
    frame, _, merged = read_run_tables(tables)
    assert "gene" in set(frame["level"].dropna())
    assert any(p.endswith("results_gene.csv") for p in merged)


def test_a_file_path_still_finds_its_siblings(tmp_path):
    """The panel is often handed results.csv itself, not the folder.

    Returning that one path alone hid the gene file sitting beside it, so
    the merge never had anything to merge and the reader saw guides only.
    """
    from spacr.qt.widgets.regression_results import find_results_tables

    tables = _run(tmp_path, _guides(3), gene=_genes(2))
    found = find_results_tables(tables[0])
    assert any(p.endswith("results_gene.csv") for p in found)
    assert found[0].endswith("results.csv"), "the chosen file must stay primary"
    frame, _, merged = read_run_tables(found)
    assert sorted(frame["level"].dropna().unique()) == ["gene", "grna"]
    assert merged


def test_a_lone_file_with_no_siblings_is_unchanged(tmp_path):
    tables = _run(tmp_path, _guides(3))
    from spacr.qt.widgets.regression_results import find_results_tables
    assert find_results_tables(tables[0]) == [tables[0]]


def test_the_table_drops_columns_blank_at_this_level():
    """A gene row has no `guide`; showing the column puts its name far right."""
    from spacr.qt.widgets.regression_results import for_table

    genes = _genes(2)
    genes["guide"] = None
    genes["wells_with_guide"] = float("nan")
    narrowed = for_table(genes)
    assert "guide" not in narrowed.columns
    assert "wells_with_guide" not in narrowed.columns
    assert "gene" in narrowed.columns and "wells_with_gene" in narrowed.columns


def test_a_kept_column_survives_even_when_blank():
    from spacr.qt.widgets.regression_results import for_table

    guides = _guides(2)
    guides["q_value"] = None
    assert "q_value" in for_table(guides).columns


def test_a_column_with_any_value_is_kept():
    from spacr.qt.widgets.regression_results import for_table

    guides = _guides(3)
    guides["note"] = ["", "", "seen"]
    assert "note" in for_table(guides).columns


def test_an_empty_frame_is_returned_as_is():
    from spacr.qt.widgets.regression_results import for_table

    empty = _guides(0)
    assert list(for_table(empty).columns) == list(empty.columns)
