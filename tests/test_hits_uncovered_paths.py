"""The hit-list paths a normal run never takes: old tables, bare terms, guards.

Three kinds of path live here. A coefficient table written before the
``feature`` column existed, or one carrying only a ``level`` column, still has
to yield a level for every row. A model term whose bracket carries no
``:gene[``/``:grna[`` label still has to be resolved to a guide or to nothing
by the shape of the identifier alone. And the row-count guards around every
metadata join have to actually refuse: they exist because a join that
multiplies a gene turns one finding into several, and each one is checked
independently of the ``validate="many_to_one"`` that normally makes it
impossible.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spacr import hits as hits_module                           # noqa: E402
from spacr.hits import (Hit, HitList, build_hit_list,           # noqa: E402
                        coefficient_levels, guide_of, join_metadata)


def _one_gene_frames():
    """A minimal ``{role: frame}`` source with a single gene-level term."""
    return {"gene": pd.DataFrame({
        "feature": ["gene_fraction:gene[233460]"],
        "coefficient": [0.42],
        "std_err": [0.05],
        "p_value": [1e-4],
        "condition": ["other"],
        "n_gene": [40],
    })}


# ---------------------------------------------------------------------------
# coefficient_levels: tables written without a `feature` column
# ---------------------------------------------------------------------------

def test_a_table_with_no_feature_column_takes_levels_from_its_level_column():
    """A table can name its level without naming its terms.

    The results panel asks for levels before it has decided which columns to
    show, so a projection that kept ``level`` but dropped ``feature`` must
    still answer 'grna' and 'gene' rather than falling back to blank.
    """
    frame = pd.DataFrame({"level": ["grna", "gene"],
                          "coefficient": [1.0, 2.0]})

    assert coefficient_levels(frame).tolist() == ["grna", "gene"]


def test_a_table_with_neither_feature_nor_level_leaves_every_level_unknown():
    """No term names and no recorded level is 'unknown', never a guess.

    Guessing 'gene' here would put nuisance rows into the gene testing family
    and correct them for multiple testing alongside real hypotheses.
    """
    frame = pd.DataFrame({"coefficient": [1.0, 2.0, 3.0]})

    levels = coefficient_levels(frame)

    assert levels.tolist() == ["", "", ""]
    assert levels.index.tolist() == frame.index.tolist()


def test_an_unrecognised_level_value_does_not_override_a_blank_level():
    """Only 'grna' and 'gene' are levels; anything else is not adopted."""
    frame = pd.DataFrame({"level": ["plate", None], "coefficient": [1.0, 2.0]})

    assert coefficient_levels(frame).tolist() == ["", ""]


# ---------------------------------------------------------------------------
# guide_of: bracketed terms carrying no explicit family label
# ---------------------------------------------------------------------------

def test_an_unlabelled_term_with_a_numeric_suffix_names_the_guide():
    """``fraction[233460_1]`` is guide 1 of gene 233460.

    Formulas built without the ``grna[...]`` spelling still reach the hit
    list; the trailing ``_1`` is the only thing marking the term as a guide,
    and losing it would leave the gene with no guide-level corroboration.
    """
    assert guide_of("fraction[233460_1]") == "233460_1"


def test_an_unlabelled_term_without_a_numeric_suffix_names_no_guide():
    """``fraction[233460]`` is a gene term, so it names no guide.

    Returning the gene id here would make the gene look like one of its own
    guides and count it as agreeing with itself.
    """
    assert guide_of("fraction[233460]") is None


def test_only_an_explicit_label_protects_a_bare_veupathdb_accession():
    """Shape alone cannot tell ``TGGT1_231640`` from a guide, and does not try.

    Both spellings end in an underscore and digits, so an unlabelled term is
    read as a guide either way. The ``gene[...]`` label is what makes the
    distinction, which is why every table spaCR writes carries one.
    """
    assert guide_of("fraction[TGGT1_231640_3]") == "TGGT1_231640_3"
    assert guide_of("fraction[TGGT1_231640]") == "TGGT1_231640"
    assert guide_of("gene_fraction:gene[TGGT1_231640]") is None


# ---------------------------------------------------------------------------
# the row-count guards
# ---------------------------------------------------------------------------

class _RowDuplicatingFrame(pd.DataFrame):
    """A frame whose ``merge`` returns twice the rows it was asked for.

    Stands in for a pandas whose ``validate='many_to_one'`` did not catch a
    fan-out, which is exactly the failure the guard in ``join_metadata``
    exists to be the second line of defence against.
    """

    @property
    def _constructor(self):
        return _RowDuplicatingFrame

    def merge(self, right, **kwargs):
        merged = pd.DataFrame.merge(pd.DataFrame(self), right, **kwargs)
        return pd.concat([merged, merged], ignore_index=True)


def test_join_metadata_refuses_a_join_that_changed_the_row_count(tmp_path):
    """A join that grew the table is refused, naming the file that grew it."""
    meta = tmp_path / "annotations.csv"
    meta.write_text("Gene ID,Gene Name\nTGME49_233460,ROP18\n",
                    encoding="utf-8")
    frame = _RowDuplicatingFrame({"gene": ["233460"], "coefficient": [0.42]})

    with pytest.raises(ValueError) as excinfo:
        join_metadata(frame, [str(meta)])

    message = str(excinfo.value)
    assert "annotations.csv" in message
    assert "changed the row count from 1 to 2" in message
    assert "not one row per gene" in message


def test_join_metadata_keeps_a_well_behaved_join(tmp_path):
    """The same guard passes silently when the annotation is one row a gene."""
    meta = tmp_path / "annotations.csv"
    meta.write_text("Gene ID,Gene Name\nTGME49_233460,ROP18\n",
                    encoding="utf-8")
    frame = pd.DataFrame({"gene": ["233460"], "coefficient": [0.42]})

    joined, notes = join_metadata(frame, [str(meta)])

    assert len(joined) == 1
    assert joined.loc[0, "Gene Name"] == "ROP18"
    assert notes == []


def test_the_bundled_toxoplasma_join_may_not_change_the_row_count(monkeypatch):
    """A bundled annotation that fanned out is refused, not published.

    ``annotate`` is joined after the caller's own metadata files, so a
    fan-out there would arrive at the very end of the build and be reported
    as findings unless the row count is checked again.
    """
    from spacr import annotation as annotation_module

    def _duplicating_annotate(frame, **kwargs):
        return pd.concat([frame, frame], ignore_index=True)

    monkeypatch.setattr(annotation_module, "annotate", _duplicating_annotate)

    with pytest.raises(ValueError) as excinfo:
        build_hit_list(_one_gene_frames(), toxoplasma=True)

    assert ("the Toxoplasma annotation changed the row count from 1 to 2"
            in str(excinfo.value))


def test_a_metadata_join_that_changed_the_row_count_is_refused(monkeypatch):
    """``build_hit_list`` re-checks the row count the join should keep.

    The check is on the joined table rather than trusted from
    ``join_metadata``, because a duplicated gene reaching the ranked list
    would be counted twice by everything downstream.
    """
    def _duplicating_join(frame, metadata_files=(), *, key="Gene ID"):
        return pd.concat([frame, frame], ignore_index=True), []

    monkeypatch.setattr(hits_module, "join_metadata", _duplicating_join)

    with pytest.raises(ValueError) as excinfo:
        build_hit_list(_one_gene_frames())

    message = str(excinfo.value)
    assert "the metadata join changed the row count from 1 to 2" in message
    assert "not one row per gene" in message


# ---------------------------------------------------------------------------
# the no-data sentinel in summary()
# ---------------------------------------------------------------------------

def test_a_summary_with_nothing_significant_reports_no_effect_size():
    """With no gene clearing the FDR there is no largest effect to quote.

    ``max_abs_effect`` is ``nan`` rather than 0.0: a screen that found
    nothing has no effect size, and zero would be read as one that was
    measured and came out flat.
    """
    listed = HitList(
        hits=(Hit(gene="233460", effect=2.4, q_value=0.9, rank=1),
              Hit(gene="100", effect=-1.8, q_value=0.7, rank=2)),
        n_genes=2, alpha=0.05)

    summary = listed.summary()

    assert summary["n_significant"] == 0
    assert math.isnan(summary["max_abs_effect"])
    assert math.isnan(summary["median_abs_effect"])
    assert summary["top_genes"] == []
