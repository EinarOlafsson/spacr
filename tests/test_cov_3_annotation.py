"""A UniProt join that cannot be made leaves the table alone and says why.

A run annotates its results and carries on either way, so every refusal here
has to come back as ``(frame, note)`` rather than as an exception or as a
silently unchanged table. The note is the only thing that reaches the user:
without it a table with no annotation columns looks identical whether UniProt
was unreachable, named no usable columns, or matched not one gene.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr import annotation, uniprot


@pytest.fixture()
def uniprot_returns(monkeypatch):
    """Hand `annotate_from_uniprot` a chosen UniProt reply."""
    def install(table, note=""):
        monkeypatch.setattr(uniprot, "annotation_for",
                            lambda *a, **k: (table, note))
    return install


# ---------------------------------------------------------------------------
# Finding the gene column
# ---------------------------------------------------------------------------

def test_the_gene_column_is_found_past_columns_the_table_lacks():
    """The preference order is a list of candidates, most of which a given
    table does not have; the finder must walk past them rather than stopping
    at the first miss."""
    frame = pd.DataFrame({"fraction": [0.1, 0.2], "grna": ["TP53_1", "TP53_2"]})

    assert annotation._uniprot_key_column(frame) == "grna"


def test_a_table_naming_no_gene_at_all_returns_no_column():
    """None is what the caller tests for; an empty string or a guess would
    make it join on a column of numbers."""
    frame = pd.DataFrame({"fraction": [0.1], "plate": ["p1"]})

    assert annotation._uniprot_key_column(frame) is None


def test_a_gene_column_of_blanks_does_not_count_as_naming_genes():
    """A column that exists and holds only empty strings names nothing, and
    joining on it would match every blank row to the same entry."""
    frame = pd.DataFrame({"gene": ["", "   ", ""]})

    assert annotation._uniprot_key_column(frame) is None


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def test_a_table_with_no_columns_is_handed_straight_back():
    """An empty result table reaches the annotator on a run that produced no
    hits; it must not be queried for a gene column it cannot have."""
    empty = pd.DataFrame()

    out, note = annotation.annotate_from_uniprot(empty, "Homo sapiens")

    assert out is empty
    assert note == ""
    assert annotation.annotate_from_uniprot(None, "Homo sapiens") == (None, "")


def test_an_empty_uniprot_reply_carries_its_own_note_through(uniprot_returns):
    """The note explaining why UniProt returned nothing is the run's only
    record of it, so it must not be replaced by a note of this module's."""
    uniprot_returns(None, "UniProt is unreachable.")
    frame = pd.DataFrame({"gene": ["TP53"], "coefficient": [0.3]})

    out, note = annotation.annotate_from_uniprot(frame, "Homo sapiens")

    assert out is frame
    assert note == "UniProt is unreachable."


def test_a_table_naming_no_gene_column_says_so(uniprot_returns):
    """UniProt answered, but there is nothing to join it onto. The note has
    to name the missing thing rather than reporting a failed download."""
    uniprot_returns(pd.DataFrame({"Entry": ["P04637"],
                                  "Gene Names": ["TP53 P53"]}))
    frame = pd.DataFrame({"plate": ["p1"], "coefficient": [0.3]})

    out, note = annotation.annotate_from_uniprot(frame, "Homo sapiens")

    assert out is frame
    assert "names no gene column" in note


def test_a_reply_with_no_recognised_columns_is_reported(uniprot_returns):
    """A reply whose columns are all unknown cannot be joined; saying
    'no usable columns' distinguishes it from an empty reply."""
    uniprot_returns(pd.DataFrame({"Something Else": ["x"]}))
    frame = pd.DataFrame({"gene": ["TP53"], "coefficient": [0.3]})

    out, note = annotation.annotate_from_uniprot(frame, "Homo sapiens")

    assert out is frame
    assert note == "UniProt returned no usable columns."


def test_annotation_the_caller_already_computed_is_not_overwritten(
        uniprot_returns):
    """Columns already on the frame belong to the caller. When UniProt offers
    nothing else, the frame comes back as it was -- not merged with itself."""
    uniprot_returns(pd.DataFrame({"Gene Names": ["TP53"]}))
    frame = pd.DataFrame({"gene": ["TP53"], "gene_name": ["mine"]})

    out, note = annotation.annotate_from_uniprot(frame, "Homo sapiens")

    assert out is frame
    assert note == ""
    assert list(out["gene_name"]) == ["mine"]


def test_matching_nothing_is_stated_with_the_number_of_identifiers(
        uniprot_returns, capsys):
    """A join that produced columns of nothing but NaN has to announce it:
    an annotated-looking table where every value is blank is the failure
    this note exists to make visible."""
    uniprot_returns(pd.DataFrame({"Entry": ["P04637"],
                                  "Gene Names": ["TP53 P53"],
                                  "Protein names": ["Cellular tumor antigen"]}))
    frame = pd.DataFrame({"gene": ["GRA14", "ROP18"], "coefficient": [.3, .1]})

    out, note = annotation.annotate_from_uniprot(frame, "Homo sapiens")

    assert len(out) == 2, "the merge must not multiply rows"
    assert "none of the 2 gene identifiers matched" in note
    assert out["product"].isna().all()
    assert "0 of 2 rows matched" in capsys.readouterr().out


def test_a_matching_gene_is_joined_and_the_note_stays_empty(uniprot_returns):
    """The contrast that makes the refusals meaningful: a gene UniProt knows
    picks up its product, and nothing is reported."""
    uniprot_returns(pd.DataFrame({"Entry": ["P04637"],
                                  "Gene Names": ["TP53 P53"],
                                  "Protein names": ["Cellular tumor antigen"]}))
    frame = pd.DataFrame({"gene": ["p53"], "coefficient": [0.3]})

    out, note = annotation.annotate_from_uniprot(frame, "Homo sapiens",
                                                 quiet=True)

    assert note == ""
    assert list(out["product"]) == ["Cellular tumor antigen"]
