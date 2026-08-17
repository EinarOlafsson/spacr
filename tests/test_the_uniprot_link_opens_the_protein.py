"""A gene's UniProt link opens THAT protein's page.

Reported 2026-08-17: "none of the uniprot links work. you have to use another
name for 224750 it is S8F0I0 for example and the link would be:
https://www.uniprot.org/uniprotkb/S8F0I0/entry"

The tile had no accessions -- `toxoplasma_metadata.csv` carries none, checked
across all 48 columns -- so it fell back to a free-text SEARCH for the gene
NUMBER. Searching UniProt for `224750` finds whatever happens to contain that
string, which is why none of them worked.

The accessions are bundled now, built from UniProt's REST API over the ME49
REFERENCE proteome. Instruction 124 H's rule is unchanged and is why a
bundled table exists rather than a URL built by formatting a gene id into a
path: an invented record URL resolves to somebody else's protein, which is
indistinguishable to the reader from a correct link.
"""
from __future__ import annotations

import pytest

from spacr.gene_tile import (UNIPROT_RECORD_URL, uniprot_accessions,
                             uniprot_reference)


# --------------------------------------------------------------------------- #
#  The reported case
# --------------------------------------------------------------------------- #

def test_the_reported_gene_resolves_to_the_reported_accession():
    label, url, is_record = uniprot_reference("224750", {})

    assert url == "https://www.uniprot.org/uniprotkb/S8F0I0/entry"
    assert is_record is True
    assert "S8F0I0" in label


@pytest.mark.parametrize("spelling", [
    "224750", "TGGT1_224750", "TGME49_224750",
])
def test_every_spelling_of_the_gene_resolves(spelling):
    """The tile is handed a bare number or a full accession depending on
    where the click came from, and ToxoDB shares the numeric suffix across
    strains -- TGGT1_224750 and TGME49_224750 are the same gene."""
    _label, url, is_record = uniprot_reference(spelling, {})

    assert is_record is True
    assert url.endswith("/S8F0I0/entry"), url


# --------------------------------------------------------------------------- #
#  The table
# --------------------------------------------------------------------------- #

def test_the_table_covers_the_proteome():
    """7,886 genes from the reference proteome. A table with a handful of
    entries would pass the test above and fail every real gene."""
    assert len(uniprot_accessions()) > 7000


def test_every_accession_looks_like_one():
    """A UniProt accession is 6 or 10 alphanumeric characters. A gene id or a
    stray column that leaked into the table would build a URL that 404s."""
    import re

    pattern = re.compile(r"^[A-NR-Z0-9][A-Z0-9]{5}(?:[A-Z0-9]{4})?$")
    bad = [(gene, acc) for gene, acc in uniprot_accessions().items()
           if not pattern.match(acc)]
    assert not bad, bad[:10]


def test_a_gene_maps_to_exactly_one_accession():
    """224750 has at least TWO real UniProt entries -- S8F0I0 on an assembled
    chromosome and A0A7J6K0I8 in an unassembled WGS proteome. Both
    cross-reference the same ToxoDB gene, and only one is what a reader
    wants. Filtering to the reference proteome is what makes that choice
    principled rather than a coin toss."""
    table = uniprot_accessions()

    assert table["224750"] == "S8F0I0"
    assert table["224750"] != "A0A7J6K0I8"


# --------------------------------------------------------------------------- #
#  What it must NOT do
# --------------------------------------------------------------------------- #

def test_an_unknown_gene_gets_a_SEARCH_and_says_so():
    """Instruction 124 H, unchanged: a record URL invented from an id opens a
    real page for a different protein and looks exactly like a correct link.
    A search makes no claim."""
    label, url, is_record = uniprot_reference("999999999", {})

    assert is_record is False
    assert "/entry" not in url
    assert "search" in label.lower()


def test_no_accession_is_fabricated_from_the_gene_id():
    """The failure this whole design exists to prevent."""
    for gene in ("999999999", "not_a_gene"):
        _label, url, is_record = uniprot_reference(gene, {})
        assert is_record is False
        assert UNIPROT_RECORD_URL.format(accession=gene) != url


def test_an_annotation_accession_still_wins():
    """If an annotation file ever carries one, it is better than the bundled
    table -- it is about the user's own data."""
    label, url, is_record = uniprot_reference(
        "224750", {"UniProt ID": "P12345"})

    assert is_record is True
    assert url.endswith("/P12345/entry"), url


def test_a_missing_table_does_not_break_the_tile(monkeypatch):
    """A screen of another organism has no reason to carry the file, and a
    gene tile without a UniProt line is still a gene tile."""
    import spacr.gene_tile as tile

    uniprot_accessions.cache_clear()
    monkeypatch.setattr(tile, "UNIPROT_TABLE", "/nonexistent/uniprot.csv")
    try:
        assert tile.uniprot_accessions() == {}
        _label, _url, is_record = tile.uniprot_reference("224750", {})
        assert is_record is False
    finally:
        uniprot_accessions.cache_clear()
