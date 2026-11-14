"""What the gene tile does with the blank cells every real side table carries.

The bundled UniProt map is a CSV somebody maintained by hand, and it ships
rows with a hole in them: a gene with no UniProt accession yet, or an
accession whose gene number was lost. The index builder skips those cells,
because a hole that is indexed anyway becomes a record link on the tile.

These drive each skip together with the neighbouring row that is NOT skipped,
so a builder that stopped skipping and a builder that started dropping good
rows both fail here.
"""
from __future__ import annotations

from spacr import gene_tile as GT

# ---------------------------------------------------------------------------
# the bundled UniProt map
# ---------------------------------------------------------------------------

def test_a_uniprot_row_missing_either_half_maps_no_gene(tmp_path, monkeypatch):
    """A half-filled row must produce a search link, never a record link.

    The UniProt table is a two-column join and either column can be blank: a
    gene with no accession yet, or an accession whose gene number was lost.
    Indexing those would put a record URL on the tile built from an empty
    accession -- and a record link is a claim that THIS is the protein, the one
    kind of wrong link a reader cannot see, because it opens a real page. The
    complete row in the same file must still resolve to its record, so this
    drives both halves of the rule from one table.
    """
    table = tmp_path / "uniprot.csv"
    table.write_text(
        "gene_nr,uniprot\n"
        "039160,Q9BJF5\n"      # complete
        "224750,\n"            # gene known, no accession yet
        ",P99999\n"            # accession with no gene number
        "411710,   \n",        # whitespace is not an accession
        encoding="utf-8")

    GT.uniprot_accessions.cache_clear()
    monkeypatch.setattr(GT, "UNIPROT_TABLE", str(table))
    try:
        assert GT.uniprot_accessions() == {"039160": "Q9BJF5",
                                           "39160": "Q9BJF5"}

        label, url, is_record = GT.uniprot_reference("TGGT1_039160", {})
        assert is_record is True
        assert label == "UniProt Q9BJF5"
        assert url.endswith("/Q9BJF5/entry")

        label, url, is_record = GT.uniprot_reference("TGGT1_224750", {})
        assert is_record is False
        assert label == "UniProt search: TGGT1_224750"
        assert "query=TGGT1_224750" in url
    finally:
        GT.uniprot_accessions.cache_clear()


def test_the_uniprot_map_resolves_a_gene_with_or_without_its_leading_zero():
    """``039160`` and ``39160`` are the same gene, and both must find the record.

    The bundled table preserves leading zeros; a click arriving from the
    volcano carries whichever spelling that screen's results table used. If
    only one spelling resolved, the same protein would show a record link on
    one plot and a bare search on another, and the user would conclude spaCR
    does not know the gene.
    """
    GT.uniprot_accessions.cache_clear()
    try:
        table = GT.uniprot_accessions()
        padded = next((g for g in table if g.startswith("0")), "")
        assert padded, "the bundled table should carry zero-padded gene numbers"
        bare = padded.lstrip("0")
        assert table[bare] == table[padded]

        _label, url, is_record = GT.uniprot_reference(f"TGGT1_{bare}", {})
        assert is_record is True
        assert table[padded] in url
    finally:
        GT.uniprot_accessions.cache_clear()
