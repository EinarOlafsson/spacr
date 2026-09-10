"""What the gene tile does with the blank cells every real side table carries.

Every source a tile joins against -- the gRNA reference, the curated
annotation export, the LOPIT localisation table, the bundled UniProt map --
is a CSV somebody maintained by hand, and every one of them ships rows with a
hole in them: a guide listed without its protospacer, an annotation row whose
``Gene ID`` cell is empty, a gene LOPIT never placed, a gene with no UniProt
accession yet. The index builders skip those cells, and skipping is the whole
point: a hole that is indexed anyway becomes a fact on the tile -- a gene
whose localisation is the empty string, a protospacer that is ``""`` and
therefore shared by every other guide missing one, which would report the
whole library as one ambiguous mapping.

These drive each skip together with the neighbouring row that is NOT skipped,
so a builder that stopped skipping and a builder that started dropping good
rows both fail here.
"""
from __future__ import annotations

import pandas as pd

from spacr import gene_tile as GT

#: Two distinct protospacers, twenty nucleotides each, as a real reference
#: writes them.
SEQUENCE = "ACGTACGTACGTACGTACGT"
OTHER_SEQUENCE = "TTTTGGGGCCCCAAAATTTT"


# ---------------------------------------------------------------------------
# the gRNA reference
# ---------------------------------------------------------------------------

def test_a_guide_listed_without_a_protospacer_is_still_a_known_guide():
    """A blank sequence cell costs the protospacer, not the whole guide row.

    References are edited by hand and rows lose their sequence. The guide must
    still resolve to its gene and accession -- that is what puts a ToxoDB link
    on the tile -- while the tile reports no protospacer at all. The blank must
    never reach the by-sequence map either: ``""`` there would be a protospacer
    every sequence-less guide in the library shares, and the tile would call
    each of them ambiguous across every gene in the file.
    """
    reference = pd.DataFrame({
        "name": ["TGGT1_239740_3", "TGGT1_411710_1"],
        "sequence": ["", SEQUENCE],
    })

    by_guide, by_sequence, strain_of = GT._reference_index(reference)

    assert by_guide["239740_3"] == ("TGGT1_239740", "")
    assert by_guide["411710_1"] == ("TGGT1_411710", SEQUENCE)
    assert strain_of == {"239740": "TGGT1", "411710": "TGGT1"}
    # The row with a sequence is indexed by it; the blank row indexed nothing.
    assert by_sequence == {SEQUENCE: [("411710", "TGGT1_411710")]}

    blank = GT.gene_tile("fraction:grna[239740_3]", barcodes=reference,
                         metadata=None, localisation=None)
    assert blank.protospacer == ""
    assert blank.ambiguous is False
    assert [(c.gene, c.accession) for c in blank.candidates] == [
        ("239740", "TGGT1_239740")]

    kept = GT.gene_tile("fraction:grna[411710_1]", barcodes=reference,
                        metadata=None, localisation=None)
    assert kept.protospacer == SEQUENCE
    assert kept.ambiguous is False


def test_one_gene_using_a_protospacer_twice_is_not_two_genes():
    """A gene whose reference lists the same sequence for two guides is one gene.

    Libraries repeat a protospacer under several guide numbers of the SAME
    gene. If each repeat were pushed onto the by-sequence bucket, the tile
    would count the bucket, find two, and tell the user their effect cannot be
    assigned to either gene -- an ambiguity warning over a gene that is not
    ambiguous at all, which invites them to throw away a real hit. The
    duplicate is collapsed; a genuine two-GENE collision still raises the
    warning, and this drives both against the same protospacer.
    """
    repeated = pd.DataFrame({
        "name": ["TGGT1_239740_3", "TGGT1_239740_7"],
        "sequence": [SEQUENCE, SEQUENCE],
    })
    _by_guide, by_sequence, _strain = GT._reference_index(repeated)
    assert by_sequence == {SEQUENCE: [("239740", "TGGT1_239740")]}

    single = GT.gene_tile("fraction:grna[239740_3]", barcodes=repeated,
                          metadata=None, localisation=None)
    assert single.protospacer == SEQUENCE
    assert single.ambiguous is False
    assert single.ambiguity == ""
    assert [c.gene for c in single.candidates] == ["239740"]

    # The same sequence under a SECOND gene is the case the bucket exists for.
    shared = pd.DataFrame({
        "name": ["TGGT1_239740_3", "TGGT1_411710_1"],
        "sequence": [SEQUENCE, SEQUENCE],
    })
    collision = GT.gene_tile("fraction:grna[239740_3]", barcodes=shared,
                             metadata=None, localisation=None)
    assert collision.ambiguous is True
    assert [c.gene for c in collision.candidates] == ["239740", "411710"]
    assert "TGGT1_239740, TGGT1_411710" in collision.ambiguity


# ---------------------------------------------------------------------------
# the curated annotation export
# ---------------------------------------------------------------------------

def test_an_annotation_row_with_no_gene_id_annotates_nothing():
    """An export row whose ``Gene ID`` cell is empty is dropped, not keyed to "".

    A curated export that gained a trailing or separator row has a product
    description sitting against no gene at all. Keying it under the empty
    string is one lookup miss away from printing that product under a real
    gene, and a product description is exactly the line a user reads as "this
    is what I clicked". The named rows in the same export must still annotate
    their genes, so this drives a good row and the blank row from one frame.
    """
    export = pd.DataFrame({
        "Gene ID": ["TGME49_239740", "", "TGME49_411710"],
        "Product Description": ["rhoptry kinase ROP18", "ghost product",
                                "apical protein"],
        "Gene Name or Symbol": ["ROP18", "GHOST", "AP1"],
    })

    index = GT._annotation_index(export)
    assert sorted(index) == ["239740", "411710"]
    assert index["239740"]["Product Description"] == "rhoptry kinase ROP18"

    tile = GT.gene_tile("gene_fraction:gene[239740]", metadata=export,
                        barcodes=None, localisation=None)
    candidate = tile.candidates[0]
    assert candidate.annotation_id == "TGME49_239740"
    assert candidate.product == "rhoptry kinase ROP18"
    assert candidate.symbol == "ROP18"
    assert ("product", "rhoptry kinase ROP18") in candidate.fields
    # The orphaned row reached no gene on the tile.
    assert "ghost product" not in str(candidate.fields)
    assert "GHOST" not in str(candidate.annotation)


def test_the_first_row_for_a_gene_wins_when_an_export_lists_it_twice():
    """A per-transcript export must annotate a gene once, with its first row.

    VEuPathDB exports carry one row per transcript, so a gene appears several
    times. The tile takes the first and drops the rest -- the same rule the hit
    list uses -- because a tile that took the last would disagree with the hit
    list about the same gene in the same run, and a user comparing the two
    would be reading two different products for one click.
    """
    export = pd.DataFrame({
        "Gene ID": ["TGME49_239740", "TGGT1_239740", "TGME49_411710"],
        "Product Description": ["first transcript", "second transcript",
                                "apical protein"],
    })

    index = GT._annotation_index(export)
    assert index["239740"]["Product Description"] == "first transcript"

    tile = GT.gene_tile("gene_fraction:gene[239740]", metadata=export,
                        barcodes=None, localisation=None)
    assert tile.candidates[0].product == "first transcript"
    assert tile.candidates[0].annotation_id == "TGME49_239740"


# ---------------------------------------------------------------------------
# the localisation table
# ---------------------------------------------------------------------------

def test_a_gene_lopit_never_placed_gets_no_localisation_line():
    """A blank TAGM cell means "unplaced", and unplaced is not a compartment.

    LOPIT places some genes and not others, and the unplaced ones sit in the
    table with an empty location. Indexed as-is, the tile would show a
    localisation line whose value is nothing -- read as "spaCR knows where this
    protein is and it is here", against a gene the experiment could not
    assign. The gene the table DID place must keep its compartment, so both
    genes come from the same frame here.
    """
    lopit = pd.DataFrame({
        "gene_nr": ["239740", "411710"],
        "tagm_location": ["rhoptries", "   "],
    })

    assert GT._localisation_index(lopit) == {"239740": "rhoptries"}

    placed = GT.gene_tile("gene_fraction:gene[239740]", localisation=lopit,
                          barcodes=None, metadata=None)
    assert placed.candidates[0].localisation == "rhoptries"

    unplaced = GT.gene_tile("gene_fraction:gene[411710]", localisation=lopit,
                            barcodes=None, metadata=None)
    assert unplaced.candidates[0].gene == "411710"
    assert unplaced.candidates[0].localisation == ""


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
