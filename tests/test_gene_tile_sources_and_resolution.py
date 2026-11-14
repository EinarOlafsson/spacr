"""Table-driven tests of how a gene tile loads its side tables and resolves a click.

Every source a tile joins against -- the gRNA reference, the curated
annotation export, the LOPIT table -- is a hand-maintained CSV with holes in
it.  A missing or unreadable source is "no such table", never a crash; a blank
cell is skipped rather than indexed; and a protospacer shared by two genes is
the one case that must read as ambiguous.  One general test per unit, each
walking a case table with ``subtests`` and asserting exact values.
"""
from __future__ import annotations

import os

import pandas as pd

from spacr import gene_tile as GT

SEQUENCE = "ACGTACGTACGTACGTACGT"
NO_RESULTS = ("no results table was supplied, so this screen's own numbers "
              "for the gene are not on this tile.")


def test_load_resolves_a_source_to_a_frame_and_a_cache_token(subtests,
                                                              tmp_path):
    """Frames get no token, absent or unparseable files are skipped, files keyed."""
    frame = pd.DataFrame({"name": ["TGGT1_239740_3"], "sequence": [SEQUENCE]})
    good = tmp_path / "reference.csv"
    frame.to_csv(good, index=False)
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    stat = os.stat(good)
    token = f"{good}|{stat.st_mtime}|{stat.st_size}"

    with subtests.test(msg="frame_used_as_is_without_token"):
        loaded, got = GT._load(frame, str(tmp_path / "unused.csv"))
        assert loaded is frame
        assert got is None

    for case_id, source in [("skipped_source", None),
                            ("missing_file", str(tmp_path / "missing.csv")),
                            ("zero_byte_file", str(empty)),
                            ("bundled_default_missing", GT.BUNDLED)]:
        with subtests.test(msg=case_id):
            assert GT._load(source, str(tmp_path / "default.csv")) == (None,
                                                                       None)

    for case_id, source in [("named_file", str(good)),
                            ("bundled_default_present", GT.BUNDLED)]:
        with subtests.test(msg=case_id):
            loaded, got = GT._load(source, str(good))
            pd.testing.assert_frame_equal(loaded, frame)
            assert got == token


def test_annotation_index_keys_each_gene_once_by_its_number(subtests):
    """First row per gene wins; blank ids and key-less tables annotate nothing."""
    cases = [
        ("no_table", None, {}),
        ("empty_table", pd.DataFrame(), {}),
        ("no_gene_column", pd.DataFrame({"product": ["a kinase"]}), {}),
        ("blank_id_dropped",
         pd.DataFrame({"Gene ID": ["TGME49_239740", "", "TGME49_411710"],
                       "product": ["ROP18", "ghost", "apical"]}),
         {"239740": {"Gene ID": "TGME49_239740", "product": "ROP18"},
          "411710": {"Gene ID": "TGME49_411710", "product": "apical"}}),
        ("first_transcript_wins_across_strains",
         pd.DataFrame({"Gene ID": ["TGME49_239740", "TGGT1_239740"],
                       "product": ["first", "second"]}),
         {"239740": {"Gene ID": "TGME49_239740", "product": "first"}}),
        ("gene_id_column_and_bare_number",
         pd.DataFrame({"gene_id": ["239740"], "product": ["bare"]}),
         {"239740": {"gene_id": "239740", "product": "bare"}}),
        ("gene_column",
         pd.DataFrame({"gene": ["TGGT1_411710"], "product": ["p"]}),
         {"411710": {"gene": "TGGT1_411710", "product": "p"}}),
    ]
    for case_id, frame, expected in cases:
        with subtests.test(msg=case_id):
            assert GT._annotation_index(frame) == expected


def _reference(names, sequences):
    return pd.DataFrame({"name": names, "sequence": sequences})


def test_gene_tile_resolves_a_guide_against_the_reference(subtests):
    """Protospacer, candidate genes and ambiguity follow the reference exactly."""
    shared_other_gene = _reference(["TGGT1_239740_3", "TGGT1_411710_1"],
                                   [SEQUENCE, SEQUENCE])
    cases = [
        # (case id, feature, barcodes, kind, [(gene, accession, reported)],
        #  protospacer, ambiguity, unresolved, notes)
        ("no_reference_says_never_checked", "fraction:grna[239740_3]", None,
         "guide", [("239740", "", True)], "", "",
         (NO_RESULTS,
          "no gRNA reference was supplied, so it is not known whether this "
          "guide's protospacer is unique to one gene."),
         ("no strain prefix was known for gene 239740, so no ToxoDB link "
          "could be built. Supply the gRNA reference the screen was counted "
          "against and the accession will resolve.",)),
        ("guide_absent_from_reference", "fraction:grna[239740_9]",
         _reference(["TGGT1_411710_1"], [SEQUENCE]),
         "guide", [("239740", "", True)], "", "",
         (NO_RESULTS,
          "guide 239740_9 is not in the gRNA reference, so its protospacer "
          "could not be checked for other genes carrying it. References may "
          "exclude guides whose protospacers map to multiple genes; verify "
          "the guide identifier and reference version."),
         ("no strain prefix was known for gene 239740, so no ToxoDB link "
          "could be built. Supply the gRNA reference the screen was counted "
          "against and the accession will resolve.",)),
        ("blank_protospacer_still_a_known_guide", "fraction:grna[239740_3]",
         _reference(["TGGT1_239740_3", "TGGT1_411710_1"], ["", SEQUENCE]),
         "guide", [("239740", "TGGT1_239740", True)], "", "",
         (NO_RESULTS,), ()),
        ("sequenced_neighbour_keeps_its_protospacer",
         "fraction:grna[411710_1]",
         _reference(["TGGT1_239740_3", "TGGT1_411710_1"], ["", SEQUENCE]),
         "guide", [("411710", "TGGT1_411710", True)], SEQUENCE, "",
         (NO_RESULTS,), ()),
        ("one_gene_repeating_a_protospacer_is_one_gene",
         "fraction:grna[239740_3]",
         _reference(["TGGT1_239740_3", "TGGT1_239740_7"],
                    [SEQUENCE, SEQUENCE]),
         "guide", [("239740", "TGGT1_239740", True)], SEQUENCE, "",
         (NO_RESULTS,), ()),
        ("two_genes_sharing_a_protospacer_are_ambiguous",
         "fraction:grna[239740_3]", shared_other_gene,
         "guide", [("239740", "TGGT1_239740", True),
                   ("411710", "TGGT1_411710", False)], SEQUENCE,
         f"protospacer {SEQUENCE} appears in the gRNA reference under 2 gene "
         f"names (TGGT1_239740, TGGT1_411710). Reads from this guide cannot "
         f"be told apart between them, so the effect below belongs to all 2 "
         f"equally — the regression attributed it to 239740 because that is "
         f"the name the counting pipeline saw last, which is a bookkeeping "
         f"fact, not a result.",
         (NO_RESULTS,), ()),
        ("reported_gene_missing_from_bucket_leads",
         "fraction:grna[AB_239740_3]",
         _reference(["TGGT1_AB_239740_3"], [SEQUENCE]),
         "guide", [("AB", "TGGT1_AB_239740", True),
                   ("AB_239740", "TGGT1_AB_239740", False)], SEQUENCE,
         f"protospacer {SEQUENCE} appears in the gRNA reference under 2 gene "
         f"names (TGGT1_AB_239740, TGGT1_AB_239740). Reads from this guide "
         f"cannot be told apart between them, so the effect below belongs to "
         f"all 2 equally — the regression attributed it to AB because that is "
         f"the name the counting pipeline saw last, which is a bookkeeping "
         f"fact, not a result.",
         (NO_RESULTS,
          "gene id 'AB' is not shaped like a Toxoplasma accession, so no "
          "ToxoDB record is offered for it. If this is a screen of another "
          "organism, spaCR has no annotation for it."), ()),
    ]
    for (case_id, feature, barcodes, kind, candidates, protospacer,
         ambiguity, unresolved, notes) in cases:
        with subtests.test(msg=case_id):
            tile = GT.gene_tile(feature, None, barcodes=barcodes,
                                metadata=None, localisation=None)
            assert tile.kind == kind
            assert [(c.gene, c.accession, c.reported)
                    for c in tile.candidates] == candidates
            assert tile.protospacer == protospacer
            assert tile.ambiguous is (len(candidates) > 1)
            assert tile.ambiguity == ambiguity
            assert tile.unresolved == unresolved
            assert tile.notes == notes

    with subtests.test(msg="ambiguous_text_and_html_name_the_reported_gene"):
        tile = GT.gene_tile("fraction:grna[AB_239740_3]", None,
                            barcodes=_reference(["TGGT1_AB_239740_3"],
                                                [SEQUENCE]),
                            metadata=None, localisation=None)
        text = tile.to_text()
        assert text.splitlines()[:2] == [
            "TGGT1_AB_239740 / TGGT1_AB_239740",
            "one protospacer, 2 genes — the effect cannot be assigned to any "
            "one of them"]
        assert ("IDENTITY — TGGT1_AB_239740 (THE GENE THE COUNTS WERE "
                "ATTRIBUTED TO)\n  gene id: TGGT1_AB_239740") in text
        assert "PROTOSPACER\n  sequence: ACGTACGTACGTACGTACGT" in text
        html = tile.to_html()
        assert "Ambiguous mapping" in html
        assert "<h2" in html


def test_gene_tile_annotates_and_localises_only_real_rows(subtests):
    """Blank annotation ids and unplaced LOPIT rows put nothing on the tile."""
    export = pd.DataFrame({
        "Gene ID": ["TGME49_239740", "", "TGGT1_239740", "TGME49_411710"],
        "Product Description": ["rhoptry kinase ROP18", "ghost product",
                                "second transcript", "apical protein"],
        "Gene Name or Symbol": ["ROP18", "GHOST", "ROP18b", "AP1"],
    })
    lopit = pd.DataFrame({"gene_nr": ["239740", "411710"],
                          "tagm_location": ["rhoptries", "   "]})
    cases = [
        # (case id, gene, metadata, localisation,
        #  (annotation_id, product, symbol, localisation))
        ("first_named_row_annotates", "239740", export, None,
         ("TGME49_239740", "rhoptry kinase ROP18", "ROP18", "")),
        ("second_gene_from_same_export", "411710", export, None,
         ("TGME49_411710", "apical protein", "AP1", "")),
        ("placed_gene_keeps_its_compartment", "239740", None, lopit,
         ("", "", "", "rhoptries")),
        ("unplaced_gene_has_no_localisation", "411710", None, lopit,
         ("", "", "", "")),
    ]
    for case_id, gene, metadata, localisation, expected in cases:
        with subtests.test(msg=case_id):
            tile = GT.gene_tile(f"gene_fraction:gene[{gene}]",
                                metadata=metadata, barcodes=None,
                                localisation=localisation)
            assert [c.gene for c in tile.candidates] == [gene]
            candidate = tile.candidates[0]
            assert (candidate.annotation_id, candidate.product,
                    candidate.symbol, candidate.localisation) == expected
            assert "ghost product" not in str(candidate.fields)
            assert "GHOST" not in str(candidate.annotation)

    with subtests.test(msg="annotation_fields_carry_the_product"):
        tile = GT.gene_tile("gene_fraction:gene[239740]", metadata=export,
                            barcodes=None, localisation=None)
        assert ("product", "rhoptry kinase ROP18") in tile.candidates[0].fields
