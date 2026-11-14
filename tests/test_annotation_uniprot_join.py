"""Table-driven tests of the UniProt join in ``spacr.annotation``.

A run annotates its results and carries on either way, so every refusal has
to come back as ``(frame, note)`` rather than as an exception or a silently
unchanged table: the note is the only thing that tells the user whether
UniProt was unreachable, named no usable columns, or matched not one gene.
One general test per unit, each walking a case table with pytest's
``subtests`` fixture.  UniProt itself is replaced by a canned reply, so
nothing here touches the network.
"""
from __future__ import annotations

import pandas as pd

from spacr import annotation, uniprot

TP53_REPLY = pd.DataFrame({"Entry": ["P04637"],
                           "Gene Names": ["TP53 P53"],
                           "Protein names": ["Cellular tumor antigen"]})


def test_the_gene_column_is_the_first_candidate_that_holds_text(subtests):
    """The finder walks its preference list past absent and blank columns."""
    cases = [
        ("found_past_absent_candidates",
         pd.DataFrame({"fraction": [0.1, 0.2], "grna": ["TP53_1", "TP53_2"]}),
         "grna"),
        ("preference_order_wins_over_column_order",
         pd.DataFrame({"feature": ["x"], "gene": ["TP53"]}), "gene"),
        ("blank_gene_column_is_skipped_for_the_next_one",
         pd.DataFrame({"gene": ["", " "], "gene_name": ["TP53", "MDM2"]}),
         "gene_name"),
        ("no_candidate_at_all",
         pd.DataFrame({"fraction": [0.1], "plate": ["p1"]}), None),
        ("only_blank_strings_name_nothing",
         pd.DataFrame({"gene": ["", "   ", ""]}), None),
        ("not_a_frame", object(), None),
    ]
    for case_id, frame, expected in cases:
        with subtests.test(msg=case_id):
            assert annotation._uniprot_key_column(frame) == expected


def test_annotate_from_uniprot_joins_or_returns_the_frame_with_its_reason(
        subtests, monkeypatch, capsys):
    """Each reply shape yields the exact frame and note the caller relies on."""
    asked = {}

    def install(table, note=""):
        def reply(source, cache_dir=None, genes=None):
            asked["genes"] = genes
            return table, note
        monkeypatch.setattr(uniprot, "annotation_for", reply)

    gene_frame = pd.DataFrame({"gene": ["TP53"], "coefficient": [0.3]})
    # (case, reply table, reply note, frame, kwargs, same object back?,
    #  expected note, expected product column or None, genes asked)
    cases = [
        ("unreachable_note_carried_through", None, "UniProt is unreachable.",
         gene_frame, {}, True, "UniProt is unreachable.", None, ["TP53"]),
        ("empty_reply_carries_its_note", pd.DataFrame(), "no entries",
         gene_frame, {}, True, "no entries", None, ["TP53"]),
        ("no_gene_column_is_named", TP53_REPLY, "",
         pd.DataFrame({"plate": ["p1"], "coefficient": [0.3]}), {}, True,
         "this table names no gene column, so the UniProt annotation was "
         "not joined onto it.", None, None),
        ("explicit_key_column_missing", TP53_REPLY, "", gene_frame,
         {"key_column": "symbol"}, True,
         "this table names no gene column, so the UniProt annotation was "
         "not joined onto it.", None, None),
        ("no_recognised_reply_columns",
         pd.DataFrame({"Something Else": ["x"]}), "", gene_frame, {}, True,
         "UniProt returned no usable columns.", None, ["TP53"]),
        ("caller_columns_are_not_overwritten",
         pd.DataFrame({"Gene Names": ["TP53"]}), "",
         pd.DataFrame({"gene": ["TP53"], "gene_name": ["mine"]}), {}, True,
         "", None, ["TP53"]),
        ("synonym_matches_case_insensitively", TP53_REPLY, "",
         pd.DataFrame({"gene": ["p53"], "coefficient": [0.3]}),
         {"quiet": True}, False, "", ["Cellular tumor antigen"], ["p53"]),
        ("accession_matches", TP53_REPLY, "",
         pd.DataFrame({"gene": [" P04637 "]}), {"quiet": True}, False, "",
         ["Cellular tumor antigen"], [" P04637 "]),
        ("nothing_matched_is_stated", TP53_REPLY, "",
         pd.DataFrame({"gene": ["GRA14", "ROP18"], "coefficient": [.3, .1]}),
         {"quiet": True}, False,
         "none of the 2 gene identifiers matched a UniProt entry for "
         "Homo sapiens.", [None, None], ["GRA14", "ROP18"]),
        ("nothing_matched_appends_to_the_reply_note", TP53_REPLY, "cached",
         pd.DataFrame({"gene": ["GRA14"]}), {"quiet": True}, False,
         "cached none of the 1 gene identifiers matched a UniProt entry for "
         "Homo sapiens.", [None], ["GRA14"]),
    ]
    for (case_id, table, reply_note, frame, kwargs, same, note, product,
         genes) in cases:
        with subtests.test(msg=case_id):
            install(table, reply_note)
            asked.clear()
            out, got = annotation.annotate_from_uniprot(
                frame, "Homo sapiens", **kwargs)
            assert got == note
            assert (out is frame) is same
            assert asked["genes"] == genes
            assert len(out) == len(frame), "the merge must not multiply rows"
            if product is None:
                assert list(out.columns) == list(frame.columns)
            else:
                assert [None if pd.isna(v) else v
                        for v in out["product"]] == product
                assert list(out.columns) == [
                    *frame.columns, "uniprot_accession", "product",
                    "gene_name"]
                assert "_key" not in out.columns

    with subtests.test(msg="frames_without_columns_skip_the_query"):
        install(TP53_REPLY)
        asked.clear()
        empty = pd.DataFrame()
        assert annotation.annotate_from_uniprot(empty, "Homo sapiens")[0] \
            is empty
        assert annotation.annotate_from_uniprot(None, "Homo sapiens") == \
            (None, "")
        assert asked == {}

    with subtests.test(msg="the_match_count_is_printed_unless_quiet"):
        install(TP53_REPLY)
        capsys.readouterr()
        annotation.annotate_from_uniprot(
            pd.DataFrame({"gene": ["TP53", "ROP18"]}), "Homo sapiens")
        assert capsys.readouterr().out == (
            "UniProt annotation (Homo sapiens): 1 of 2 rows matched, "
            "3 column(s) joined.\n")
        annotation.annotate_from_uniprot(
            pd.DataFrame({"gene": ["TP53"]}), "Homo sapiens", quiet=True)
        assert capsys.readouterr().out == ""


def test_a_matching_join_keeps_the_callers_values(monkeypatch):
    """Joined rows keep the caller's values and gain UniProt's, row for row."""
    monkeypatch.setattr(uniprot, "annotation_for",
                        lambda *a, **k: (TP53_REPLY, ""))
    frame = pd.DataFrame({"gene": ["TP53", "P53", "MDM2"],
                          "coefficient": [0.3, 0.2, 0.1]})
    out, note = annotation.annotate_from_uniprot(frame, "Homo sapiens",
                                             quiet=True)
    assert note == ""
    expected = frame.assign(
        uniprot_accession=["P04637", "P04637", None],
        product=["Cellular tumor antigen", "Cellular tumor antigen", None],
        gene_name=["TP53 P53", "TP53 P53", None])
    pd.testing.assert_frame_equal(out.astype(object).where(out.notna(), None),
                                  expected.astype(object), check_dtype=False)
