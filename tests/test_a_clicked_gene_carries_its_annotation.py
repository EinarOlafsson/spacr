"""Clicking a gene shows what spaCR knows about it, and says when it knows nothing.

Instruction 121 wants "all the information on that gene" on the tile.
:mod:`spacr.gene_tile` already answers WHICH gene a dot is; this file is about
:mod:`spacr.gene_facts`, which answers what that gene IS -- product, DeepTMHMM
topology with each segment's coordinates, hyperLOPIT compartment, the
published CRISPR fitness screens and the stage expression.

The four things that have to be true, and each of them has been false in this
project before:

* ONE KEY SPACE. ``TGGT1_239740``, ``TGME49_239740``,
  ``gene_fraction:gene[239740]`` and the guide ``239740_3`` are one gene, and
  they must return one record. Two key spaces is how a volcano ends up
  colouring the wrong dots while looking entirely plausible.
* A GAP IS SAID OUT LOUD. A gene with no annotation row gets a sentence, not a
  panel of blank fields -- blank fields read as "measured, found nothing".
* NOTHING IS INVENTED AND NOTHING IS DROPPED. Every column
  ``spacr.annotation`` can add lands in exactly one group on the tile,
  including one this module has never heard of.
* THE GUI THREAD DOES NOT READ FILES. After :func:`spacr.gene_facts.warm`,
  a click must not touch the disk or the network.

The numbers asserted here are read off the SHIPPED tables, so a test that
starts failing means the bundled annotation changed -- which is a thing worth
being told about.
"""
from __future__ import annotations

import math

import pytest

from spacr import annotation, gene_facts


#: GRA14. A dense granule protein with a name, a product, a compartment, a
#: signal peptide and one transmembrane helix -- one gene that exercises every
#: group on the tile at once. It is also the gene the instruction's own worked
#: example uses.
GRA14 = "239740"

#: ROM1, a rhomboid protease: seven transmembrane helices and no signal
#: peptide, which is what makes it the per-segment coordinate case.
ROM1 = "200290"

#: Two of the three genes the ambiguous protospacer could belong to. 241310
#: has a product description and 411710 has none, which is the pair that shows
#: a missing identity is reported rather than filled in.
NAMED_OF_THE_AMBIGUOUS_TRIO = "241310"
UNNAMED_OF_THE_AMBIGUOUS_TRIO = "411710"


@pytest.fixture(autouse=True)
def _warm():
    """Every test runs against the real bundled tables, loaded once."""
    gene_facts.warm()
    yield


def test_every_spelling_of_one_gene_gives_one_record():
    """The whole point of leaning on `annotation.gene_number`.

    A design term, both strains' accessions, a guide id and the bare number
    are five names for gene 239740, and a tile that answered them differently
    would be a second key space.
    """
    spellings = ["239740", "TGGT1_239740", "TGME49_239740",
                 "gene_fraction:gene[239740]", "fraction:grna[239740_3]"]
    records = [gene_facts.facts(one) for one in spellings]
    assert {record.gene for record in records} == {GRA14}
    first = records[0].values
    for record, spelling in zip(records[1:], spellings[1:]):
        assert record.values == first, f"{spelling} produced a different record"


def test_the_identity_is_the_first_thing_on_the_tile():
    """Identity, then everything else -- the instruction's ORDER MATTERS."""
    known = gene_facts.facts(GRA14)
    headings = [heading for heading, _rows in known.sections()]
    assert headings[0] == "identity"
    assert known.value("gene_name") == "GRA14"
    assert "dense granule" in known.value("product_description").lower()


def test_the_compartment_comes_from_hyperlopit():
    assert gene_facts.facts(GRA14).value("hyperlopit") == "dense granules"


def test_all_seven_published_fitness_screens_are_on_the_tile():
    """The seven `fit_` columns of the bundled phenotype table, under one
    heading, with no invented eighth and none of them quietly dropped."""
    known = gene_facts.facts(GRA14)
    rows = dict(dict(known.sections())["CRISPR fitness screens"])
    assert len(rows) == 7, rows
    assert set(rows) == {"in vitro (HFF)", "in vivo (PE)", "in vivo (lung)",
                         "in vivo (liver)", "in vivo (spleen)", "naive BMDM",
                         "IFN-γ"}
    assert rows["in vivo (liver)"] == "-5.8949"


def test_the_five_enteroepithelial_stages_are_named_not_numbered():
    """`expr_ees3` is a column name; "enteroepithelial stage 3" is a fact."""
    rows = dict(dict(gene_facts.facts(GRA14).sections())["expression"])
    assert rows["tachyzoite"] == "681.59"
    assert rows["tissue cyst (bradyzoite)"] == "489.47"
    assert [label for label in rows if label.startswith("enteroepithelial")] == [
        f"enteroepithelial stage {n}" for n in range(1, 6)]


def test_a_multipass_protein_carries_every_segment_in_residue_order():
    """The per-segment coordinates are the thing `annotate` leaves out.

    ROM1's seven helices, from DeepTMHMM's own columns: they must arrive in
    residue order, non-overlapping, and each one's length must be the length
    the source file recorded rather than one this module recomputed.
    """
    known = gene_facts.facts(ROM1)
    helices = [s for s in known.segments if s.kind == "transmembrane"]
    assert len(helices) == 7 == known.value("n_transmembrane")
    assert [s.index for s in helices] == list(range(1, 8))
    assert helices[0].start == 62 and helices[0].end == 78
    previous = 0
    for helix in helices:
        assert previous < helix.start <= helix.end
        assert helix.length == helix.end - helix.start + 1
        previous = helix.end


def test_the_signal_peptide_span_does_not_reuse_the_summary_row_s_label():
    """Two rows labelled "signal peptide" read as the panel printing twice."""
    known = gene_facts.facts(GRA14)
    rows = dict(dict(known.sections())["membrane topology"])
    assert rows["signal peptide"] == "yes"
    assert rows["signal peptide length"] == "34"
    assert rows["signal peptide span"] == "residues 1–34 (34 aa)"
    assert rows["TM 1"] == "residues 290–304 (15 aa)"


def test_a_protein_with_no_signal_peptide_says_no_rather_than_saying_nothing():
    """`False` is an answer and `0` is an answer.

    Dropping them as "empty" would turn "DeepTMHMM looked and found no signal
    peptide" into silence, which reads as "nobody looked".
    """
    rows = dict(dict(gene_facts.facts(ROM1).sections())["membrane topology"])
    assert rows["signal peptide"] == "no"
    rows = dict(dict(
        gene_facts.facts(NAMED_OF_THE_AMBIGUOUS_TRIO).sections()
    )["membrane topology"])
    assert rows["transmembrane"] == "no"
    assert rows["transmembrane helices"] == "0"


def test_a_gene_the_annotation_never_heard_of_says_so_in_a_sentence():
    """Not an empty record: an empty panel reads as a bug."""
    known = gene_facts.facts("999999")
    assert not known.known
    assert known.gene == "999999"
    assert "999999" in known.reason
    assert "annotation" in known.reason
    assert known.to_text() == known.reason


def test_a_term_that_names_no_gene_says_that_instead():
    for term in ("Intercept", "rowID[T.r03]", "", None):
        known = gene_facts.facts(term)
        assert not known.known
        assert known.gene == ""
        assert "does not name a Toxoplasma gene" in known.reason


def test_a_gene_with_only_part_of_the_annotation_shows_only_that_part():
    """411710 has four fitness scores and no name, no product, no topology.

    The groups it cannot fill are ABSENT rather than present and blank -- the
    rule the annotation module states and this one has to keep.
    """
    known = gene_facts.facts(UNNAMED_OF_THE_AMBIGUOUS_TRIO)
    assert known.known
    headings = [heading for heading, _rows in known.sections()]
    assert headings == ["CRISPR fitness screens"]
    assert known.value("gene_name") is None
    for _heading, rows in known.sections():
        assert rows, "a heading was emitted over no rows"


def test_no_group_is_ever_emitted_empty():
    for gene in (GRA14, ROM1, NAMED_OF_THE_AMBIGUOUS_TRIO,
                 UNNAMED_OF_THE_AMBIGUOUS_TRIO):
        for heading, rows in gene_facts.facts(gene).sections():
            assert rows, f"{gene}: {heading} had no rows"
            for label, value in rows:
                assert label and value, f"{gene}: {heading} had a blank cell"


def test_every_column_the_annotation_can_add_lands_in_exactly_one_group():
    """A source added to `annotation.SOURCES` cannot fall off the tile."""
    from collections import Counter

    placed = Counter(column for _heading, columns in gene_facts._layout()
                     for column in columns)
    assert set(placed) == set(annotation.columns())
    assert not [c for c, n in placed.items() if n != 1]


def test_a_column_this_module_has_never_heard_of_still_gets_a_heading(
        monkeypatch):
    """The negative of the test above, driven rather than asserted.

    A column matching neither a name in GROUPS nor a prefix must land in
    "other annotation" -- silently dropping it is how a new annotation source
    ships and is never seen.
    """
    real = list(annotation.columns())
    monkeypatch.setattr(annotation, "columns", lambda: real + ["moon_phase"])
    gene_facts._layout.cache_clear()
    try:
        layout = dict(gene_facts._layout())
        assert layout[gene_facts.OTHER] == ("moon_phase",)
        assert "moon_phase" in gene_facts.available()
    finally:
        gene_facts._layout.cache_clear()


def test_an_unknown_column_is_labelled_by_its_own_name():
    """So an eighth published screen reads as a screen, not as `fit_gut`."""
    assert gene_facts._label("fit_gut_2027") == "gut 2027"
    assert gene_facts._label("expr_merozoite") == "merozoite"
    assert gene_facts._label("moon_phase") == "moon phase"


def test_several_genes_come_back_in_the_order_they_were_named():
    """The ambiguous case asks for three at once and the order is the tile's."""
    found = gene_facts.facts_for(
        ["fraction:grna[411710_2]", "TGME49_241310", "411210", "Intercept"])
    assert list(found) == ["411710", "241310", "411210"]
    assert found["241310"].value("product_description") == "hypothetical protein"


def test_a_repeated_gene_is_asked_for_once():
    found = gene_facts.facts_for(["239740", "TGGT1_239740",
                                  "gene_fraction:gene[239740]"])
    assert list(found) == [GRA14]


def test_nothing_in_the_values_names_a_gene_returns_nothing():
    assert gene_facts.facts_for(["Intercept", "rowID[T.r03]"]) == {}


def test_a_warmed_click_reads_no_file(monkeypatch):
    """THE RULE THIS MODULE EXISTS FOR.

    Cold, the first click is 360 ms of CSV reading inside a mouse press.
    After `warm`, a click must not reach the disk at all -- so `read_csv` is
    replaced with something that raises and a click still answers.
    """
    import pandas as pd

    gene_facts.warm([GRA14, ROM1])

    def refuse(*args, **kwargs):
        raise AssertionError("the GUI thread read a file on a click")

    monkeypatch.setattr(pd, "read_csv", refuse)
    known = gene_facts.facts(f"fraction:grna[{GRA14}_3]")
    assert known.value("gene_name") == "GRA14"
    assert len(gene_facts.facts(ROM1).segments) == 7


def test_warming_a_whole_screen_costs_one_join(monkeypatch):
    """400 genes and one gene both cost one `annotate` call.

    Not a micro-optimisation: `annotate` re-verifies all five right-hand keys
    for uniqueness on every call, so a per-click join is 20 ms of a mouse
    press and a per-screen join is 21 ms once.
    """
    calls = []
    real = annotation.annotate
    monkeypatch.setattr(annotation, "annotate",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    gene_facts.clear_cache()
    gene_facts.warm([f"fraction:grna[{200000 + n}_1]" for n in range(400)])
    assert len(calls) == 1
    gene_facts.facts("200290")
    assert len(calls) == 1, "a warmed gene was joined again"


def test_no_click_touches_the_network(monkeypatch):
    """The instruction: "Nothing in the path makes a network call while the
    user waits"."""
    import socket

    gene_facts.warm([GRA14])

    def refuse(*args, **kwargs):
        raise AssertionError("a gene lookup opened a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert gene_facts.facts(GRA14).value("gene_name") == "GRA14"


def test_the_html_escapes_what_the_annotation_put_in_it():
    """A product description is text from a curated export, not markup."""
    known = gene_facts.GeneFacts(
        gene="1", values={"product_description": "<b>not bold</b> & co"})
    html = known.to_html()
    assert "&lt;b&gt;not bold&lt;/b&gt; &amp; co" in html
    assert "<b>not bold" not in html


def test_the_html_of_an_unknown_gene_is_the_sentence_not_a_blank_table():
    known = gene_facts.facts("999999")
    html = known.to_html()
    assert "<table>" not in html
    assert "999999" in html


def test_clearing_the_cache_drops_the_records_and_the_indices():
    gene_facts.warm([GRA14])
    assert gene_facts.facts(GRA14) is gene_facts.facts(GRA14)
    gene_facts.clear_cache()
    assert not gene_facts._CACHE
    assert gene_facts.facts(GRA14).value("gene_name") == "GRA14"


def test_the_record_cache_is_bounded(monkeypatch):
    """A session that loads twenty screens must not grow without limit.

    Driven rather than asserted about: the bound is dropped to two and the
    third gene is watched to clear the first two.
    """
    gene_facts.clear_cache()
    gene_facts.warm()
    monkeypatch.setattr(gene_facts, "_CACHE_MAX", 2)
    gene_facts.facts_for(["239740", "200290"])
    assert len(gene_facts._CACHE) == 2
    gene_facts.facts("241310")
    assert list(gene_facts._CACHE) == ["241310"]
    gene_facts.clear_cache()


def test_numbers_print_as_numbers_and_not_as_pandas():
    """`7.0 helices` is a dtype leaking onto a figure caption."""
    assert gene_facts._show("n_transmembrane", 7.0) == "7"
    assert gene_facts._show("fit_ifng", -0.2742) == "-0.2742"
    assert gene_facts._show("signal_peptide", True) == "yes"
    assert gene_facts._show("signal_peptide", False) == "no"


def test_a_gap_is_a_gap_and_zero_is_not():
    assert gene_facts._is_gap(None)
    assert gene_facts._is_gap(float("nan"))
    assert gene_facts._is_gap("N/A")
    assert gene_facts._is_gap("  ")
    assert not gene_facts._is_gap(0.0)
    assert not gene_facts._is_gap(False)
    assert not gene_facts._is_gap("dense granules")


def test_a_missing_annotation_install_says_so_rather_than_showing_nothing(
        monkeypatch):
    """An install without the bundled tables is a state, not a failure."""
    monkeypatch.setattr(annotation, "columns", lambda: [])
    gene_facts._layout.cache_clear()
    gene_facts._CACHE.clear()
    try:
        assert gene_facts.available() == ()
        reason = gene_facts.unavailable_reason()
        assert "not installed" in reason
        known = gene_facts.facts(GRA14)
        assert not known.known and known.reason == reason
        assert gene_facts.warm([GRA14]) == ()
    finally:
        gene_facts.clear_cache()


def test_the_segment_index_survives_a_table_without_topology(monkeypatch):
    """A DeepTMHMM run that is not bundled leaves segments empty, not broken."""
    monkeypatch.setattr(annotation, "supplementary", lambda *a, **k: None)
    gene_facts._segment_index.cache_clear()
    try:
        assert gene_facts._segment_index() == {}
    finally:
        gene_facts._segment_index.cache_clear()


def test_a_segment_row_missing_its_length_is_measured_rather_than_dropped(
        monkeypatch):
    """The length column is the file's; without one the span still stands."""
    import pandas as pd

    frame = pd.DataFrame({"gene_nr": [123456], "tm_1_start": [10.0],
                          "tm_1_end": [20.0]})
    monkeypatch.setattr(annotation, "supplementary", lambda *a, **k: frame)
    gene_facts._segment_index.cache_clear()
    try:
        segments = gene_facts._segment_index()["123456"]
        assert segments[0].length == 11
        assert segments[0].text == "residues 10–20 (11 aa)"
        assert segments[0].label == "TM 1"
    finally:
        gene_facts._segment_index.cache_clear()


def test_the_facts_never_carry_a_nan():
    """A NaN reaching the tile prints as "nan", which reads as a value."""
    for gene in (GRA14, ROM1, NAMED_OF_THE_AMBIGUOUS_TRIO):
        for value in gene_facts.facts(gene).values.values():
            assert not (isinstance(value, float) and math.isnan(value))
