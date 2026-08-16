"""Clicking a point in the regression says what that gene IS, or why it cannot.

Instruction 121: "in the interactive regression, when a gene is clicked a tile
should appear with all the information on that gene." A volcano answers "which
guides moved" and cannot answer "what IS 411710", which is the question the
user has at the moment they click.

THE PART THAT CAN BE WRONG IS THE MAPPING. The regression fits GUIDES and the
gene is a prefix of them, so a resolver has to go guide -> gene and be right
about it, and in this screen's own data it sometimes cannot be. Three
protospacers in the TSG101 reference each sit in two or three genes:

    GCCGGCGATAGAGCCCCGCCC   TGGT1_241310_2   TGGT1_411210_2   TGGT1_411710_2
    GCGATAGAGCCCCGCCCTGG                     TGGT1_411210_3   TGGT1_411710_3
    GTCGCTAGGACATCCTCCAAG   TGGT1_241310_10  TGGT1_411210_10  TGGT1_411710_10

Those eight rows are REAL. They are quoted verbatim from
``spacr/resources/data/barcodes_grna.csv`` as it shipped before commit
``cd5a8920`` ("data: drop ambiguous shared guides, instruction 100 D2")
removed them, and they are still live in the screen's own reference at
``.../tsg101_screen/test/grna_barcodes.csv``. Nothing here is a synthetic
fixture: :data:`AMBIGUOUS_ROWS` is the diff of that commit and
:data:`REAL_ROWS` is transcribed from ``plate1_dv/ols/list/results.csv``,
where ``411710_2``, ``411710_3`` and ``411710_10`` — all three of the
ambiguous guides — appear as fitted coefficients.

So the two answers a click on ``411710_2`` can get are both tested:

* against the screen's own reference, which still holds the duplicates: THREE
  genes, and the tile says the mapping is ambiguous;
* against the shipped reference, which no longer holds them: "this guide is
  not in the reference", which is a different true statement and not a
  silently clean one.

And a miss is an answer everywhere else too — a non-targeting control, a gene
with no annotation row, an id from a screen of another organism. An empty
panel reads as a bug; a sentence reads as a finding.
"""
from __future__ import annotations

import math
import os
import socket

import pandas as pd
import pytest

from spacr.gene_tile import gene_tile, is_toxoplasma_gene_id
from spacr.hits import gene_of, guide_of

#: The eight rows commit cd5a8920 removed from the shipped reference, verbatim.
#: Recoverable at any time with
#: ``git show cd5a8920^:spacr/resources/data/barcodes_grna.csv``.
AMBIGUOUS_ROWS = [
    ("TGGT1_241310_10", "GTCGCTAGGACATCCTCCAAG"),
    ("TGGT1_241310_2", "GCCGGCGATAGAGCCCCGCCC"),
    ("TGGT1_411210_10", "GTCGCTAGGACATCCTCCAAG"),
    ("TGGT1_411210_2", "GCCGGCGATAGAGCCCCGCCC"),
    ("TGGT1_411210_3", "GCGATAGAGCCCCGCCCTGG"),
    ("TGGT1_411710_10", "GTCGCTAGGACATCCTCCAAG"),
    ("TGGT1_411710_2", "GCCGGCGATAGAGCCCCGCCC"),
    ("TGGT1_411710_3", "GCGATAGAGCCCCGCCCTGG"),
]

#: Unambiguous rows quoted from the SHIPPED reference, so the fixture is a
#: reference and not just the pathological corner of one. Note
#: ``TGGT1_241310_28``: the same gene as one of the ambiguous rows, on its own
#: unshared protospacer, which is why 241310 keeps a row in the shipped file.
CLEAN_ROWS = [
    ("TGGT1_000000_22", "GTCCATATAGTAGTATTAGAC"),
    ("TGGT1_225160_1", "GTGCCTACTGAATGCTACCA"),
    ("TGGT1_239740_1", "GCAACATCAACCGCAGCAGGA"),
    ("TGGT1_239740_3", "GCCCGTAAGCTATCTAGGAGA"),
    ("TGGT1_241310_28", "GGAAACGAAGGACCTCTTGG"),
]

#: Rows transcribed from the real screen, ``plate1_dv/ols/list/results.csv``:
#: ``(feature, coefficient, p_value, grna, condition, gene, n_grna, n_gene)``.
#: The whole 411710 family, the GRA14 positive control, the top-ranked
#: hypothetical protein, one control guide and the intercept.
REAL_ROWS = [
    ("Intercept", 0.1901544055440546, 3.153920935895919e-46,
     None, "other", None, None, None),
    ("fraction:grna[000000_22]", 4.370849812608784, 3.1222414615895375e-05,
     "000000_22", "control", None, 1.0, None),
    ("fraction:grna[225160_1]", 0.2095486129795587, 0.4858057713238733,
     "225160_1", "other", None, 5.0, None),
    ("fraction:grna[225160_2]", 0.207826520493141, 0.1955889789381013,
     "225160_2", "other", None, 9.0, None),
    ("fraction:grna[225160_3]", 0.2866982278649471, 0.1623789537747346,
     "225160_3", "other", None, 4.0, None),
    ("fraction:grna[239740_1]", 0.1443270629242229, 0.6031324094051667,
     "239740_1", "pc", None, 10.0, None),
    ("fraction:grna[239740_3]", 0.7288503312667108, 3.886509059233938e-05,
     "239740_3", "pc", None, 4.0, None),
    ("fraction:grna[411710_10]", 0.0102805901694449, 0.9763151604702928,
     "411710_10", "other", None, 6.0, None),
    ("fraction:grna[411710_2]", -0.0033716357862716, 0.9806973349380717,
     "411710_2", "other", None, 13.0, None),
    ("fraction:grna[411710_3]", -0.0208172903187161, 0.8779636976799623,
     "411710_3", "other", None, 12.0, None),
    ("gene_fraction:gene[225160]", 0.7040733613376482, 7.058846511259459e-09,
     None, "other", "225160", None, 18.0),
    ("gene_fraction:gene[239740]", 0.8731773941909292, 5.000024935549653e-09,
     None, "pc", "239740", None, 15.0),
    ("gene_fraction:gene[411710]", -0.0139083359354999, 0.9076974906968912,
     None, "other", "411710", None, 31.0),
]

#: The real screen, when this machine has it. Only used to prove the
#: transcription above is faithful; every other test runs everywhere.
REAL_SCREEN = ("/mnt/firecuda2/Claude/toxoplasma_projects/tsg101_screen/test")


@pytest.fixture()
def results() -> pd.DataFrame:
    """The real screen's own coefficients for the genes under test."""
    frame = pd.DataFrame(REAL_ROWS, columns=[
        "feature", "coefficient", "p_value", "grna", "condition", "gene",
        "n_grna", "n_gene"])
    # The run applied NO correction, so its q_value column is the p-values
    # unchanged. Reproduced rather than tidied away: the tile has to say so.
    frame["q_value"] = frame["p_value"]
    frame["multiple_testing_method"] = "none"
    return frame


@pytest.fixture()
def screen_reference(tmp_path) -> str:
    """The gRNA reference AS THE SCREEN WAS COUNTED, duplicates included."""
    path = tmp_path / "grna_barcodes.csv"
    pd.DataFrame(AMBIGUOUS_ROWS + CLEAN_ROWS,
                 columns=["name", "sequence"]).to_csv(path, index=False)
    return str(path)


# --------------------------------------------------------------------------- #
#  Identity: the first line has to be a name a human recognises
# --------------------------------------------------------------------------- #

def test_clicking_a_guide_puts_a_recognisable_gene_name_on_the_first_line(results):
    """`239740_3` means nothing to anyone. The tile's first line says GRA14."""
    tile = gene_tile("fraction:grna[239740_3]", results)

    assert tile.title == "GRA14", (
        f"the tile's first line is {tile.title!r}. The whole point of the "
        "instruction is that TGGT1_239740 is not a name a human recognises "
        "and the tile makes it one.")
    assert "dense granule protein GRA14" in tile.subtitle


def test_the_identity_comes_before_this_screens_numbers(results):
    """"identity first, then this screen's numbers, then everything else."""
    headings = [heading for heading, _ in gene_tile(
        "gene_fraction:gene[239740]", results).sections()]

    assert headings[0].startswith("identity"), headings
    assert "this screen" in headings, headings
    assert headings.index("this screen") < headings.index("external records"), (
        f"the bibliography is above the numbers: {headings}. A user who "
        "clicked a point wants to know what they clicked first.")


def test_the_tile_carries_what_spacr_already_knew_about_the_gene(results):
    """Not just the accession: product, symbol and the LOPIT localisation."""
    text = gene_tile("gene_fraction:gene[225160]", results).to_text()

    assert "hypothetical protein" in text
    assert "dense granules" in text, (
        "the bundled lopit.csv puts TGME49_225160 in the dense granules and "
        f"the tile did not say so:\n{text}")


def test_a_missing_results_frame_still_resolves_the_gene():
    """Identity does not depend on the numbers, and says the numbers are gone."""
    tile = gene_tile("gene_fraction:gene[239740]", None)

    assert tile.title == "GRA14"
    assert any("no results table" in note for note in tile.unresolved), (
        f"no numbers and no explanation either: {tile.unresolved}")


# --------------------------------------------------------------------------- #
#  This screen's own numbers
# --------------------------------------------------------------------------- #

def test_the_tile_carries_this_screens_effect_p_and_q(results):
    """"the effect, the p-value, the q-value ... cost nothing"."""
    tile = gene_tile("gene_fraction:gene[239740]", results)

    assert tile.effect == pytest.approx(0.8731773941909292)
    assert tile.p_value == pytest.approx(5.000024935549653e-09)
    assert tile.q_value == pytest.approx(5.000024935549653e-09)
    assert tile.condition == "pc"


def test_the_tile_lists_the_guides_behind_the_gene_and_how_each_moved(results):
    """"the guides that carry it and how each behaved"."""
    tile = gene_tile("gene_fraction:gene[239740]", results)

    assert [g.guide for g in tile.guides] == ["239740_1", "239740_3"]
    assert [g.direction for g in tile.guides] == ["up", "up"]
    assert tile.n_agree == 2, (
        "both of GRA14's guides push the same way as the gene effect and the "
        f"tile counted {tile.n_agree}")


def test_the_clicked_guide_is_marked_among_its_siblings(results):
    """Three guides look alike in a list; the user clicked one of them."""
    tile = gene_tile("fraction:grna[411710_3]", results)

    clicked = [g.guide for g in tile.guides if g.clicked]
    assert clicked == ["411710_3"], (
        f"clicked 411710_3 and the tile marked {clicked}")


def test_a_run_with_no_correction_does_not_call_its_q_value_corrected(results):
    """`multiple_testing_method` is 'none' on this screen, so the q-values ARE
    the p-values. A tile printing a bare "q-value" over them launders that."""
    tile = gene_tile("gene_fraction:gene[239740]", results)

    assert tile.correction == "none", (
        f"the correction method came back {tile.correction!r}; 'none' is a "
        "meaningful answer, not an empty cell")
    labels = [label for _, rows in tile.sections() for label, _ in rows]
    assert any("NO correction" in label for label in labels), (
        f"nothing on the tile warns the q-values are uncorrected: {labels}")


# --------------------------------------------------------------------------- #
#  THE AMBIGUOUS MAPPING, on the real rows
# --------------------------------------------------------------------------- #

def test_a_shared_protospacer_names_every_gene_it_could_be(results,
                                                          screen_reference):
    """`411710_2`'s protospacer sits in three genes. All three go on the tile.

    Instruction 121: "A guide that could belong to three genes gets a tile
    that SAYS SO, listing all three -- not the first one, and not a silent
    pick."
    """
    tile = gene_tile("fraction:grna[411710_2]", results,
                     barcodes=screen_reference)

    assert tile.ambiguous, (
        "GCCGGCGATAGAGCCCCGCCC is in the reference three times, under "
        "TGGT1_241310_2, TGGT1_411210_2 and TGGT1_411710_2, and the tile "
        "reported an unambiguous gene")
    assert {c.gene for c in tile.candidates} == {"411710", "241310", "411210"}, (
        f"the tile listed {[c.gene for c in tile.candidates]}")
    assert tile.protospacer == "GCCGGCGATAGAGCCCCGCCC"


def test_the_ambiguous_tile_says_the_mapping_is_ambiguous_in_words(
        results, screen_reference):
    """Listing three genes is not enough if nothing says why there are three."""
    tile = gene_tile("fraction:grna[411710_2]", results,
                     barcodes=screen_reference)

    assert "cannot be told apart" in tile.ambiguity, tile.ambiguity
    assert "TGGT1_241310" in tile.ambiguity and "TGGT1_411210" in tile.ambiguity
    assert "/" in tile.title, (
        f"the title is {tile.title!r} — one of the three genes wearing the "
        "confidence of a resolved one")


def test_the_two_shared_guides_of_one_gene_resolve_to_their_own_gene_sets(
        results, screen_reference):
    """`411710_3`'s protospacer is in two genes, `411710_2`'s is in three.

    A resolver that keyed on the GENE rather than on the guide's own sequence
    would give both the same answer, and one of them would be wrong.
    """
    two = gene_tile("fraction:grna[411710_3]", results,
                    barcodes=screen_reference)
    three = gene_tile("fraction:grna[411710_2]", results,
                      barcodes=screen_reference)

    assert {c.gene for c in two.candidates} == {"411710", "411210"}, (
        f"GCGATAGAGCCCCGCCCTGG is in the reference twice, not "
        f"{len(two.candidates)} times: {[c.gene for c in two.candidates]}")
    assert len(three.candidates) == 3


def test_the_gene_the_counts_were_attributed_to_is_named_as_such(
        results, screen_reference):
    """One of the three IS the one the pipeline booked the reads under. Saying
    which is useful; implying it is therefore the right one is not."""
    tile = gene_tile("fraction:grna[411710_2]", results,
                     barcodes=screen_reference)

    reported = [c.gene for c in tile.candidates if c.reported]
    assert reported == ["411710"], reported
    assert tile.candidates[0].gene == "411710", (
        "the reported gene should lead the list, not be buried in it")
    assert "bookkeeping fact, not a result" in tile.ambiguity


def test_an_unshared_guide_is_not_reported_as_ambiguous(results,
                                                        screen_reference):
    """The refusal has to be specific or it is just noise on every tile."""
    tile = gene_tile("fraction:grna[239740_3]", results,
                     barcodes=screen_reference)

    assert not tile.ambiguous
    assert [c.gene for c in tile.candidates] == ["239740"]
    assert tile.ambiguity == ""


def test_each_of_the_three_ambiguous_genes_carries_its_own_annotation(
        results, screen_reference):
    """241310 has a metadata row and 411210 and 411710 do not. The tile shows
    what is known about each rather than one summary for the group."""
    tile = gene_tile("fraction:grna[411710_2]", results,
                     barcodes=screen_reference)
    by_gene = {c.gene: c for c in tile.candidates}

    assert by_gene["241310"].product == "hypothetical protein"
    assert by_gene["411210"].product == ""
    assert any("no annotation row" in note
               for note in by_gene["411210"].notes), by_gene["411210"].notes


# --------------------------------------------------------------------------- #
#  A miss is an answer
# --------------------------------------------------------------------------- #

def test_a_guide_dropped_from_the_shipped_reference_does_not_look_clean(results):
    """The shipped reference lost those eight rows to commit cd5a8920, so
    against it `411710_2` cannot be checked at all. That is a third answer,
    and it must not be silently the same as "unique to one gene"."""
    tile = gene_tile("fraction:grna[411710_2]", results)

    assert not tile.ambiguous
    assert any("not in the gRNA reference" in note
               for note in tile.unresolved), (
        "the shipped reference does not contain 411710_2 and the tile said "
        f"nothing about it: {tile.unresolved}")


def test_a_non_targeting_control_says_it_is_a_control(results):
    """"a panel saying 'no gene record for this guide' reads as an answer"."""
    tile = gene_tile("fraction:grna[000000_22]", results)

    assert tile.kind == "control"
    assert tile.title == "non-targeting control"
    assert any("non-targeting control" in note for note in tile.unresolved)
    assert tile.candidates == (), (
        "the control block is not a gene and must not be given a gene record")


def test_a_control_is_not_given_a_meaningless_gene_level_effect(results):
    """The control block is fitted as if it were one gene, so an effect and a
    sign agreement exist for it arithmetically and mean nothing."""
    tile = gene_tile("fraction:grna[000000_22]", results)

    assert math.isnan(tile.gene_effect), tile.gene_effect
    assert all(g.agrees is None for g in tile.guides)


def test_an_id_from_another_organism_says_what_it_could_not_resolve(results):
    """"a screen that is not Toxoplasma ... the tile should say what it could
    not resolve rather than appearing empty"."""
    tile = gene_tile("fraction:grna[HsCtrl_1]", results)

    assert not tile.resolved
    assert tile.unresolved, "an unrecognised id produced an empty tile"
    assert any("not shaped like a Toxoplasma accession" in note
               for note in tile.unresolved), tile.unresolved


def test_no_toxodb_link_is_offered_for_an_id_that_is_not_toxoplasma(results):
    """A link to a record that does not exist is worse than no link."""
    assert gene_tile("fraction:grna[HsCtrl_1]", results).references == ()
    assert not is_toxoplasma_gene_id("HsCtrl")
    assert is_toxoplasma_gene_id("239740")
    assert is_toxoplasma_gene_id("201180A"), (
        "201180A is a real gene id in this screen — the paralog letter is "
        "part of the id, not a malformed one")


def test_the_intercept_says_it_is_a_covariate_not_a_gene(results):
    """The intercept is a model term the user can click and is not a gene.

    It is also the term whose p of 3e-46 is 3.6x the tallest real hit, so it
    is exactly the dot someone clicks first when a plot goes wrong.
    """
    tile = gene_tile("Intercept", results)

    assert tile.kind == "nuisance"
    assert not tile.resolved
    assert any("covariate" in note for note in tile.unresolved), tile.unresolved


def test_a_gene_with_no_annotation_row_says_so_rather_than_showing_blanks(
        results):
    """TGME49_411710 is not in the bundled metadata: the ME49 genome has no
    counterpart for that GT1 id. Blank fields would read as "not measured"."""
    tile = gene_tile("gene_fraction:gene[411710]", results)

    assert tile.resolved, "the gene id is real even where the annotation is not"
    assert tile.candidates[0].product == ""
    assert any("no annotation row" in note
               for note in tile.candidates[0].notes), tile.candidates[0].notes


def test_an_empty_tile_is_never_produced(results):
    """Every branch says something. This is the invariant the instruction is
    actually about: "an empty panel reads as a bug"."""
    for feature in ("fraction:grna[239740_3]", "gene_fraction:gene[411710]",
                    "fraction:grna[000000_22]", "Intercept",
                    "fraction:grna[HsCtrl_1]", "", "???", None):
        tile = gene_tile(feature, results)
        assert tile.title, f"{feature!r} produced a tile with no title"
        assert tile.to_text().strip(), f"{feature!r} produced an empty tile"
        assert tile.resolved or tile.unresolved, (
            f"{feature!r} resolved nothing and did not say why")


# --------------------------------------------------------------------------- #
#  One rule for guide -> gene, and no network
# --------------------------------------------------------------------------- #

def test_the_tile_uses_the_repos_own_guide_to_gene_rule(results,
                                                        screen_reference):
    """`spacr.hits.gene_of` is the canonical rule. A second copy is how the
    volcano and the hit list start naming different genes for one dot."""
    for feature in ("fraction:grna[239740_3]", "gene_fraction:gene[225160]",
                    "fraction:grna[411710_2]"):
        tile = gene_tile(feature, results, barcodes=screen_reference)
        assert tile.gene == gene_of(feature), (
            f"{feature}: tile says {tile.gene!r}, hits.gene_of says "
            f"{gene_of(feature)!r}")


def test_guide_of_names_the_guide_and_refuses_a_gene_term():
    """The companion to `gene_of`, added beside it so both live in one place."""
    assert guide_of("fraction:grna[233460_1]") == "233460_1"
    assert guide_of("fraction:grna[T.233460_1]") == "233460_1"
    assert guide_of("gene_fraction:gene[233460]") is None, (
        "a gene term names no guide, and returning the gene id from a "
        "function called guide_of is how a gene ends up plotted as a guide")
    assert guide_of("Intercept") is None
    assert guide_of(None) is None
    assert guide_of(float("nan")) is None


def test_gene_of_and_tested_family_still_behave_as_they_did():
    """`guide_of` was added beside them; neither was allowed to change."""
    from spacr.hits import tested_family

    assert gene_of("gene_fraction:gene[233460]") == "233460"
    assert gene_of("fraction:grna[233460_1]") == "233460"
    assert gene_of("Intercept") is None
    assert tested_family(["Intercept", "fraction:grna[233460_1]"]).tolist() == [
        False, True]


def test_nothing_in_the_path_opens_a_socket(results, monkeypatch):
    """"Nothing in the path makes a network call while the user waits."

    The ToxoDB reference is a URL to display, not a fetch. Enforced by making
    a socket impossible rather than by reading the code, because the next
    person to add an annotation source will not read this file.
    """
    def refuse(*args, **kwargs):
        raise AssertionError(
            "gene_tile opened a socket. The external reference is a URL to "
            "show and open on demand, never one to fetch while the user is "
            "waiting for a tile.")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)

    tile = gene_tile("fraction:grna[239740_3]", results)

    assert tile.references, "the tile offered no external record at all"
    assert tile.references[0].url.startswith("https://toxodb.org/toxo/app/")


def test_the_second_click_does_not_rebuild_the_annotation_index(results):
    """Every click was re-reading 8,800 annotation rows into a fresh dict --
    ~90 ms of work per click, which is a plot that feels broken. The index is
    built once per file and reused."""
    from spacr import gene_tile as module

    calls = []
    original = module._annotation_index
    module._INDEX_CACHE.clear()

    def counted(frame):
        calls.append(1)
        return original(frame)

    try:
        module._annotation_index = counted
        for feature in ("fraction:grna[239740_3]", "gene_fraction:gene[225160]",
                        "fraction:grna[411710_2]"):
            module.gene_tile(feature, results)
    finally:
        module._annotation_index = original

    assert len(calls) == 1, (
        f"three clicks on one screen rebuilt the annotation index "
        f"{len(calls)} times")


def test_the_toxodb_link_names_the_accession_the_library_used(
        results, screen_reference):
    """The library writes TGGT1 and the annotation file writes TGME49. Both
    are offered, because a user following the link wants the strain they
    screened and the strain the annotation came from."""
    tile = gene_tile("fraction:grna[239740_3]", results,
                     barcodes=screen_reference)
    urls = [r.url for r in tile.references]

    assert "https://toxodb.org/toxo/app/record/gene/TGGT1_239740" in urls, urls
    assert "https://toxodb.org/toxo/app/record/gene/TGME49_239740" in urls, urls


# --------------------------------------------------------------------------- #
#  The transcription above is faithful to the real files
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.isdir(REAL_SCREEN),
                    reason="the real TSG101 screen is not on this machine")
def test_the_fixtures_match_the_real_screen_on_disk(results):
    """Every number and every barcode row in this file is quoted from the real
    screen. If the transcription drifts, everything above tests fiction."""
    real = pd.read_csv(os.path.join(
        REAL_SCREEN, "results", "plate1_dv", "ols", "list", "results.csv"))
    for row in results.itertuples(index=False):
        match = real[real["feature"] == row.feature]
        assert len(match) == 1, f"{row.feature} is not in the real results"
        assert float(match.iloc[0]["coefficient"]) == pytest.approx(
            row.coefficient), row.feature
        assert float(match.iloc[0]["p_value"]) == pytest.approx(
            row.p_value), row.feature

    reference = pd.read_csv(os.path.join(REAL_SCREEN, "grna_barcodes.csv"))
    live = set(zip(reference["name"], reference["sequence"]))
    for name, sequence in AMBIGUOUS_ROWS:
        assert (name, sequence) in live, (
            f"{name} is no longer in the screen's own reference; the "
            "ambiguity these tests are built on has moved")


@pytest.mark.heavy
@pytest.mark.skipif(not os.path.isdir(REAL_SCREEN),
                    reason="the real TSG101 screen is not on this machine")
def test_every_feature_in_the_real_screen_produces_a_tile():
    """1,213 rows, every one of them clickable. None may raise and none may
    come back empty."""
    real = pd.read_csv(os.path.join(
        REAL_SCREEN, "results", "plate1_dv", "ols", "list", "results.csv"))
    reference = os.path.join(REAL_SCREEN, "grna_barcodes.csv")

    ambiguous = 0
    for feature in real["feature"]:
        tile = gene_tile(feature, real, barcodes=reference)
        assert tile.title, feature
        assert tile.resolved or tile.unresolved, feature
        ambiguous += bool(tile.ambiguous)

    assert ambiguous == 3, (
        "exactly three fitted guides in this screen carry a shared "
        f"protospacer -- 411710_2, _3 and _10 -- and {ambiguous} were flagged")
