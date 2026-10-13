"""What a clicked point resolves to when the inputs are thin or contradictory.

The gene tile is the only place spaCR says out loud what a dot on a volcano IS,
so every gap has to read as a gap rather than as a fact. These drive the paths
where something is missing -- no reference, no annotation column, an unparseable
guide name -- and the one where two genes carry the same protospacer, which is
the case a naive resolver reports as a clean single hit.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from spacr import gene_tile as GT
from spacr.gene_tile import (GeneCandidate, GeneTile, GuideRow,
                             is_toxoplasma_gene_id, uniprot_reference)


SEQUENCE = 'ACGTACGTACGTACGTACGT'


# ---------------------------------------------------------------------------
# cells and ids
# ---------------------------------------------------------------------------

def test_a_missing_cell_is_an_empty_string_not_the_word_nan():
    """A float NaN reads as a gap, so no field shows the string "nan".

    Every annotation column arrives through this helper; a NaN rendered
    verbatim would put "nan" under "product" and read as a product name.
    """
    assert GT._text(float('nan')) == ''
    assert GT._text(None) == ''
    assert GT._text('  none  ') == 'none'


def test_only_toxoplasma_shaped_ids_are_offered_a_toxodb_link():
    """A blank or foreign id is rejected, an accession or bare number accepted.

    A ToxoDB link built for a gene from another organism opens a page for a
    different gene entirely, which is worse than offering no link at all.
    """
    assert is_toxoplasma_gene_id('') is False
    assert is_toxoplasma_gene_id(None) is False
    assert is_toxoplasma_gene_id('TGGT1_239740') is True
    assert is_toxoplasma_gene_id('239740') is True
    assert is_toxoplasma_gene_id('ENSG00000141510') is False


def test_an_annotation_that_is_not_a_row_falls_back_to_a_search_link():
    """A non-mapping annotation yields a search link, not an AttributeError.

    Callers pass whatever their table gave them; a scalar or a list there must
    cost the record link and nothing else.
    """
    label, url, is_record = uniprot_reference('TGGT1_999999',
                                              annotation=['not a row'])
    assert is_record is False
    assert label.startswith('UniProt search:')
    assert 'TGGT1_999999' in url


# ---------------------------------------------------------------------------
# loading and caching the side tables
# ---------------------------------------------------------------------------

def test_a_frame_handed_in_directly_is_used_without_a_cache_token(tmp_path):
    """An in-memory frame has no file to key a cache on, so it gets no token.

    Caching it under a stale token would serve one caller's frame to the next
    caller who passed a different one.
    """
    frame = pd.DataFrame({'name': ['TGGT1_239740_3'], 'sequence': [SEQUENCE]})
    loaded, token = GT._load(frame, str(tmp_path / 'unused.csv'))
    assert loaded is frame
    assert token is None


def test_a_source_that_is_not_a_file_is_skipped_rather_than_raising(tmp_path):
    """A path that does not exist means "no such table", not a crash.

    A settings file naming a reference that was moved must still open the tile
    with the identity it can resolve.
    """
    assert GT._load(str(tmp_path / 'missing.csv'), 'default.csv') == (None,
                                                                      None)


def test_an_unreadable_table_is_skipped_and_leaves_no_cache_token(tmp_path):
    """A file that pandas refuses to parse is treated as absent.

    A truncated or zero-byte reference is a real thing to find on disk; letting
    the parser error escape would make the whole tile unopenable.
    """
    empty = tmp_path / 'empty.csv'
    empty.write_text('')
    assert GT._load(str(empty), 'default.csv') == (None, None)


def test_the_index_cache_is_emptied_rather_than_grown_without_limit():
    """Past its ceiling the cache is cleared, so it cannot grow unbounded.

    The cache holds whole parsed reference indexes; a session that opens tiles
    against many reference files would otherwise hold every one of them for the
    life of the process.
    """
    saved = dict(GT._INDEX_CACHE)
    GT._INDEX_CACHE.clear()
    try:
        for i in range(GT._INDEX_CACHE_MAX + 1):
            GT._indexed('probe', lambda frame: frame, i, f'token-{i}')
        assert len(GT._INDEX_CACHE) == 1
        assert ('probe', f'token-{GT._INDEX_CACHE_MAX}') in GT._INDEX_CACHE
    finally:
        GT._INDEX_CACHE.clear()
        GT._INDEX_CACHE.update(saved)


# ---------------------------------------------------------------------------
# the record's own wording
# ---------------------------------------------------------------------------

def test_a_guide_with_no_effect_has_no_direction_to_report():
    """A NaN coefficient is directionless, and an exactly zero one is flat.

    The direction drives an arrow in the tile; calling a missing effect "down"
    would put a claim on the screen that the fit never made.
    """
    assert GuideRow(guide='239740_3', feature='f',
                    effect=float('nan')).direction == ''
    assert GuideRow(guide='239740_3', feature='f',
                    effect=0.0).direction == 'flat'
    assert GuideRow(guide='239740_3', feature='f',
                    effect=-0.2).direction == 'down'


def test_an_ambiguous_tile_says_the_effect_cannot_be_assigned():
    """The subtitle of a multi-gene guide names the count and the problem.

    This is the one line that stops a shared protospacer being read as a clean
    single-gene hit.
    """
    tile = GeneTile(feature='fraction:grna[239740_3]', kind='guide',
                    candidates=(GeneCandidate(gene='239740'),
                                GeneCandidate(gene='411710')),
                    ambiguous=True)
    assert 'cannot be assigned' in tile.subtitle
    assert '2 genes' in tile.subtitle


def test_a_gene_with_no_product_still_shows_which_accession_it_is():
    """With no product description the accession carries the subtitle.

    An empty subtitle under a bare gene number leaves nothing to check the
    identity against.
    """
    tile = GeneTile(feature='239740', kind='gene',
                    candidates=(GeneCandidate(gene='239740',
                                              accession='TGGT1_239740'),))
    assert tile.subtitle == 'no product description · TGGT1_239740'


def test_a_tile_that_resolved_nothing_has_an_empty_subtitle():
    """No candidate and no note leaves the subtitle blank rather than invented.

    The title already carries the clicked string; repeating it as a subtitle
    would look like a resolved identity.
    """
    tile = GeneTile(feature='mystery', kind='unresolved')
    assert tile.subtitle == ''
    assert tile.resolved is False


# ---------------------------------------------------------------------------
# parsing a clicked string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('feature,expected', [
    ('C(row_name)[T.B]', ('nuisance', '', '')),
    ('fraction:grna[]', ('unresolved', '', '')),
    ('TGGT1_239740_3', ('guide', '239740', '239740_3')),
    ('TGGT1_239740', ('gene', '239740', '')),
    ('239740_3', ('guide', '239740', '239740_3')),
    ('239740', ('gene', '239740', '')),
    ('not a term at all', ('unresolved', '', '')),
])
def test_what_a_clicked_string_is_taken_to_be(feature, expected):
    """Each accepted spelling resolves to the same kind, gene and guide.

    The tile and the hit list have to agree about which gene a dot is; the
    bare-id spellings are what a user pastes in when the click came from
    somewhere other than the table.
    """
    assert GT._parse(feature) == expected


# ---------------------------------------------------------------------------
# the side indexes
# ---------------------------------------------------------------------------

def test_a_reference_without_the_expected_columns_indexes_nothing():
    """A CSV that is not a gRNA reference yields empty maps, not a KeyError.

    Users point this at whatever CSV is nearest; the tile must then say the
    protospacer could not be checked rather than fail to open.
    """
    wrong = pd.DataFrame({'guide': ['239740_3'], 'seq': [SEQUENCE]})
    assert GT._reference_index(wrong) == ({}, {}, {})


def test_reference_rows_with_no_name_are_skipped_and_odd_names_kept_whole():
    """A blank name is dropped; a name that is not strain_gene_number is kept.

    Custom libraries name guides their own way. Splitting such a name as though
    it carried a strain prefix would file the guide under a gene that does not
    exist.
    """
    frame = pd.DataFrame({
        'name': ['', 'customlib_7'],
        'sequence': [SEQUENCE, SEQUENCE],
    })
    by_guide, by_sequence, strain = GT._reference_index(frame)
    assert by_guide == {'customlib_7': ('', SEQUENCE)}
    assert by_sequence == {SEQUENCE: [('customlib', '')]}
    assert strain == {}


def test_annotation_without_a_gene_column_annotates_nothing():
    """A metadata table with no id column yields an empty index.

    Joining on a guessed column would attach one gene's product description to
    every other gene in the export.
    """
    assert GT._annotation_index(
        pd.DataFrame({'product': ['a kinase']})) == {}


def test_localisation_without_both_of_its_columns_locates_nothing():
    """A LOPIT table missing the gene key or the location column is skipped.

    A localisation shown against the wrong gene is a claim about cell biology
    that nothing in the file supports.
    """
    assert GT._localisation_index(
        pd.DataFrame({'gene_nr': ['239740']})) == {}
    assert GT._localisation_index(
        pd.DataFrame({'tagm_location': ['nucleus']})) == {}


# ---------------------------------------------------------------------------
# the whole tile
# ---------------------------------------------------------------------------

def test_without_a_reference_the_tile_says_uniqueness_was_never_checked():
    """Skipping the gRNA reference is recorded, not passed over in silence.

    Whether a protospacer is unique to one gene is exactly what the reference
    answers; a tile built without one must not read like a tile that checked.
    """
    tile = GT.gene_tile('fraction:grna[239740_3]', None, barcodes=None,
                        metadata=None, localisation=None)
    assert tile.kind == 'guide'
    assert any('no gRNA reference was supplied' in note
               for note in tile.unresolved)


def test_a_protospacer_in_another_gene_puts_both_genes_on_the_tile():
    """The reported gene leads, the gene the reference names follows.

    The reference here files the guide under a longer gene id than the model
    term does, so the gene the counts were attributed to is not in the
    protospacer's own list; dropping it would lose the gene the number is
    actually about.
    """
    barcodes = pd.DataFrame({'name': ['TGGT1_AB_239740_3'],
                             'sequence': [SEQUENCE]})
    tile = GT.gene_tile('fraction:grna[AB_239740_3]', None, barcodes=barcodes,
                        metadata=None, localisation=None)

    assert tile.ambiguous is True
    assert [c.gene for c in tile.candidates] == ['AB', 'AB_239740']
    assert tile.candidates[0].reported is True
    assert tile.protospacer == SEQUENCE

    text = tile.to_text()
    # to_text upper-cases each section heading.
    assert ('IDENTITY — TGGT1_AB_239740 (THE GENE THE COUNTS WERE '
            'ATTRIBUTED TO)') in text

    html = tile.to_html()
    assert 'Ambiguous mapping' in html
    assert '<h2' in html
