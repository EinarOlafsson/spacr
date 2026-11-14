"""What a clicked point resolves to when the inputs are thin or contradictory.

The gene tile is the only place spaCR says out loud what a dot on a volcano IS,
so every gap has to read as a gap rather than as a fact. These drive the paths
where something is missing -- no reference, no annotation column, an unparseable
guide name -- and the one where two genes carry the same protospacer, which is
the case a naive resolver reports as a clean single hit.
"""
from __future__ import annotations

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


def test_localisation_without_both_of_its_columns_locates_nothing():
    """A LOPIT table missing the gene key or the location column is skipped.

    A localisation shown against the wrong gene is a claim about cell biology
    that nothing in the file supports.
    """
    assert GT._localisation_index(
        pd.DataFrame({'gene_nr': ['239740']})) == {}
    assert GT._localisation_index(
        pd.DataFrame({'tagm_location': ['nucleus']})) == {}
