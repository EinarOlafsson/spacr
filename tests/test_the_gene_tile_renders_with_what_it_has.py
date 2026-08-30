"""A gene tile rendered from a partial result, in text and in HTML.

The tile is what the user reads after clicking a dot on a volcano, and every
field on it is optional -- an unresolved feature has no gene, a permutation run
has no p-value, a single-guide gene has no agreement. Each of those must render
as an absence rather than as "nan" or an empty heading, because the tile is
prose the user takes at face value.
"""
from __future__ import annotations

import math

import pytest


def _tile(**changes):
    from spacr.gene_tile import GeneTile

    fields = {"feature": "grna[TGGT1_231640_1]"}
    fields.update(changes)
    return GeneTile(**fields)


def test_a_tile_with_no_subtitle_renders_without_an_empty_line():
    """Arcs 829 -> 831 and 847 -> 850, in both renderers.

    An unresolved feature has no subtitle. Emitting one anyway gives a blank
    line in the text tile and an empty grey paragraph in the HTML -- which
    reads as a value that failed to load rather than one that does not exist.
    """
    tile = _tile()          # unresolved: no candidates, so no subtitle

    text = tile.to_text()
    html = tile.to_html()

    assert text.splitlines()[0] == tile.title
    assert "<p style='margin-top:0;color:#999'></p>" not in html


def test_a_tile_with_a_subtitle_shows_it_in_both_renderers():
    """The taken sides, so the omission above is visibly a decision.

    A tile whose protospacer matches several genes gets its subtitle from the
    ambiguity, which is the case the reader most needs the line for.
    """
    from spacr.gene_tile import GeneCandidate

    tile = _tile(ambiguous=True,
                 candidates=(GeneCandidate(gene="TGGT1_231640"),
                             GeneCandidate(gene="TGGT1_231650")))

    assert tile.subtitle
    assert tile.subtitle in tile.to_text()
    assert tile.subtitle.split(" —")[0] in tile.to_html()


def test_a_guide_with_no_p_value_shows_its_effect_and_no_p():
    """Arc 788 -> 790.

    A permutation run reports no per-guide p-value, and a mixed model gives a
    guide a shrunken BLUP with none either. Printing "p nan" beside a real
    effect would read as a computed non-significance.
    """
    from spacr.gene_tile import GuideRow

    tile = _tile(guides=(GuideRow(guide="g1", effect=0.5,
                                  p_value=float("nan")),))

    text = tile.to_text()

    assert "+0.5" in text
    assert "nan" not in text.lower()
    assert " p " not in text


def test_a_guide_with_a_p_value_shows_it():
    """The taken side."""
    from spacr.gene_tile import GuideRow

    tile = _tile(guides=(GuideRow(guide="g1", effect=0.5, p_value=0.001),))

    assert "p 0.001" in tile.to_text()


def test_a_guide_with_no_effect_shows_a_dash_rather_than_nan():
    """The same rule on the effect itself, which shares the line."""
    from spacr.gene_tile import GuideRow

    tile = _tile(guides=(GuideRow(guide="g1", effect=float("nan"),
                                  p_value=float("nan")),))

    text = tile.to_text()

    assert "—" in text
    assert "nan" not in text.lower()


def test_a_tile_with_nothing_at_all_still_renders_its_title():
    """The minimum: an unresolved feature is still a tile.

    Returning nothing here would leave the user clicking a dot and seeing an
    empty panel, with no way to tell that from a panel that failed to open.
    """
    tile = _tile()

    assert tile.to_text().strip()
    assert "<h2" in tile.to_html()
