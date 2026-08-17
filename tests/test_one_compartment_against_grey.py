"""Colour the volcano by localisation -- ONE compartment at a time.

Asked for on 2026-08-16: "make a new version of your volcanoplot where the
data are colord by localization (LOPIT or at least make it an option when
right clicking on the graph)".

THE HOUSE RULE CAPS IT. "Everything is grey except what the sentence is
about", and the bundled TAGM/LOPIT table names 26 real compartments. A
26-colour volcano is exactly what that rule exists to forbid -- no reader can
hold 26 hues apart, and the two that matter are never adjacent in the legend.
It is also, measured, the slow version: a 27-entry legend cost 40 ms of a
49 ms redraw.

So one compartment against grey, chosen from the ones this screen actually
has.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import localisation


def _screen(n=400, seed=0):
    """A coefficient table whose genes are really in the LOPIT table."""
    genes = list(localisation.table())[:n]
    if not genes:
        pytest.skip("the bundled LOPIT table is not present")
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature": [f"gene_fraction:gene[{g}]" for g in genes],
        "coefficient": rng.normal(0, .5, len(genes)),
        "p_value": rng.uniform(size=len(genes)),
    })


# --------------------------------------------------------------------------- #
#  The table
# --------------------------------------------------------------------------- #

def test_the_table_loads():
    assert len(localisation.table()) > 1000


def test_unknown_is_not_offered_as_a_place():
    """"unknown" is the table's own way of saying it could not place the
    protein. Offering it is offering to colour by "we do not know"."""
    places = set(localisation.table().values())

    assert "unknown" not in places
    assert not any(place.strip().lower() in localisation.NOT_A_COMPARTMENT
                   for place in places)


def test_the_malformed_rows_are_dropped():
    """The CSV has rows whose location cell holds a number. Left in, they
    appear on the menu as compartments called "49.59"."""
    for place in localisation.table().values():
        with pytest.raises(ValueError):
            float(place)


def test_a_missing_table_is_not_an_exception(monkeypatch):
    """A screen of another organism has no reason to carry this file, and a
    volcano is still a volcano without compartment colouring."""
    localisation.table.cache_clear()
    monkeypatch.setattr("spacr.gene_tile.BUNDLED_LOCALISATION",
                        "/nonexistent/lopit.csv")
    try:
        assert localisation.table() == {}
    finally:
        localisation.table.cache_clear()


# --------------------------------------------------------------------------- #
#  The join
# --------------------------------------------------------------------------- #

def test_a_gene_row_resolves():
    frame = _screen()
    places = localisation.of(frame)

    assert (places != "").sum() > len(frame) * 0.8


def test_a_guide_row_resolves_to_its_genes_compartment():
    """A guide's gene is where its protein lives too, and a screen's table is
    mostly guide rows -- so a join that only handled the gene rows would
    colour a handful of dots and look like the annotation was missing."""
    gene, place = next(iter(localisation.table().items()))
    frame = pd.DataFrame({"feature": [f"fraction:grna[{gene}_2]"]})

    assert list(localisation.of(frame)) == [place]


def test_a_nuisance_term_resolves_to_nothing():
    frame = pd.DataFrame({"feature": ["Intercept", "rowID[T.r02]"]})

    assert list(localisation.of(frame)) == ["", ""]


def test_a_float_shaped_gene_number_still_joins():
    """`gene_nr` reads as a float when the column has a blank in it, so
    244480 arrives as "244480.0" and joins to nothing at all."""
    localisation.table.cache_clear()
    try:
        table = localisation.table()
        gene = next(iter(table))
        assert not gene.endswith(".0"), gene
    finally:
        localisation.table.cache_clear()


# --------------------------------------------------------------------------- #
#  What is offered
# --------------------------------------------------------------------------- #

def test_only_the_compartments_this_screen_has():
    """A menu offering choices that would colour nothing is a menu where a
    choice that colours nothing is indistinguishable from a broken one."""
    frame = _screen()
    offered = localisation.present(frame)
    places = set(localisation.of(frame))

    assert offered
    assert set(offered) <= places


def test_they_are_ordered_commonest_first():
    frame = _screen()
    counts = localisation.of(frame).value_counts()

    offered = localisation.present(frame)
    sizes = [counts[name] for name in offered]
    assert sizes == sorted(sizes, reverse=True), offered


def test_a_compartment_with_almost_nothing_in_it_is_not_offered():
    """Colouring three dots out of twelve hundred produces a figure whose
    sentence rests on three points, and the eye reads the highlight as a
    finding rather than as three points."""
    frame = _screen()
    counts = localisation.of(frame).value_counts()

    for name in localisation.present(frame):
        assert counts[name] >= localisation.MIN_GENES


def test_a_screen_with_no_annotated_genes_offers_nothing():
    frame = pd.DataFrame({"feature": ["gene_fraction:gene[999999999]"] * 50})

    assert localisation.present(frame) == []


# --------------------------------------------------------------------------- #
#  The mask
# --------------------------------------------------------------------------- #

def test_no_compartment_selects_nothing():
    """None is the menu's "none" state and must pass straight through without
    the caller branching on it."""
    frame = _screen()

    assert localisation.mask(frame, None).sum() == 0
    assert localisation.mask(frame, "").sum() == 0


def test_the_mask_picks_only_that_compartment():
    frame = _screen()
    name = localisation.present(frame)[0]
    picked = localisation.mask(frame, name)

    assert picked.sum() > 0
    assert set(localisation.of(frame)[picked]) == {name}


# --------------------------------------------------------------------------- #
#  The drawn panel
# --------------------------------------------------------------------------- #

def test_the_panel_colours_one_compartment():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from spacr.figures import build_panel

    frame = _screen()
    name = localisation.present(frame)[0]
    figure, panel = build_panel("volcano", frame, compartment=name)
    try:
        assert name in panel.caption
        assert "TAGM/LOPIT" in panel.caption
    finally:
        matplotlib.pyplot.close(figure)


def test_the_legend_has_two_entries_not_twenty_six():
    """THE MEASURED REASON, not only the style rule: the 27-entry legend cost
    40 ms of a 49 ms redraw."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from spacr.figures import build_panel

    frame = _screen()
    name = localisation.present(frame)[0]
    figure, _panel = build_panel("volcano", frame, compartment=name)
    try:
        texts = [t.get_text() for t in figure.axes[0].texts]
        coloured = [t for t in texts if name in t or "elsewhere" in t]
        assert len(coloured) == 2, texts
    finally:
        matplotlib.pyplot.close(figure)


def test_a_compartment_nothing_matches_says_so_rather_than_drawing_grey():
    """A volcano that is entirely grey and a volcano whose compartment
    matched nothing look identical."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from spacr.figures import build_panel

    figure, panel = build_panel("volcano", _screen(),
                                compartment="not a place")
    try:
        assert "No coefficient" in panel.caption
    finally:
        matplotlib.pyplot.close(figure)


def test_compartment_colouring_replaces_up_down_rather_than_joining_it():
    """A reader cannot tell whether a coloured dot is coloured for being
    called or for being a rhoptry, so a volcano carrying both has no
    sentence."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from spacr.figures import build_panel

    frame = _screen()
    name = localisation.present(frame)[0]
    figure, _panel = build_panel("volcano", frame, compartment=name)
    try:
        texts = [t.get_text() for t in figure.axes[0].texts]
        assert not any("up" == t or t.endswith(" up") for t in texts), texts
    finally:
        matplotlib.pyplot.close(figure)
