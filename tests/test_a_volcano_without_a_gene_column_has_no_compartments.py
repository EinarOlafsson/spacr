"""Compartment colouring on a results table that cannot name a gene.

Colouring a volcano by LOPIT compartment needs a column the reference table
can be looked up in. The regression's own output carries one; a table a user
opened in the explorer may not -- a term/coefficient/p-value frame names
nothing the localisation table has ever heard of.

A volcano is still a volcano without compartment colouring, so the answer is
"no compartments", not an exception and not a menu of the 27 that would
colour nothing. A menu entry that colours nothing is indistinguishable from
a broken one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.localisation import table
from spacr.volcano_style import (VolcanoStyle, localizations_present,
                                 point_localizations)


def _nameless():
    """A regression output whose identifier column is called something else."""
    return pd.DataFrame({
        "term": ["fraction:x[1]", "fraction:x[2]", "fraction:x[3]"],
        "coefficient": [0.8, -0.4, 0.1],
        "p_value": [0.001, 0.2, 0.7],
    })


def _named(n=6):
    """The same shape, keyed by the gene numbers the bundled table holds."""
    genes = [gene for gene, _place in list(table().items())[:n]]
    return pd.DataFrame({
        "gene": genes,
        "coefficient": [0.5] * n,
        "p_value": [0.01] * n,
    })


def test_a_table_that_names_no_gene_gets_no_localisations():
    assert point_localizations(_nameless(), VolcanoStyle()) is None


def test_no_localisations_means_no_compartment_menu():
    """An empty menu, not 27 entries that would each colour nothing."""
    assert localizations_present(_nameless(), VolcanoStyle()) == []


def test_a_named_column_does_produce_compartments():
    """The control: the empty answer must mean 'no gene column', not 'broken'."""
    frame = _named()

    places = point_localizations(frame, VolcanoStyle())

    assert places is not None
    assert len(places) == len(frame)
    assert all(str(place) for place in places)
    assert localizations_present(frame, VolcanoStyle(), minimum=1)


def test_a_named_column_the_style_points_at_is_used():
    """`localization_column` is tried before the conventional names."""
    frame = _named().rename(columns={"gene": "target"})
    style = VolcanoStyle(localization_column="target")

    assert point_localizations(frame, style) is not None
    assert point_localizations(frame, VolcanoStyle()) is None
