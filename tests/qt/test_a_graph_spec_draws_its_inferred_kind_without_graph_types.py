"""A chart spec draws its inferred kind when the graph-type table is missing.

Pinned behaviour of :class:`spacr.qt.widgets.graph_spec.GraphSpec`: with
:mod:`spacr.graph_types` unimportable, the Default Graph Type preference
cannot be consulted, so two continuous columns are drawn as the inferred
scatter and no fallback note is added to the caption.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.graph_spec import (  # noqa: E402
    CONTINUOUS,
    SCATTER,
    GraphSpec,
)

pytestmark = pytest.mark.qt


def test_two_continuous_columns_draw_the_inferred_scatter(monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.graph_types", None)
    spec = GraphSpec(x="cell_area", y="nucleus_area")
    kinds = {"cell_area": CONTINUOUS, "nucleus_area": CONTINUOUS}

    assert spec.resolved_kind(kinds) == SCATTER
    assert spec._kind_note(kinds) == ""
