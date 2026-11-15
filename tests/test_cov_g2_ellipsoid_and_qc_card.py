"""Ellipsoid gate edges and the image-QC classifier card's report reading."""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import gate_spec as gs  # noqa: E402


def _ellipsoid(**extra):
    values = dict(name="e", x_column="a", y_column="b", z_column="c",
                  x_centre=1.0, y_centre=2.0, z_centre=0.0,
                  x_radius=-2.0, y_radius=2.0, z_radius=2.0)
    values.update(extra)
    return gs._EllipsoidGate(**values)


def test_an_ellipsoid_keeps_radii_positive_and_moves_and_scales():
    gate = _ellipsoid()
    assert gate.kind == gs._ELLIPSOID and gate.x_radius == 2.0
    moved = gate.translated(1, -1)
    assert moved.centre() == (2.0, 1.0)
    grown = gate.scaled(2.0)
    assert (grown.x_radius, grown.z_radius) == (4.0, 4.0)
    assert grown.centre() == gate.centre()
    anchored = gate.scaled(2.0, about=(0.0, 0.0))
    assert anchored.centre() == (2.0, 4.0)


def test_a_flat_ellipsoid_holds_no_rows():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [0.0]})
    assert _ellipsoid().mask(frame).tolist() == [True]
    assert _ellipsoid(z_radius=0).mask(frame).tolist() == [False]
