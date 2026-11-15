"""The pixel size a TIFF records, read with tifffile alone."""

import pytest

from spacr import tiff_io


@pytest.mark.parametrize("unit, nanometres", [
    ("nm", 1.0), (" µm ", 1000.0), ("um", 1000.0), ("Å", 0.1),
    ("mm", 1e6), (None, None), ("", None), ("furlong", None)])
def test_a_length_unit_is_nanometres_or_nothing(unit, nanometres):
    assert tiff_io._nm_per(unit) == nanometres
