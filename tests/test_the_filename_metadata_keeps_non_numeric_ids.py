"""Filename ids that do not start with a digit, and the padding rule beside them.

``_extract_filename_metadata`` un-pads numeric ids so ``001`` and ``1`` key the
same well, and the comment above it records what that guard is FOR: without
``_int_or_token`` every unreadable well used to collapse onto well ``0``. The
``[0].isdigit()`` test is the other half of the same protection -- an id that is
not a number at all must be kept as its own token rather than coerced -- and
the non-numeric side of it had never run for timeID or sliceID.

Timepoints named ``T0``/``Tpre`` and slices named ``Z01``/``top`` are ordinary
in exported microscope filenames, so this is a real vocabulary rather than an
edge case.
"""
from __future__ import annotations

import re

import pytest


REGEX = re.compile(
    r"(?P<plateID>[^_]+)_(?P<wellID>[^_]+)_(?P<fieldID>[^_]+)"
    r"_(?P<chanID>[^_]+)_(?P<timeID>[^_]+)_(?P<sliceID>[^_.]+)\.tif")


def _keys(filenames, src="/data/plate1"):
    from spacr.utils import _extract_filename_metadata

    return _extract_filename_metadata(filenames, src, REGEX)


def test_a_non_numeric_timepoint_and_slice_are_kept_as_written():
    """Arcs 2081 -> 2086 and 2088 -> 2093: neither is coerced.

    ``Tpre`` is not a number, and turning it into one is impossible -- the
    guard is what keeps it as its own key instead. Two timepoints that both
    failed to parse would otherwise merge into one.
    """
    keys = _keys(["p1_A01_F001_C1_Tpre_top.tif",
                  "p1_A01_F001_C1_Tpost_top.tif"])

    flat = " ".join(str(k) for k in keys)
    assert "Tpre" in flat and "Tpost" in flat
    assert len(keys) == 2, "the two timepoints must not collapse into one key"


def test_a_numeric_timepoint_and_slice_are_un_padded():
    """The taken side, which is why the guard is conditional.

    ``T001`` and ``T1`` are the same timepoint written two ways, and the whole
    point of the un-padding is that they key together.
    """
    keys = _keys(["p1_A01_F001_C1_001_01.tif",
                  "p1_A01_F001_C1_1_1.tif"])

    assert len(keys) == 1, "zero-padded and bare ids must key the same"


def test_a_non_numeric_well_or_field_is_also_kept():
    """The same rule on the ids that had already been covered, as a contrast."""
    keys = _keys(["p1_A01_edge_C1_1_1.tif"])

    flat = " ".join(str(k) for k in keys)
    assert "A01" in flat
    assert "edge" in flat


def test_a_filename_that_does_not_match_is_skipped():
    """The ``if match:`` above everything, so the tests above are deliberate."""
    keys = _keys(["not_a_recognised_name.txt",
                  "p1_A01_F001_C1_1_1.tif"])

    assert len(keys) == 1
