"""The lines a conversion summary prints, and the ones it leaves out.

This text is what a user reads after a conversion, and every line in it names a
file they can go and look at. A line naming an empty path is worse than no
line: they go looking for it.

The uncovered arcs are the absences -- no map file, no checkpoint -- which is
what a plain conversion with neither resume nor a written map produces.
"""
from __future__ import annotations

import pytest


def _result(**changes):
    from spacr.convert import ConversionPlan, ConversionResult

    plan = ConversionPlan(mappings=(), errors=(), warnings=(), notes=(),
                          sources=(), unreadable=(), well_map={},
                          plate_map={}, channel_map={})
    fields = dict(plan=plan, dst="/data/plate1", written=(), existing=(),
                  failed=(), skipped=(), resumed_fields=())
    fields.update(changes)
    return ConversionResult(**fields)


def test_a_conversion_with_no_map_and_no_checkpoint_names_neither():
    """Arcs 933 -> 935 and 939 -> 941.

    A plain conversion writes no map file and keeps no checkpoint. Printing
    "Map file: " with nothing after it sends the user looking for a path that
    does not exist.
    """
    text = _result().summary()

    assert "Map file:" not in text
    assert "Checkpoint:" not in text
    assert "Resumed" not in text


def test_a_written_map_is_named():
    """The taken side of the first."""
    text = _result(map_path="/data/plate1/conversion_map.csv").summary()

    assert "Map file: /data/plate1/conversion_map.csv" in text


def test_a_checkpoint_that_was_not_resumed_from_is_named_as_a_checkpoint():
    """The ``elif``: kept for next time, but nothing was resumed this time.

    The two states are different and the user acts on them differently -- one
    says work was skipped, the other says work can be skipped later.
    """
    text = _result(checkpoint_path="/data/plate1/convert.ckpt").summary()

    assert "Checkpoint: /data/plate1/convert.ckpt" in text
    assert "Resumed" not in text


def test_a_resumed_conversion_says_how_much_it_skipped():
    """The taken side of the ``if``, which suppresses the plain checkpoint line.

    Both lines would name the same file twice and say different things about
    it, so only the more specific one is printed.
    """
    text = _result(checkpoint_path="/data/plate1/convert.ckpt",
                   resumed_fields=("f1", "f2")).summary()

    assert "Resumed 2 completed field(s)" in text
    assert "/data/plate1/convert.ckpt" in text
    assert "Checkpoint: " not in text


def test_skipped_sources_are_listed_with_their_reasons():
    """The block above, so the omissions are visibly not "nothing is printed"."""
    text = _result(skipped=(("/data/raw/a.tif", "unreadable header"),)).summary()

    assert "Skipped 1 source(s):" in text
    assert "a.tif: unreadable header" in text
