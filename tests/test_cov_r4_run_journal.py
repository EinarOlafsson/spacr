"""The journal's inventory root and value comparison at their platform edges."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from spacr import run_journal as RJ


def test_an_output_at_the_top_of_a_drive_does_not_inventory_the_whole_drive():
    """`dst="/outputs"` before it exists has the filesystem root as its parent,
    and walking that inventoried the whole disk -- on Windows it then died on
    the first protected file under C:\\."""
    top = Path(Path.cwd().anchor)
    assert RJ.Run._inventory_root("dst", top / "spacr-not-made-yet", True) is None
    assert RJ.Run._inventory_root("dst", top / "a" / "b", True) == top / "a"


@pytest.mark.posix_only
def test_only_the_filesystem_root_stops_being_an_inventory_root(tmp_path):
    """POSIX: "/" is refused as an output parent; real output folders still walk."""
    assert RJ.Run._inventory_root("dst", Path("/spacr-not-made-yet"), True) is None
    assert RJ.Run._inventory_root("dst", Path("/"), True) == Path("/")
    existing = tmp_path / "outputs"
    existing.mkdir()
    assert RJ.Run._inventory_root("dst", existing, True) == existing
    assert RJ.Run._inventory_root("dst", existing / "not-yet", True) == existing
    assert RJ.Run._inventory_root("dst", tmp_path / "new-out", False) == tmp_path
    assert RJ.Run._inventory_root("src", tmp_path / "input.tif", False) is None
    assert RJ.Run._inventory_root("dst", Path("/tmp/spacr-new"), True) == Path("/tmp")


def test_values_equal_compares_paths_as_before_off_windows():
    assert RJ.values_equal(Path("/tmp/x"), "/tmp/x")
    assert RJ.values_equal("a", "a") and not RJ.values_equal("a", "b")
    if os.name != "nt":
        assert RJ._same_windows_path(Path("/tmp/x"), "/tmp/x") is False
