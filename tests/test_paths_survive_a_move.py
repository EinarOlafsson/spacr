"""A run folder moved to another machine still finds its own pixels.

Instruction 155 F. The structure a spaCR run writes is::

    <root>/measurements/measurements.db
    <root>/data/                <- the exported crop PNGs
    <root>/merged/              <- the arrays a crop is cut from

so the ROOT is derivable from the database and both anchor folders live under
it. The logic that re-anchors a recorded path existed in two places with two
strategies, and MEASURED on 2026-08-18 it held for the case asked about and
failed two others -- one of them silently:

    OK     same-OS move
    FAIL   a Windows path read on Linux: ``split('/data/')`` cannot match
           ``\\data\\`` and ``os.path.basename`` returns the whole string
    WRONG  ``/old/data/exp1/data/plate1/a.png`` -> ``<root>/data/exp1``,
           because ``split`` took the FIRST ``/data/`` -- a path that has
           lost the file name and now names a DIRECTORY
    FAIL   no ``data/`` component at all, returned unchanged and unreported

The third is the one these tests exist for: it did not raise, it produced a
plausible path, and it failed much later as a missing file with no context.
The anchor is now found from the RIGHT.
"""
from __future__ import annotations

import os
import shutil
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr.crops import (
    ALREADY_ANCHORED, NO_ANCHOR, PATH_ANCHORS, REANCHORED, MergedCropSource,
    PngCropSource, basename_any, normalise_separators, path_components,
    path_is_under, reanchor_frame, reanchor_path,
)
from spacr.utils import correct_paths


# --------------------------------------------------------------------------- #
#  The four measured cases, through the real correct_paths
# --------------------------------------------------------------------------- #

def test_the_same_os_move_still_works():
    """The case that already worked, held so the fix cannot break it."""
    assert correct_paths(["/old/home/exp1/data/plate1/a.png"],
                         "/mnt/newdisk/exp1") == [
        "/mnt/newdisk/exp1/data/plate1/a.png"]


def test_a_windows_path_read_on_linux_re_anchors():
    """FAIL before: the separator was hard-coded, so nothing matched."""
    assert correct_paths([r"C:\lab\exp1\data\plate1\a.png"],
                         "/mnt/newdisk/exp1") == [
        "/mnt/newdisk/exp1/data/plate1/a.png"]


def test_an_old_root_containing_its_own_data_folder_re_anchors_to_the_file():
    """THE SILENT CORRUPTION, and the reason this file exists.

    ``split`` took the FIRST ``/data/`` and produced
    ``/mnt/newdisk/exp1/data/exp1`` -- the file name gone and a directory in
    its place. It raised nothing. The anchor is the one nearest the FILE.
    """
    got = correct_paths(["/old/data/exp1/data/plate1/a.png"],
                        "/mnt/newdisk/exp1")
    assert got == ["/mnt/newdisk/exp1/data/plate1/a.png"]
    assert got != ["/mnt/newdisk/exp1/data/exp1"]
    assert os.path.basename(got[0]) == "a.png"


def test_a_path_with_no_anchor_is_left_alone_but_counted_and_named(capsys):
    """A silent pass-through is how the case above stayed invisible."""
    got = correct_paths(["/old/home/exp1/data/p1/a.png",
                         "/old/home/exp1/crops/p1/b.png"],
                        "/mnt/newdisk/exp1")
    assert got[1] == "/old/home/exp1/crops/p1/b.png"     # still untouched
    said = capsys.readouterr().out
    assert "1 of 2 recorded paths could not be re-anchored" in said
    assert "/old/home/exp1/crops/p1/b.png" in said       # ONE of them named


def test_a_root_that_is_a_substring_of_the_old_path_is_not_already_anchored():
    """``base_path not in path`` is a substring test standing in for a
    question about folders, and it answers wrongly."""
    got = correct_paths(["/backup/mnt/new/data/p1/a.png"], "/mnt/new")
    assert got == ["/mnt/new/data/p1/a.png"]


def test_a_sibling_root_with_a_shared_prefix_is_not_mistaken_for_the_root():
    assert not path_is_under("/mnt/newer/data/a.png", "/mnt/new")
    assert path_is_under("/mnt/new/data/a.png", "/mnt/new")


# --------------------------------------------------------------------------- #
#  The primitives
# --------------------------------------------------------------------------- #

def test_the_outcome_says_which_of_the_three_things_happened():
    assert reanchor_path("/old/exp/data/p1/a.png", "/new/exp")[1] == REANCHORED
    assert reanchor_path("/new/exp/data/p1/a.png", "/new/exp")[1] == ALREADY_ANCHORED
    assert reanchor_path("/old/exp/crops/a.png", "/new/exp")[1] == NO_ANCHOR
    assert reanchor_path("", "/new/exp") == ("", NO_ANCHOR)
    assert reanchor_path("/old/exp/data/a.png", "") [1] == NO_ANCHOR


def test_an_anchor_with_nothing_after_it_names_a_folder_and_is_not_used():
    """``.../data`` is a directory; re-anchoring it would invent a file."""
    assert reanchor_path("/old/exp/data", "/new/exp")[1] == NO_ANCHOR


def test_both_anchors_are_recognised_and_the_rightmost_wins():
    assert reanchor_path("/old/merged/exp/merged/f1.npy", "/new/exp")[0] == \
        "/new/exp/merged/f1.npy"
    assert reanchor_path("/old/x/data/p1/a.png", "/new/exp",
                         anchors=PATH_ANCHORS)[0] == "/new/exp/data/p1/a.png"


def test_separator_and_component_helpers():
    assert normalise_separators(r"C:\a\b") == "C:/a/b"
    assert normalise_separators(None) == ""
    assert basename_any(r"C:\lab\exp1\merged\x.npy") == "x.npy"
    assert basename_any("/a/b/c/") == "c"
    assert path_components("/a/./b/../c") == ("", "a", "c")
    assert path_components("") == ()
    assert not path_is_under("", "/x") and not path_is_under("/x", "")


# --------------------------------------------------------------------------- #
#  Generalised from the COLUMN to the ROOT
# --------------------------------------------------------------------------- #

def test_every_path_bearing_column_is_re_anchored_not_only_png_path():
    """``correct_paths`` re-anchors ``png_path`` alone, so a moved folder
    showed its exported crops and failed on the arrays they are cut from."""
    frame = pd.DataFrame({
        "png_path": ["/old/exp/data/p1/a.png"],
        "path_name": ["/old/exp/merged/p1_f1.npy"],
        "merged_path": [r"C:\old\exp\merged\p1_f1.npy"],
        "pred": [0.5],
    })
    out, report = reanchor_frame(frame, "/new/exp")
    assert out["png_path"][0] == "/new/exp/data/p1/a.png"
    assert out["path_name"][0] == "/new/exp/merged/p1_f1.npy"
    assert out["merged_path"][0] == "/new/exp/merged/p1_f1.npy"
    assert report.n_reanchored == 3 and report.n_failed == 0
    assert report.describe() == ""


def test_the_report_counts_what_it_could_not_place_and_names_one():
    frame = pd.DataFrame({"png_path": ["/old/exp/data/p1/a.png",
                                       "/somewhere/else/b.png",
                                       "/nowhere/c.png",
                                       None, float("nan"), ""]})
    _out, report = reanchor_frame(frame, "/new/exp")
    assert report.n_paths == 3
    assert report.n_reanchored == 1
    assert report.n_failed == 2
    said = report.describe()
    assert "2 of 3 recorded paths could not be re-anchored under /new/exp" in said
    assert "/somewhere/else/b.png" in said


def test_a_frame_without_the_columns_is_left_alone():
    frame = pd.DataFrame({"pred": [0.1, 0.2]})
    out, report = reanchor_frame(frame, "/new/exp")
    assert list(out.columns) == ["pred"]
    assert report.n_paths == 0 and report.describe() == ""


# --------------------------------------------------------------------------- #
#  The two crop sources' own resolvers
# --------------------------------------------------------------------------- #

def test_the_png_source_re_anchors_from_the_right_and_across_separators():
    source = PngCropSource(root="/new/exp")
    assert source.resolve({"png_path": "/old/data/exp/data/p1/a.png"}) == \
        "/new/exp/data/p1/a.png"
    assert source.resolve({"png_path": r"C:\old\exp\data\p1\a.png"}) == \
        "/new/exp/data/p1/a.png"
    # Nothing to anchor on: untouched, and the failure surfaces on read.
    assert source.resolve({"png_path": "/old/exp/crops/a.png"}) == \
        "/old/exp/crops/a.png"


def test_the_merged_source_finds_a_windows_path_after_a_move(tmp_path):
    """``os.path.basename`` on Linux hands back the WHOLE backslashed string,
    so the flat fallback could not find the file either."""
    root = tmp_path / "exp"
    merged = root / "merged"
    merged.mkdir(parents=True)
    array = np.zeros((8, 8, 7), dtype=np.uint16)
    np.save(str(merged / "plate1_r1_c1_1.npy"), array)

    source = MergedCropSource(merged_root=str(merged))
    windows = {"path_name": r"D:\lab\exp\merged\plate1_r1_c1_1.npy"}
    assert source.resolve_path(windows) == str(merged / "plate1_r1_c1_1.npy")


def test_the_merged_source_prefers_the_anchor_over_the_flat_basename(tmp_path):
    """A sub-folder under ``merged/`` is preserved, which the old flat
    ``<merged_root>/<basename>`` retry silently flattened."""
    root = tmp_path / "exp"
    nested = root / "merged" / "plate1"
    nested.mkdir(parents=True)
    np.save(str(nested / "f1.npy"), np.zeros((8, 8, 7), dtype=np.uint16))
    # A decoy at the flat location, so preferring it would be visible.
    np.save(str(root / "merged" / "f1.npy"), np.ones((8, 8, 7), dtype=np.uint16))

    source = MergedCropSource(merged_root=str(root / "merged"))
    got = source.resolve_path({"path_name": "/old/place/merged/plate1/f1.npy"})
    assert got == str(nested / "f1.npy")
