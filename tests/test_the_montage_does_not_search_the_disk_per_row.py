"""Re-anchoring a plate asks the filesystem about folders, not about rows.

GITHUB ISSUE 116, and the reason "Show the cells" sat on "reading 4
database(s)" until the reporter gave up -- twice, once for four and a half
minutes and once for the rest of the log.

`spacr.crops.reanchor_frame` rewrites every recorded path in a measurement
frame so a project that has MOVED still finds its crops. When a path cannot
be placed structurally it falls back to
`spacr.portable_paths._reroot_with_prefix`, which asks the disk about roughly
twenty-two candidate locations. That was run PER ROW: 16,000 recorded paths
produced 360,000 stat calls, measured. The reporter had four plates -- about
a million paths -- and had just renamed the databases, so not one path was
already anchored. Some twenty-two million filesystem probes stood between
pressing the button and selecting the few hundred cells it would draw.

The two mechanisms that fix it were not invented here: `portable_paths`'s own
frame walker already had both, with its own measurement beside them (8.2 s
over 60,816 rows against 0.6 s once a prefix is known). This file is the
statement that the function the montage actually calls has them too.

Measured on this machine, paths that do not resolve: 122 us/row before,
11.7 us/row after. A million rows: two minutes to eleven seconds, on a fast
local disk -- and the reporter's is neither of those things.
"""
from __future__ import annotations

import os

import pytest

pd = pytest.importorskip("pandas")

from spacr.crops import reanchor_frame                      # noqa: E402


class _CountingExists:
    """Counts `os.path.exists` calls without changing what it answers."""

    def __init__(self, monkeypatch):
        self.calls = 0
        real = os.path.exists

        def counted(path):
            self.calls += 1
            return real(path)

        monkeypatch.setattr(os.path, "exists", counted)
        # `crops` and `portable_paths` both call it through the module, so
        # patching `os.path` covers every caller.


def _plate(tmp_path, wells=("w1", "w2"), per_well=25):
    """A real plate tree, and the paths a database would have recorded for
    it before the project moved."""
    root = tmp_path / "new" / "plate1"
    recorded = []
    for well in wells:
        folder = root / "data" / well
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_well):
            (folder / f"o{index}.png").write_bytes(b"x")
            recorded.append(f"/old/run7/plate1/data/{well}/o{index}.png")
    return root, recorded


def test_every_recorded_crop_is_still_found(tmp_path):
    """The behaviour first: speed that loses a crop is not speed."""
    root, recorded = _plate(tmp_path)
    frame = pd.DataFrame({"png_path": recorded})

    out, report = reanchor_frame(frame, str(root))

    assert report.n_reanchored == len(recorded)
    assert not report.failures
    for value in out["png_path"]:
        assert os.path.isfile(value), f"{value} was not placed"


def test_the_search_runs_once_per_folder_not_once_per_row(tmp_path,
                                                          monkeypatch):
    """The fix, stated as the count it changed.

    Two folders, fifty rows: the expensive resolver may run for the first
    row of each folder and must not run again once its prefix is known.

    Driven from the SCREEN folder rather than the plate folder, because that
    is the case that needs the search at all -- from the plate folder the
    structural pass places every row and nothing reaches the resolver, which
    would make this test pass without measuring anything.
    """
    root, recorded = _plate(tmp_path)
    root = root.parent
    frame = pd.DataFrame({"png_path": recorded})

    from spacr import portable_paths

    searches = []
    real = portable_paths._reroot_with_prefix

    def counted(path, src_root):
        searches.append(path)
        return real(path, src_root)

    monkeypatch.setattr(portable_paths, "_reroot_with_prefix", counted)

    reanchor_frame(frame, str(root))

    assert len(searches) <= 2, (
        f"the disk was searched {len(searches)} times for {len(recorded)} "
        f"rows in 2 folders")


def test_a_route_that_is_not_on_this_machine_is_searched_once(tmp_path,
                                                              monkeypatch):
    """A screen with PNG crops and no `merged/` is HEALTHY and common, and
    it was the worst case: nothing resolves, so nothing ever discovered a
    prefix and every single row paid for a full search."""
    root, _recorded = _plate(tmp_path)
    frame = pd.DataFrame({
        "path_name": [f"/old/run7/plate1/merged/f{i}.npy" for i in range(60)]})

    from spacr import portable_paths

    searches = []
    real = portable_paths._reroot_with_prefix

    def counted(path, src_root):
        searches.append(path)
        return real(path, src_root)

    monkeypatch.setattr(portable_paths, "_reroot_with_prefix", counted)

    _out, _report = reanchor_frame(frame, str(root))

    assert len(searches) <= 3, (
        f"a route with nothing on disk was searched {len(searches)} times "
        f"for 60 rows")


def test_a_missing_crop_beside_a_present_one_still_places_the_present_one(
        tmp_path):
    """The risk the folder memo introduces, and it was real.

    A folder written off after ONE failed search takes every later row in it
    down as well -- and the first row of a folder is not guaranteed to be one
    whose file was exported. Measured while writing this: one never-exported
    crop at the head of a folder lost all three real crops behind it, in the
    case that needs the search (a root one level above the plate, where the
    structural rewrite lands somewhere plausible that is not there).
    """
    root, recorded = _plate(tmp_path, wells=("w1",), per_well=3)
    screen_root = str(root.parent)          # one level up: the search is
    #: A row whose file was never exported, ahead of three that exist.
    mixed = ["/old/run7/plate1/data/w1/gone.png"] + recorded

    out, _report = reanchor_frame(pd.DataFrame({"png_path": mixed}),
                                  screen_root)

    placed = [p for p in out["png_path"] if os.path.isfile(p)]
    assert len(placed) == len(recorded), (
        "a missing crop suppressed the real ones beside it")


def test_a_folder_is_not_condemned_by_one_unlucky_row(tmp_path, monkeypatch):
    """The same property from the other side: the allowance is spent, not
    infinite. Ten missing rows in one folder must not cost ten searches."""
    root, _recorded = _plate(tmp_path, wells=("w1",), per_well=1)
    frame = pd.DataFrame({
        "png_path": [f"/old/run7/plate1/data/nowhere/o{i}.png"
                     for i in range(10)]})

    from spacr import portable_paths

    searches = []
    real = portable_paths._reroot_with_prefix
    monkeypatch.setattr(
        portable_paths, "_reroot_with_prefix",
        lambda path, src: (searches.append(path), real(path, src))[1])

    reanchor_frame(frame, str(root))

    assert len(searches) <= 3, (
        f"{len(searches)} searches for 10 rows in one absent folder")


def test_the_probe_count_does_not_grow_twentyfold_with_the_rows(tmp_path,
                                                                monkeypatch):
    """The number in the issue: ~22 filesystem probes per row.

    Bounded rather than pinned exactly -- the structural pass legitimately
    checks whether its own rewrite exists -- but a per-row SEARCH is 20+ and
    is what nine minutes of waiting was made of.
    """
    root, recorded = _plate(tmp_path, wells=("w1",), per_well=40)
    counter = _CountingExists(monkeypatch)

    reanchor_frame(pd.DataFrame({"png_path": recorded}), str(root))

    per_row = counter.calls / len(recorded)
    assert per_row < 5, f"{per_row:.1f} filesystem probes per row"
