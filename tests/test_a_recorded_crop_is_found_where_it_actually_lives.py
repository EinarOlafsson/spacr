"""Recorded crop paths resolved against a root that is not the plate folder.

Re-anchoring rewrites a path on structure alone: it takes the recorded path's
``data/`` tail and hangs it under the root it was given. That is right when
the root is the folder holding ``data/`` and wrong the moment a caller holds
the screen folder above it, the ``measurements/`` folder, or the database
file -- the rewrite then lands somewhere plausible that is not there, and the
failure arrives much later as a missing file.

Both the frame-wide pass and the single-row crop source fall back to the
resolver that searches the recorded structure under every folder the root
could mean and returns ONLY what exists, so a path that does not exist is
never accepted as a resolution.

Also here: a percentile setting that is not two numbers. A picture is the
last thing a montage produces and the least important, so a mistyped
percentile falls back to the shipped pair rather than losing the montage.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from spacr.crops import (DEFAULT_PERCENTILES, PngCropSource, percentile_pair,
                         reanchor_frame, reanchor_path)


# ---------------------------------------------------------------------------
# a screen copied to another machine
# ---------------------------------------------------------------------------

RECORDED = "/old/machine/plate1/data/cell_png/object_1.png"


def _screen(tmp_path):
    """A screen folder with the crop one level further down than the root."""
    real = tmp_path / "screen" / "plate1" / "data" / "cell_png" / "object_1.png"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"\x89PNG\r\n\x1a\n")
    return tmp_path / "screen", real


def test_the_structural_rewrite_alone_lands_somewhere_that_is_not_there(
        tmp_path):
    """The premise of both fallbacks below, pinned so it cannot drift."""
    root, real = _screen(tmp_path)

    rewritten, outcome = reanchor_path(RECORDED, str(root))

    assert outcome == "reanchored"
    assert rewritten != str(real)
    assert not os.path.exists(rewritten)


def test_a_frame_re_anchored_from_the_screen_folder_finds_the_real_files(
        tmp_path):
    """A caller holding the screen folder is not doing anything wrong."""
    root, real = _screen(tmp_path)
    frame = pd.DataFrame({"png_path": [RECORDED]})

    out, report = reanchor_frame(frame, str(root))

    assert out["png_path"].tolist() == [str(real)]
    assert os.path.exists(out["png_path"].iloc[0])
    assert report.n_reanchored == 1
    assert report.failures == ()


def test_the_crop_source_resolves_a_row_the_rewrite_could_not_place(tmp_path):
    """One row, same fallback: resolve() may only return a path that exists."""
    root, real = _screen(tmp_path)
    source = PngCropSource(root=str(root))

    assert source.resolve({"png_path": RECORDED}) == str(real)


def test_a_path_that_is_nowhere_comes_back_as_recorded(tmp_path):
    """The fallback may not invent a resolution.

    A caller's error has to keep naming what was actually recorded, otherwise
    the message points at a path the database never held.
    """
    root, _real = _screen(tmp_path)
    missing = "/old/machine/plate9/data/cell_png/object_404.png"
    source = PngCropSource(root=str(root))

    resolved = source.resolve({"png_path": missing})

    assert not os.path.exists(resolved)
    assert resolved.endswith(os.path.join("data", "cell_png", "object_404.png"))


# ---------------------------------------------------------------------------
# a percentile setting that is not two numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("typed", ["2%, 98%", "low,high", "[a b]", "None,None"])
def test_a_percentile_that_is_not_a_number_falls_back_to_the_shipped_pair(
        typed):
    """Losing a montage to a mistyped percentile is the worst trade going."""
    assert percentile_pair(typed) == (float(DEFAULT_PERCENTILES[0]),
                                      float(DEFAULT_PERCENTILES[1]))


def test_a_stated_default_is_honoured_when_the_pair_cannot_be_read():
    """The caller's own default, not the module's, when one is given."""
    assert percentile_pair("2%, 98%", default=(5, 95)) == (5.0, 95.0)


def test_two_numbers_still_read_as_two_numbers():
    """The control: the fallback must not be swallowing good input."""
    assert percentile_pair("1, 99") == (1.0, 99.0)
    assert percentile_pair([2, 98]) == (2.0, 98.0)
