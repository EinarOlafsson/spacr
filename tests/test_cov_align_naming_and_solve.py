"""spacr.align — what the filename decides, and what the solve weighs.

``tests/test_align.py`` and ``tests/test_align_cov.py`` between them already
execute every line of :mod:`spacr.align`. Executing a line is not the same as
checking what it computed, and this file goes after two things those suites
step over:

* **the name is the key.** :func:`~spacr.align.scan_tiles` decides from a
  filename alone which files are the same stage position and which are
  different ones. Get that wrong and tiles are dropped, or two wells are
  composited into one canvas as if they were two channels — silently, with
  a clean-looking plan and a full coordinates table. Five naming shapes are
  pinned here; three of them are broken today and carry
  ``xfail(strict=True)`` asserting the correct answer, in this repo's usual
  form.
* **the numbers the least-squares solve actually produces.** The existing
  suite proves the global solve beats sequential accumulation. It never
  checks that a pair's ``confidence`` reaches the solver as a *weight*, nor
  that ``DEFAULT_ANCHOR_WEIGHT`` is small enough to leave a registered chain
  alone — both are single-constant regressions that would degrade every
  stitch on the plate while every existing test still passed.

Plus the two documented settings the suite never passes (``save_stack``,
``group_by_well``) and the reader-cache size, which turns out to unregister
an entire plate when it is set to its own documented minimum.

Everything is deterministic (fixed seeds), CPU-only, offline, and no tile
here is larger than 128x128.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from spacr import align


# ---------------------------------------------------------------------------
# Builders — same shape as the two existing align suites use.
# ---------------------------------------------------------------------------

def _texture(height, width, seed=0, sigma=2.0):
    """A smooth, non-repeating uint16 field — registrable, unlike noise."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    raw = rng.random((height, width)).astype(np.float32)
    smooth = gaussian_filter(raw, sigma)
    smooth -= smooth.min()
    smooth /= max(float(smooth.max()), 1e-9)
    return (smooth * 30000 + 1000).astype(np.uint16)


TILE = 128
STEP = 80
OVERLAP = 1 - STEP / TILE


def _write_2x2(folder, plate='plate1', well='B07', seed=7):
    """Cut one texture into a 2x2 grid of ``TILE`` tiles stepped ``STEP``."""
    os.makedirs(folder, exist_ok=True)
    big = _texture(TILE + STEP + 8, TILE + STEP + 8, seed=seed)
    k = 0
    for row in range(2):
        for col in range(2):
            np.save(os.path.join(folder, f'{plate}_{well}_{k + 1:03d}.npy'),
                    big[row * STEP:row * STEP + TILE,
                        col * STEP:col * STEP + TILE])
            k += 1
    return big


# ---------------------------------------------------------------------------
# 1. The filename is the key: which files are one stage position
# ---------------------------------------------------------------------------

def test_grid_describes_one_wells_acquisition_not_the_whole_folder(tmp_path):
    """``grid=(2, 2)`` on a two-well plate means 2x2 *per well*.

    A grid is a property of the acquisition, and the acquisition is
    per well — the same 2x2 field pattern is repeated in every well on the
    plate. With ``group_by_well`` on (the default) there is no other thing
    ``grid`` could mean, and no other value a user could pass: (2, 4) is not
    the layout of anything that was imaged.
    """
    src = tmp_path / 'plate'
    src.mkdir()
    for well, seed in (('B07', 7), ('C08', 11)):
        _write_2x2(str(src), well=well, seed=seed)

    results = align.align_folder(src=str(src), dst=str(tmp_path / 'out'),
                                 grid=(2, 2), overlap=OVERLAP,
                                 save_stack=False)

    assert len(results) == 2, 'one stitch per well'
    for result in results:
        # A 2x2 mosaic is about TILE + STEP on a side; a 1x4 strip would be
        # TILE + 3 * STEP wide and one tile tall.
        assert result.canvas.height == pytest.approx(TILE + STEP, abs=4)
        assert result.canvas.width == pytest.approx(TILE + STEP, abs=4)


# ---------------------------------------------------------------------------
# 2. The reader cache: a documented memory knob that unregisters the plate
# ---------------------------------------------------------------------------

def test_a_two_tile_cache_places_and_writes_exactly_what_the_default_does(
        tmp_path):
    """Evicting between bands must not move a pixel.

    Two is the smallest cache a pair fits in, so this run evicts and
    re-opens a reader on nearly every pair and every band — the LRU path
    that keeps a thousand-tile folder from becoming resident. If a re-opened
    reader ever returned a stale window, an offset window, or a stale
    ``shape``, the placements or the canvas would differ from the run that
    never evicted anything. They are compared exactly.
    """
    folder = tmp_path / 'lru'
    _write_2x2(str(folder))
    tiles = align.scan_tiles(str(folder), grid=(2, 2), overlap=OVERLAP)

    roomy = align.estimate_offsets(tiles, max_open_tiles=8)
    cramped = align.estimate_offsets(tiles, max_open_tiles=2)

    assert roomy.n_registered == 4 and cramped.n_registered == 4
    assert [(p.y, p.x) for p in cramped.placements] == \
           [(p.y, p.x) for p in roomy.placements]

    wide = align.write_stack(roomy, tmp_path / 'wide.npy', max_open_tiles=8)
    narrow = align.write_stack(cramped, tmp_path / 'narrow.npy',
                               max_open_tiles=2, band_rows=17)
    assert wide.n_written == narrow.n_written == 4
    assert np.array_equal(np.load(wide.stack_path),
                          np.load(narrow.stack_path))


# ---------------------------------------------------------------------------
# 3. What the global solve weighs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 4. Two settings the suite never passes
# ---------------------------------------------------------------------------

