"""End-to-end regression test for the make-masks screen.

Flow: create a folder of synthetic 16-bit images → open it in
MakeMasksScreen → brush pixels into the mask → save → re-open a
fresh MakeMasksScreen against the same folder → assert the mask on
disk contains the pixels we painted.

This exercises MakeMasksScreen ↔ mask_engine.save_mask ↔ imageio
round-trip in one shot.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic folder
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_mask_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "expt_masks"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        img = rng.integers(0, 65535, size=(80, 100), dtype=np.uint16)
        imageio.imwrite(folder / f"img_{i:02d}.tif", img)
    return folder


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------

def test_open_paint_save_reload_persists(
    qtbot, qt_theme_applied, synth_mask_folder: Path,
):
    """Paint a labeled blob into image 0's mask, save, re-open, verify
    that the mask on disk has connected components covering the
    painted region."""
    from spacr.qt.screens.make_masks import MakeMasksScreen

    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    assert screen._canvas.mask is not None
    H, W = screen._canvas.mask.shape[:2]

    # Paint two obvious blobs
    screen._canvas.mask[10:30, 20:40] = 255
    screen._canvas.mask[60:70, 70:90] = 255

    # Save mask
    screen._on_save()

    # File must exist under masks/
    saved = synth_mask_folder / "masks" / "img_00.tif"
    assert saved.is_file(), f"expected saved mask at {saved}"

    # Reload from disk and confirm the two blobs are present as
    # connected components (save_mask relabels to uint16 labels).
    disk = imageio.imread(saved)
    assert disk.shape == (H, W)
    unique_nonzero = sorted(int(v) for v in np.unique(disk) if v > 0)
    assert unique_nonzero == [1, 2], (
        f"expected two labeled components, got {unique_nonzero}"
    )

    # Open a FRESH MakeMasksScreen against the same folder and confirm
    # image 0's mask now has the two blobs already loaded.
    fresh = MakeMasksScreen()
    qtbot.addWidget(fresh)
    fresh._open_folder(str(synth_mask_folder))
    assert (fresh._canvas.mask > 0).sum() > 0, (
        "reopened mask has no non-zero pixels; save didn't round-trip"
    )


def test_engine_ops_via_screen(
    qtbot, qt_theme_applied, synth_mask_folder: Path,
):
    """Cover the object-op handlers wired to the screen — the buttons
    the user actually clicks."""
    from spacr.qt.screens.make_masks import MakeMasksScreen

    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen._open_folder(str(synth_mask_folder))
    # Paint a tiny blob (4 px) and a big one (100 px)
    screen._canvas.mask[5:7, 5:7] = 255
    screen._canvas.mask[20:30, 20:30] = 255

    # Remove-small at 10 px cutoff drops the tiny one
    screen._min_area.setValue(10)
    screen._on_remove_small()
    assert not (screen._canvas.mask[5:7, 5:7] > 0).any(), \
        "remove_small should have dropped the 4-pixel blob"
    assert (screen._canvas.mask[20:30, 20:30] > 0).any(), \
        "remove_small should have kept the 100-pixel blob"

    # Relabel — the surviving blob gets label 1
    screen._on_relabel()
    unique_nz = sorted(int(v) for v in np.unique(screen._canvas.mask) if v > 0)
    assert unique_nz == [1]

    # Fill holes — draw a ring and confirm the interior fills
    screen._canvas.mask[:] = 0
    screen._canvas.mask[10:20, 10:12] = 255
    screen._canvas.mask[10:20, 20:22] = 255
    screen._canvas.mask[10:12, 10:22] = 255
    screen._canvas.mask[18:20, 10:22] = 255
    screen._on_fill_holes()
    # Interior of the ring is filled
    assert screen._canvas.mask[15, 15] > 0
