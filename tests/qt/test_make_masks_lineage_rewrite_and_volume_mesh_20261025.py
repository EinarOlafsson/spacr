"""Two Make Masks worker paths, run on real masks (item 43 coverage ceiling).

``_rewrite_lineage_request`` embeds a sequence's lineage graph in every saved
mask of the sequence; a frame nobody has saved a mask for yet is skipped,
not created and not counted as a failure. ``_volume_mesh`` builds an
object's editable 3-D boundary off the GUI thread; called without a
cancellation token it returns exactly what the session builds directly.
"""
from __future__ import annotations

import os

import numpy as np

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as screen


def test_a_lineage_rewrite_skips_a_frame_with_no_saved_mask(tmp_path):
    folder = str(tmp_path)
    mask = np.zeros((8, 8), np.uint16)
    mask[2:5, 2:5] = 1
    engine.save_mask(folder, "frame_01.tif", mask)
    lineage, founder = engine.new_lineage_cell(
        None, "founder", 1, os.path.join(folder, "frame_01.tif"))
    missing = engine.mask_save_path(folder, "frame_02.tif")

    failed = screen._rewrite_lineage_request({
        "folder": folder, "names": ["frame_01.tif", "frame_02.tif"],
        "lineage": lineage, "layout": {}})

    assert failed == 0
    saved = engine.load_mask_lineage(folder, "frame_01.tif")
    assert saved["cells"] == engine.validate_lineage(lineage)["cells"]
    assert founder in saved["cells"]
    assert list(saved["frames"]) == [os.path.join(folder, "frame_01.tif")]
    assert not os.path.exists(missing)


def test_a_volume_mesh_without_a_token_is_the_sessions_own_mesh():
    labels = np.zeros((6, 6, 6), np.uint16)
    labels[1:5, 1:5, 1:5] = 3
    session = engine.VolumeMaskSession(np.zeros(labels.shape, np.uint16), labels)

    built = screen._volume_mesh((session, 3))

    assert built == session.geometry_for_label(3)
