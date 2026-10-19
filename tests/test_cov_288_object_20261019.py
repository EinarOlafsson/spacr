"""Timepose is the tracker generate_cellpose_masks_sam hands a timelapse to when asked."""
from __future__ import annotations

import numpy as np

import spacr.object as O
import spacr.timelapse as TL
from tests.test_cov_object_masks_sam import (  # noqa: F401  (fixtures)
    _tl_settings, _write_npz, fake_model, fake_timelapse)


def test_timelapse_timepose_gets_masks_images_and_the_release_key(
        tmp_path, fake_model, fake_timelapse, monkeypatch):
    seen = []

    def timepose(**kw):
        seen.append(kw)
        return [np.asarray(m, dtype=np.uint16) for m in kw["masks"]]

    monkeypatch.setattr(TL, "_timepose_track_cells", timepose)
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=2)
    settings = _tl_settings(src, batch_size=2, timelapse_mode="timepose_v1",
                            timelapse_remove_transient=True)

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(seen) == 1
    kw = seen[0]
    assert kw["mode"] == "timepose_v1"
    assert kw["batch_filenames"] == names
    assert kw["object_type"] == "cell"
    assert kw["timelapse_remove_transient"] is True
    assert np.asarray(kw["images"]).shape == (2, 32, 32, 2)
    assert len(kw["masks"]) == 2
    assert fake_timelapse["btrack"] == [] and fake_timelapse["trackastra"] == []
