"""Item 288: the SAM mask generator hands a timelapse to Timeflows when asked.

``timelapse_mode='timeflows'`` is spaCR's opt-in temporal tracker. The SAM
generator must hand it what it needs and nothing it does not: the label
stack AND the image batch (Timeflows reads the images, unlike the overlap
trackers), the checkpoint named by ``timeflows_model``, and the run's plot,
save and remove-transient switches. The tracker itself is a recorder here;
its linking is pinned in ``test_timeflows_movie_tracking.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

import spacr.object as O
from tests.test_cov_object_cellpose_masks import (  # noqa: F401
    _mask_files, _settings, _write_npz, fake_sam_model, fake_timelapse)


@pytest.fixture
def timeflows(monkeypatch):
    import spacr.timelapse as TL

    calls = []

    def track(**kw):
        calls.append(kw)
        return [np.asarray(m, dtype=np.uint16) for m in kw["masks"]]

    monkeypatch.setattr(TL, "_timeflows_track_cells", track)
    return calls


def test_the_sam_generator_routes_to_timeflows_with_its_checkpoint(
        tmp_path, fake_sam_model, fake_timelapse, timeflows):
    src = tmp_path / "stack"
    _, names = _write_npz(src, n=3)
    settings = _settings(
        src, timelapse=True, timelapse_objects=["cell"],
        timelapse_mode="timeflows", timelapse_displacement=None,
        timelapse_memory=3, timelapse_remove_transient=True,
        timelapse_frame_limits=[0, 3], timeflows_model="/models/tf.pt",
        cell_min_split_area=0, nucleus_min_split_area=0)

    O.generate_cellpose_masks_sam(str(src), settings, "cell")

    assert len(timeflows) == 1
    kw = timeflows[0]
    assert kw["object_type"] == "cell"
    assert kw["model_path"] == "/models/tf.pt"
    assert kw["mode"] == "timeflows"
    assert kw["timelapse_remove_transient"] is True
    assert kw["batch_filenames"] == names
    assert len(kw["masks"]) == 3
    assert np.asarray(kw["images"]).shape == (3, 32, 32, 2)
    assert fake_timelapse["ultrack"] == [] and fake_timelapse["btrack"] == []
    assert _mask_files(src, "cell") == sorted(names)
