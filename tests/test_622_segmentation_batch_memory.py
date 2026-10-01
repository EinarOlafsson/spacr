"""Completed segmentation batches must not survive into the next allocation."""
import types
import weakref

import numpy as np
import pytest

import spacr.object as objects
from tests.test_cov_object_masks_sam import _settings, force_cpu  # noqa: F401


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("save", [False, True])
def test_completed_arrays_are_released_before_next_batch_and_archive(
        tmp_path, monkeypatch, channels, save):
    src = tmp_path / "masks"
    src.mkdir()
    for archive in range(2):
        pixels = np.full((2, 24, 25, channels), .25, dtype=np.float32)
        if channels > 1:
            pixels[..., 1] = .75
        np.savez(src / f"batch{archive}.npz", data=pixels,
                 filenames=[f"plate1_A01_{archive * 2 + i}.npy" for i in range(2)])

    completed = []
    live = []
    loaded = []
    auxiliary = []
    seen = []
    original_load = np.load

    class Archive:
        def __init__(self, path, *args, **kwargs):
            self.archive = original_load(path, *args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __getitem__(self, key):
            if key == "data":
                assert all(ref() is None for ref in loaded), "previous archive retained"
            array = self.archive[key]
            if key == "data":
                loaded.append(weakref.ref(array))
            return array

    def load(path, *args, **kwargs):
        return (Archive(path, *args, **kwargs) if str(path).endswith(".npz")
                else original_load(path, *args, **kwargs))

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["gpu"] is False

        def eval(self, x, **kwargs):
            assert all(ref() is None for ref in live), "previous batch retained"
            assert all(ref() is None for ref in auxiliary), "unused flow outputs retained"
            assert len(x) == 1
            np.testing.assert_allclose(x[0][..., 0], .25 if channels == 1 else .75)
            if channels > 1:
                np.testing.assert_allclose(x[0][..., 1], .25)
            seen.append(x[0].shape)
            live.extend(weakref.ref(image) for image in x)
            masks = [np.zeros(image.shape[:2], dtype=np.uint16) for image in x]
            for mask in masks:
                mask[3:10, 3:10] = 1
            flows = [[np.zeros((*image.shape[:2], 3), dtype=np.uint8),
                      np.zeros((2, *image.shape[:2]), dtype=np.float32),
                      np.zeros(image.shape[:2], dtype=np.float32),
                      np.zeros(image.shape[:2], dtype=np.float32)] for image in x]
            live.extend(weakref.ref(mask) for mask in masks)
            live.extend(weakref.ref(flow[0]) for flow in flows)
            auxiliary.extend(weakref.ref(array) for flow in flows for array in flow[1:])
            return masks, flows, np.zeros((1, 256), dtype=np.float32)

    original_filter = objects.merge_split_filter_masks

    def filter_masks(**kwargs):
        assert all(ref() is None for ref in auxiliary), "unused flows retained during filtering"
        return original_filter(**kwargs)

    def batch_done(path):
        assert all(ref() is None for ref in loaded), "archive retained after completion"
        assert all(ref() is None for ref in live), "batch retained after completion"
        completed.append(path)

    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(objects, "cp_models", types.SimpleNamespace(CellposeModel=Model))
    monkeypatch.setattr(objects, "merge_split_filter_masks", filter_masks)
    settings = _settings(src, batch_size=1, cell_channel=0,
                         nucleus_channel=None if channels == 1 else 1,
                         pathogen_channel=None if channels == 1 else 2, save=save)
    objects.generate_cellpose_masks_sam(str(src), settings, "cell",
                                      on_batch_done=batch_done, run_qc=False)
    assert seen == [(24, 25, 1 if channels == 1 else 2)] * 4
    assert len(completed) == 2
    if save:
        files = sorted((src / "cell_mask_stack").glob("*.npy"))
        assert len(files) == 4
        for path in files:
            mask = original_load(path)
            assert mask.dtype == np.uint16
            assert mask.shape == (24, 25)
            assert np.count_nonzero(mask) == 49
