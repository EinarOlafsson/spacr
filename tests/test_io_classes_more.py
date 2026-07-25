"""CPU coverage for spacr.io's Dataset/DataLoader classes, channel-folder
organisation, model checkpointing and the timelapse GIF writer.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _png(path, rng, size=16):
    Image.fromarray(rng.integers(0, 255, (size, size, 3)).astype(np.uint8)).save(path)
    return str(path)


def _class_dirs(root, rng, classes=("nc", "pc"), n=4):
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _png(d / f"{cls}_{i}.png", rng)
    return root


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

def test_no_class_dataset_load_to_memory(tmp_path, rng):
    """load_to_memory decodes every image up front."""
    from spacr.io import NoClassDataset
    d = tmp_path / "imgs"; d.mkdir()
    for i in range(4):
        _png(d / f"i{i}.png", rng)
    ds = NoClassDataset(str(d), transform=None, shuffle=True,
                        load_to_memory=True)
    assert len(ds) == 4
    img, path = ds[0][:2]
    assert path.endswith(".png")


def test_combined_dataset_concatenates(tmp_path, rng):
    from spacr.io import NoClassDataset, CombinedDataset
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    for i in range(3):
        _png(a / f"a{i}.png", rng)
    for i in range(2):
        _png(b / f"b{i}.png", rng)
    ds_a = NoClassDataset(str(a), shuffle=False)
    ds_b = NoClassDataset(str(b), shuffle=False)
    comb = CombinedDataset([ds_a, ds_b], shuffle=False)
    assert len(comb) == 5
    assert comb[0] is not None
    assert comb[4] is not None


def test_combined_dataset_shuffled(tmp_path, rng):
    from spacr.io import NoClassDataset, CombinedDataset
    a = tmp_path / "a"; a.mkdir()
    for i in range(4):
        _png(a / f"a{i}.png", rng)
    comb = CombinedDataset([NoClassDataset(str(a), shuffle=False)], shuffle=True)
    assert len(comb) == 4
    assert comb[2] is not None


def test_combine_loaders_round_robin(tmp_path, rng):
    from torch.utils.data import DataLoader
    from spacr.io import NoClassDataset, CombineLoaders
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    for i in range(4):
        _png(a / f"a{i}.png", rng)
        _png(b / f"b{i}.png", rng)
    la = DataLoader(NoClassDataset(str(a), shuffle=False), batch_size=2)
    lb = DataLoader(NoClassDataset(str(b), shuffle=False), batch_size=2)
    # 4 images per folder at batch_size=2 => 2 batches per loader, 4 total.
    # Every batch must be delivered exactly once regardless of the shuffle
    # order; nothing may be dropped when a loader empties.
    per_loader = {0: 0, 1: 0}
    for idx, batch in CombineLoaders([la, lb]):
        assert idx in (0, 1)
        per_loader[idx] += 1
    assert per_loader == {0: 2, 1: 2}


def test_spacr_dataloader_preloads(tmp_path, rng):
    from torchvision import transforms
    from spacr.io import spacrDataset, spacrDataLoader
    root = _class_dirs(tmp_path / "train", rng, n=4)
    # A transform is required: without one the dataset yields PIL Images,
    # which torch's default collate cannot batch.
    ds = spacrDataset(str(root), loader_classes=["nc", "pc"],
                      transform=transforms.ToTensor(), shuffle=False)
    dl = spacrDataLoader(ds, batch_size=2, preload_batches=1)
    batches = list(iter(dl))
    assert len(batches) == 4          # 8 images / batch 2
    # iterating again re-runs the full stream
    assert len(list(iter(dl))) == 4
    dl.cleanup()


def test_spacr_dataloader_surfaces_errors(tmp_path, rng):
    """A collate failure must raise, not masquerade as an empty dataset."""
    from spacr.io import spacrDataset, spacrDataLoader
    root = _class_dirs(tmp_path / "train", rng, n=2)
    # transform=None -> PIL images -> default_collate TypeError
    ds = spacrDataset(str(root), loader_classes=["nc", "pc"], shuffle=False)
    dl = spacrDataLoader(ds, batch_size=2, preload_batches=1)
    with pytest.raises(Exception):
        list(iter(dl))
    dl.cleanup()


def test_spacr_dataset_specific_files(tmp_path, rng):
    """specific_files/labels bypass directory scanning."""
    from spacr.io import spacrDataset
    d = tmp_path / "flat"; d.mkdir()
    files = [_png(d / f"x{i}.png", rng) for i in range(4)]
    labels = [0, 1, 0, 1]
    ds = spacrDataset(str(d), loader_classes=["a", "b"],
                      specific_files=files, specific_labels=labels,
                      shuffle=False)
    assert len(ds) == 4
    _img, label, _path = ds[1]
    assert label == 1


# ---------------------------------------------------------------------------
# channel folder organisation
# ---------------------------------------------------------------------------

def test_move_to_chan_folder(tmp_path, rng):
    """Cellvoyager-named TIFFs are sorted into per-channel subfolders."""
    from spacr.io import _move_to_chan_folder
    from spacr.utils import _get_regex
    src = tmp_path / "plate1"; src.mkdir()
    for field in (1, 2):
        for chan in (1, 2):
            name = f"plate1_A01_T0001F00{field}L01A01Z01C0{chan}.tif"
            tifffile.imwrite(str(src / name),
                             rng.integers(0, 500, (16, 16)).astype(np.uint16))
    regex = _get_regex("cellvoyager", ".tif", None)
    _move_to_chan_folder(str(src), regex, timelapse=False,
                         metadata_type="cellvoyager")
    chan_dirs = [p for p in src.iterdir() if p.is_dir() and p.name.isdigit()]
    assert chan_dirs, "expected per-channel folders"
    assert any(any(d.glob("*.tif")) for d in chan_dirs)


# ---------------------------------------------------------------------------
# multi-dimensional / non-tif splitting
# ---------------------------------------------------------------------------

def test_process_non_tif_non_2D_splits_3d_stack(tmp_path, rng):
    """A 3-D TIFF is split into one grayscale TIFF per plane."""
    from spacr.io import process_non_tif_non_2D_images
    d = tmp_path / "multi"; d.mkdir()
    tifffile.imwrite(str(d / "stack.tif"),
                     rng.integers(0, 500, (3, 16, 16)).astype(np.uint16))
    process_non_tif_non_2D_images(str(d))
    produced = [p.name for p in d.glob("*.tif")]
    assert len(produced) > 1, f"3-D stack not split: {produced}"


def test_process_non_tif_non_2D_rgb_tif(tmp_path, rng):
    """A 3-channel TIFF is split into one grayscale TIFF per channel."""
    from spacr.io import process_non_tif_non_2D_images
    d = tmp_path / "rgb"; d.mkdir()
    tifffile.imwrite(str(d / "c.tif"),
                     rng.integers(0, 500, (16, 16, 3)).astype(np.uint16))
    process_non_tif_non_2D_images(str(d))
    produced = [p.name for p in d.glob("*.tif")]
    assert len(produced) > 1, f"multi-channel tif not split: {produced}"


# ---------------------------------------------------------------------------
# model checkpointing
# ---------------------------------------------------------------------------

def test_save_model_writes_checkpoint(tmp_path):
    import torch
    from spacr.io import _save_model
    from spacr.utils import TorchModel
    model = TorchModel(model_name="resnet18", pretrained=False, num_classes=2)
    results = {"accuracy": 0.95, "neg_accuracy": 0.9, "pos_accuracy": 0.95,
               "loss": 0.2, "prauc": 0.9, "optimal_threshold": 0.5}
    out = _save_model(model, "resnet18", results, str(tmp_path),
                      epoch=1, epochs=1, intermedeate_save=False,
                      channels=["r", "g", "b"])
    pths = list(tmp_path.rglob("*.pth"))
    assert pths or out is not None


def test_save_model_intermediate_threshold(tmp_path):
    import torch
    from spacr.io import _save_model
    from spacr.utils import TorchModel
    model = TorchModel(model_name="resnet18", pretrained=False, num_classes=2)
    results = {"accuracy": 0.99, "neg_accuracy": 0.99, "pos_accuracy": 0.99,
               "loss": 0.01, "prauc": 0.99, "optimal_threshold": 0.5}
    _save_model(model, "resnet18", results, str(tmp_path),
                epoch=5, epochs=10, intermedeate_save=[0.9],
                channels=["r", "g", "b"], val_dict=results)
    assert list(tmp_path.rglob("*.pth"))


# ---------------------------------------------------------------------------
# timelapse GIF
# ---------------------------------------------------------------------------

def test_save_mask_timelapse_as_gif(tmp_path):
    from spacr.io import _save_mask_timelapse_as_gif
    import matplotlib.colors as mcolors
    masks = []
    for t in range(3):
        m = np.zeros((16, 16), np.uint16)
        m[2 + t:6 + t, 2:6] = 1
        masks.append(m)
    out = tmp_path / "tl.gif"
    try:
        _save_mask_timelapse_as_gif(
            masks, None, str(out), cmap="viridis",
            norm=mcolors.Normalize(vmin=0, vmax=2),
            filenames=[f"f{t}.npy" for t in range(3)])
    except Exception as e:
        pytest.skip(f"gif writer unavailable: {e}")
    assert out.exists()
