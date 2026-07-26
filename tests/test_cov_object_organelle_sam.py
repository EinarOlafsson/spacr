"""Coverage for ``spacr.object.generate_organelle_masks_sam`` and its two
settings helpers (``_validate_organelle_settings``/``_build_object_settings``).

Everything here is CPU-only and offline: the ``.npz`` batches are built from
deterministic synthetic blob fields, the Cellpose model is replaced by a fake
whose ``eval`` returns canned masks, and the U-Net is a 1x1 convolution saved
with ``torch.save``. No network, no CUDA, no plt.show() that blocks.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
#  Fixtures / helpers
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _close_figures():
    """Never let Agg figures accumulate between tests."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """Keep every device decision on the CPU, whatever the host has."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# Four well-separated blobs -> four objects for every classical method.
BLOB_CENTERS = ((16, 16), (16, 48), (48, 16), (48, 48))
TOP_CENTERS = ((16, 16), (16, 48))


def _blob_field(h=64, w=64, centers=BLOB_CENTERS, radius=3, bg=100, fg=3000):
    """Deterministic uint16 field: flat background + `radius`-px bright disks."""
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w), bg, dtype=np.uint16)
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = fg
    return img


def _flat_field(h=64, w=64, value=100):
    return np.full((h, w), value, dtype=np.uint16)


def _make_npz(src, filenames, channel_specs=("blob",), name="stack1",
              centers=BLOB_CENTERS):
    """Write one merged-stack .npz under `src`.

    `channel_specs` is a tuple of 'blob'/'flat' (one entry per channel) giving a
    4-D ``(N, H, W, C)`` stack, or ``None`` for a channel-less 3-D
    ``(N, H, W)`` stack.
    """
    os.makedirs(src, exist_ok=True)
    imgs = []
    for _ in filenames:
        if channel_specs is None:
            imgs.append(_blob_field(centers=centers))
        else:
            chans = [_blob_field(centers=centers) if spec == "blob" else _flat_field()
                     for spec in channel_specs]
            imgs.append(np.stack(chans, axis=-1))
    data = np.stack(imgs, axis=0)
    path = os.path.join(src, f"{name}.npz")
    np.savez(path, data=data, filenames=np.array(list(filenames)))
    return path


def _base_settings(**over):
    """Minimal settings dict; everything else comes from _set_organelle_defaults."""
    settings = {
        "verbose": False,
        "save": True,
        "plot": False,
        "batch_size": 4,
        "n_jobs": 1,
        "nucleus_channel": None,
        "cell_channel": None,
        "pathogen_channel": None,
        "organelle_channel": 0,
        "organelle_morphology": "spots",
        "organelle_method": "otsu",
        "organelle_model_name": "cpsam",
    }
    settings.update(over)
    return settings


def _n_objects(mask):
    uniq = np.unique(mask)
    return int(len(uniq) - (1 if 0 in uniq else 0))


def _object_counts(db_path):
    con = sqlite3.connect(str(db_path))
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "object_counts" not in tables:
            return None
        return sorted(con.execute(
            "SELECT file_name, count_type, object_count FROM object_counts").fetchall())
    finally:
        con.close()


def _run(src, settings, object_type="organelle"):
    from spacr.object import generate_organelle_masks_sam
    return generate_organelle_masks_sam(str(src), settings, object_type)


# --------------------------------------------------------------------------- #
#  Early exits
# --------------------------------------------------------------------------- #

def test_returns_early_when_src_has_no_npz(tmp_path, capsys):
    """No .npz in src -> print + return before the database is created."""
    src = tmp_path / "masks"
    src.mkdir()
    (src / "not_a_stack.txt").write_text("ignored")

    assert _run(src, _base_settings()) is None

    out = capsys.readouterr().out
    assert "No .npz files found" in out
    # The measurements DB is only created after the early return.
    assert not (tmp_path / "measurements").exists()
    assert not (src / "organelle_mask_stack").exists()


def test_invalid_morphology_raises_before_any_io(tmp_path):
    """_validate_organelle_settings is reached before the .npz listing."""
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    with pytest.raises(ValueError, match="organelle_morphology must be one of"):
        _run(src, _base_settings(organelle_morphology="mesh"))
    assert not (tmp_path / "measurements").exists()


def test_invalid_method_for_morphology_raises(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    with pytest.raises(ValueError, match="method must be one of"):
        _run(src, _base_settings(organelle_morphology="irregular",
                                 organelle_method="log"))


# --------------------------------------------------------------------------- #
#  Happy path: spots / otsu
# --------------------------------------------------------------------------- #

def test_spots_otsu_writes_uint16_masks_and_db_counts(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy", "f2.npy"])

    _run(src, _base_settings())

    out_dir = src / "organelle_mask_stack"
    assert sorted(os.listdir(out_dir)) == ["f1.npy", "f2.npy"]

    mask = np.load(out_dir / "f1.npy")
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint16
    assert _n_objects(mask) == 4
    assert mask[0, 0] == 0                       # background stays background
    # every blob centre carries a distinct label
    labels = {int(mask[cy, cx]) for cy, cx in BLOB_CENTERS}
    assert 0 not in labels and len(labels) == 4

    rows = _object_counts(tmp_path / "measurements" / "measurements.db")
    assert rows == [("f1.npy", "organelle", 4), ("f2.npy", "organelle", 4)]


def test_batch_size_one_processes_every_image(tmp_path):
    """batch_size < n_images exercises the inner batching loop repeatedly."""
    src = tmp_path / "masks"
    _make_npz(str(src), ["a.npy", "b.npy", "c.npy"])

    _run(src, _base_settings(batch_size=1))

    out_dir = src / "organelle_mask_stack"
    assert sorted(os.listdir(out_dir)) == ["a.npy", "b.npy", "c.npy"]
    rows = _object_counts(tmp_path / "measurements" / "measurements.db")
    assert rows == [("a.npy", "organelle", 4),
                    ("b.npy", "organelle", 4),
                    ("c.npy", "organelle", 4)]


def test_two_npz_stacks_both_get_their_own_output_folder(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["a.npy"], name="stack1")
    _make_npz(str(src), ["b.npy"], name="stack2")

    _run(src, _base_settings())

    out_dir = src / "organelle_mask_stack"
    assert sorted(os.listdir(out_dir)) == ["a.npy", "b.npy"]


def test_verbose_displays_organelle_settings_and_clamps_n_jobs(tmp_path, capsys):
    """verbose=True renders the organelle_* table; n_jobs<1 is clamped to 1."""
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(verbose=True, n_jobs=0))

    out = capsys.readouterr().out
    assert "setting_key" in out and "setting_value" in out
    assert "organelle_morphology" in out
    # the per-batch summary reports the clamped worker count
    assert "n_jobs=1]" in out
    assert (src / "organelle_mask_stack" / "f1.npy").exists()


# --------------------------------------------------------------------------- #
#  Channel selection / remapping
# --------------------------------------------------------------------------- #

def test_raw_organelle_channel_is_remapped_to_dense_stack_index(tmp_path):
    """organelle_channel=3 with only nucleus+organelle enabled -> dense index 1.

    Indexing the compacted 2-channel stack with the raw channel 3 would raise
    IndexError, so a successful run that finds the blobs (which live in dense
    channel 1) proves the remap happened.
    """
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"], channel_specs=("flat", "blob"))

    _run(src, _base_settings(nucleus_channel=1, organelle_channel=3))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert _n_objects(mask) == 4


def test_channel_none_falls_back_to_first_channel_of_4d_stack(tmp_path):
    """organelle_channel=None -> channel 0 (blobs), not the flat channel 1."""
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"], channel_specs=("blob", "flat"))

    _run(src, _base_settings(organelle_channel=None))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert _n_objects(mask) == 4


def test_three_dim_stack_with_channel_set_uses_whole_image(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"], channel_specs=None)

    _run(src, _base_settings(organelle_channel=0))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert mask.shape == (64, 64)
    assert _n_objects(mask) == 4


def test_three_dim_stack_without_channel_uses_whole_image(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"], channel_specs=None)

    _run(src, _base_settings(organelle_channel=None))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert mask.shape == (64, 64)
    assert _n_objects(mask) == 4


# --------------------------------------------------------------------------- #
#  Resume / skip behaviour
# --------------------------------------------------------------------------- #

def test_fully_processed_stack_is_skipped_untouched(tmp_path, capsys):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy", "f2.npy"])
    out_dir = src / "organelle_mask_stack"
    out_dir.mkdir()
    sentinel = np.full((64, 64), 7, dtype=np.uint16)
    for fn in ("f1.npy", "f2.npy"):
        np.save(out_dir / fn, sentinel)

    _run(src, _base_settings())

    assert "already processed" in capsys.readouterr().out
    # nothing was recomputed or overwritten
    assert np.array_equal(np.load(out_dir / "f1.npy"), sentinel)
    assert np.array_equal(np.load(out_dir / "f2.npy"), sentinel)
    assert _object_counts(tmp_path / "measurements" / "measurements.db") is None


def test_partially_processed_stack_still_runs(tmp_path):
    """One missing output is enough to process the stack."""
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy", "f2.npy"])
    out_dir = src / "organelle_mask_stack"
    out_dir.mkdir()
    np.save(out_dir / "f1.npy", np.zeros((64, 64), dtype=np.uint16))

    _run(src, _base_settings())

    assert _n_objects(np.load(out_dir / "f2.npy")) == 4
    rows = _object_counts(tmp_path / "measurements" / "measurements.db")
    assert ("f2.npy", "organelle", 4) in rows


# --------------------------------------------------------------------------- #
#  Per-cell masking
# --------------------------------------------------------------------------- #

def test_mask_within_cells_zeroes_organelles_outside_cells(tmp_path, capsys):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])
    cell_dir = tmp_path / "cell_mask_stack"
    cell_dir.mkdir()
    cell_mask = np.zeros((64, 64), dtype=np.uint16)
    cell_mask[:32, :] = 1                      # only the top half is "inside a cell"
    np.save(cell_dir / "f1.npy", cell_mask)

    _run(src, _base_settings(organelle_mask_within_cells=True))

    assert "Per-cell masking enabled" in capsys.readouterr().out
    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert _n_objects(mask) == 2                # only the two top blobs survive
    assert mask[32:, :].max() == 0
    for cy, cx in TOP_CENTERS:
        assert mask[cy, cx] != 0


def test_mask_within_cells_without_cell_folder_warns_and_continues(tmp_path, capsys):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(organelle_mask_within_cells=True))

    out = capsys.readouterr().out
    assert "no cell_mask_stack found" in out
    # segmentation still ran on the unmasked image
    assert _n_objects(np.load(src / "organelle_mask_stack" / "f1.npy")) == 4


# --------------------------------------------------------------------------- #
#  Preprocessing
# --------------------------------------------------------------------------- #

def test_rolling_ball_and_clahe_preprocessing_still_finds_blobs(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(organelle_rolling_ball=True,
                             organelle_rolling_ball_radius=10,
                             organelle_clahe=True))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    labels = {int(mask[cy, cx]) for cy, cx in BLOB_CENTERS}
    assert 0 not in labels and len(labels) == 4


# --------------------------------------------------------------------------- #
#  Post-processing wiring
# --------------------------------------------------------------------------- #

def test_max_size_filter_empties_masks_and_reports_zero_stats(tmp_path, capsys):
    """Every object is larger than organelle_max_size -> empty stack, 0 counts."""
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(organelle_max_size=5))

    out = capsys.readouterr().out
    assert "Found 0.0 organelle/FOV" in out
    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert mask.max() == 0
    rows = _object_counts(tmp_path / "measurements" / "measurements.db")
    assert rows == [("f1.npy", "organelle", 0)]


def test_remove_border_drops_edge_touching_objects(tmp_path):
    centers = BLOB_CENTERS + ((2, 32),)          # extra blob clipped by row 0

    keep_src = tmp_path / "keep" / "masks"
    _make_npz(str(keep_src), ["f1.npy"], centers=centers)
    _run(keep_src, _base_settings(organelle_remove_border=False))
    kept = np.load(keep_src / "organelle_mask_stack" / "f1.npy")

    drop_src = tmp_path / "drop" / "masks"
    _make_npz(str(drop_src), ["f1.npy"], centers=centers)
    _run(drop_src, _base_settings(organelle_remove_border=True))
    dropped = np.load(drop_src / "organelle_mask_stack" / "f1.npy")

    assert _n_objects(kept) == 5
    assert _n_objects(dropped) == 4
    assert dropped[0, :].max() == 0
    assert kept[2, 32] != 0 and dropped[2, 32] == 0


def test_save_false_keeps_counts_but_writes_no_files(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(save=False))

    assert os.listdir(src / "organelle_mask_stack") == []
    rows = _object_counts(tmp_path / "measurements" / "measurements.db")
    assert rows == [("f1.npy", "organelle", 4)]


# --------------------------------------------------------------------------- #
#  Segmenter returning nothing
# --------------------------------------------------------------------------- #

def test_none_from_segmenter_skips_the_batch(tmp_path, monkeypatch):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])
    monkeypatch.setattr("spacr.object._segment_classical_parallel",
                        lambda *a, **k: None)

    _run(src, _base_settings())

    assert os.listdir(src / "organelle_mask_stack") == []
    assert _object_counts(tmp_path / "measurements" / "measurements.db") is None


def test_empty_list_from_segmenter_skips_the_batch(tmp_path, monkeypatch):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])
    monkeypatch.setattr("spacr.object._segment_classical_parallel",
                        lambda *a, **k: [])

    _run(src, _base_settings())

    assert os.listdir(src / "organelle_mask_stack") == []
    assert _object_counts(tmp_path / "measurements" / "measurements.db") is None


# --------------------------------------------------------------------------- #
#  Classical dispatch / n_jobs
# --------------------------------------------------------------------------- #

def test_classical_settings_subset_is_handed_to_the_workers(tmp_path, monkeypatch):
    seen = {}

    def fake_parallel(img_batch, classical_settings, n_jobs=1):
        seen["settings"] = classical_settings
        seen["n_jobs"] = n_jobs
        seen["shape"] = img_batch.shape
        seen["dtype"] = img_batch.dtype
        return [np.zeros(img_batch.shape[1:], dtype=np.int32) for _ in range(len(img_batch))]

    monkeypatch.setattr("spacr.object._segment_classical_parallel", fake_parallel)
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy", "f2.npy"])

    _run(src, _base_settings(n_jobs=-4))

    assert seen["n_jobs"] == 1                    # clamped from -4
    assert seen["shape"] == (2, 64, 64)
    assert seen["dtype"] == np.float32
    assert seen["settings"]["organelle_morphology"] == "spots"
    assert seen["settings"]["organelle_method"] == "otsu"
    # deep-learning-only keys never reach the pickled worker payload
    assert "organelle_model_name" not in seen["settings"]
    assert "organelle_unet_model_path" not in seen["settings"]


@pytest.mark.parametrize(
    "morphology,method",
    [("network", "hysteresis"), ("irregular", "otsu"), ("ring", "otsu")],
)
def test_other_morphologies_run_end_to_end(tmp_path, morphology, method):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(organelle_morphology=morphology,
                             organelle_method=method,
                             organelle_min_size=5))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint16
    assert _n_objects(mask) >= 1
    assert mask[0, 0] == 0


# --------------------------------------------------------------------------- #
#  Cellpose branch
# --------------------------------------------------------------------------- #

class _FakeCellposeModel:
    """Stands in for a CellposeModel; records the eval kwargs it was given."""

    def __init__(self, masks):
        self._masks = masks
        self.calls = []

    def eval(self, **kwargs):
        self.calls.append(kwargs)
        n = len(kwargs["x"])
        masks = [self._masks[i] for i in range(n)]
        flows = [np.zeros((3,) + m.shape, dtype=np.float32) for m in masks]
        return masks, flows, None, None


def _canned_masks(n=2):
    """n label images with two square objects each (areas 100 and 64)."""
    out = []
    for _ in range(n):
        m = np.zeros((64, 64), dtype=np.int32)
        m[5:15, 5:15] = 1
        m[40:48, 40:48] = 2
        out.append(m)
    return out


def test_cellpose_method_loads_model_once_and_uses_its_masks(tmp_path, monkeypatch):
    chosen = {}
    fake_model = _FakeCellposeModel(_canned_masks(2))

    def fake_choose_model(model_name, device, object_type=None, restore_type=None,
                          object_settings=None):
        chosen["model_name"] = model_name
        chosen["object_type"] = object_type
        chosen["object_settings"] = object_settings
        chosen["device"] = device
        return fake_model

    monkeypatch.setattr("spacr.utils._choose_model", fake_choose_model)

    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy", "f2.npy"])
    _run(src, _base_settings(organelle_method="cellpose",
                             organelle_model_name="cpsam"))

    # _choose_model got the object_settings built by _build_object_settings
    assert chosen["model_name"] == "cpsam"
    assert chosen["object_type"] == "organelle"
    assert chosen["object_settings"]["model_name"] == "cpsam"
    assert chosen["object_settings"]["diameter"] == 30
    assert chosen["object_settings"]["merge"] is False
    assert str(chosen["device"]) == "cpu"      # _force_cpu fixture

    # one eval call for the single batch, with the organelle thresholds
    assert len(fake_model.calls) == 1
    kwargs = fake_model.calls[0]
    assert len(kwargs["x"]) == 2
    assert kwargs["batch_size"] == 2
    assert kwargs["normalize"] is False
    assert kwargs["channel_axis"] == -1
    assert kwargs["diameter"] is None
    assert kwargs["flow_threshold"] == 0.4
    assert kwargs["cellprob_threshold"] == 0.0
    assert kwargs["resample"] is True

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert _n_objects(mask) == 2
    assert mask[10, 10] != 0 and mask[44, 44] != 0
    assert mask[30, 30] == 0
    rows = _object_counts(tmp_path / "measurements" / "measurements.db")
    assert rows == [("f1.npy", "organelle", 2), ("f2.npy", "organelle", 2)]


def test_cellpose_model_is_loaded_once_for_several_stacks(tmp_path, monkeypatch):
    calls = []
    fake_model = _FakeCellposeModel(_canned_masks(2))

    def fake_choose_model(model_name, device, **kwargs):
        calls.append(model_name)
        return fake_model

    monkeypatch.setattr("spacr.utils._choose_model", fake_choose_model)

    src = tmp_path / "masks"
    _make_npz(str(src), ["a.npy"], name="stack1")
    _make_npz(str(src), ["b.npy"], name="stack2")
    _run(src, _base_settings(organelle_method="cellpose"))

    assert len(calls) == 1                      # loaded outside the file loop
    assert len(fake_model.calls) == 2           # but evaluated per batch


# --------------------------------------------------------------------------- #
#  U-Net branch
# --------------------------------------------------------------------------- #

def _save_tiny_unet(path, out_channels=1, weight=5.0):
    """A 1x1 conv that thresholds z-scored intensity: sigmoid(w*x) > 0.5 <=> x > 0."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = nn.Sequential(nn.Conv2d(1, out_channels, kernel_size=1, bias=True))
    with torch.no_grad():
        model[0].weight.fill_(0.0)
        model[0].weight[0, 0, 0, 0] = weight
        if out_channels > 1:
            model[0].weight[1, 0, 0, 0] = -weight
        model[0].bias.fill_(0.0)
    torch.save(model, str(path))
    return str(path)


def test_unet_method_segments_with_the_loaded_model(tmp_path):
    model_path = _save_tiny_unet(tmp_path / "unet.pt")
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(organelle_morphology="network",
                             organelle_method="unet",
                             organelle_unet_model_path=model_path))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    assert _n_objects(mask) == 4
    assert mask[0, 0] == 0
    for cy, cx in BLOB_CENTERS:
        assert mask[cy, cx] != 0


def test_unet_multichannel_logits_and_skeletonize(tmp_path):
    model_path = _save_tiny_unet(tmp_path / "unet2.pt", out_channels=2)
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(organelle_morphology="network",
                             organelle_method="unet",
                             organelle_skeletonize=True,
                             organelle_min_size=1,
                             organelle_unet_model_path=model_path))

    mask = np.load(src / "organelle_mask_stack" / "f1.npy")
    # skeleton + 1-px dilation: one thin component per blob, much smaller
    # than the original 29-px disks.
    assert _n_objects(mask) == 4
    assert 0 < int((mask > 0).sum()) < 4 * 29


def test_unet_without_model_path_raises(tmp_path):
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    with pytest.raises(ValueError, match="organelle_unet_model_path"):
        _run(src, _base_settings(organelle_morphology="network",
                                 organelle_method="unet"))
    assert not (src / "organelle_mask_stack" / "f1.npy").exists()


# --------------------------------------------------------------------------- #
#  Plotting
# --------------------------------------------------------------------------- #

def test_plot_true_forwards_batch_and_masks_to_plot_organelle_output(tmp_path, monkeypatch):
    calls = []

    def spy(img_batch, masks, settings, **kwargs):
        calls.append((img_batch, masks, settings, kwargs))

    monkeypatch.setattr("spacr.plot.plot_organelle_output", spy)
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy", "f2.npy"])

    _run(src, _base_settings(plot=True, examples_to_plot=5))

    assert len(calls) == 1
    img_batch, masks, settings, kwargs = calls[0]
    assert img_batch.shape == (2, 64, 64)
    assert len(masks) == 2
    assert settings["organelle_morphology"] == "spots"
    assert kwargs["nr"] == 2                     # min(examples_to_plot, n_masks)
    assert kwargs["cmap"] == "inferno"
    assert kwargs["figuresize"] == 10
    assert kwargs["print_object_number"] is True


def test_plot_true_renders_a_real_figure(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy"])

    _run(src, _base_settings(plot=True))

    assert len(plt.get_fignums()) == 1
    assert _n_objects(np.load(src / "organelle_mask_stack" / "f1.npy")) == 4


# --------------------------------------------------------------------------- #
#  Settings helpers
# --------------------------------------------------------------------------- #

def test_build_object_settings_maps_organelle_keys():
    from spacr.object import _build_object_settings

    settings = {
        "organelle_model_name": "cpsam",
        "organelle_diameter": 17,
        "organelle_min_size": 3,
        "organelle_max_size": 900,
        "organelle_resample": False,
        "organelle_remove_border": True,
    }
    out = _build_object_settings(settings)
    assert out == {
        "model_name": "cpsam",
        "diameter": 17,
        "minimum_size": 3,
        "maximum_size": 900,
        "resample": False,
        "filter_size": False,
        "filter_intensity": False,
        "remove_border_objects": True,
        "merge": False,
    }


# --------------------------------------------------------------------------- #
#  generate_cellpose_masks (the legacy sibling whose save/cleanup tail shares
#  this line region) — currently unreachable, see the xfail reason.
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True,
                   reason="BUG: generate_cellpose_masks calls "
                          "utils._get_cellpose_channels(src, nucleus, pathogen, cell) "
                          "but that helper takes a single settings dict and returns "
                          "(channels_to_extract, cellpose_channels) -> TypeError on "
                          "every call, so the whole function is dead code")
def test_generate_cellpose_masks_saves_masks_and_counts(tmp_path, monkeypatch):
    from spacr.object import generate_cellpose_masks
    from spacr.settings import set_default_settings_preprocess_generate_masks

    fake_model = _FakeCellposeModel(_canned_masks(2))
    monkeypatch.setattr("spacr.utils._choose_model", lambda *a, **k: fake_model)

    src = tmp_path / "masks"
    _make_npz(str(src), ["f1.npy", "f2.npy"])

    settings = set_default_settings_preprocess_generate_masks({
        "src": str(src),
        "nucleus_channel": 0, "cell_channel": None,
        "pathogen_channel": None, "organelle_channel": None,
        "batch_size": 2, "plot": False, "save": True, "verbose": False,
        "filter": False, "timelapse": False,
    })

    generate_cellpose_masks(str(src), settings, "nucleus")

    out_dir = src / "nucleus_mask_stack"
    assert sorted(os.listdir(out_dir)) == ["f1.npy", "f2.npy"]
    assert np.load(out_dir / "f1.npy").dtype == np.uint16
    rows = _object_counts(tmp_path / "measurements" / "measurements.db")
    assert ("f1.npy", "nucleus_before_filtration", 2) in rows


def test_default_organelle_model_name_is_a_model_not_a_docstring():
    """The tooltip text had been pasted in as the value, so the default
    organelle model name was a paragraph of prose. _choose_model would have
    treated it as an unknown name and quietly used cpsam anyway — which is
    what it happens to want, but for the wrong reason and with no warning."""
    from spacr.settings import _set_organelle_defaults
    from spacr.object import _build_object_settings

    filled = _set_organelle_defaults({})
    assert filled["organelle_model_name"] == "cpsam"
    assert _build_object_settings(filled)["model_name"] == "cpsam"
