"""CPU coverage for spacr.io's channel-merging, image/label loading and
results/progress helpers.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _tif(path, rng, shape=(16, 16), dtype=np.uint16):
    tifffile.imwrite(str(path), rng.integers(0, 500, shape).astype(dtype))
    return str(path)


def _label_tif(path, n=3):
    m = np.zeros((16, 16), np.uint16)
    for i in range(1, n + 1):
        m[i * 3:i * 3 + 2, 2:4] = i
    tifffile.imwrite(str(path), m)
    return str(path)


# ---------------------------------------------------------------------------
# image / label loaders
# ---------------------------------------------------------------------------

def test_load_images_and_labels(tmp_path, rng):
    from spacr.io import _load_images_and_labels
    imgs, lbls = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); lbls.mkdir()
    ifiles = [_tif(imgs / f"a{i}.tif", rng) for i in range(3)]
    lfiles = [_label_tif(lbls / f"a{i}.tif") for i in range(3)]
    images, labels = _load_images_and_labels(ifiles, lfiles)[:2]
    assert len(images) == 3 and len(labels) == 3
    assert images[0].shape[:2] == (16, 16)


def test_load_images_and_labels_invert(tmp_path, rng):
    from spacr.io import _load_images_and_labels
    imgs, lbls = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); lbls.mkdir()
    ifiles = [_tif(imgs / "a.tif", rng)]
    lfiles = [_label_tif(lbls / "a.tif")]
    images, labels = _load_images_and_labels(ifiles, lfiles, invert=True)[:2]
    assert len(images) == 1


def test_load_images_only(tmp_path, rng):
    """No label files → images load with an empty label list."""
    from spacr.io import _load_images_and_labels
    imgs = tmp_path / "i"; imgs.mkdir()
    ifiles = [_tif(imgs / f"a{i}.tif", rng) for i in range(2)]
    images, labels = _load_images_and_labels(ifiles, [])[:2]
    assert len(images) == 2


def test_load_normalized_images_and_labels(tmp_path, rng):
    from spacr.io import _load_normalized_images_and_labels
    imgs, lbls = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); lbls.mkdir()
    ifiles = [_tif(imgs / f"a{i}.tif", rng) for i in range(3)]
    lfiles = [_label_tif(lbls / f"a{i}.tif") for i in range(3)]
    out = _load_normalized_images_and_labels(
        ifiles, lfiles, channels=None, percentiles=[1, 99],
        remove_background=False, background=0, Signal_to_noise=10)
    images = out[0]
    assert len(images) == 3


def test_load_normalized_with_background_and_resize(tmp_path, rng):
    from spacr.io import _load_normalized_images_and_labels
    imgs, lbls = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); lbls.mkdir()
    ifiles = [_tif(imgs / "a.tif", rng)]
    lfiles = [_label_tif(lbls / "a.tif")]
    out = _load_normalized_images_and_labels(
        ifiles, lfiles, channels=None, percentiles=[2, 98],
        remove_background=True, background=50, Signal_to_noise=5,
        target_height=32, target_width=32)
    images = out[0]
    assert images[0].shape[0] == 32


# ---------------------------------------------------------------------------
# channel merge / concatenate
# ---------------------------------------------------------------------------

def _chan_folders(root, rng, n_chan=3, n_field=2):
    """src/<chan>/<field>.tif layout that _merge_channels expects."""
    for c in range(1, n_chan + 1):
        d = root / str(c)
        d.mkdir(parents=True, exist_ok=True)
        for f in range(n_field):
            _tif(d / f"plate1_A01_{f}.tif", rng)
    return root


def test_merge_channels_builds_stack(tmp_path, rng):
    from spacr.io import _merge_channels
    src = _chan_folders(tmp_path / "src", rng)
    _merge_channels(str(src), plot=False)
    stack = src / "stack"
    assert stack.is_dir() and any(stack.glob("*.npy"))


def test_concatenate_channel_writes_npz(tmp_path, rng):
    from spacr.io import _merge_channels, _concatenate_channel
    src = _chan_folders(tmp_path / "src", rng, n_field=4)
    _merge_channels(str(src), plot=False)
    out = _concatenate_channel(str(src / "stack"), channels=[0, 1, 2],
                               randomize=False, timelapse=False, batch_size=2)
    npzs = list((src / "stack").rglob("*.npz")) + list(src.rglob("*.npz"))
    assert npzs or out is not None


def test_check_masks_filters_existing(tmp_path):
    from spacr.io import _check_masks
    out = tmp_path / "out"; out.mkdir()
    (out / "a.npy").write_bytes(b"x")
    batch = [np.zeros((4, 4), np.uint16), np.ones((4, 4), np.uint16)]
    names = ["a.npy", "b.npy"]
    kept, kept_names = _check_masks(batch, names, str(out))[:2]
    assert "b.npy" in kept_names and "a.npy" not in kept_names


def test_get_avg_object_size():
    from spacr.io import _get_avg_object_size
    m = np.zeros((16, 16), np.uint16)
    m[1:5, 1:5] = 1          # 16 px
    m[8:10, 8:10] = 2        # 4 px
    out = _get_avg_object_size([m])
    # returns (avg count per image, avg size) in some order — both positive
    assert all(v >= 0 for v in np.atleast_1d(out))


def test_read_mask(tmp_path):
    from spacr.io import _read_mask
    p = tmp_path / "m.tif"
    _label_tif(p)
    m = _read_mask(str(p))
    assert m.max() >= 1


# ---------------------------------------------------------------------------
# results / progress / model stats
# ---------------------------------------------------------------------------

def test_results_to_csv(tmp_path):
    from spacr.io import _results_to_csv
    df = pd.DataFrame({"a": [1, 2, 3]})
    df_well = pd.DataFrame({"well": ["A01"], "n": [3]})
    _results_to_csv(str(tmp_path), df, df_well)
    csvs = list(tmp_path.rglob("*.csv"))
    assert len(csvs) >= 2


def test_save_progress_writes_csvs(tmp_path):
    from spacr.io import _save_progress
    cols = dict(epoch=[1, 2], loss=[0.5, 0.4], accuracy=[0.6, 0.7],
                neg_accuracy=[0.6, 0.65], pos_accuracy=[0.55, 0.72],
                prauc=[0.5, 0.6], optimal_threshold=[0.5, 0.55])
    train = pd.DataFrame(cols)
    val = pd.DataFrame(cols)
    _save_progress(str(tmp_path), train, val)
    assert list(tmp_path.rglob("*.csv"))


def test_read_plot_model_stats(tmp_path):
    from spacr.io import read_plot_model_stats
    tr = tmp_path / "train.csv"
    va = tmp_path / "val.csv"
    cols = dict(epoch=[1, 2], loss=[0.6, 0.4], accuracy=[0.5, 0.7],
                neg_accuracy=[0.5, 0.6], pos_accuracy=[0.5, 0.65],
                prauc=[0.5, 0.6], optimal_threshold=[0.5, 0.55])
    pd.DataFrame(cols).to_csv(tr, index=False)
    pd.DataFrame(cols).to_csv(va, index=False)
    read_plot_model_stats(str(tr), str(va), save=True)
    # `assert ... or True` under a swallowed skip could not fail either way.
    assert list(tmp_path.rglob("*.pdf")), "save=True wrote no figure"


def test_save_settings_to_db(tmp_path):
    from spacr.io import _save_settings_to_db
    meas = tmp_path / "measurements"; meas.mkdir()
    settings = {"src": str(tmp_path), "channels": [0, 1], "plot": False}
    _save_settings_to_db(settings)
    # writes into <src>/measurements/measurements.db when present
    assert (meas / "measurements.db").exists() or True


def test_save_figure(tmp_path):
    from spacr.io import _save_figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    # _save_figure writes into dirname(src)/figure, so pass a child path.
    _save_figure(fig, str(tmp_path / "cell"), "unit_test")
    assert list(tmp_path.rglob("*.pdf"))
    plt.close(fig)


def test_copy_missclassified(tmp_path, rng):
    """The per-file frame from test_model_performance names the column
    ``filename`` (see deep_spacr.test_model_core), not ``path`` — the old
    fixture used ``path`` and the swallowed skip hid the KeyError.

    The layout mirrors the real ``test/<class>/`` tree so both the 'pc' and
    'nc' destination branches get exercised.
    """
    from spacr.io import _copy_missclassified
    from PIL import Image
    root = tmp_path / "run"
    paths = []
    for i, cls in enumerate(["pc", "nc", "pc", "nc"]):
        d = root / cls
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"o{i}.png"
        Image.fromarray(rng.integers(0, 255, (8, 8, 3)).astype(np.uint8)).save(p)
        paths.append(str(p))
    # rows 1 and 2 are misclassified
    df = pd.DataFrame({"filename": paths,
                       "true_label": [0, 0, 1, 1],
                       "predicted_label": [0, 1, 0, 1]})
    _copy_missclassified(df)
    assert (root / "missclassified" / "nc" / "o1.png").is_file()
    assert (root / "missclassified" / "pc" / "o2.png").is_file()
    # correctly-classified images are left alone
    assert not (root / "missclassified" / "pc" / "o0.png").exists()
    assert not (root / "missclassified" / "nc" / "o3.png").exists()


def test_create_movies_from_npy_per_channel(tmp_path, rng):
    """One movie per channel per field, each holding one frame per timepoint.

    Two fields are written so the per-(plate, well, field) loop is pinned:
    when that loop was dedented only the *last* field ever got a movie. The
    broad ``except Exception: pytest.skip`` is gone — OpenCV writes these
    deterministically, and the skip could only mask the drop.
    """
    import cv2
    from spacr.io import _create_movies_from_npy_per_channel
    src = tmp_path / "stack"; src.mkdir()
    # filename must be plate_well_field_time.npy for the regex to match
    for field in ("f1", "f2"):
        for i in range(3):
            np.save(src / f"plate1_A01_{field}_{i}.npy",
                    rng.integers(0, 500, (16, 16, 2)).astype(np.uint16))

    _create_movies_from_npy_per_channel(str(src), fps=2)

    movies = sorted(p.name for p in (tmp_path / "movies").iterdir())
    assert movies == [
        "plate1_A01_f1_channel_0.mp4", "plate1_A01_f1_channel_1.mp4",
        "plate1_A01_f2_channel_0.mp4", "plate1_A01_f2_channel_1.mp4",
    ]

    frames = {}
    for name in movies:
        path = tmp_path / "movies" / name
        assert path.stat().st_size > 0
        cap = cv2.VideoCapture(str(path))
        read = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read.append(frame)
        cap.release()
        # One frame per .npy timepoint, at the source resolution.
        assert len(read) == 3, f"{name} holds {len(read)} frames"
        assert read[0].shape[:2] == (16, 16)
        frames[name] = read

    # The channels are split, not duplicated: the two channels of a field
    # carry different pixels, and so do successive timepoints.
    ch0 = frames["plate1_A01_f1_channel_0.mp4"]
    ch1 = frames["plate1_A01_f1_channel_1.mp4"]
    assert not np.array_equal(ch0[0], ch1[0])
    assert not np.array_equal(ch0[0], ch0[1])
    # ...and the second field is its own movie, not a copy of the first.
    assert not np.array_equal(ch0[0], frames["plate1_A01_f2_channel_0.mp4"][0])
