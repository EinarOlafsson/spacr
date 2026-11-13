"""CPU-only coverage for the tail of :mod:`spacr.io` (``read_plot_model_stats``
onwards).

Everything in this module runs offline on synthetic SQLite databases, tiny
TIFFs and stub reader objects — no GPU, no network, no Cellpose weights.
The focus is the branches the rest of the suite never reaches:

* the checkpoint / progress writers (``_save_model``, ``_save_progress``)
* ``_read_db`` chunk assembly and its identifier guard
* every single-table path through ``_read_and_merge_data``
* the ``generate_dataset`` / ``generate_loaders`` guard rails
* every mode + helper of ``generate_training_dataset``
* the Yokogawa converters, including well overflow onto a second plate and
  the ND2 / CZI / LIF vendor branches (driven through stub readers)
* ``prepare_cellpose_dataset`` augmentation maths

"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io as IO  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _headless_figures(monkeypatch):
    """No blocking windows, no figure leaks."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _tiny_model():
    torch = pytest.importorskip("torch")
    return torch.nn.Linear(2, 2)


# ---------------------------------------------------------------------------
# read_plot_model_stats
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _save_model
# ---------------------------------------------------------------------------

def test_save_model_defaults_and_below_every_threshold(tmp_path):
    """intermedeate_save=None/channels=None fill in defaults; a low accuracy
    saves nothing and returns None."""
    model = _tiny_model()
    out = IO._save_model(model, "cnn", {"accuracy": 0.10}, str(tmp_path),
                         epoch=3, epochs=10,
                         intermedeate_save=None, channels=None)
    assert out is None
    assert list(tmp_path.iterdir()) == []


def test_save_model_uses_val_dict_not_train_dict(tmp_path):
    """A memorising train accuracy must not trigger a checkpoint; the
    validation dict is the signal."""
    model = _tiny_model()
    assert IO._save_model(model, "cnn", {"accuracy": 0.999}, str(tmp_path),
                          epoch=3, epochs=10,
                          val_dict={"accuracy": 0.10}) is None
    assert list(tmp_path.iterdir()) == []

    path = IO._save_model(model, "cnn", {"accuracy": 0.0}, str(tmp_path),
                          epoch=3, epochs=10,
                          val_dict={"accuracy": 0.995})
    assert path is not None and os.path.isfile(path)
    assert os.path.basename(path) == "cnn_epoch_3_acc_99.5000_channels_rgb.pth"


# ---------------------------------------------------------------------------
# _save_progress
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _read_db
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _read_and_merge_data
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _read_mask / convert_numpy_to_tiff
# ---------------------------------------------------------------------------

def test_read_mask_upcasts_non_uint16(tmp_path):
    """A uint8 mask is promoted to uint16 by img_as_uint (values rescaled)."""
    p = tmp_path / "m.tif"
    mask = np.zeros((8, 8), np.uint8)
    mask[2:5, 2:5] = 3
    tifffile.imwrite(str(p), mask)

    out = IO._read_mask(str(p))
    assert out.dtype == np.uint16
    assert (out > 0).sum() == 9
    assert out.max() == 3 * 257  # skimage's uint8 -> uint16 scaling


def test_convert_numpy_to_tiff_honours_limit(tmp_path, rng):
    """limit stops the conversion loop early."""
    folder = tmp_path / "npys"
    folder.mkdir()
    for i in range(4):
        np.save(folder / f"a{i}.npy", rng.integers(0, 500, (4, 4)).astype(np.uint16))

    IO.convert_numpy_to_tiff(str(folder), limit=1)
    tiffs = list((folder / "tiff").glob("*.tif"))
    assert len(tiffs) == 1


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_loaders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# generate_training_dataset
# ---------------------------------------------------------------------------

N = 40


# ---------------------------------------------------------------------------
# training_dataset_from_annotation_metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# convert_separate_files_to_yokogawa
# ---------------------------------------------------------------------------

_SEP_REGEX = (r"W(?P<wellID>\d+)_F(?P<fieldID>\d+)_T(?P<timeID>\d+)"
              r"_C(?P<chanID>\d+)_Z(?P<sliceID>\d+)")


# ---------------------------------------------------------------------------
# convert_to_yokogawa
# ---------------------------------------------------------------------------


# --- vendor formats: ND2 / CZI / LIF, driven through stub readers ----------


# ---------------------------------------------------------------------------
# prepare_cellpose_dataset
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# make_validation_holdout -- the seed moves the split, misses are announced
# ---------------------------------------------------------------------------


