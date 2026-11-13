"""Organelle settings helpers: ``_build_object_settings`` and the default model name.

Also holds the synthetic blob-field builders and ``_base_settings`` that
``tests/test_organelle_raw_intensity_filtering.py`` and
``tests/test_pipeline_mask_label_capacity.py`` import.  The organelle mask
generator itself is tested in ``tests/test_object_organelle_masks.py``.
"""
from __future__ import annotations

import os

import numpy as np


# Four well-separated blobs -> four objects for every classical method.
BLOB_CENTERS = ((16, 16), (16, 48), (48, 16), (48, 48))


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


# --------------------------------------------------------------------------- #
#  Settings helpers
# --------------------------------------------------------------------------- #

def test_build_object_settings_maps_organelle_keys():
    from spacr.object import _build_object_settings

    settings = {
        "organelle_model_name": "cpsam",
        "organelle_diameter": 17,
        "organelle_min_area": 3,
        "organelle_max_area": 900,
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
#  this line region). It used to call utils._get_cellpose_channels with four
#  positional arguments left over from an older signature and died with a
#  TypeError on every call; that is fixed, the xfail(strict=True) marker that
#  documented it has been retired, and the full behaviour suite for this
#  generator now lives in tests/test_cov_object_cellpose_masks.py.
# --------------------------------------------------------------------------- #

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
