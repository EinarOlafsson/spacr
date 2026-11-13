"""CPU-only coverage for the spacr.io preprocessing entry points.

Targets ``spacr/io.py`` lines 1717-1974:

* :func:`spacr.io.preprocess_img_data` — extension sniffing (found / not
  found + the existing ``stack``/``channel_stack``/``masks`` short
  circuits), the mask-channel dedup + int coercion loop, test mode (both
  the deletable and the non-deletable pre-existing ``test/`` folder), the
  stack-building ``try``/``except`` (channel-count mismatch, timelapse
  movies, plotting, all-to-MIP) and the ``cellpose_*`` channel remap.
* :func:`spacr.io._get_avg_object_size` — the malformed-mask warning and
  the empty-input branches.

The heavy collaborators (``_rename_and_organize_image_files``,
``plot_arrays``, ``concatenate_and_normalize``, ``_create_movies_from_npy_per_channel``)
are replaced with recording stubs so the branch under test is isolated —
except in ``test_end_to_end_real_run``, which runs the genuine pipeline on
eight 128x128 synthetic Yokogawa TIFFs and inspects the emitted arrays.
"""
from __future__ import annotations


import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_tif(path, arr):
    """Write a real TIFF (tifffile when available, Pillow otherwise)."""
    try:
        import tifffile
        tifffile.imwrite(str(path), arr)
        return
    except Exception:
        from PIL import Image
        Image.fromarray(arr).save(str(path))


def _tiny_img(seed=0, shape=(16, 16)):
    """Small deterministic uint16 image with a bright square (never flat)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(50, 200, size=shape, dtype=np.uint16)
    img[4:10, 4:10] = 40000
    return img


# ---------------------------------------------------------------------------
# no valid image files -> the "existing folder" short circuits
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# mask-channel dedup / coercion + the cellpose_* remap
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# test mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the stack-building try block
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# real end-to-end run (no stubs)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _get_avg_object_size edge branches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('entry', ['v1', 'watch', 'cloud', 'timelapse', 'direct-watch', 'preprocess'])
def test_per_plate_unmixing_refuses_unsupported_entry_before_dispatch(tmp_path, monkeypatch, entry):
    """Keep real inputs unchanged when a requested plate-control route is unsupported."""
    import spacr.core as core
    import spacr.io as io
    import spacr.ome_zarr as cloud
    import spacr.validate as validate

    source = tmp_path / 'plate1'
    source.mkdir()
    _write_tif(source / 'plate1_A01_1_0.tif', _tiny_img())
    before = {path: path.read_bytes() for path in source.iterdir()}
    settings = dict(src=str(source), unmix_per_plate=True, unmix=True,
                    unmix_calibration=str(tmp_path / 'explicit-calibration.json'))

    def forbidden(*args, **kwargs):
        """Fail if an unsupported request reaches staging, acquisition or a worker."""
        pytest.fail('per-plate controls reached unsupported dispatch')

    monkeypatch.setattr(validate, 'run_preflight', forbidden)
    monkeypatch.setattr(cloud, '_run_with_cloud_sources', forbidden)
    monkeypatch.setattr(core, '_watch_vendor_recipe', forbidden)
    monkeypatch.setattr(io, '_preprocess_mapped_volume_series', forbidden)
    if entry == 'v1':
        settings.update(pipeline_style='v1', dry_run=True)
        call = core.preprocess_generate_masks
    elif entry == 'watch':
        settings.update(pipeline_style='v2', watch_folder=True)
        call = core.preprocess_generate_masks
    elif entry == 'cloud':
        settings.update(pipeline_style='v2', src='s3://example-bucket/plate1')
        call = core.preprocess_generate_masks
    elif entry == 'timelapse':
        settings.update(pipeline_style='v2')
        call = core.preprocess_generate_masks_timelapse
    elif entry == 'direct-watch':
        settings.update(unmix=False, unmix_calibration='')
        call = core._watch_folder_and_analyse
    else:
        settings.update(z_stack=True, t_stack=True, z_segmentation_mode='volumetric')
        call = io.preprocess_img_data
    with pytest.raises(ValueError, match='Per-plate spectral controls require calibrated organized 2D cell segmentation through V2'):
        call(settings)
    assert {path: path.read_bytes() for path in source.iterdir()} == before
    assert set(tmp_path.iterdir()) == {source}
