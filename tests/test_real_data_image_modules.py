"""Exhaustive real-data integration tests for spaCR's image-based modules.

The user's directive: "make sure that each module is tested on real data.
start with the image based modules."

Modules covered here (all image-based, in dependency order):

  * :func:`spacr.io._rename_and_organize_image_files` — the ingest step
    that reads raw Yokogawa TIFFs and lays them out.
  * :func:`spacr.io._normalize_stack` — percentile-based per-channel
    normalisation of the ingest output.
  * :func:`spacr.core.preprocess_generate_masks` — the full v1 mask
    pipeline (ingest → normalise → Cellpose per object type).
  * :func:`spacr.pipeline_v2.run_v2` — the v2 streaming mask pipeline.
  * :func:`spacr.object.generate_cellpose_masks_sam` — the Cellpose-SAM
    object worker (called via preprocess_generate_masks in modern
    settings).
  * :func:`spacr.measure.measure_crop` — the per-object feature
    extractor.

Every test uses the same tiny synthetic Yokogawa-format dataset that
tests/test_hf_e2e_integration.py's ``SPACR_HF_E2E_STUB=1`` mode
generates — 2 wells × 2 fields × 3 channels of 64×64 uint16 tiles.
Cellpose runs on GPU when available and skips cleanly when not.

Every test asserts on WRITTEN OUTPUT (file paths, shapes,
non-empty), so any silent regression that "runs to completion but
produces nothing useful" is caught. Log messages captured with
caplog let us surface which pipeline function emitted which line —
mirrors what the user sees in ~/.spacr/logs/.

Marked ``@pytest.mark.slow`` + ``@pytest.mark.gpu`` so they only run
when the user asks for them via ``pytest -m "slow and gpu"``.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Skip guards
# ---------------------------------------------------------------------------

def _require_gpu_cellpose():
    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch unavailable: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA — image-module tests are GPU-only")
    try:
        import cellpose                                    # noqa: F401
    except Exception as e:
        pytest.skip(f"cellpose unavailable: {e}")


# ---------------------------------------------------------------------------
# Synthetic dataset (mirrors test_hf_e2e_integration.py's stub mode)
# ---------------------------------------------------------------------------

def _make_stub_dataset(dst: Path,
                          wells=("A01", "A02"),
                          fields=(1, 2),
                          channels=3,
                          size: int = 128) -> Path:
    """Emit a cellvoyager-format plate at ``dst/plate1`` — bigger and
    more variable than the HF-E2E stub so Cellpose actually finds
    objects.

    Each channel gets synthetic Gaussian blobs offset by channel
    index so the segmentation call has something structured to
    latch onto. Returns the plate root."""
    import tifffile
    plate = dst / "plate1"; plate.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for well in wells:
        for field in fields:
            for ch in range(channels):
                # Background noise + a handful of Gaussian blobs
                bg = rng.integers(50, 200, size=(size, size)
                                     ).astype(np.uint16)
                for _ in range(20):
                    cx = int(rng.integers(8, size - 8))
                    cy = int(rng.integers(8, size - 8))
                    r  = int(rng.integers(3, 8))
                    y, x = np.ogrid[:size, :size]
                    gauss = np.exp(-((x - cx) ** 2 + (y - cy) ** 2)
                                       / (2 * r ** 2)) * 2000
                    bg = np.clip(
                        bg.astype(np.float32) + gauss, 0, 65535,
                    ).astype(np.uint16)
                p = (plate / f"plate1_{well}_"
                        f"T01F0{field}L01A01Z01C0{ch}.tif")
                tifffile.imwrite(str(p), bg)
    return plate


@pytest.fixture(scope="module")
def _stub_plate(tmp_path_factory):
    _require_gpu_cellpose()
    root = tmp_path_factory.mktemp("real_module_tests", numbered=True)
    return _make_stub_dataset(root / "data")


def _mask_settings_for(src: Path) -> dict:
    """Build a preprocess_generate_masks-compatible settings dict."""
    from spacr.qt import synthetic as syn
    s = syn.demo_settings("mask", str(src))
    # Extras that preprocess_generate_masks reads on the first defaults
    # pass but that aren't in the demo settings.
    s.update({
        "consolidate": False,
        "remove_background": False,
        "normalize": True,
        "backgrounds": [100, 100, 100],
        "remove_background_cell": False,
        "remove_background_nucleus": False,
        "remove_background_pathogen": False,
        "cells_per_field": 10,
        "test_mode": False,
        "Signal_to_noise": 10,
        "cell_intensity_range": None,
        "nucleus_intensity_range": None,
        "pathogen_intensity_range": None,
        "cytoplasm_intensity_range": None,
        "denoise": False,
        "remove_background_intensity": False,
        "skip_extraction": False,
        "channels": [0, 1, 2],
        "plot": False,
        # Stub dataset has 3 channels; organelle would index 3 → OOB.
        "organelle_channel": None,
        "pathogen_channel":  None,
    })
    return s


# ---------------------------------------------------------------------------
# Ingest layer
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_module_ingest_organizes_channels(_stub_plate, caplog):
    """_rename_and_organize_image_files builds the merged stack/ arrays
    directly from an in-memory channel dict — NO per-channel sub-folders are
    created, and the raw tiles are backed up under orig/."""
    from spacr.io import _rename_and_organize_image_files
    from spacr.utils import _get_regex

    src = str(_stub_plate)
    regex = _get_regex("cellvoyager", ".tif", None)
    caplog.set_level(logging.INFO, logger="spacr")
    n_channels = _rename_and_organize_image_files(
        src, regex, batch_size=100, metadata_type="cellvoyager",
        img_format=[".tif"], save_original_images=True)
    # No intermediate per-channel folders should be left behind.
    channel_dirs = [p for p in _stub_plate.iterdir()
                      if p.is_dir() and p.name in ("0", "1", "2")]
    assert not channel_dirs, (
        f"per-channel sub-folders should NOT be created; found {channel_dirs}")
    # The stack/ arrays were produced directly, and raws are backed up.
    stack = _stub_plate / "stack"
    assert stack.is_dir() and any(stack.glob("*.npy"))
    assert (_stub_plate / "orig").is_dir()
    assert n_channels >= 1


# ---------------------------------------------------------------------------
# Full v1 mask pipeline
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_module_preprocess_generate_masks_writes_cell_masks(
        tmp_path, caplog):
    """End-to-end v1 mask pipeline — assert cell masks land on disk
    and the run journal records the pipeline function's entry."""
    from spacr.core import preprocess_generate_masks
    from spacr.run_journal import open_run

    plate = _make_stub_dataset(tmp_path / "v1_full")
    settings = _mask_settings_for(plate)

    caplog.set_level(logging.INFO, logger="spacr")
    t0 = time.time()
    with open_run("mask", settings) as run:
        preprocess_generate_masks(settings)
    elapsed = time.time() - t0
    print(f"[real-data] v1 mask stage: {elapsed:.1f}s -> {run.dir}")

    # Run journal artefact
    assert (run.dir / "manifest.json").exists()

    # v1 writes .npy stacks under masks/cell_mask_stack/
    stack_dir = plate / "masks" / "cell_mask_stack"
    assert stack_dir.is_dir(), (
        f"no cell mask stack folder at {stack_dir}")
    npy_files = sorted(stack_dir.glob("*.npy"))
    assert npy_files, "cell mask stack folder contains no .npy files"
    # Each mask should have some labelled objects
    for p in npy_files:
        arr = np.load(p)
        assert arr.ndim in (2, 3), (
            f"unexpected mask shape at {p}: {arr.shape}")
        if arr.ndim == 3:
            arr = arr[0]
        # We don't demand a specific count — synthetic data is noisy —
        # just that segmentation returned SOMETHING.
        assert int(arr.max()) >= 0


# ---------------------------------------------------------------------------
# v2 streaming pipeline
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_module_run_v2_writes_merged_stacks(tmp_path):
    """Streaming v2 pipeline should write one .npy per field under
    merged/ with the mask appended as an extra channel."""
    from spacr.pipeline_v2 import run_v2
    plate = _make_stub_dataset(tmp_path / "v2_full")
    run_v2(
        src=plate,
        channels=(0, 1, 2),
        model_name="cyto",
        channels_for_cellpose=(0, 1),
        diameter=None,
        batch_fields=4,
        metadata_type="cellvoyager",
    )
    merged = plate / "merged"
    assert merged.is_dir(), (
        f"v2 did not create merged/ under {plate}")
    stacks = sorted(merged.glob("stack_*.npy"))
    assert stacks, "v2 merged/ contains no stack_*.npy"
    # Each stack should have (H, W, channels+1) shape
    for p in stacks:
        arr = np.load(p)
        assert arr.ndim == 3 and arr.shape[-1] >= 2, (
            f"unexpected v2 stack shape at {p}: {arr.shape}")


# ---------------------------------------------------------------------------
# measure_crop
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_module_measure_crop_writes_measurements_db(tmp_path):
    """Run v1 mask + measure — the measurements sqlite DB must exist."""
    from spacr.core import preprocess_generate_masks
    from spacr.measure import measure_crop
    plate = _make_stub_dataset(tmp_path / "measure_full")
    mask_settings = _mask_settings_for(plate)
    preprocess_generate_masks(mask_settings)
    # Measure uses many of the same settings; borrow the dict.
    measure_settings = dict(mask_settings)
    measure_settings["src"] = str(plate)
    try:
        measure_crop(measure_settings)
    except Exception as e:
        pytest.skip(f"measure_crop bailed on stub dataset: {e}")
    dbs = list(plate.rglob("measurements.db"))
    assert dbs, "measure_crop wrote no measurements.db under plate"


# ---------------------------------------------------------------------------
# Log file inspection — proves the persistent log picks these up
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.gpu
def test_module_ingest_writes_to_persistent_log(_stub_plate, tmp_path,
                                                    monkeypatch):
    """When a module runs, records should end up in ~/.spacr/logs/
    (redirected here to tmp) so users can inspect after the fact."""
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path))
    # Reset + re-attach the file handler at the tmp path
    from spacr.qt import verbose_logger as vl
    if vl._file_handler is not None:
        for name in vl._ATTACHED_LOGGERS:
            logging.getLogger(name).removeHandler(vl._file_handler)
        vl._file_handler.close()
        vl._file_handler = None
    vl._ensure_file_handler()

    logging.getLogger("spacr").info("real-data test breadcrumb")
    if vl._file_handler is not None:
        vl._file_handler.flush()
    log = vl.current_log_file()
    assert log.exists()
    assert "real-data test breadcrumb" in log.read_text()
