"""Item 0aj HARD RULE: fov.ome.zarr keeps the acquired pixels, byte for byte.

A Mask run from raw TIFFs writes every channel plane into fov.ome.zarr
exactly as the TIFF holds it (values, dtype, shape, plane order). Background
removal, segmentation illumination correction and the retired upscale /
pick_slice settings act on the normalised segmentation batches only, never
on the stored channels. ``spacr store verify`` checks the recorded
checksums and the originals, and fails on a corrupted plane.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
tifffile = pytest.importorskip("tifffile")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

from tests.test_a_rerun_trusts_nothing_a_killed_run_left import (  # noqa: E402,F401
    _RUN, _settings, fake_cellpose,
)


def _originals(manifest):
    """``{field file name: [plane per channel]}`` read straight from the TIFFs."""
    fields = {}
    for entry in sorted(manifest, key=lambda e: (e["well"], e["field"], e["channel"])):
        name = f"plate1_{entry['well']}_{int(entry['field'])}_1.npy"
        fields.setdefault(name, []).append(tifffile.imread(entry["path"]))
    return fields


def _illumination_model(tmp_path, shape):
    from spacr.illumination import IlluminationField, IlluminationModel

    rows, cols = np.indices(shape, dtype=np.float32)
    radius = np.sqrt((rows - shape[0] / 2) ** 2 + (cols - shape[1] / 2) ** 2)
    flat = np.clip(1.0 - 0.004 * radius, 0.4, 1.0).astype(np.float32)
    model = IlluminationModel(
        fields={"plate1": IlluminationField(
            plate="plate1", channels=(0, 1), flatfield=np.stack([flat, flat]),
            dark=np.zeros(2, dtype=np.float32), n_fields=20,
            estimator="polynomial", degree=2, bin_size=1)},
        meta={"application_contract_version": 1,
              "channel_index_space": "persisted-intensity-axis",
              "estimated_from_intensity_state": "raw"})
    return str(model.save(tmp_path / "flat.npz"))


def _assert_store_is_the_tiffs(store, originals):
    from spacr.image_store import _store_entries, read_image

    entries = _store_entries(store)
    assert sorted(entries) == sorted(originals)
    for name, planes in originals.items():
        image = read_image(store / entries[name]["key"])
        assert image.shape[0] == len(planes)
        for stored, original in zip(image, planes):
            assert stored.dtype == original.dtype and stored.shape == original.shape
            assert np.ascontiguousarray(stored).tobytes() == np.ascontiguousarray(original).tobytes()


PROCESSING = {
    "plain": {},
    "remove_background": dict(remove_background_cell=True, remove_background_nucleus=True,
                              cell_background=40000, nucleus_background=40000,
                              background=40000),
    "retired_upscale_pick_slice": dict(upscale=True, upscale_factor=2, pick_slice=True),
    "illumination": "illumination",
}


@pytest.mark.parametrize("processing", sorted(PROCESSING))
def test_fov_store_channels_equal_the_original_tiffs_byte_for_byte(
        yokogawa_cellvoyager_dir, fake_cellpose, tmp_path, processing):
    from spacr.core import preprocess_generate_masks
    from spacr.image_store import _merged_store_path, verify_project

    plate = yokogawa_cellvoyager_dir["src"]
    originals = _originals(yokogawa_cellvoyager_dir["manifest"])
    over = PROCESSING[processing]
    if over == "illumination":
        shape = next(iter(originals.values()))[0].shape
        over = dict(illumination_correction=True, illumination_qc=False,
                    illumination_model=_illumination_model(tmp_path, shape))
    preprocess_generate_masks(_settings(plate, preprocess=True, **dict(_RUN, **over)))

    store = _merged_store_path(plate)
    if processing == "illumination":
        # The correction did run, on the segmentation batches only.
        assert (plate / "illumination" / "segmentation_application.json").is_file()
    _assert_store_is_the_tiffs(store, originals)
    # The originals were moved to orig/ untouched; verify reads them there.
    assert sorted(p.name for p in (plate / "orig").glob("*.tif")) == sorted(
        Path(e["path"]).name for e in yokogawa_cellvoyager_dir["manifest"])
    report = verify_project(plate)
    assert report["mismatches"] == [] and report["unrecorded"] == []
    # Every new store is OME-Zarr 0.6 (maintainer, 2026-11-23 16:50).
    root = json.loads((store / "zarr.json").read_text())["attributes"]["ome"]
    assert root["version"] == "0.6"
    assert report["planes"] == 8 and report["originals_compared"] == 8


def test_store_verify_command_passes_then_catches_a_corrupted_plane(
        yokogawa_cellvoyager_dir, fake_cellpose, capsys):
    import zarr

    from spacr.cli_store import main
    from spacr.core import preprocess_generate_masks
    from spacr.image_store import _merged_store_path, _store_entries
    from spacr.qt import run

    plate = yokogawa_cellvoyager_dir["src"]
    preprocess_generate_masks(_settings(plate, preprocess=True, **_RUN))
    capsys.readouterr()
    assert main(["verify", str(plate)]) == 0
    out = capsys.readouterr().out
    assert "fields checked: 4" in out and "compared with their original images: 8" in out
    assert "OK: every stored channel plane matches." in out
    # `spacr store verify` reaches the same command without starting the GUI.
    assert run(["store", "verify", str(plate), "--no-originals"]) == 0
    capsys.readouterr()

    store = _merged_store_path(plate)
    entry = next(iter(_store_entries(store).values()))
    array = zarr.open_array(str(store / entry["key"] / "0"), mode="r+")
    array[1, 5, 5] = int(array[1, 5, 5]) ^ 1
    assert main(["verify", str(store), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert f"{entry['name']} plane 1: checksum" in report["mismatches"]
    assert f"{entry['name']} plane 1: differs from the original" in report["mismatches"]
    assert main(["verify", str(plate)]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_store_verify_refuses_a_folder_without_a_store(tmp_path, capsys):
    from spacr.cli_store import main

    assert main(["verify", str(tmp_path)]) == 1
    assert "is not a spaCR fov.ome.zarr" in capsys.readouterr().err
