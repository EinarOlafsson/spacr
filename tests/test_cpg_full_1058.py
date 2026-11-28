"""Item 1058: the full OPS phases on the Cell Painting Gallery cropped-site layout.

A synthetic cropped-site well (two sites, the painting stains and the
sequencing cycles) runs objects, ph_flatfield, ph_segment, ph_measure and the
spacer-prefix decode through the engine's own phases, with the threshold
stand-in for Cellpose and fake pinned weights, so no model is loaded. The
well report carries the md5 of the weights of every role that ran.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("cellpose.models")

from scipy import ndimage

from spacr import cpg_full, ops_engine
from tests.test_cpg_frame_919 import BATCH, LOGGED, write_image_csv
from tests.test_ops_provenance_975 import V2, weights  # noqa: F401
from tests.test_ops_spacer_prefix_decode_920 import (BASES, SPACERS, library_table,
                                                     synthetic_site)

class _HalfMaxModel:
    """A stand-in for Cellpose: labels the last channel above half its maximum."""

    def eval(self, image, batch_size=8, resample=True, channels=None,
             channel_axis=None, z_axis=None, normalize=True, rescale=None,
             diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
             do_3D=False, anisotropy=None, flow3D_smooth=0,
             stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
             niter=None, augment=False, tile_overlap=0.1, bsize=None,
             compute_masks=True, progress=None):
        plane = np.asarray(image, float)
        if plane.ndim == 3:
            plane = plane[..., -1]
        return ndimage.label(plane > plane.max() / 2.0)[0], None, None


MD5_V2 = hashlib.md5(V2).hexdigest()
STAINS = ("CorrDNA", "CorrPhalloidin")


def _tree(top, sites, cycles=12):
    stack, labels = synthetic_site(cycles=cycles)
    well = top / "images" / BATCH / "images_corrected_cropped" / "CP9_Well1"
    bright = np.where(labels > 0, 1000, 100).astype(np.uint16)
    for site in sites:
        for c in range(cycles):
            for k, base in enumerate(BASES):
                name = f"Cycle{c + 1:02d}_{base}"
                (well / name).mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff", stack[c, ..., k] + 100)
        for name in ("Cycle01_DAPI",) + STAINS:
            (well / name).mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(well / name / f"{name}_Site_{site}.tiff", bright)
        write_image_csv(top, site, LOGGED.get(site, LOGGED[1]))
    return well.parent


def _settings(top, library):
    return {
        "ops_layout": "cpg_cropped", "cpg_root": str(top / "images" / BATCH / "images_corrected_cropped"),
        "cpg_phenotype_channels": list(STAINS), "dst_root": str(top / "out"), "plate": "CP9",
        "ops_library_mode": "spacer_prefix", "ops_library": str(library), "ops_prefix_length": 10,
        "ops_read_threshold": 10.0, "ops_prefix_threshold_tuning": "none", "ops_prefix_decoy_k": 0,
        "ops_prefix_quality_method": "recipe_simple", "ops_decode_threshold_mode": "manual",
        "ops_cellpose_model": "cpsam_v2", "ops_ph_downsample": 1, "ops_ph_nucleus_downsample": "same",
        "ops_ph_segment_flatfield": False, "ops_store_raw_masks": True, "ops_gpu": False,
        "ops_sbs_nucleus_model": "cpsam", "ops_ph_nucleus_model": "cpsam", "ops_ph_cell_model": "cpsam",
        "ops_flatfield": "none", "ops_ph_texture": False, "n_workers": 1,
        "ops_read_mode": "log", "ops_sbs_cell_method": "off"}


@pytest.fixture
def run(tmp_path, weights, monkeypatch):  # noqa: F811
    top = tmp_path / "top"
    _tree(top, (1, 2))
    library = tmp_path / "library.csv"
    library_table(SPACERS).to_csv(library, index=False)
    monkeypatch.setattr(ops_engine, "_cellpose_model", lambda s, gpu, key=None: _HalfMaxModel())
    phases = ["objects", "ph_segment", "ph_measure", "prefix_decode"]
    result = ops_engine.run_ops(_settings(top, library), phases=phases)
    return top, result


def test_every_role_that_ran_has_the_weights_md5(run):
    top, result = run
    with open(top / "out" / "W1" / "ops_report.json", encoding="utf-8") as handle:
        report = json.load(handle)
    models = report["provenance"]["cellpose_models"]
    assert set(models) == {"ph_nuclei", "ph_cells", "sbs_nuclei"}
    assert {record["md5"] for record in models.values()} == {MD5_V2}
    assert report["stitch"]["placement"] == report["phenotype"]["placement"] == "cpg_cropped"
    assert report["objects"]["objects"] > 0 and report["ph_fields"]


def test_the_phases_ran_on_both_sites(run):
    top, result = run
    for site in (1, 2):
        with np.load(top / "out" / "W1" / "phenotype_masks" / f"site-{site}.npz") as saved:
            assert saved["cells"].max() == 4 and saved["nuclei"].max() == 4
    cells = [p for p in (top / "out").rglob("cells_CP9_W1.csv")]
    assert cells and len(cells[0].read_text().splitlines()) > 1


def test_the_painting_shift_is_never_silent(tmp_path, weights, monkeypatch):  # noqa: F811
    top = tmp_path / "top"
    _tree(top, (1, 2))
    (top / "workspace" / "analysis" / BATCH / "CP9-Well1-2" / "Image.csv").unlink()
    library = tmp_path / "library.csv"
    library_table(SPACERS).to_csv(library, index=False)
    monkeypatch.setattr(ops_engine, "_cellpose_model", lambda s, gpu, key=None: _HalfMaxModel())
    with pytest.raises(ValueError, match="no 04Align record"):
        ops_engine.run_ops(_settings(top, library), phases=["objects"])


def test_the_default_cropped_phases_do_not_use_the_engine_phases(tmp_path):
    top = tmp_path / "top"
    _tree(top, (1,))
    library = tmp_path / "library.csv"
    library_table(SPACERS).to_csv(library, index=False)
    result = ops_engine.run_ops(_settings(top, library), phases=["cpg_index", "cpg_merged"])
    assert result["layout"] == "cpg_cropped" and cpg_full.current() is None
    assert not os.path.exists(top / "out" / "W1" / "ops_report.json")

