"""Every OPS run records the Cellpose weights and flat-field map it segmented on (item 975).

``ops_report.json`` of each well and the attributes of the mask stores carry,
per segmentation role (phenotype nuclei and cells, SBS nuclei, SBS cells when
``ops_sbs_cell_method`` is ``'cellpose'``), the full path and md5 of the
weights, and the md5 of the flat-field maps phenotype segmentation divided by
(item 1035). The weights are tiny fake files whose md5 is pinned in place of
the real ones, and Cellpose is the threshold stand-in of the 1035 tests, so
no model is loaded.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

pytest.importorskip("cellpose.models")
pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")

from cellpose import models as cp_models

from spacr import ops_engine, utils
from tests.test_ph_segment_flatfield_1035 import (  # noqa: F401  (fixtures and helpers)
    PLATE, SETTINGS, _maps_folder, _run, _run_ops, models as threshold_models)

V1, V2 = b"v1 weights", b"v2 weights!"


@pytest.fixture
def weights(monkeypatch, tmp_path_factory):
    """Fake cpsam and cpsam_v2 files in a fake Cellpose folder, pinned by md5.

    :returns: ``{"v1": path, "v2": path, "reads": [paths hashed]}``.
    """
    folder = tmp_path_factory.mktemp("cellpose_models")
    pins, paths = {}, {}
    for choice, name, body in (("cpsam_v1", "cpsam", V1), ("cpsam_v2", "cpsam_v2", V2)):
        (folder / name).write_bytes(body)
        paths[choice[-2:]] = str(folder / name)
        pins[choice] = {"file": name, "size": len(body), "md5": hashlib.md5(body).hexdigest()}
    monkeypatch.setattr(utils, "CPSAM_WEIGHTS", pins)
    monkeypatch.setattr(cp_models, "MODEL_DIR", folder)
    monkeypatch.setattr(utils, "_CPSAM_MD5_SEEN", {})
    reads = []
    real = utils._file_md5
    monkeypatch.setattr(utils, "_file_md5", lambda path: (reads.append(path), real(path))[1])
    return {**paths, "reads": reads, "folder": folder}


def test_default_records_the_cpsam_v2_path_and_md5(weights):
    """The stock name resolves to the verified full path of the pinned file."""
    record = ops_engine._cellpose_provenance({}, "ops_ph_nucleus_model")
    assert record == {"setting": "ops_ph_nucleus_model", "model": "cpsam",
                      "weights": "cpsam_v2", "path": weights["v2"],
                      "md5": hashlib.md5(V2).hexdigest()}


def test_v1_replay_records_the_v1_file(weights):
    """A settings file that keeps cpsam_v1 records that file, not v2."""
    record = ops_engine._cellpose_provenance(
        {"ops_cellpose_model": "cpsam_v1"}, "ops_sbs_nucleus_model")
    assert record["path"] == weights["v1"] and record["md5"] == hashlib.md5(V1).hexdigest()
    assert record["weights"] == "cpsam_v1"


def test_custom_checkpoint_records_its_own_path_and_md5(weights, tmp_path):
    """A fine-tuned checkpoint is recorded with its absolute path and md5."""
    custom = tmp_path / "my_model"
    custom.write_bytes(b"fine tuned")
    record = ops_engine._cellpose_provenance(
        {"ops_ph_cell_model": str(custom)}, "ops_ph_cell_model")
    assert record["path"] == str(custom)
    assert record["md5"] == hashlib.md5(b"fine tuned").hexdigest()
    assert "weights" not in record


def test_missing_weights_are_said_not_invented(weights):
    """Without the pinned file there is no path or md5, and the reason is kept."""
    (weights["folder"] / "cpsam_v2").unlink()
    record = ops_engine._cellpose_provenance({}, "ops_ph_nucleus_model")
    assert record["path"] is None and record["md5"] is None
    assert "cpsam_v2" in record["unverified"]


def test_the_weights_are_hashed_once_for_every_role_and_field(weights):
    """Four roles and a model build share one read of the file."""
    settings = {"ops_sbs_cell_method": "cellpose"}
    phases = ("objects", "decode", "ph_segment")
    for _ in range(3):
        found = ops_engine._run_provenance(settings, phases)
    utils.cpsam_weights("cpsam_v2")
    assert set(found) == {"ph_nuclei", "ph_cells", "sbs_nuclei", "sbs_cells"}
    assert {r["md5"] for r in found.values()} == {hashlib.md5(V2).hexdigest()}
    assert weights["reads"] == [weights["v2"]]


def test_roles_follow_the_phases_and_the_sbs_cell_method(weights):
    """SBS cells are listed only for the cellpose method; phases pick the roles."""
    assert set(ops_engine._run_provenance({}, ("objects", "decode"))) == {"sbs_nuclei"}
    assert set(ops_engine._run_provenance(
        {"ops_sbs_cell_method": "cellpose"}, ("decode",))) == {"sbs_cells"}
    assert ops_engine._run_provenance({}, ("stitch",)) == {}
    separate = ops_engine._run_provenance(
        {"ops_ph_cell_model": str(weights["v1"])}, ("ph_segment",))
    assert separate["ph_nuclei"]["weights"] == "cpsam_v2"
    assert separate["ph_cells"]["path"] == weights["v1"]


def test_ph_fields_report_the_weights_and_the_flat_field_md5(tmp_path, weights, threshold_models):
    """The fields' report and the mask store both carry the weights and the map md5."""
    zarr = pytest.importorskip("zarr")
    pytest.importorskip("spacr.image_store")
    from tests.test_ph_segment_flatfield_1035 import _write_fields, _placed
    from spacr.ops_engine import _index_tiles, _ph_fields

    _maps_folder(tmp_path / "flatfield")
    tiles = _write_fields(tmp_path / "pheno")
    db = str(tmp_path / "measurements.db")
    _placed(db)
    index = _index_tiles(str(tiles), "uncycled")
    report = _ph_fields(db, PLATE, "A1", index["A1"],
                        {**SETTINGS, "ops_ph_image_format": "ome-zarr"}, False,
                        str(tmp_path), segment=True, measure=False)
    block = report["provenance"]
    nuclei, cells = block["cellpose_models"]["ph_nuclei"], block["cellpose_models"]["ph_cells"]
    assert nuclei["path"] == cells["path"] == weights["v2"]
    assert nuclei["md5"] == hashlib.md5(V2).hexdigest()
    assert set(block["segment_flatfield_md5"]) == {"DAPI", "AF750"}
    store = zarr.open_group(str(tmp_path / "A1" / "phenotype_masks" / f"{PLATE}.ome.zarr"),
                            mode="r")
    assert json.loads(json.dumps(dict(store.attrs["spacr"]))) == {
        "cellpose_models": block["cellpose_models"],
        "software": block["software"],
        "gpu_deterministic": False,
        "segment_flatfield_md5": block["segment_flatfield_md5"]}


def test_raw_segmentation_records_no_flat_field_map(tmp_path, weights, threshold_models):
    """Off, or without maps, the map md5 is None: the fields were segmented raw."""
    report, _masks, _marker = _run(
        tmp_path, {**SETTINGS, "ops_ph_segment_flatfield": False})
    assert report["provenance"]["segment_flatfield_md5"] is None
    assert report["provenance"]["cellpose_models"]["ph_nuclei"]["md5"] == hashlib.md5(V2).hexdigest()


def test_run_ops_writes_provenance_to_ops_report_json(tmp_path, weights, threshold_models,
                                                      monkeypatch):
    """ops_report.json of the well names the weights of the run and the map it used."""
    _maps_folder(tmp_path / "flatfield")
    _run_ops(tmp_path, monkeypatch, ("ph_segment",))
    saved = json.loads((tmp_path / "A1" / "ops_report.json").read_text())["provenance"]
    assert saved["cellpose_models"]["ph_cells"]["path"] == weights["v2"]
    assert saved["cellpose_models"]["ph_cells"]["md5"] == hashlib.md5(V2).hexdigest()
    assert set(saved["segment_flatfield_md5"]) == {"DAPI", "AF750"}
    assert saved["software"] == ops_engine._software_versions()
    assert "sbs_nuclei" not in saved["cellpose_models"]


def test_well_provenance_keeps_what_an_earlier_phase_recorded(weights):
    """A later run of other phases adds its roles and leaves the phenotype record."""
    earlier = {"cellpose_models": {"ph_nuclei": {"md5": "x"}}, "segment_flatfield_md5": {"DAPI": "y"}}
    block = ops_engine._well_provenance({}, ("objects",), {"provenance": earlier})
    assert block["cellpose_models"]["ph_nuclei"] == {"md5": "x"}
    assert block["segment_flatfield_md5"] == {"DAPI": "y"}
    assert block["cellpose_models"]["sbs_nuclei"]["md5"] == hashlib.md5(V2).hexdigest()


def test_sbs_raw_mask_store_carries_the_weights(tmp_path, weights):
    """The SBS raw window store's root attributes hold the SBS weights."""
    zarr = pytest.importorskip("zarr")
    attrs = {"cellpose_models": ops_engine._run_provenance({}, ("objects",)),
             "software": ops_engine._software_versions()}
    writer = ops_engine._objects_raw_writer(str(tmp_path / "measurements.db"), PLATE, attrs)
    stored = zarr.open_group(str(writer.path), mode="r").attrs["spacr"]
    assert stored["cellpose_models"]["sbs_nuclei"]["path"] == weights["v2"]
    assert stored["cellpose_models"]["sbs_nuclei"]["md5"] == hashlib.md5(V2).hexdigest()
    assert stored["software"]["cellpose"]


def test_an_unresolvable_model_name_is_recorded_as_unverified(weights):
    """A name that matches no model gives no path, and the reason, instead of stopping."""
    record = ops_engine._cellpose_provenance({"ops_ph_cell_model": "no-such-model"},
                                             "ops_ph_cell_model")
    assert record["path"] is None and record["md5"] is None
    assert "no-such-model" in record["unverified"]


def test_software_versions_are_the_installed_ones():
    """spacr, cellpose and torch versions come from the packages themselves."""
    from importlib import metadata

    import torch

    import spacr

    found = ops_engine._software_versions()
    assert found["spacr"] == spacr.__version__
    assert found["cellpose"] == metadata.version("cellpose")
    assert found["torch"] == torch.__version__
    assert found["torch_cuda"] == (torch.version.cuda or None)
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    assert found["cudnn"] == (None if cudnn is None else str(cudnn))


def test_software_versions_are_read_once_per_run(monkeypatch):
    """Many wells and stores share one read of the metadata."""
    from importlib import metadata

    ops_engine._software_versions.cache_clear()
    reads = []
    real = metadata.version
    monkeypatch.setattr(metadata, "version", lambda name: (reads.append(name), real(name))[1])
    first = ops_engine._software_versions()
    for _ in range(5):
        assert ops_engine._well_provenance({}, ("stitch",), {})["software"] == first
    assert reads == ["cellpose"]
    ops_engine._software_versions.cache_clear()


def test_missing_packages_give_none_not_an_error(monkeypatch):
    """A cellpose that is not installed is recorded as None."""
    from importlib import metadata

    def absent(name):
        raise metadata.PackageNotFoundError(name)

    ops_engine._software_versions.cache_clear()
    monkeypatch.setattr(metadata, "version", absent)
    try:
        assert ops_engine._software_versions()["cellpose"] is None
    finally:
        ops_engine._software_versions.cache_clear()
