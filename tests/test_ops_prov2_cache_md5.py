"""Provenance merge, md5 cache keys, the stale-objects guard and the replay rule (items 975, 1054).

Runs the synthetic five-field well of the 958 tests with the threshold stand-in
for Cellpose, and fake pinned weight files, so no model is loaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil

import pytest

pytest.importorskip("tifffile")
pytest.importorskip("scipy")
pytest.importorskip("pyarrow")
pytest.importorskip("cellpose.models")

from spacr import ops_engine
from tests.test_ops_objects_resume_958 import (  # noqa: F401  (fixtures and helpers)
    _CountingModel, stitched)
from tests.test_sbs_cells_1025 import _ProbabilityModel
from tests.test_ops_provenance_975 import V1, V2, weights  # noqa: F401

MD5_V1, MD5_V2 = hashlib.md5(V1).hexdigest(), hashlib.md5(V2).hexdigest()


def _run(stitched, folder, monkeypatch, phases, model=None, **extra):
    """Run phases on a copy of the stitched well; returns (result, model)."""
    root, settings, library = stitched
    if not os.path.exists(folder / "measurements"):
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root / "stitched" / "measurements", folder / "measurements")
    model = model or _CountingModel()
    cells = _ProbabilityModel()
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda s, gpu, key="ops_sbs_nucleus_model":
                        cells if key == "ops_sbs_cell_model" else model)
    monkeypatch.setattr(ops_engine, "_SBS_CELL_MODELS", {})
    result = ops_engine.run_ops({**settings, "dst_root": str(folder), **extra},
                                phases=phases, library=library)
    return result, model


def _report(folder):
    with open(folder / "A1" / "ops_report.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_keys_hold_the_resolved_md5_not_the_name(weights, monkeypatch):
    """Same model name, other weights md5: other window basis, live and cell keys."""
    from spacr import ops_live

    settings = {"ops_sbs_nucleus_model": "cpsam", "ops_cellpose_model": "cpsam_v2"}
    first = ops_engine._objects_window_basis(settings, "average", {}, (10, 10), (5, 5))
    real = ops_engine._cellpose_provenance
    monkeypatch.setattr(ops_engine, "_cellpose_provenance",
                        lambda s, key: {**real(s, key), "md5": "f" * 32})
    second = ops_engine._objects_window_basis(settings, "average", {}, (10, 10), (5, 5))
    assert first["model"]["name"] == second["model"]["name"]
    assert first != second and first["model"]["md5"] == MD5_V2
    assert ops_engine._resolved_weights_md5(settings, "ops_ph_cell_model") == "f" * 32
    monkeypatch.setattr(ops_engine, "_cellpose_provenance", real)
    v1 = {**settings, "ops_cellpose_model": "cpsam_v1"}
    assert ops_engine._resolved_weights_md5(v1, "ops_sbs_cell_model") == MD5_V1
    assert ops_live._signature({**settings, "_live_root": "x"}, "spots", {}) != \
        ops_live._signature({**v1, "_live_root": "x"}, "spots", {})
    assert ops_engine._ops_calibration_context(settings, "s") != \
        ops_engine._ops_calibration_context(v1, "s")


def test_sbs_cell_model_cache_and_live_models_split_by_md5(weights, monkeypatch):
    built = []
    monkeypatch.setattr(ops_engine, "_SBS_CELL_MODELS", {})
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda s, gpu, key=None: built.append(s["ops_cellpose_model"]) or object())
    for choice in ("cpsam_v2", "cpsam_v1", "cpsam_v2"):
        ops_engine._sbs_cell_model({"mask_settings": {"ops_cellpose_model": choice}}, False)
    assert built == ["cpsam_v2", "cpsam_v1"]


def test_phenotype_marker_records_the_md5(weights):
    options = {"cellpose_md5": {"ph_nuclei": MD5_V2, "ph_cells": MD5_V2}}
    other = {"cellpose_md5": {"ph_nuclei": MD5_V1, "ph_cells": MD5_V1}}
    import tempfile

    with tempfile.TemporaryDirectory() as folder:
        ops_engine._ph_options_marker(folder, options)
        with pytest.raises(ValueError):
            ops_engine._ph_options_marker(folder, other)


def test_phases_run_separately_merge_their_roles(stitched, weights, tmp_path, monkeypatch):
    """objects then decode (cellpose cells): ops_report.json keeps both roles."""
    _run(stitched, tmp_path, monkeypatch, ("objects",))
    models = _report(tmp_path)["provenance"]["cellpose_models"]
    assert set(models) == {"sbs_nuclei"}
    _run(stitched, tmp_path, monkeypatch, ("decode",), ops_sbs_cell_method="cellpose")
    models = _report(tmp_path)["provenance"]["cellpose_models"]
    assert set(models) == {"sbs_nuclei", "sbs_cells"}
    assert models["sbs_nuclei"]["md5"] == MD5_V2 == models["sbs_cells"]["md5"]


def test_phases_run_together_record_both_roles(stitched, weights, tmp_path, monkeypatch):
    _run(stitched, tmp_path, monkeypatch, ("objects", "decode"), ops_sbs_cell_method="cellpose")
    assert set(_report(tmp_path)["provenance"]["cellpose_models"]) == {"sbs_nuclei", "sbs_cells"}


def test_a_rerun_replaces_only_its_own_role(weights):
    earlier = {"cellpose_models": {"sbs_nuclei": {"md5": "old"}, "ph_cells": {"md5": "keep"}}}
    block = ops_engine._well_provenance({}, ("objects",), {"provenance": earlier})
    assert block["cellpose_models"]["sbs_nuclei"]["md5"] == MD5_V2
    assert block["cellpose_models"]["ph_cells"] == {"md5": "keep"}


def test_the_phases_log_their_resolved_weights(stitched, weights, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=ops_engine.LOG.name)
    _run(stitched, tmp_path, monkeypatch, ("objects", "decode"), ops_sbs_cell_method="cellpose")
    text = "\n".join(caplog.messages)
    assert f"objects weights: sbs_nuclei model cpsam -> path {weights['v2']} md5 {MD5_V2}" in text
    assert f"decode weights: sbs_cells model cpsam -> path {weights['v2']} md5 {MD5_V2}" in text


def test_the_guard_refuses_stale_objects_without_the_objects_phase(
        stitched, weights, tmp_path, monkeypatch):
    """(b) v1 objects, then a v2 decode without objects: refused. (a) with objects: redone."""
    keep = {"ops_objects_keep_windows": True}
    _result, first = _run(stitched, tmp_path, monkeypatch, ("objects",),
                          ops_cellpose_model="cpsam_v1", **keep)
    with pytest.raises(ValueError, match="md5 " + MD5_V1):
        _run(stitched, tmp_path, monkeypatch, ("decode",), ops_cellpose_model="cpsam_v2")
    _result, again = _run(stitched, tmp_path, monkeypatch, ("objects", "decode"),
                          ops_cellpose_model="cpsam_v2")
    assert again.calls == first.calls > 0
    assert _report(tmp_path)["provenance"]["cellpose_models"]["sbs_nuclei"]["md5"] == MD5_V2
    _run(stitched, tmp_path, monkeypatch, ("decode",), ops_cellpose_model="cpsam_v2")


def test_objects_without_provenance_are_refused_unless_allowed(
        stitched, weights, tmp_path, monkeypatch):
    _run(stitched, tmp_path, monkeypatch, ("objects",))
    (tmp_path / "A1" / "ops_report.json").unlink()
    with pytest.raises(ValueError, match="ops_reuse_unprovenanced_objects"):
        _run(stitched, tmp_path, monkeypatch, ("decode",))
    result, _ = _run(stitched, tmp_path, monkeypatch, ("decode",),
                     ops_reuse_unprovenanced_objects=True)
    assert result["wells"]["A1"]["decode"]


def test_the_new_setting_defaults_to_off():
    from spacr.ops_settings import OPS_DEFAULTS

    assert OPS_DEFAULTS["ops_reuse_unprovenanced_objects"] is False
    assert OPS_DEFAULTS["ops_replay_cellpose_weights"] is None


@pytest.mark.parametrize("version, expected", [
    ("4.2.1.1", "cpsam_v2"), ("4.2.0", "cpsam_v2"), ("4.0.9", "cpsam_v1"),
    ("4.1.1", "cpsam_v1"), ("3.1.1.1", None), ("", None)])
def test_cellpose_default_weights_by_version(version, expected):
    assert ops_engine._cellpose_default_weights(version) == expected


def _record_version(folder, version):
    (folder / "A1").mkdir(parents=True, exist_ok=True)
    (folder / "A1" / "ops_report.json").write_text(json.dumps(
        {"provenance": {"software": {"cellpose": version}}}))


@pytest.mark.parametrize("version, expected", [("4.2.1.1", "cpsam_v2"), ("4.0.9", "cpsam_v1")])
def test_replay_reads_the_recorded_cellpose_version(tmp_path, version, expected):
    _record_version(tmp_path, version)
    out = ops_engine._resolve_replay_weights(
        {"ops_cellpose_model": "replay", "dst_root": str(tmp_path)})
    assert out["ops_cellpose_model"] == expected


def test_replay_refuses_when_no_version_was_recorded(tmp_path):
    with pytest.raises(ValueError, match="ops_replay_cellpose_weights"):
        ops_engine._resolve_replay_weights(
            {"ops_cellpose_model": "replay", "dst_root": str(tmp_path)})
    _record_version(tmp_path, "3.1.1.1")
    with pytest.raises(ValueError, match="ops_replay_cellpose_weights"):
        ops_engine._resolve_replay_weights(
            {"ops_cellpose_model": "replay", "dst_root": str(tmp_path)})


def test_replay_explicit_setting_wins(tmp_path):
    _record_version(tmp_path, "4.2.1.1")
    out = ops_engine._resolve_replay_weights(
        {"ops_cellpose_model": "replay", "dst_root": str(tmp_path),
         "ops_replay_cellpose_weights": "cpsam_v1"})
    assert out["ops_cellpose_model"] == "cpsam_v1"
    with pytest.raises(ValueError):
        ops_engine._resolve_replay_weights(
            {"ops_cellpose_model": "replay", "ops_replay_cellpose_weights": "v3"})


def test_an_old_stamped_file_folds_to_replay_and_run_ops_resolves_it(tmp_path, weights):
    from spacr.ops_settings import OPS_DEFAULTS
    from spacr.settings import _fold_older_defaults

    given = {}
    _fold_older_defaults(given, 974, known=set(OPS_DEFAULTS))
    assert given["ops_cellpose_model"] == "replay"
    with pytest.raises(ValueError, match="ops_replay_cellpose_weights"):
        ops_engine.run_ops({**given, "dst_root": str(tmp_path), "genotype_source": str(tmp_path)},
                           phases=("objects",))
