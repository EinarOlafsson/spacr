"""Item 1040: ops_require_versions refuses a run made with the wrong cellpose or torch."""
import pytest

from spacr import ops_engine
from spacr.ops_settings import OPS_DEFAULTS, OPS_TYPES

PINNED = {"cellpose": "4.2.1.1", "torch": "2.13.0+cu130"}


def _software(cellpose="4.2.1.1", torch="2.13.0+cu130"):
    return {"spacr": "1", "cellpose": cellpose, "torch": torch,
            "torch_cuda": None, "cudnn": None}


def test_default_is_no_check():
    assert OPS_DEFAULTS["ops_require_versions"] is None
    assert type(None) in OPS_TYPES["ops_require_versions"]
    assert ops_engine._version_check({}) is None
    assert ops_engine._version_check({"ops_require_versions": None}) is None


def test_a_match_passes_and_reports_both_sides(monkeypatch):
    monkeypatch.setattr(ops_engine, "_software_versions", lambda: _software())
    got = ops_engine._version_check({"ops_require_versions": PINNED})
    assert got == {"expected": PINNED, "found": PINNED}


def test_a_mismatch_names_each_expected_and_found(monkeypatch):
    monkeypatch.setattr(ops_engine, "_software_versions",
                        lambda: _software("4.0.9", "2.10.0+cu128"))
    with pytest.raises(RuntimeError) as info:
        ops_engine._version_check({"ops_require_versions": PINNED})
    text = str(info.value)
    for piece in ("cellpose: expected 4.2.1.1, found 4.0.9",
                  "torch: expected 2.13.0+cu130, found 2.10.0+cu128"):
        assert piece in text


def test_a_mismatch_stops_run_ops_before_any_field(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(ops_engine, "_software_versions",
                        lambda: _software("4.0.9", "2.10.0+cu128"))
    for name in ("_index_tiles", "_stitch_well", "_decode_well", "_run_cpg"):
        if hasattr(ops_engine, name):
            monkeypatch.setattr(ops_engine, name,
                                lambda *a, _n=name, **k: started.append(_n))
    with pytest.raises(RuntimeError, match="ops_require_versions"):
        ops_engine.run_ops({"genotype_source": str(tmp_path), "dst_root": str(tmp_path),
                            "ops_require_versions": PINNED})
    assert started == []


def test_a_malformed_setting_is_refused():
    with pytest.raises(ValueError, match="ops_require_versions"):
        ops_engine._version_check({"ops_require_versions": "cellpose"})


def test_provenance_records_expected_and_found(monkeypatch):
    monkeypatch.setattr(ops_engine, "_software_versions", lambda: _software())
    block = ops_engine._well_provenance({"ops_require_versions": PINNED}, ["decode"], {})
    assert block["required_versions"] == {"expected": PINNED, "found": PINNED}
    assert "required_versions" not in ops_engine._well_provenance({}, ["decode"], {})
