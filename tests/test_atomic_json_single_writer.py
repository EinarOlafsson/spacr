"""One atomic JSON writer: crash safety and byte-identical migrated formats."""
import json
import re
from pathlib import Path

import pytest

from spacr import checkpoint
from spacr.checkpoint import _atomic_json

PAYLOAD = {"b": [1, 2.5, "é"], "a": {"z": None, "y": True}}

LEGACY = {
    "ops_report": (dict(indent=1, sort_keys=False, default=str, normalize=False,
                        ensure_ascii=True, newline=False),
                   lambda: json.dumps(PAYLOAD, indent=1, default=str)),
    "ops_field": (dict(indent=None, sort_keys=False, default=str, normalize=False,
                       ensure_ascii=True, newline=False),
                  lambda: json.dumps(PAYLOAD, default=str)),
    "sorted_indent2": (dict(normalize=False, ensure_ascii=True, newline=False),
                       lambda: json.dumps(PAYLOAD, indent=2, sort_keys=True)),
}


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_migrated_formats_write_the_legacy_bytes(tmp_path, name):
    options, legacy = LEGACY[name]
    target = tmp_path / "sub" / "out.json"
    _atomic_json(str(target), PAYLOAD, **options)
    assert target.read_text(encoding="utf-8") == legacy()
    assert [p.name for p in target.parent.iterdir()] == ["out.json"]


def test_wrappers_keep_their_formats(tmp_path):
    from spacr import _lineage_measurements, fit_resources, ome_zarr_convert

    fit_resources._atomic_json(tmp_path / "f.json", PAYLOAD)
    assert (tmp_path / "f.json").read_text(encoding="utf-8") == json.dumps(
        PAYLOAD, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    _lineage_measurements._atomic_json(tmp_path / "l.json", PAYLOAD)
    assert (tmp_path / "l.json").read_text(encoding="utf-8") == json.dumps(
        PAYLOAD, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ome_zarr_convert._atomic_json(tmp_path / "o.json", PAYLOAD)
    assert (tmp_path / "o.json").read_text(encoding="utf-8") == json.dumps(
        PAYLOAD, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ome_zarr_convert._atomic_json(tmp_path / "k.json", PAYLOAD, sort_keys=False)
    assert list(json.loads((tmp_path / "k.json").read_text())) == ["b", "a"]


def test_crash_mid_write_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "ops_report.json"
    _atomic_json(target, {"old": 1})
    before = target.read_bytes()

    def torn(value, stream, **kwargs):
        stream.write('{"new": [1, 2')
        stream.flush()
        raise RuntimeError("killed mid-write")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint.json, "dump", torn)
        with pytest.raises(RuntimeError):
            _atomic_json(target, {"new": 2}, normalize=False)
    assert target.read_bytes() == before
    assert json.loads(target.read_text()) == {"old": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["ops_report.json"]


def test_crash_before_rename_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    _atomic_json(target, {"old": 1})

    def boom(*_args):
        raise OSError("disk gone")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoint.os, "replace", boom)
        with pytest.raises(OSError):
            _atomic_json(target, {"new": 2})
    assert json.loads(target.read_text()) == {"old": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_resume_read_json_is_never_written_in_place():
    root = Path(checkpoint.__file__).parent
    pattern = re.compile(r'open\((?:report_path|done_path|layout_path|record_path)'
                         r'(?: \+ "\.part")?, "w"')
    text = (root / "ops_engine.py").read_text(encoding="utf-8")
    assert pattern.search(text) is None
