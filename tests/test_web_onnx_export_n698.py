"""Browser ONNX export: selection, weight trust, manifest merging and the export call."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools.web_mask_generator import export_onnx as e


def _row(key, content, *, kind="cellpose", family="fam", repo_type="model", filename=None):
    return {"key": key, "family": family, "kind": kind, "repo_id": "owner/repo", "repo_type": repo_type,
            "revision": "a" * 40, "filename": filename or f"weights/{key}_file",
            "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


@pytest.fixture
def catalogue(tmp_path):
    contents = {"fam_v1": b"first checkpoint", "det_v1": b"detector checkpoint"}
    rows = [_row("fam_v1", contents["fam_v1"]),
            _row("det_v1", contents["det_v1"], kind="detector", family="det", filename="best.pt")]
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema": 1, "models": rows}))
    return path, {row["key"]: row for row in rows}, contents


def test_shipped_catalogue_lists_only_exportable_kinds():
    rows = e.load_catalogue()
    assert rows and {row["kind"] for row in rows.values()} <= {"cellpose", "detector"}
    assert "nuclei_from_cellmask_v1" in rows


def test_key_selection(catalogue):
    path, rows, _ = catalogue
    loaded = e.load_catalogue(path)
    assert e.select_keys(loaded, [], True) == ["det_v1", "fam_v1"]
    assert e.select_keys(loaded, ["fam_v1", "fam_v1"], False) == ["fam_v1"]
    for keys, everything in (([], False), (["nope"], False), (["fam_v1"], True)):
        with pytest.raises(e.ExportError):
            e.select_keys(loaded, keys, everything)


def test_duplicate_catalogue_key_is_refused(tmp_path):
    row = _row("k_v1", b"x")
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"schema": 1, "models": [row, row]}))
    with pytest.raises(e.ExportError, match="duplicate"):
        e.load_catalogue(path)


def test_source_url_respects_repo_type():
    model = _row("k_v1", b"x", filename="weights/a b.pt")
    dataset = {**model, "repo_type": "dataset"}
    assert e.source_url(model) == "https://huggingface.co/owner/repo/resolve/" + "a" * 40 + "/weights/a%20b.pt"
    assert e.source_url(dataset).startswith("https://huggingface.co/datasets/owner/repo/resolve/")


def test_weights_pairs_are_parsed_strictly():
    assert e.parse_weights(["k=/a/b"]) == {"k": Path("/a/b")}
    for bad in ("k", "=p", "k="):
        with pytest.raises(e.ExportError):
            e.parse_weights([bad])


def test_weights_resolution_order_and_no_silent_download(tmp_path, catalogue, monkeypatch):
    _, rows, contents = catalogue
    row = rows["fam_v1"]
    directory = tmp_path / "weights"
    directory.mkdir()
    (directory / "fam_v1_file").write_bytes(contents["fam_v1"])
    explicit = tmp_path / "explicit"
    explicit.write_bytes(contents["fam_v1"])
    monkeypatch.setattr(e, "download_weights", lambda *a, **k: pytest.fail("downloaded without --download"))
    common = {"download": False, "download_dir": tmp_path / "dl"}
    assert e.resolve_weights(row, explicit={"fam_v1": explicit}, weights_dir=directory, **common) == (explicit, "explicit")
    assert e.resolve_weights(row, explicit={}, weights_dir=directory, **common) == (directory / "fam_v1_file", "weights-dir")
    with pytest.raises(e.ExportError, match="--download"):
        e.resolve_weights(row, explicit={}, weights_dir=tmp_path, **common)
    with pytest.raises(e.ExportError, match="not found"):
        e.resolve_weights(row, explicit={"fam_v1": tmp_path / "missing"}, weights_dir=None, **common)


def test_weights_must_match_published_sha(tmp_path, catalogue):
    _, rows, contents = catalogue
    row = rows["fam_v1"]
    good = tmp_path / "good"
    good.write_bytes(contents["fam_v1"])
    assert e.verify_weights(good, row) == row["sha256"]
    same_size = tmp_path / "same_size"
    same_size.write_bytes(b"x" * row["bytes"])
    with pytest.raises(e.ExportError, match="SHA-256 mismatch"):
        e.verify_weights(same_size, row)
    short = tmp_path / "short"
    short.write_bytes(b"x")
    with pytest.raises(e.ExportError, match="bytes"):
        e.verify_weights(short, row)


class _Response(io.BytesIO):
    status = 200

    def __init__(self, raw, url):
        super().__init__(raw)
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_opt_in_download_uses_public_url_and_verifies(tmp_path, catalogue, monkeypatch):
    from tools.web_mask_generator import provision
    _, rows, contents = catalogue
    row = rows["fam_v1"]
    calls = []

    def transport(url, timeout):
        calls.append(url)
        return _Response(served, url)

    monkeypatch.setattr(provision, "_transport", transport)
    served = contents["fam_v1"]
    target = e.download_weights(row, tmp_path / "dl")
    assert target.read_bytes() == served and calls == [e.source_url(row)]
    served = b"y" * row["bytes"]
    with pytest.raises(e.ExportError, match="does not match"):
        e.download_weights(row, tmp_path / "other")
    assert not list((tmp_path / "other").iterdir())


def test_manifest_is_created_then_merged_per_key(tmp_path):
    path = tmp_path / "m" / "browser-manifest.json"
    first = e.merge_manifest(path, {"b_v1": {"file": "b.onnx"}})
    assert first == {"schema": 1, "generated_by": e.GENERATED_BY, "base_url": None, "models": {"b_v1": {"file": "b.onnx"}}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["base_url"] = "https://example.invalid/web-onnx/"
    path.write_text(json.dumps(data))
    merged = e.merge_manifest(path, {"a_v1": {"file": "a.onnx"}, "b_v1": {"file": "b2.onnx"}})
    assert merged["base_url"] == "https://example.invalid/web-onnx/"
    assert list(merged["models"]) == ["a_v1", "b_v1"] and merged["models"]["b_v1"] == {"file": "b2.onnx"}
    path.write_text(json.dumps({"schema": 2, "models": {}}))
    with pytest.raises(e.ExportError):
        e.merge_manifest(path, {})


class _TinyNet:
    def __new__(cls):
        import torch

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(0)
                self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

            def forward(self, x):
                return self.conv(x), torch.zeros((x.shape[0], 256))

        return Tiny().eval()


def test_cellpose_export_call_shape(tmp_path, monkeypatch):
    import torch
    seen = {}

    def fake_export(module, args, path, **kwargs):
        seen.update(kwargs, output=module(*args), path=path)
        Path(path).write_bytes(b"onnx")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    destination = tmp_path / "x.onnx"
    e.export_cellpose(_TinyNet(), destination)
    assert destination.read_bytes() == b"onnx"
    assert seen["input_names"] == ["image"] and seen["output_names"] == ["flows"]
    assert seen["dynamic_axes"] == {"image": {0: "N"}, "flows": {0: "N"}}
    assert seen["opset_version"] == e.OPSET and seen["dynamo"] is False
    assert tuple(seen["output"].shape) == (1, 3, e.BSIZE, e.BSIZE)
    assert [p.name for p in tmp_path.iterdir()] == ["x.onnx"]


def test_entries_follow_schema(monkeypatch, catalogue):
    _, rows, _ = catalogue
    monkeypatch.setattr(e, "spacr_defaults", lambda family: {"desktop": {}, "web_app": {}})
    monkeypatch.setattr(e, "detector_defaults", lambda: {"confidence": 0.25, "imgsz": 640, "min_axis_ratio": 0.7})
    keys = {"family", "kind", "file", "bytes", "sha256", "source", "input", "output", "preprocess", "postprocess", "check"}
    cell = e.cellpose_entry(rows["fam_v1"], file="f.onnx", size=1, digest="0" * 64, check=None)
    assert keys <= set(cell) and cell["output"]["channels"] == ["dY", "dX", "cellprob"]
    assert cell["preprocess"]["bsize"] == 256 and cell["preprocess"]["diam_mean"] == 30.0
    assert cell["preprocess"]["normalize"]["percentiles"] == [1.0, 99.0]
    assert cell["postprocess"]["min_size"] == 15 and cell["postprocess"]["max_size_fraction"] == 0.4
    assert set(cell["source"]) >= {"repo_id", "repo_type", "revision", "filename", "sha256"}
    info = {"num_classes": 1, "class_names": ["well"], "end2end": False, "max_det": 300, "stride": 32, "head": "Detect"}
    det = e.detector_entry(rows["det_v1"], file="d.onnx", size=1, digest="0" * 64, check=None, info=info)
    assert keys <= set(det) and det["input"]["shape"] == ["N", 3, "H", "W"] and det["output"]["shape"][:2] == ["N", 5]
    assert det["preprocess"]["letterbox"]["pad_value"] == 114 and det["preprocess"]["letterbox"]["padding"].startswith("rect")
    static = e.detector_entry(rows["det_v1"], file="d.onnx", size=1, digest="0" * 64, check=None, info=info, dynamic=False)
    assert static["output"]["shape"] == [1, 5, 8400] and static["input"]["shape"] == [1, 3, 640, 640]
    json.dumps([cell, det, static], allow_nan=False)


def test_main_refuses_without_weights_and_never_downloads(tmp_path, catalogue, monkeypatch, capsys):
    path, _, _ = catalogue
    monkeypatch.setattr(e, "download_weights", lambda *a, **k: pytest.fail("downloaded without --download"))
    code = e.main(["--catalogue", str(path), "--key", "fam_v1", "--out", str(tmp_path / "out")])
    assert code == 2 and "--download" in capsys.readouterr().err
    assert not (tmp_path / "out" / "browser-manifest.json").exists()


def test_main_refuses_mismatched_weights_before_loading(tmp_path, catalogue, monkeypatch):
    path, _, _ = catalogue
    wrong = tmp_path / "wrong"
    wrong.write_bytes(b"not the published checkpoint")
    monkeypatch.setattr(e, "export_one", lambda *a, **k: pytest.fail("loaded unverified weights"))
    assert e.main(["--catalogue", str(path), "--key", "fam_v1", "--weights", f"fam_v1={wrong}", "--out", str(tmp_path)]) == 2


def test_main_writes_manifest_for_verified_weights(tmp_path, catalogue, monkeypatch):
    path, _, contents = catalogue
    directory = tmp_path / "w"
    directory.mkdir()
    (directory / "fam_v1_file").write_bytes(contents["fam_v1"])
    (directory / "best.pt").write_bytes(contents["det_v1"])
    monkeypatch.setattr(e, "export_one", lambda row, weights, out, **k: {"file": row["key"] + ".onnx", "weights": str(weights)})
    out = tmp_path / "out"
    assert e.main(["--catalogue", str(path), "--all", "--weights-dir", str(directory), "--out", str(out)]) == 0
    manifest = json.loads((out / "browser-manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_url"] is None and sorted(manifest["models"]) == ["det_v1", "fam_v1"]


def test_content_named_output_replaces_older_export(tmp_path):
    (tmp_path / "k_v1-000000000000.onnx").write_bytes(b"old")
    staged = tmp_path / ".k_v1.onnx.partial"
    staged.write_bytes(b"new")
    name, size, digest = e._publish_file(staged, tmp_path, "k_v1")
    assert name == f"k_v1-{digest[:12]}.onnx" and size == 3
    assert [p.name for p in tmp_path.iterdir()] == [name]


def test_tiny_network_round_trips_through_onnxruntime(tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    net = _TinyNet()
    destination = tmp_path / "tiny.onnx"
    e.export_cellpose(net, destination)
    result = e.check_cellpose(net, destination)
    assert result["passed"] and result["outputs"]["flows"]["max_abs_diff"] < 1e-4
