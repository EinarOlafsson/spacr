"""Real disposable provisioning files with bounded public-transport doubles only."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import urllib.request

import pytest

from tools.web_mask_generator import provision as p


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, url, *, advertised=None):
        super().__init__(raw)
        self.url = url
        self.headers = {} if advertised is None else {"Content-Length": str(advertised)}

    def geturl(self):
        return self.url


@pytest.fixture
def specimen(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "bound.py").write_text("source-bound fixture\n")
    monkeypatch.setattr(p, "SOURCE_FILES", ("bound.py",))
    contents = {"family_v1": b"first exact checkpoint", "family_v2": b"second exact checkpoint"}
    rows = [{"family": "family", "key": key, "kind": "cellpose", "repo_type": "model",
             "repo_id": "owner/approved", "filename": "weights/" + key,
             "sha256": hashlib.sha256(raw).hexdigest(), "revision": str(i + 1) * 40,
             "bytes": len(raw)} for i, (key, raw) in enumerate(contents.items())]
    identities = [{key: value for key, value in row.items() if key not in ("revision", "bytes")} for row in rows]
    monkeypatch.setattr(p, "_catalogue", lambda: copy.deepcopy(identities))
    manifest = {"schema": 1, "catalogue_sha256": p._sha(p._encoded(identities)),
                "source_sha256": p._source_pins(root), "models": rows}
    manifest_path = tmp_path / "models.json"
    manifest_path.write_bytes(p._encoded(manifest))
    cache = tmp_path / "controlled-cache"
    calls = []

    def transport(url, timeout):
        assert 0 < timeout <= 60
        assert url.startswith("https://huggingface.co/owner/approved/resolve/")
        key = url.rsplit("/", 1)[-1]
        calls.append(key)
        return Response(contents[key], url, advertised=len(contents[key]))

    monkeypatch.setattr(p, "_transport", transport)
    return SimpleNamespace(root=root, contents=contents, manifest=manifest, path=manifest_path,
                           cache=cache, calls=calls, transport=transport)


def namespace(s):
    return s.cache / p._sha(s.path.read_bytes())


def rewrite(s):
    s.path.write_bytes(p._encoded(s.manifest))


def test_actual_complete_files_and_source_receipt_then_verified_restart_without_network(specimen, monkeypatch):
    s = specimen
    paths = p.provision_models(s.path, s.cache, root=s.root)
    assert s.calls == ["family_v1", "family_v2"]
    assert set(paths) == set(s.contents)
    for key, name in paths.items():
        assert Path(name).read_bytes() == s.contents[key]
        assert Path(name).stat().st_mode & 0o077 == 0
    saved = (namespace(s) / "paths.json").read_bytes()
    assert json.loads(saved)["source_sha256"] == s.manifest["source_sha256"]
    monkeypatch.setattr(p, "_transport", lambda *a: pytest.fail("verified restart downloaded again"))
    assert p.provision_models(s.path, s.cache, root=s.root) == paths
    assert (namespace(s) / "paths.json").read_bytes() == saved


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown", "repo_type", "repository", "filename",
                                    "revision", "sha", "size_bool", "size_limit", "source", "schema"])
def test_invalid_manifest_refuses_before_network_or_cache_mutation(specimen, change):
    s = specimen
    if change == "missing":
        s.manifest["models"].pop()
    elif change == "duplicate":
        s.manifest["models"].append(copy.deepcopy(s.manifest["models"][0]))
    elif change == "unknown":
        s.manifest["models"][0]["key"] = "unadvertised"
    elif change == "repo_type":
        s.manifest["models"][0]["repo_type"] = "dataset"
    elif change == "repository":
        s.manifest["models"][0]["repo_id"] = "other/unapproved"
    elif change == "filename":
        s.manifest["models"][0]["filename"] = "../escape"
    elif change == "revision":
        s.manifest["models"][0]["revision"] = "main"
    elif change == "sha":
        s.manifest["models"][0]["sha256"] = "0" * 64
    elif change == "size_bool":
        s.manifest["models"][0]["bytes"] = True
    elif change == "size_limit":
        s.manifest["models"][0]["bytes"] = p.MAX_MODEL_BYTES + 1
    elif change == "source":
        (s.root / "bound.py").write_text("changed actual source\n")
    else:
        s.manifest["schema"] = True
    rewrite(s)
    with pytest.raises(p.ProvisioningError):
        p.provision_models(s.path, s.cache, root=s.root)
    assert s.calls == [] and not s.cache.exists()


def test_duplicate_json_fields_refuse_before_network(specimen):
    s = specimen
    s.path.write_text('{"schema":1,"schema":1}')
    with pytest.raises(p.ProvisioningError, match="duplicate_json_key"):
        p.provision_models(s.path, s.cache, root=s.root)
    assert s.calls == [] and not s.cache.exists()


@pytest.mark.parametrize("fault", ["truncated", "oversized", "hash", "declared_size", "redirect", "timeout"])
def test_failed_second_download_preserves_first_bytes_and_never_publishes_partial_paths(specimen, monkeypatch, fault):
    s = specimen

    def transport(url, timeout):
        if url.endswith("family_v1"):
            return s.transport(url, timeout)
        if fault == "timeout":
            raise TimeoutError("controlled startup transport timeout")
        raw = s.contents["family_v2"]
        if fault == "truncated":
            raw = raw[:-1]
        elif fault == "oversized":
            raw += b"extra"
        elif fault == "hash":
            raw = b"x" * len(raw)
        return Response(raw, "https://untrusted.example/escape" if fault == "redirect" else url,
                        advertised=len(raw) + 1 if fault == "declared_size" else None)

    monkeypatch.setattr(p, "_transport", transport)
    with pytest.raises((p.ProvisioningError, TimeoutError)):
        p.provision_models(s.path, s.cache, root=s.root)
    row = s.manifest["models"][0]
    assert (namespace(s) / (row["key"] + "_" + row["sha256"])).read_bytes() == s.contents["family_v1"]
    assert not (namespace(s) / "paths.json").exists()
    assert not list(namespace(s).glob(".provision-*"))
    assert len(list(namespace(s).iterdir())) == 1


@pytest.mark.parametrize("fault", ["corrupt", "symlink", "foreign_receipt", "foreign_entry"])
def test_existing_foreign_or_corrupt_cache_is_not_overwritten(specimen, monkeypatch, fault):
    s = specimen
    paths = p.provision_models(s.path, s.cache, root=s.root)
    first = Path(paths["family_v1"])
    other = s.root / "unrelated"
    other.write_bytes(b"unrelated preserved data")
    if fault == "corrupt":
        first.write_bytes(b"x" * len(s.contents["family_v1"]))
    elif fault == "symlink":
        first.unlink()
        first.symlink_to(other)
    elif fault == "foreign_receipt":
        (namespace(s) / "paths.json").write_bytes(b"foreign receipt")
    else:
        (namespace(s) / "foreign").write_bytes(b"foreign bytes")
    snapshot = {str(x): x.read_bytes() for x in namespace(s).iterdir()}
    monkeypatch.setattr(p, "_transport", lambda *a: pytest.fail("foreign cache reached network"))
    with pytest.raises((p.ProvisioningError, OSError)):
        p.provision_models(s.path, s.cache, root=s.root)
    assert {str(x): x.read_bytes() for x in namespace(s).iterdir()} == snapshot
    assert other.read_bytes() == b"unrelated preserved data"


@pytest.mark.parametrize("fault", ["foreign_root", "root_symlink", "no_space", "deadline"])
def test_root_capacity_and_deadline_refusals_do_not_download(specimen, monkeypatch, fault):
    s = specimen
    if fault == "foreign_root":
        s.cache.mkdir(mode=0o700)
        (s.cache / "unrelated").write_bytes(b"keep")
    elif fault == "root_symlink":
        s.cache.symlink_to(s.root, target_is_directory=True)
    elif fault == "no_space":
        monkeypatch.setattr(p.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    else:
        monkeypatch.setattr(p, "STARTUP_SECONDS", 0)
    with pytest.raises((p.ProvisioningError, OSError)):
        p.provision_models(s.path, s.cache, root=s.root)
    assert s.calls == []
    if fault == "foreign_root":
        assert (s.cache / "unrelated").read_bytes() == b"keep"
        assert not (s.cache / ".lease").exists()


def test_real_other_process_cannot_take_an_active_cache_lease(specimen):
    s = specimen
    script = '''
import sys
from tools.web_mask_generator.provision import _cache_lease, ProvisioningError
try:
    with _cache_lease(sys.argv[1]):
        raise SystemExit(9)
except ProvisioningError as error:
    print(error)
    raise SystemExit(0)
'''
    with p._cache_lease(s.cache):
        child = subprocess.run([sys.executable, "-c", script, str(s.cache)], capture_output=True, text=True, timeout=5)
        assert child.returncode == 0, child.stdout + child.stderr
        assert "cache_in_use" in child.stdout
    assert s.calls == []


@pytest.mark.parametrize("url", ["http://huggingface.co/file", "https://huggingface.co.attacker.example/file",
                                  "https://user:password@huggingface.co/file", "https://huggingface.co:444/file",
                                  "file:///tmp/model"])
def test_redirects_are_verified_before_following_them(url):
    request = urllib.request.Request("https://huggingface.co/owner/model/resolve/" + "1" * 40 + "/model")
    with pytest.raises(p.ProvisioningError, match="download_redirect_identity"):
        p._PublicRedirect().redirect_request(request, None, 302, "Found", {}, url)


def test_public_cdn_redirect_has_no_authorization_or_cookie_and_dataset_url_is_correct():
    request = urllib.request.Request("https://huggingface.co/owner/model/resolve/" + "1" * 40 + "/model")
    redirected = p._PublicRedirect().redirect_request(request, None, 302, "Found", {}, "https://cas-bridge.xethub.hf.co/public?signature=public")
    assert redirected.full_url.startswith("https://cas-bridge.xethub.hf.co/")
    assert not any(k.lower() in ("authorization", "cookie") for k in redirected.headers)
    assert p._url({"repo_type": "dataset", "repo_id": "einarolafsson/models", "revision": "2" * 40,
                   "filename": "weights/chosen model"}) == "https://huggingface.co/datasets/einarolafsson/models/resolve/" + "2" * 40 + "/weights/chosen%20model"
    request.add_header("Authorization", "controlled-fixture-token")
    with pytest.raises(p.ProvisioningError, match="download_credentials"):
        p._PublicRedirect().redirect_request(request, None, 302, "Found", {}, "https://cdn-lfs.hf.co/public")


@pytest.mark.parametrize("zero,selected", [(None, None), ("0", "1"), ("1", None), ("yes", "1"), ("1", "0")])
def test_default_and_cpu_startup_never_provision_or_change_operator_paths(monkeypatch, zero, selected):
    for key, value in (("SPACR_WEB_ZERO_GPU", zero), ("SPACR_WEB_PROVISION_MODELS", selected)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("SPACR_WEB_MODELS_JSON", '{"existing":"untouched"}')
    monkeypatch.setattr(p, "provision_models", lambda *a: pytest.fail("unselected startup provisioned"))
    p.managed_startup()
    assert os.environ["SPACR_WEB_MODELS_JSON"] == '{"existing":"untouched"}'


def test_selected_startup_sets_only_complete_map_and_preserves_conflicting_operator_value(monkeypatch, tmp_path):
    monkeypatch.setenv("SPACR_WEB_ZERO_GPU", "1")
    monkeypatch.setenv("SPACR_WEB_PROVISION_MODELS", "1")
    monkeypatch.setenv("SPACR_WEB_PROVISION_CACHE", str(tmp_path))
    monkeypatch.delenv("SPACR_WEB_MODELS_JSON", raising=False)
    expected = {"all_advertised": "/controlled/verified"}
    monkeypatch.setattr(p, "provision_models", lambda *a: expected)
    p.managed_startup()
    assert json.loads(os.environ["SPACR_WEB_MODELS_JSON"]) == expected
    with pytest.raises(p.ProvisioningError, match="conflicting_operator_model_paths"):
        p.managed_startup()
    assert json.loads(os.environ["SPACR_WEB_MODELS_JSON"]) == expected


def test_selected_startup_failure_never_installs_partial_map(monkeypatch):
    monkeypatch.setenv("SPACR_WEB_ZERO_GPU", "1")
    monkeypatch.setenv("SPACR_WEB_PROVISION_MODELS", "1")
    monkeypatch.delenv("SPACR_WEB_MODELS_JSON", raising=False)

    def fail(*args):
        raise p.ProvisioningError("controlled partial acquisition")

    monkeypatch.setattr(p, "provision_models", fail)
    with pytest.raises(p.ProvisioningError):
        p.managed_startup()
    assert "SPACR_WEB_MODELS_JSON" not in os.environ


def test_entry_calls_bootstrap_before_main_and_never_on_import(monkeypatch):
    events = []
    fake = ModuleType("tools.web_mask_generator.app")
    fake.main = lambda: events.append("main")
    monkeypatch.setitem(sys.modules, "tools.web_mask_generator.app", fake)
    monkeypatch.setattr(p, "managed_startup", lambda: events.append("provision"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delenv("SPACR_WEB_ZERO_GPU", raising=False)
    source = Path(__file__).parents[1] / "tools/web_mask_generator/space_app.py"
    runpy.run_path(str(source), run_name="imported_entry")
    assert events == []
    runpy.run_path(str(source), run_name="__main__")
    assert events == ["provision", "main"]


def test_generator_accepts_only_verified_public_metadata_and_is_deterministic(specimen, tmp_path):
    s = specimen
    rows = []
    for row in s.manifest["models"]:
        rows.append({"key": row["key"], "uri_repo_type": row["repo_type"], "uri_repo_id": row["repo_id"],
                     "filename": row["filename"], "sha256": row["sha256"], "revision": row["revision"],
                     "private": False, "gated": False, "catalog_sha_matches_lfs": True,
                     "file_metadata": {"size": row["bytes"], "lfs": {"sha256": row["sha256"]}}})
    snapshot = tmp_path / "public-metadata.json"
    snapshot.write_bytes(p._encoded({"models": rows}))
    first = p.prepare_manifest(snapshot, root=s.root)
    assert first == s.manifest == p.prepare_manifest(snapshot, root=s.root)
    rows[0]["private"] = True
    snapshot.write_bytes(p._encoded({"models": rows}))
    with pytest.raises(p.ProvisioningError, match="unverified_public_metadata"):
        p.prepare_manifest(snapshot, root=s.root)
    assert s.calls == [] and not s.cache.exists()


@pytest.mark.parametrize("filename,kind,suffix", [("weights/published_detector.pt", "detector", ".pt"),
                                                 ("weights/published_cpsam", "cellpose", ""),
                                                 ("weights/legacy.CP_model", "cellpose", ".CP_model")])
def test_provisioned_paths_retain_the_real_loader_format(specimen, monkeypatch, filename, kind, suffix):
    s = specimen
    s.manifest["models"][0]["filename"] = filename
    s.manifest["models"][0]["kind"] = kind
    identities = [{key: value for key, value in row.items() if key not in ("revision", "bytes")}
                  for row in s.manifest["models"]]
    monkeypatch.setattr(p, "_catalogue", lambda: copy.deepcopy(identities))
    s.manifest["catalogue_sha256"] = p._sha(p._encoded(identities))
    rewrite(s)
    by_url = {p._url(row): s.contents[row["key"]] for row in s.manifest["models"]}
    monkeypatch.setattr(p, "_transport", lambda url, timeout: Response(by_url[url], url))
    paths = p.provision_models(s.path, s.cache, root=s.root)
    path = Path(paths["family_v1"])
    if kind == "detector":
        from spacr.plaque import _load_detector
        calls = []

        def yolo(weights):
            if str(weights).rpartition(".")[-1] != "pt":
                raise ValueError("controlled vendor .pt dispatch refused unsupported checkpoint format")
            calls.append(weights)
            assert Path(weights).read_bytes() == s.contents["family_v1"]
            return "controlled detector model"

        module = ModuleType("ultralytics")
        module.YOLO = yolo
        monkeypatch.setitem(sys.modules, "ultralytics", module)
        assert _load_detector(str(path)) == "controlled detector model"
        assert calls == [str(path)]
    assert path.suffix == suffix
    assert path.read_bytes() == s.contents["family_v1"]
    assert json.loads((namespace(s) / "paths.json").read_bytes())["models"] == paths
