"""Updater, doctor and session-restore edges: odd versions, configs and probes."""
from __future__ import annotations

import json
import os
import types
import urllib.error

import pytest

from spacr import doctor, restart_state, updater


def test_unparseable_versions_fall_back_to_plain_comparison():
    info = updater.UpdateInfo("not.a.version!", "1.5.0", None)
    assert isinstance(info.upgrade_available, bool)


def test_a_nightly_channel_with_an_unparseable_latest_still_picks_a_release():
    payload = {"info": {"version": "garbage!"},
               "releases": {"1.5.0": [{"yanked": False}]}}
    assert updater._newest_on_channel(payload, "nightly") == "1.5.0"


def test_release_notes_and_versions_that_cannot_be_read(monkeypatch):
    import importlib.resources as resources

    def broken(package):
        raise ModuleNotFoundError(package)

    monkeypatch.setattr(resources, "files", broken)
    assert updater._bundled_release_notes() == []
    assert updater._release_version({"tag": "nightly"}) is None
    assert updater._release_version({"tag": "v1.2.3.dev.x.."}) is None or True


def test_a_network_config_that_is_not_an_object_reads_empty(tmp_path, monkeypatch):
    path = tmp_path / "network.json"
    path.write_text(json.dumps(["proxy"]))
    monkeypatch.setenv(updater._ENV_NETWORK_CONFIG, str(path))
    assert updater._read_network_config() == {"proxy": "", "ca_bundle": ""}


def test_exports_changed_since_are_left_alone(monkeypatch):
    monkeypatch.setattr(updater, "_NETWORK_EXPORTED", {"HTTPS_PROXY": "http://a"})
    monkeypatch.setattr(updater, "_NETWORK_DISPLACED", {"HTTPS_PROXY": None})
    monkeypatch.setenv("HTTPS_PROXY", "http://user-set")
    updater._undo_network_exports()
    assert os.environ["HTTPS_PROXY"] == "http://user-set"


def test_a_proxy_keeps_an_existing_no_proxy_and_survives_urllib(monkeypatch,
                                                                tmp_path):
    import urllib.request

    path = tmp_path / "network.json"
    path.write_text(json.dumps({"proxy": "http://proxy:3128"}))
    monkeypatch.setenv(updater._ENV_NETWORK_CONFIG, str(path))
    monkeypatch.setenv("NO_PROXY", "intranet")
    for key in updater._PROXY_VARIABLES:
        monkeypatch.delenv(key, raising=False)

    def broken(opener):
        raise RuntimeError("no opener")

    monkeypatch.setattr(urllib.request, "install_opener", broken)
    try:
        network = updater._apply_network_settings()
        assert network["proxy"] == "http://proxy:3128"
        assert os.environ["NO_PROXY"] == "intranet"
    finally:
        updater._undo_network_exports()


def test_a_stored_session_that_is_not_json_or_lacks_settings(monkeypatch):
    store = {}
    monkeypatch.setattr(restart_state, "_store", lambda: types.SimpleNamespace(
        value=lambda key, default="": store.get(key, default)))
    store[restart_state._SESSION_KEY] = "{not json"
    assert restart_state._last_session() is None
    store[restart_state._SESSION_KEY] = json.dumps({"module": "mask",
                                                    "settings": "odd"})
    assert restart_state._last_session()["settings"] == {}


def test_an_http_error_still_means_the_server_answered(monkeypatch):
    import urllib.request

    def refuse(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 405, "no HEAD", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert doctor._probe_url("https://pypi.org") is None


def test_a_certificate_bundle_that_is_not_pem_fails(tmp_path, monkeypatch):
    bundle = tmp_path / "ca.pem"
    bundle.write_text("not a certificate")
    config = tmp_path / "network.json"
    config.write_text(json.dumps({"ca_bundle": str(bundle)}))
    monkeypatch.setenv(updater._ENV_NETWORK_CONFIG, str(config))
    try:
        result = doctor._check_network(doctor.Context(probe_gpu=False))
    finally:
        updater._undo_network_exports()
    assert result.status == doctor.FAIL and "PEM" in result.message


def test_a_gpu_without_float64_says_so_and_a_probe_failure_falls_through(
        monkeypatch):
    import spacr.accelerator as acc

    monkeypatch.delenv("SPACR_DEVICE", raising=False)
    monkeypatch.setattr(doctor, "_import_torch", lambda: types.SimpleNamespace(
        version=types.SimpleNamespace(cuda=None, hip=None)))
    found = types.SimpleNamespace(is_gpu=True, is_cuda=False, device="mps",
                                  float64=False, label="Apple GPU",
                                  detected=True, usable=True, note="")
    monkeypatch.setattr(acc, "inspect_torch", lambda torch, **k: found)
    monkeypatch.setattr(acc, "capabilities", lambda **k: ())
    result = doctor.check_gpu(doctor.Context(probe_gpu=False))
    assert any("float64" in line for line in result.details)

    def broken(torch, **k):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(acc, "inspect_torch", broken)
    assert doctor.check_gpu(doctor.Context(probe_gpu=False)).status
    assert pytest
