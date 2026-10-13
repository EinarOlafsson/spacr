"""The update channel, the What's new notes, and proxy/CA export, with no network."""
import io
import json
import os
import ssl
import urllib.request

import pytest

from spacr import doctor, updater


@pytest.fixture(autouse=True)
def _network_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    for key in (updater._PROXY_VARIABLES + updater._CA_VARIABLES
                + updater._NO_PROXY_VARIABLES):
        monkeypatch.delenv(key, raising=False)
    updater._NETWORK_EXPORTED.clear()
    updater._NETWORK_DISPLACED.clear()
    yield
    updater._undo_network_exports()


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _pypi_payload():
    return {"info": {"version": "1.5.1.3"},
            "releases": {"1.5.1.3": [{"yanked": False}],
                         "1.5.2rc1": [{"yanked": False}],
                         "1.6.0": [{"yanked": True}],
                         "1.7.0": [],
                         "not-a-version": [{}]}}


def test_stable_offers_the_release_and_nightly_the_newest_prerelease():
    assert updater._newest_on_channel(_pypi_payload(), "stable") == "1.5.1.3"
    assert updater._newest_on_channel(_pypi_payload(), "nightly") == "1.5.2rc1"


@pytest.mark.parametrize("channel,expected", [("stable", "1.5.1.3"),
                                              ("nightly", "1.5.2rc1")])
def test_check_for_updates_follows_the_channel(monkeypatch, channel, expected):
    def fake_urlopen(request, timeout=None):
        if "pypi" in request.full_url:
            return _Answer(json.dumps(_pypi_payload()).encode())
        return _Answer(json.dumps({"sha": "abcdef123"}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(updater, "_installed_version", lambda: "1.5.1.3")
    info = (updater.check_for_updates(timeout=0.01) if channel == "stable"
            else updater._check_on_channel(channel, timeout=0.01))
    assert info.latest_release == expected
    assert info.upgrade_available is (channel == "nightly")
    assert info.nightly_sha == "abcdef1"


def test_release_notes_between_two_versions_prefer_fetched_records(monkeypatch):
    bundled = [{"tag": "v1.5.1.3", "body": "bundled"},
               {"tag": "v1.5.1.2", "body": "two"},
               {"tag": "v1.5.1.0", "body": "zero"}]
    monkeypatch.setattr(updater, "_bundled_release_notes", lambda: bundled)
    fetched = [{"tag": "v1.5.1.3", "body": "fetched"},
               {"tag": "v1.6.0", "body": "future"}]
    notes = updater._release_notes_between("1.5.1.0", "1.5.1.3", fetched)
    assert [n["body"] for n in notes] == ["fetched", "two"]
    assert updater._release_notes_between("unknown", "1.5.1.3") == []


def test_the_bundled_notes_are_readable():
    notes = updater._bundled_release_notes()
    assert notes and all(updater._release_version(n) for n in notes[:3])


def test_environment_proxy_and_ca_reach_every_downloader(tmp_path, monkeypatch):
    bundle = tmp_path / "corp.pem"
    bundle.write_text("x")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.org:3128")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    network = updater._apply_network_settings()
    assert network["proxy_source"] == "HTTPS_PROXY"
    for key in updater._PROXY_VARIABLES:
        assert os.environ[key] == "http://proxy.example.org:3128"
    for key in updater._CA_VARIABLES:
        assert os.environ[key] == str(bundle)
    assert "localhost" in os.environ["NO_PROXY"]
    assert urllib.request.getproxies()["https"] == "http://proxy.example.org:3128"
    import requests
    session = requests.Session()
    settings = session.merge_environment_settings(
        "https://huggingface.co/x", {}, None, None, None)
    assert settings["verify"] == str(bundle)
    assert settings["proxies"]["https"] == "http://proxy.example.org:3128"


def test_saved_choice_wins_and_clearing_it_restores_the_environment(tmp_path, monkeypatch):
    bundle = tmp_path / "corp.pem"
    bundle.write_text("x")
    monkeypatch.setenv("HTTPS_PROXY", "http://env:1")
    updater._write_network_config("http://saved:2", str(bundle))
    assert json.loads((tmp_path / "network.json").read_text())["proxy"] == "http://saved:2"
    assert os.environ["HTTPS_PROXY"] == "http://saved:2"
    assert os.environ["PIP_CERT"] == str(bundle)
    updater._write_network_config("", "")
    assert os.environ["HTTPS_PROXY"] == "http://env:1"
    assert "PIP_CERT" not in os.environ
    assert "localhost" in os.environ["NO_PROXY"]


def test_a_missing_ca_file_is_not_exported(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "absent.pem"))
    updater._apply_network_settings()
    assert "SSL_CERT_FILE" not in os.environ


def test_installers_inherit_the_proxy(monkeypatch):
    seen = {}

    class Done:
        returncode, stdout, stderr = 0, "ok", ""

    def fake_run(args, **kwargs):
        seen["proxy"] = os.environ.get("HTTPS_PROXY")
        return Done()

    updater._write_network_config("http://saved:2", "")
    monkeypatch.setattr(updater.subprocess, "run", fake_run)
    assert updater.run_install_command(["pip", "install", "x"]) == (0, "ok")
    assert seen["proxy"] == "http://saved:2"


def test_doctor_passes_a_direct_connection_without_a_socket(monkeypatch):
    monkeypatch.setattr(doctor, "_probe_url",
                        lambda *a, **k: pytest.fail("no socket expected"))
    row = doctor._check_network(doctor.Context())
    assert row.status == doctor.PASS


def test_doctor_fails_a_missing_bundle(monkeypatch, tmp_path):
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "absent.pem"))
    row = doctor._check_network(doctor.Context())
    assert row.status == doctor.FAIL and row.fix


def test_doctor_reports_an_unreachable_proxy_and_hides_credentials(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@proxy.example.org:3128")
    monkeypatch.setattr(doctor, "_probe_url", lambda *a, **k: "Connection refused")
    row = doctor._check_network(doctor.Context())
    assert row.status == doctor.WARN and "Connection refused" in row.message
    assert row.fix and "secret" not in " ".join(row.details)


def test_doctor_passes_a_reachable_proxy_with_a_valid_bundle(monkeypatch, tmp_path):
    import certifi
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.org:3128")
    monkeypatch.setattr(doctor, "_probe_url", lambda *a, **k: None)
    row = doctor._check_network(doctor.Context())
    assert row.status == doctor.PASS
    assert any("certificate bundle" in d for d in row.details)


def test_the_probe_goes_through_urllib(monkeypatch):
    opened = []
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: opened.append(req.full_url) or _Answer(b""))
    assert doctor._probe_url("https://pypi.org/simple/spacr/") is None

    def refuse(req, timeout=None):
        raise ssl.SSLError("CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    assert "CERTIFICATE_VERIFY_FAILED" in doctor._probe_url("https://pypi.org")
    assert opened == ["https://pypi.org/simple/spacr/"]
