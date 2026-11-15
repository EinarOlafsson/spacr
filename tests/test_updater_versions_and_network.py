"""Table-driven tests of the version comparison and the proxy/certificate exports in ``spacr.updater``.

``UpdateInfo.upgrade_available`` is walked over parseable, unparseable and
missing versions; the network exports are applied from a saved
``network.json`` and undone, including the case where the user changed a
variable after spaCR exported it, and the doctor's certificate check reads
the same exports.  No socket is opened.
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

from spacr import doctor, updater


@pytest.fixture()
def clean_network(monkeypatch, tmp_path):
    """No proxy, no-proxy or CA variable in the environment; a private network.json path."""
    for key in (updater._PROXY_VARIABLES + updater._NO_PROXY_VARIABLES
                + updater._CA_VARIABLES):
        monkeypatch.delenv(key, raising=False)
    config = tmp_path / "network.json"
    monkeypatch.setenv(updater._ENV_NETWORK_CONFIG, str(config))
    monkeypatch.setattr(updater, "_NETWORK_EXPORTED", {})
    monkeypatch.setattr(updater, "_NETWORK_DISPLACED", {})
    yield config
    updater._undo_network_exports()


def test_update_info_upgrade_available_compares_versions_with_a_plain_fallback():
    """PEP 440 versions compare normally; unparseable ones fall back to their numeric parts."""
    cases = [
        ("newer-release", "1.4.1", "1.4.2", True),
        ("four-part-newer", "1.4.1", "1.4.1.1", True),
        ("same", "9.9.9", "9.9.9", False),
        ("older-release", "2.0", "1.0", False),
        ("no-release", "1.0.0", None, False),
        ("empty-release", "1.0.0", "", False),
        ("installed-unparseable", "not.a.version!", "1.5.0", True),
        ("latest-unparseable", "1.5.0", "garbage!", False),
        ("both-unparseable", "x!", "y!", False),
    ]
    for case_id, installed, latest, expected in cases:
        info = updater.UpdateInfo(installed, latest, None)
        assert info.upgrade_available is expected, case_id
        assert info.error is None


def test_network_exports_are_applied_then_undone_unless_changed_since(
        clean_network, monkeypatch, tmp_path, subtests):
    """Undo restores or removes what spaCR exported, leaves a user's later edit alone, and clears its memory."""
    undo_cases = [
        # id, exported, displaced, value now, expected value after undo
        ("changed-since-is-kept", {"HTTPS_PROXY": "http://a"},
         {"HTTPS_PROXY": None}, "http://user-set", "http://user-set"),
        ("unchanged-new-variable-is-removed", {"HTTPS_PROXY": "http://a"},
         {"HTTPS_PROXY": None}, "http://a", None),
        ("unchanged-displaced-value-is-restored", {"HTTPS_PROXY": "http://a"},
         {"HTTPS_PROXY": "http://before"}, "http://a", "http://before"),
    ]
    for case_id, exported, displaced, now, after in undo_cases:
        with subtests.test(msg=case_id):
            updater._NETWORK_EXPORTED.update(exported)
            updater._NETWORK_DISPLACED.update(displaced)
            monkeypatch.setenv("HTTPS_PROXY", now)
            updater._undo_network_exports()
            assert os.environ.get("HTTPS_PROXY") == after
            assert updater._NETWORK_EXPORTED == {}
            assert updater._NETWORK_DISPLACED == {}
            monkeypatch.delenv("HTTPS_PROXY", raising=False)

    def broken(opener):
        raise RuntimeError("no opener")

    proxy_cases = [
        # id, NO_PROXY already set, expected NO_PROXY / no_proxy after apply
        ("existing-no-proxy-kept", "intranet", ("intranet", None)),
        ("local-hosts-exempted", None,
         (updater._LOCAL_HOSTS, updater._LOCAL_HOSTS)),
    ]
    for case_id, no_proxy, expected_no_proxy in proxy_cases:
        with subtests.test(msg=case_id):
            clean_network.write_text(json.dumps(
                {"proxy": " http://proxy:3128 "}), encoding="utf-8")
            if no_proxy is not None:
                monkeypatch.setenv("NO_PROXY", no_proxy)
            monkeypatch.setattr(urllib.request, "install_opener", broken)
            network = updater._apply_network_settings()
            assert network == {"proxy": "http://proxy:3128", "ca_bundle": "",
                               "proxy_source": "preferences", "ca_source": ""}
            assert {k: os.environ.get(k) for k in updater._PROXY_VARIABLES} \
                == dict.fromkeys(updater._PROXY_VARIABLES, "http://proxy:3128")
            assert (os.environ.get("NO_PROXY"),
                    os.environ.get("no_proxy")) == expected_no_proxy
            updater._undo_network_exports()
            assert [os.environ.get(k) for k in updater._PROXY_VARIABLES] == \
                [None] * 4
            assert os.environ.get("NO_PROXY") == no_proxy
            assert os.environ.get("no_proxy") is None
            monkeypatch.delenv("NO_PROXY", raising=False)

    with subtests.test(msg="certificate-bundle-not-pem"):
        bundle = tmp_path / "ca.pem"
        bundle.write_text("not a certificate", encoding="utf-8")
        clean_network.write_text(json.dumps({"ca_bundle": str(bundle)}),
                                 encoding="utf-8")
        result = doctor._check_network(doctor.Context(probe_gpu=False))
        assert result.status == doctor.FAIL
        assert result.message.startswith(
            f"The certificate bundle {bundle} is not a readable PEM file: ")
        assert result.details == (
            f"certificate bundle {bundle} (from preferences)",)
        assert {k: os.environ.get(k) for k in updater._CA_VARIABLES} == \
            dict.fromkeys(updater._CA_VARIABLES, str(bundle))
        updater._undo_network_exports()
        assert [os.environ.get(k) for k in updater._CA_VARIABLES] == \
            [None] * len(updater._CA_VARIABLES)
