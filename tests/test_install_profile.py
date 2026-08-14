"""Installer backend persistence and doctor reporting."""
from __future__ import annotations

import json
import types

import pytest

from spacr import doctor, install_profile


def _fake_torch(*, cuda=False, mps=False, cuda_build=None):
    return types.SimpleNamespace(
        __version__="9.8.7",
        version=types.SimpleNamespace(cuda=cuda_build),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
    )


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    ((True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")),
)
def test_build_profile_records_actual_torch_backend(
    monkeypatch, cuda, mps, expected
):
    fake = _fake_torch(cuda=cuda, mps=mps, cuda_build="12.8" if cuda else None)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)

    profile = install_profile.build_profile("auto", "nvidia")

    assert profile["schema"] == install_profile.PROFILE_SCHEMA
    assert profile["requested_backend"] == "auto"
    assert profile["active_backend"] == expected
    assert profile["cuda_available"] is cuda
    assert profile["mps_available"] is mps


@pytest.mark.parametrize(
    ("requested", "detected"),
    (("../cpu", "nvidia"), ("cpu", "amd"), ("", "none")),
)
def test_build_profile_rejects_unknown_values(requested, detected):
    with pytest.raises(ValueError, match="unsupported"):
        install_profile.build_profile(requested, detected)


def test_write_and_read_profile_roundtrip_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(
        install_profile, "_torch_facts", lambda: {"active_backend": "cpu"}
    )
    target = tmp_path / "nested" / "install-profile.json"

    written = install_profile.write_profile(target, "cpu", "none")

    assert install_profile.read_profile(target) == written
    assert json.loads(target.read_text(encoding="utf-8")) == written
    assert list(target.parent.glob("*.tmp")) == []
    assert written["consent"] == {
        "collected": False,
        "share_diagnostics": False,
        "report_issues": False,
        "sign_in_now": False,
    }


def test_consent_choices_roundtrip_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(
        install_profile, "_torch_facts", lambda: {"active_backend": "cpu"}
    )
    target = tmp_path / "profile.json"
    install_profile.write_profile(
        target,
        "cpu",
        "none",
        consent_collected=True,
        share_diagnostics=True,
        report_issues=True,
        sign_in_now=True,
    )
    assert install_profile.read_profile(target)["consent"] == {
        "collected": True,
        "share_diagnostics": True,
        "report_issues": True,
        "sign_in_now": True,
    }


def test_explicit_uv_backend_name_is_preserved(monkeypatch):
    monkeypatch.setattr(
        install_profile, "_torch_facts", lambda: {"active_backend": "cuda"}
    )
    assert install_profile.build_profile("cu128", "nvidia")[
        "requested_backend"
    ] == "cu128"


@pytest.mark.parametrize(
    "contents",
    ("not json", "[]", '{"schema": 99}', '{"schema": 1}'),
)
def test_read_profile_treats_malformed_or_unknown_schema_as_absent(
    tmp_path, contents
):
    target = tmp_path / "install-profile.json"
    target.write_text(contents, encoding="utf-8")
    assert install_profile.read_profile(target) is None


def test_default_profile_path_honours_explicit_override(tmp_path, monkeypatch):
    target = tmp_path / "chosen.json"
    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(target))
    assert install_profile.default_profile_path() == target


def test_doctor_reports_installer_choice(monkeypatch):
    profile = {
        "schema": 1,
        "requested_backend": "cpu",
        "active_backend": "cpu",
        "detected_accelerator": "none",
    }
    monkeypatch.setattr(install_profile, "read_profile", lambda _path: profile)

    row = doctor.check_installer_backend(doctor.Context())

    assert row.status == doctor.PASS
    assert "selected cpu" in row.message
    assert "uses cpu" in row.message


def test_doctor_warns_when_accelerated_choice_fell_back_to_cpu(monkeypatch):
    profile = {
        "schema": 1,
        "requested_backend": "auto",
        "active_backend": "cpu",
        "detected_accelerator": "nvidia",
    }
    monkeypatch.setattr(install_profile, "read_profile", lambda _path: profile)

    row = doctor.check_installer_backend(doctor.Context())

    assert row.status == doctor.WARN
    assert "13x" in row.fix and "20x" in row.fix


def test_doctor_skips_non_installer_environment(monkeypatch):
    monkeypatch.setattr(install_profile, "read_profile", lambda _path: None)
    assert doctor.check_installer_backend(doctor.Context()).status == doctor.SKIP
