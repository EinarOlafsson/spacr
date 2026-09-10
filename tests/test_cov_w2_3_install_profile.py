"""The installer's hand-off file: writing it atomically and refusing bad ones.

The desktop installers live outside the environment they build, so the only
durable record of the hardware choice is a small JSON file beside the private
``venv``. A half-written one would make ``spacr-doctor`` report a choice
nobody made, so the write is atomic and the read is fussy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spacr import install_profile as IP


def test_a_torch_build_whose_mps_probe_raises_is_read_as_no_mps(monkeypatch):
    """An optional backend that errors when asked is not an available one."""
    import torch

    class RaisingMps:
        @staticmethod
        def is_available():
            raise RuntimeError("the MPS backend is not built into this torch")

    monkeypatch.setattr(torch.backends, "mps", RaisingMps(), raising=False)
    facts = IP._torch_facts()
    assert facts["mps_available"] is False
    assert facts["active_backend"] in {"cpu", "cuda"}


def test_a_write_that_fails_leaves_no_temporary_file_behind(
        tmp_path, monkeypatch):
    """A failed write must not leave a stray dotfile beside the target.

    The temporary is what makes the write atomic; leaving one behind on
    failure would accumulate one per failed install.
    """
    target = tmp_path / "install-profile.json"

    def failing_replace(src, dst):
        raise OSError("the volume is read-only")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError):
        IP.write_profile(target, "cpu", "none")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_temporary_already_gone_does_not_mask_the_original_failure(
        tmp_path, monkeypatch):
    """Cleanup is best-effort; the write's own error is what propagates."""
    target = tmp_path / "install-profile.json"

    def failing_replace(src, dst):
        os.unlink(src)
        raise OSError("the volume went away mid-write")

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="went away mid-write"):
        IP.write_profile(target, "cpu", "none")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("mutate,why", [
    ({"schema": 99}, "a schema from another version"),
    ({"requested_backend": "CUDA 12 please"}, "a backend that is not a token"),
    ({"active_backend": "vulkan"}, "a backend spaCR cannot run on"),
])
def test_a_profile_that_does_not_match_the_schema_is_read_as_absent(
        tmp_path, mutate, why):
    """Half-right is not right: the reader returns None rather than a guess."""
    target = tmp_path / "install-profile.json"
    IP.write_profile(target, "cpu", "none")
    payload = json.loads(target.read_text())
    payload.update(mutate)
    target.write_text(json.dumps(payload))
    assert IP.read_profile(target) is None, why


def test_a_profile_that_was_written_reads_back_whole(tmp_path):
    """What the installer wrote is what ``spacr-doctor`` gets back."""
    target = tmp_path / "install-profile.json"
    written = IP.write_profile(target, "CPU", "None",
                               consent_collected=True, report_issues=True)
    read = IP.read_profile(target)
    assert read == written
    assert read["requested_backend"] == "cpu"
    assert read["detected_accelerator"] == "none"
    assert read["consent"] == {"collected": True, "share_diagnostics": False,
                               "report_issues": True, "sign_in_now": False}


def test_the_installer_command_writes_the_profile_and_prints_it(
        tmp_path, capsys):
    """The installer reads the JSON off stdout; the file is the durable copy."""
    target = tmp_path / "install-profile.json"
    code = IP.main([
        "--path", str(target),
        "--requested", "cpu",
        "--detected", "none",
        "--consent-collected", "1",
        "--share-diagnostics", "1",
        "--report-issues", "0",
        "--sign-in-now", "1",
    ])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(target.read_text())
    assert printed["consent"] == {"collected": True, "share_diagnostics": True,
                                  "report_issues": False, "sign_in_now": True}


def test_the_installer_command_refuses_an_accelerator_it_does_not_know(tmp_path):
    """``--detected`` is a closed vocabulary, enforced by the parser."""
    with pytest.raises(SystemExit):
        IP.main(["--path", str(tmp_path / "p.json"),
                 "--requested", "cpu", "--detected", "quantum"])


def test_the_profile_path_follows_the_environment_override(monkeypatch, tmp_path):
    """An explicit override wins over the location beside ``sys.prefix``."""
    monkeypatch.setenv("SPACR_INSTALL_PROFILE", str(tmp_path / "elsewhere.json"))
    assert IP.default_profile_path() == tmp_path / "elsewhere.json"
    monkeypatch.setenv("SPACR_INSTALL_PROFILE", "   ")
    assert IP.default_profile_path().name == IP.PROFILE_NAME
    assert IP.read_profile() is None or isinstance(IP.read_profile(), dict)


def test_a_backend_name_that_is_not_a_token_is_refused_before_anything_is_written():
    """The profile is a schema, so both names are checked before the write."""
    with pytest.raises(ValueError, match="unsupported requested backend"):
        IP.build_profile("cuda 12.4", "nvidia")
    with pytest.raises(ValueError, match="unsupported detected accelerator"):
        IP.build_profile("cpu", "a graphics card")


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, False, "cuda"),
    (False, True, "mps"),
    (True, True, "cuda"),
])
def test_the_active_backend_is_the_fastest_one_torch_can_actually_use(
        monkeypatch, cuda, mps, expected):
    """CUDA wins over MPS; the profile records what torch would really use."""
    import torch

    class Backend:
        def __init__(self, available):
            self._available = available

        def is_available(self):
            return self._available

    monkeypatch.setattr(torch, "cuda", Backend(cuda), raising=False)
    monkeypatch.setattr(torch.backends, "mps", Backend(mps), raising=False)
    facts = IP._torch_facts()
    assert facts["active_backend"] == expected
    assert facts["cuda_available"] is cuda
