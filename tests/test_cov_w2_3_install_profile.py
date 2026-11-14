"""The installer's hand-off file: writing it atomically and refusing bad ones.

The desktop installers live outside the environment they build, so the only
durable record of the hardware choice is a small JSON file beside the private
``venv``. A half-written one would make ``spacr-doctor`` report a choice
nobody made, so the write is atomic and the read is fussy.
"""
from __future__ import annotations

import json

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
    assert printed == json.loads(target.read_text(encoding="utf-8"))
    assert printed["consent"] == {"collected": True, "share_diagnostics": True,
                                  "report_issues": False}


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
