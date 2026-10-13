"""F546: provision only backend-owned Python after an explicit install."""
import os
import subprocess
import threading
from pathlib import Path

import pytest

from spacr import _segmentation_backends as backend


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Keep interpreter discovery, network probes and installs off the host."""
    monkeypatch.setenv(backend._ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(backend, "_CANDIDATES", {backend._CELLPROFILER: []})
    monkeypatch.setattr(backend, "_PROBED", {})
    monkeypatch.setattr(backend, "_probe_network", lambda: "")
    monkeypatch.setattr(backend, "_provisioning_uv", lambda spec: "/fake/uv"
                        if spec.name == backend._CELLPROFILER else None)
    monkeypatch.setattr(backend.shutil, "disk_usage",
                        lambda root: type("Disk", (), {"free": 10 * 2**30})())
    return tmp_path


def fake_runner(root, calls, *, failure=None, version="3.9", external=False):
    """Emulate uv and package installation without running a subprocess."""
    python = root / ("outside" if external else ".cellprofiler-python") / "python"

    def run(argv, **kwargs):
        """Return command output while preserving the runner's normal shape."""
        calls.append((tuple(argv), kwargs))
        if failure is not None and len(calls) == failure:
            return 1, ["download failed"]
        if "find" in argv:
            python.parent.mkdir(parents=True, exist_ok=True)
            python.write_text("fake interpreter")
            result = [str(python)]
        elif "-c" in argv:
            result = [version]
        elif "venv" in argv:
            env_python = Path(backend._env_python(argv[-1]))
            env_python.parent.mkdir(parents=True, exist_ok=True)
            env_python.write_text("fake environment")
            result = []
        elif "--selftest" in argv:
            result = ['{"ok": true, "python": "3.9", "packages": {}}']
        else:
            result = []
        for line in result:
            kwargs["on_line"](line)
        return 0, result

    return run


def test_local_compatible_python_wins_without_provisioning(isolated, monkeypatch):
    """The existing interpreter route never invokes uv."""
    monkeypatch.setitem(backend._CANDIDATES, backend._CELLPROFILER, [("local",)])
    found = backend._preflight(
        backend._spec("cellprofiler"), str(isolated),
        run=lambda *a, **k: subprocess.CompletedProcess(a, 0, "3.8\n", ""),
        runner=lambda *a, **k: pytest.fail("uv must not run"))
    assert found == ("local",)


def test_managed_python_is_private_verified_and_used_by_install(isolated, monkeypatch):
    """The complete fake install uses managed Python, never the app interpreter."""
    monkeypatch.setenv("PYTHONPATH", "/unrelated")
    monkeypatch.setenv("PYTHONHOME", "/unrelated")
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated")
    calls, progress = [], []
    state = backend._install_backend(
        "cellprofiler", runner=fake_runner(isolated, calls),
        progress=lambda *args: progress.append(args), torch_index="")
    assert state.ready
    managed = isolated / ".cellprofiler-python"
    assert calls[0][0] == ("/fake/uv", "--no-config", "python", "install",
                            "3.9", "--managed-python", "--no-bin", "--no-registry")
    assert "--no-python-downloads" in calls[1][0]
    assert calls[2][0][1:3] == ("-I", "-c")
    assert calls[3][0][:3] == (str(managed / "python"), "-m", "venv")
    for _, kwargs in calls[:3]:
        assert kwargs["env"]["UV_PYTHON_INSTALL_DIR"] == str(managed)
        assert kwargs["env"]["UV_CACHE_DIR"] == str(managed / "cache")
        assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & kwargs["env"].keys()
    assert state.record["interpreter"] == [str(managed / "python")]
    assert (0, 1, "Prepare compatible Python") in progress
    log = (isolated / "cellprofiler.log").read_text()
    assert "--no-bin" in log and "--selftest" in log
    assert not (isolated / "cellprofiler.lock").exists()


@pytest.mark.parametrize("failure", [1, 2, 3])
def test_failed_provision_preserves_existing_environment(isolated, failure):
    """No replacement starts while download, discovery or validation fails."""
    sentinel = isolated / "cellprofiler" / "existing.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve me")
    with pytest.raises(backend._InstallFailed, match="download failed"):
        backend._install_backend("cellprofiler", runner=fake_runner(
            isolated, [], failure=failure), torch_index="")
    assert sentinel.read_text() == "preserve me"
    assert not (isolated / "cellprofiler.lock").exists()
    assert not backend._read_marker(str(sentinel.parent))


@pytest.mark.parametrize("version,external,match", [
    ("3.12", False, "not Python 3.9"),
    ("3.9", True, "outside its backend folder"),
])
def test_invalid_managed_python_is_refused(isolated, version, external, match):
    """A uv response cannot silently select another interpreter."""
    with pytest.raises(backend._InstallBlocked, match=match):
        backend._install_backend("cellprofiler", runner=fake_runner(
            isolated, [], version=version, external=external), torch_index="")
    assert not (isolated / "cellprofiler").exists()


@pytest.mark.parametrize("phase", [0, 1, 2, 3])
def test_cancellation_preserves_existing_environment(isolated, phase):
    """Cancellation at each provisioning boundary never replaces the old env."""
    sentinel = isolated / "cellprofiler" / "existing.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve")
    cancel = threading.Event()
    calls = []
    base = fake_runner(isolated, calls)

    def run(argv, **kwargs):
        """Set cancellation after the selected completed command."""
        assert kwargs["cancel"] is cancel
        result = base(argv, **kwargs)
        if len(calls) == phase:
            cancel.set()
        return result

    if phase == 0:
        cancel.set()
    with pytest.raises(backend._InstallCancelled):
        backend._install_backend("cellprofiler", runner=run, cancel=cancel,
                                 torch_index="")
    assert sentinel.read_text() == "preserve"
    assert len(calls) == phase
    assert not (isolated / "cellprofiler.lock").exists()


def test_network_failure_prevents_provisioning(isolated, monkeypatch):
    """Existing network preflight still blocks before any download command."""
    monkeypatch.setattr(backend, "_probe_network", lambda: "no network")
    with pytest.raises(backend._InstallBlocked, match="no network"):
        backend._install_backend("cellprofiler", runner=lambda *a, **k:
                                 pytest.fail("must not run"))


def test_listing_without_local_python_is_cheap_when_uv_exists(isolated, monkeypatch):
    """Model Zoo offers Install without triggering subprocesses or downloads."""
    monkeypatch.setattr(backend.subprocess, "run", lambda *a, **k:
                        pytest.fail("no subprocess during listing"))
    assert backend._backend_state("cellprofiler").state == backend._INSTALLABLE
    monkeypatch.setattr(backend, "_provisioning_uv", lambda spec: None)
    assert backend._backend_state("cellprofiler").state == backend._UNAVAILABLE


def test_uv_absence_keeps_actionable_manual_python_path(isolated, monkeypatch):
    """Without uv, missing Python is an explicit preflight error."""
    monkeypatch.setattr(backend, "_provisioning_uv", lambda spec: None)
    with pytest.raises(backend._InstallBlocked, match="no Python 3.8 to 3.9"):
        backend._preflight(backend._spec("cellprofiler"), str(isolated))


def test_bundled_uv_precedes_path_and_other_backends_are_unchanged(tmp_path, monkeypatch):
    """Only the online install's adjacent bootstrap or PATH uv is discovered."""
    monkeypatch.setattr(backend.sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(backend.shutil, "which", lambda name: "/path/uv")
    spec = backend._spec("cellprofiler")
    assert backend._provisioning_uv(spec) == "/path/uv"
    uv = tmp_path / "bootstrap" / ("uv.exe" if os.name == "nt" else "uv")
    uv.parent.mkdir()
    uv.write_text("test")
    uv.chmod(0o755)
    assert backend._provisioning_uv(spec) == str(uv)
    assert backend._provisioning_uv(backend._spec("cellpose3")) is None


@pytest.mark.parametrize("cache_link", [False, True])
def test_symlinked_managed_destination_is_refused_before_download(isolated, cache_link):
    """Existing links cannot redirect managed Python or its cache elsewhere."""
    outside = isolated / "unrelated"
    outside.mkdir()
    managed = isolated / ".cellprofiler-python"
    if cache_link:
        managed.mkdir()
    target = managed / "cache" if cache_link else managed
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(backend._InstallBlocked, match="outside its backend"):
        backend._install_backend("cellprofiler", runner=lambda *a, **k:
                                 pytest.fail("no download"))
    assert list(outside.iterdir()) == []


def test_relative_root_resolves_before_provisioning(isolated, monkeypatch):
    """All uv paths and returned interpreter stay absolute for relative input."""
    monkeypatch.chdir(isolated.parent)
    calls = []
    result = backend._provision_python(backend._spec("cellprofiler"),
        isolated.name, runner=fake_runner(isolated, calls))
    assert result == (str(isolated / ".cellprofiler-python" / "python"),)
    assert all(os.path.isabs(kwargs["cwd"]) for _, kwargs in calls)


def test_failed_reinstall_keeps_finished_backend_usable(isolated):
    """The previous ready marker and executable survive preflight failure."""
    env = isolated / "cellprofiler"
    python = Path(backend._env_python(str(env)))
    python.parent.mkdir(parents=True)
    python.write_text("old Python")
    backend._write_marker(str(env), {"backend": "cellprofiler", "old": True})
    with pytest.raises(backend._InstallFailed):
        backend._install_backend("cellprofiler", reinstall=True,
                                 runner=fake_runner(isolated, [], failure=1))
    assert python.read_text() == "old Python"
    assert backend._backend_state("cellprofiler").record["old"] is True


def test_unusable_local_candidate_falls_back_to_managed_python(isolated, monkeypatch):
    """A discovered Python without ensurepip does not prevent the fallback."""
    monkeypatch.setitem(backend._CANDIDATES, backend._CELLPROFILER, [("broken",)])
    calls = []
    chosen = backend._preflight(backend._spec("cellprofiler"), str(isolated),
        run=lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "no ensurepip"),
        runner=fake_runner(isolated, calls))
    assert chosen == (str(isolated / ".cellprofiler-python" / "python"),)
    assert len(calls) == 3
