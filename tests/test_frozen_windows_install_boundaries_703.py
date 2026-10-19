"""Unknown filesystem entries survive frozen Windows replacement failures."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from spacr import install_cleanup as cleanup


def _payload(root, version):
    root.mkdir()
    (root / "spacr.exe").write_text(version)
    metadata = root / "_internal" / ("spacr-" + version + ".dist-info") / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: spacr\nVersion: " + version + "\n")
    (root / "_internal" / "runtime.dll").write_bytes(version.encode())
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in root.rglob("*") if path.is_file()}
    (root / "spacr-payload.json").write_text(json.dumps(
        {"schema": 1, "version": version, "files": files}))
    return root


def _identity(argv):
    root = Path(argv[0]).parent
    Path(argv[-1]).write_text(json.dumps({
        "schema": 1, "version": (root / "spacr.exe").read_text(encoding="utf-8"), "frozen": True,
        "executable": str(root / "spacr.exe"), "bundle_root": str(root),
        "import_root": str(root / "_internal"),
        "module": str(root / "_internal" / "spacr" / "__init__.py"),
    }))
    return 0, ""


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO receipt boundary")
def test_fifo_identity_receipt_is_refused_without_waiting_for_a_writer(tmp_path):
    program = '''import os, sys
from spacr import install_cleanup as cleanup
def fifo(argv):
    os.mkfifo(argv[-1])
    return 0, ''
try:
    cleanup._frozen_windows_identity(sys.argv[1], '1.5.1.5', run=fifo)
except (OSError, ValueError):
    print('FIFO refused')
else:
    raise AssertionError('FIFO receipt was accepted')
'''
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path / "bundle")],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "FIFO refused"


def test_dangling_foreign_link_is_preserved_instead_of_restored_over(tmp_path):
    source = _payload(tmp_path / "source", "1.5.1.5")
    installed = _payload(tmp_path / "installed", "1.5.1.4")
    missing = tmp_path / "uncreated-user-file"
    sentinel = installed / "unrelated.csv"
    sentinel.write_bytes(b"user output")
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(missing)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is unavailable")
    probe.unlink()

    def replace_activated(argv):
        if Path(argv[0]).parent == installed:
            executable = installed / "spacr.exe"
            executable.unlink()
            executable.symlink_to(missing)
            return 2, "controlled import refusal"
        return _identity(argv)

    with pytest.raises(ValueError):
        cleanup._install_frozen_windows(source, installed, "1.5.1.5", run=replace_activated)
    assert (installed / "spacr.exe").is_symlink()
    assert os.readlink(installed / "spacr.exe") == str(missing)
    assert not missing.exists()
    assert sentinel.read_bytes() == b"user output"
    backups = list(tmp_path.glob(".spacr-frozen-install-*/previous/spacr.exe"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "1.5.1.4"
    assert (tmp_path / ".installed.spacr-install-lock").is_dir()
    assert (backups[0].parent.parent / "recovery.json").is_file()


def test_known_old_manifest_does_not_authorize_removing_unknown_empty_directory(tmp_path):
    source = _payload(tmp_path / "source", "1.5.1.5")
    installed = _payload(tmp_path / "installed", "1.5.1.4")
    user_directory = installed / "_internal" / "user-created-empty-directory"
    user_directory.mkdir()
    original_manifest = (installed / "spacr-payload.json").read_bytes()
    result = cleanup._install_frozen_windows(source, installed, "1.5.1.5", run=_identity)
    assert result["ok"] is True
    assert (installed / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.5"
    assert result["previous_runtime"] is not None
    backup = Path(result["previous_runtime"])
    assert (backup / "_internal" / user_directory.name).is_dir()
    assert list((backup / "_internal" / user_directory.name).iterdir()) == []
    assert (backup / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4"
    assert (backup / "spacr-payload.json").read_bytes() == original_manifest


@pytest.mark.parametrize("startup", ["ready", "failed", "uncertain"])
def test_old_manifest_payload_survives_until_actual_desktop_acknowledgment(tmp_path, startup):
    source = _payload(tmp_path / "source", "1.5.1.5")
    installed = _payload(tmp_path / "installed", "1.5.1.4")
    sentinel = installed / "results.csv"
    sentinel.write_bytes(b"user data")
    events = []

    def prepared():
        assert (installed / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4"
        stages = list(tmp_path.glob(".spacr-frozen-install-*/new/spacr.exe"))
        assert len(stages) == 1
        assert stages[0].read_text(encoding="utf-8") == "1.5.1.5"
        events.append("prepared")

    def activated():
        assert (installed / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.5"
        backups = list(tmp_path.glob(".spacr-frozen-install-*/previous/spacr.exe"))
        assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "1.5.1.4"
        events.append("activated")
        if startup == "failed":
            raise RuntimeError("desktop child stopped after startup failure")
        if startup == "uncertain":
            raise cleanup._UpdateActivationUnconfirmed("desktop outcome unknown")

    if startup == "ready":
        result = cleanup._install_frozen_windows(source, installed, "1.5.1.5", run=_identity,
            before_activate=prepared, after_activate=activated, preserve_previous=True)
        assert (Path(result["previous_runtime"]) / "spacr.exe").read_text(encoding="utf-8") == "1.5.1.4"
    else:
        with pytest.raises(RuntimeError):
            cleanup._install_frozen_windows(source, installed, "1.5.1.5", run=_identity,
                before_activate=prepared, after_activate=activated, preserve_previous=True)
    assert events == ["prepared", "activated"]
    assert sentinel.read_bytes() == b"user data"
    assert (installed / "spacr.exe").read_text(encoding="utf-8") == ("1.5.1.4" if startup == "failed" else "1.5.1.5")
    if startup == "uncertain":
        assert (tmp_path / ".installed.spacr-install-lock").is_dir()
        backups = list(tmp_path.glob(".spacr-frozen-install-*/previous/spacr.exe"))
        assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == "1.5.1.4"
