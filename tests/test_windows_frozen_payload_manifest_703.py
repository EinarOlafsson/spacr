"""The Windows builder binds the exact payload consumed by its transaction."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "packaging/build_windows.ps1"


def _run(source, version="1.5.1.4"):
    text = BUILDER.read_text()
    assert "$ManifestBuilder = @'\n" in text, "Frozen payload ownership must be recorded before packaging"
    script = text.split("$ManifestBuilder = @'\n", 1)[1].split("\n'@", 1)[0]
    script_path = source.parent / "manifest builder.py"
    script_path.write_text(script, encoding="utf-8")
    return subprocess.run([sys.executable, "-I", str(script_path), str(source), version],
                          capture_output=True, text=True, timeout=15)


def _payload(tmp_path):
    source = tmp_path / "payload"
    (source / "_internal").mkdir(parents=True)
    (source / "spacr.exe").write_bytes(b"controlled executable")
    (source / "_internal/spacr-update-helper.exe").write_bytes(b"controlled onefile helper")
    (source / "_internal/dependency.dll").write_bytes(b"controlled dependency")
    return source


def test_manifest_binds_every_payload_file_and_exact_requested_version(tmp_path):
    source = _payload(tmp_path)
    before = {path.relative_to(source).as_posix(): path.read_bytes()
              for path in source.rglob("*") if path.is_file()}
    result = _run(source)
    assert result.returncode == 0, result.stderr
    protocol = source / "_internal/spacr-uninstall-protocol.json"
    assert json.loads(protocol.read_text()) == {"schema": 1, "adapter": "windows-frozen-uninstall-v1"}
    before[protocol.relative_to(source).as_posix()] = protocol.read_bytes()
    manifest = json.loads((source / "spacr-payload.json").read_text())
    assert manifest == {"schema": 1, "version": "1.5.1.4", "files": {
        name: hashlib.sha256(content).hexdigest() for name, content in before.items()}}
    assert all((source / name).read_bytes() == content for name, content in before.items())
    assert "spacr-payload.json" not in manifest["files"]


@pytest.mark.parametrize("missing", ["spacr.exe", "_internal/spacr-update-helper.exe"])
def test_incomplete_payload_is_not_given_a_manifest(tmp_path, missing):
    source = _payload(tmp_path)
    (source / missing).unlink()
    result = _run(source)
    assert result.returncode != 0
    assert not (source / "spacr-payload.json").exists()


def test_existing_manifest_is_not_overwritten(tmp_path):
    source = _payload(tmp_path)
    manifest = source / "spacr-payload.json"
    manifest.write_bytes(b"original ownership record")
    result = _run(source)
    assert result.returncode != 0
    assert manifest.read_bytes() == b"original ownership record"


def test_linked_payload_is_refused_and_target_is_preserved(tmp_path):
    source = _payload(tmp_path)
    unrelated = tmp_path / "user-data"
    unrelated.write_bytes(b"unrelated bytes")
    (source / "_internal/linked-data").symlink_to(unrelated)
    result = _run(source)
    assert result.returncode != 0
    assert not (source / "spacr-payload.json").exists()
    assert unrelated.read_bytes() == b"unrelated bytes"
