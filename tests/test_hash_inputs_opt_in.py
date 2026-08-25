"""Input hashing is opt-in, and the manifest says which it was.

Hashing every file under every path-valued setting is proportional to the
DATA, not to the run. Measured on 126 MB of TIFFs with a warm page cache it
adds ~0.55 s/GB; cold it is bounded by disk read speed, and it reads every
file up front whether or not the run itself will ever touch them. That is
where "sometimes takes a very long time" comes from.

So it is off unless asked for. What must never happen is a manifest that
QUIETLY lacks hashes -- that is indistinguishable from one whose hashes
were computed and matched, and telling those apart is the whole value of
the record.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture()
def source(tmp_path):
    folder = tmp_path / "plate"
    folder.mkdir()
    (folder / "a.tif").write_bytes(b"x" * 4096)
    (folder / "b.tif").write_bytes(b"y" * 4096)
    return folder


def _manifest(run):
    with open(os.path.join(run.dir, "manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_hashing_is_off_unless_asked_for(source):
    from spacr.run_journal import open_run

    with open_run("mask", {"src": str(source)}) as run:
        pass
    manifest = _manifest(run)
    assert manifest["input_hashing"] == "skipped"
    assert manifest["performance"]["input_files"] == 0


def test_hashing_happens_when_asked_for(source):
    from spacr.run_journal import open_run

    with open_run("mask", {"src": str(source), "hash_inputs": True}) as run:
        pass
    manifest = _manifest(run)
    assert manifest["input_hashing"] == "on"
    assert manifest["performance"]["input_files"] >= 2


def test_the_manifest_is_written_either_way(source):
    """A run without hashes still has a reproducibility record."""
    from spacr.run_journal import open_run

    with open_run("mask", {"src": str(source)}) as run:
        pass
    manifest = _manifest(run)
    for key in ("app_key", "env", "settings_sha256", "seeds", "status"):
        assert key in manifest, f"{key} went missing when hashing was off"


def test_the_pipeline_never_reads_qsettings():
    """The preference is passed down as a setting, never read from Qt.

    A `from PySide6 import` in a pipeline module makes the package
    unimportable on a cluster.
    """
    import ast
    import pathlib

    import spacr.run_journal as run_journal

    # Parsed, not grepped. The module names PySide6 in a version-string
    # lookup for the environment snapshot and again in a comment saying not
    # to import it -- a substring search calls both a violation, which is
    # how this test failed on its first run.
    tree = ast.parse(pathlib.Path(run_journal.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "PySide6" not in imported, (
        "run_journal imports Qt; the package becomes unimportable on a "
        "cluster")


def test_the_cli_can_force_it_either_way():
    """Or a headless run cannot reproduce a GUI run."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "spacr.cli", "--help"],
                         capture_output=True, text=True, timeout=300).stdout
    assert "--hash-inputs" in out
    assert "--no-hash-inputs" in out


MESSAGE = "Recording reproducibility input hashes"


def test_the_slow_line_is_only_printed_when_hashing():
    """The line names a pause. With hashing off there is no pause to name.

    Printing it anyway claims a record that was not made, which is the
    same failure as a manifest that silently lacks hashes.

    Asked of the job runner that emits the line -- ``spacr.qt.bridge``, the
    one place a run is started with somebody watching. Parsed rather than
    read as text, and rather than imported: the enclosing ``if`` is the
    thing under test, a character window around the message would pass on a
    condition that merely sits nearby, and importing the module would drag
    Qt into a test that only needs the source.
    """
    import ast
    import pathlib

    import spacr

    bridge = pathlib.Path(spacr.__file__).resolve().parent / "qt" / "bridge.py"
    tree = ast.parse(bridge.read_text(encoding="utf-8"))

    def announces(node):
        return any(isinstance(sub, ast.Constant)
                   and isinstance(sub.value, str)
                   and MESSAGE in sub.value
                   for sub in ast.walk(node))

    assert announces(tree), (
        f"nothing in {bridge.name} announces the hashing pause any more; "
        "either the line moved or the announcement was lost")

    guarded = [node for node in ast.walk(tree)
               if isinstance(node, ast.If)
               and any(announces(stmt) for stmt in node.body)
               and "hash_inputs" in ast.dump(node.test)]
    assert guarded, (
        "the message is printed unconditionally, so a run that skips "
        "hashing still announces it")
