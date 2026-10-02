"""The headless archive workflow builds checked packages without altering runs."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from spacr import cli, report


@pytest.fixture
def archive_run(tmp_path):
    source = tmp_path / "screen"
    (source / "settings").mkdir(parents=True)
    (source / "plate1_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"original pixels")
    (source / "settings" / "gen_mask_settings.csv").write_text(
        "Key,Value\nexperiment,archive-fixture\nmetadata_type,cellvoyager\nchannels,[0]\n")
    return source


def _metadata(**overrides):
    """Return a complete minimal form, inheriting microscope and title from settings."""
    return {"description": "Small archive fixture", "authors": "Doe Jane",
            "email": "jane@example.org", **overrides}


def _snapshot(folder):
    """Return source filenames and bytes to detect writes or overwrites."""
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


def _arguments(source, out, metadata):
    """Build inline metadata arguments for the real public CLI entry point."""
    return ["archive-package", "--src", str(source), "--out", str(out),
            "--metadata-json", json.dumps(metadata)]


@pytest.mark.parametrize("copy_images", [False, True])
def test_real_cli_package_is_valid_and_source_is_unchanged(archive_run, tmp_path, capsys, copy_images):
    before = _snapshot(archive_run)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps(_metadata()), encoding="utf-8")
    (tmp_path / "exports").mkdir()  # The parent may exist; the named package may not.
    args = ["archive-package", "--src", str(archive_run), "--out", str(tmp_path / "exports"),
            "--metadata", str(metadata)]
    if copy_images:
        args.append("--copy-images")
    assert cli.main(args) == cli.EXIT_OK
    package = tmp_path / "exports" / "archive-fixture"
    assert report._validate_archive_package(package) == []
    manifest = json.loads((package / "archive_manifest.json").read_text())
    assert manifest["form"]["microscope"] == "Yokogawa CellVoyager"
    assert manifest["copy_images"] is copy_images
    assert (package / "data").exists() is copy_images
    assert _snapshot(archive_run) == before
    output = capsys.readouterr().out
    assert str(package) in output and "Nothing was uploaded" in output
    assert "curator/service approval is still required" in output


@pytest.mark.parametrize("raw, expected", [
    ("{", "Expecting"),
    ("[]", "must be an object"),
    ('{"authors": 12}', "must contain text or null"),
    ('{"email": "a", "email": "b"}', "duplicate metadata field"),
    ('{"email": NaN}', "must contain text or null"),
    ('{"description": "' + "x" * (1024 * 1024) + '"}', "exceeds 1 MiB"),
])
def test_invalid_json_never_writes(archive_run, tmp_path, capsys, raw, expected):
    before = _snapshot(archive_run)
    args = _arguments(archive_run, tmp_path / "exports", {})
    args[-1] = raw
    assert cli.main(args) == cli.EXIT_USAGE
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "exports").exists()
    assert _snapshot(archive_run) == before


@pytest.mark.parametrize("metadata, expected", [
    ({}, "missing required metadata"),
    (_metadata(email="  "), "email"),
    (_metadata(authro="typo"), "unknown metadata fields: authro"),
])
def test_form_errors_are_actionable_before_output(archive_run, tmp_path, capsys, metadata, expected):
    assert cli.main(_arguments(archive_run, tmp_path / "exports", metadata)) == cli.EXIT_USAGE
    assert expected in capsys.readouterr().err
    assert not (tmp_path / "exports").exists()


def test_unreadable_metadata_and_bad_source_return_usage(archive_run, tmp_path, capsys):
    assert cli.main(["archive-package", "--src", str(archive_run), "--out", str(tmp_path / "out"),
                     "--metadata", str(tmp_path / "missing.json")]) == cli.EXIT_USAGE
    assert "missing.json" in capsys.readouterr().err
    assert cli.main(_arguments(tmp_path / "missing", tmp_path / "out", _metadata())) == cli.EXIT_USAGE
    assert "Not a folder" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_cli_preserves_source_and_existing_destination(archive_run, tmp_path, capsys):
    before = _snapshot(archive_run)
    assert cli.main(_arguments(archive_run, archive_run / "exports", _metadata())) == cli.EXIT_RUNTIME
    assert "outside the source" in capsys.readouterr().err
    out = tmp_path / "exports"
    package = out / "archive-fixture"
    package.mkdir(parents=True)
    (package / "notes.txt").write_bytes(b"keep existing package")
    assert cli.main(_arguments(archive_run, out, _metadata())) == cli.EXIT_RUNTIME
    assert "already exists" in capsys.readouterr().err
    assert _snapshot(package) == {"notes.txt": b"keep existing package"}
    assert _snapshot(archive_run) == before


def test_real_postwrite_checksum_failure_is_nonzero(archive_run, tmp_path, monkeypatch, capsys):
    original = report._write_archive_package

    def corrupt(*args, **kwargs):
        """Write a real package, then simulate corruption before validation."""
        package = original(*args, **kwargs)
        (package / "README.txt").write_text("corrupted")
        return package

    monkeypatch.setattr(report, "_write_archive_package", corrupt)
    assert cli.main(_arguments(archive_run, tmp_path / "exports", _metadata())) == cli.EXIT_RUNTIME
    message = capsys.readouterr().err
    assert "local checks failed" in message and "README.txt does not match" in message
    assert (tmp_path / "exports" / "archive-fixture").is_dir()


def test_archive_help_is_light_and_discoverable(capsys):
    assert cli.main(["--help"]) == cli.EXIT_OK
    assert "archive-package --help" in capsys.readouterr().out
    script = """
import sys
from spacr.cli import main
assert main(['archive-package', '--help']) == 0
assert not any(name in sys.modules for name in ('PySide6', 'tkinter', 'torch', 'numpy', 'pandas'))
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--metadata-json" in result.stdout


def test_notebook_cell_is_opt_in_and_uses_the_working_cli(archive_run, tmp_path):
    notebook = json.loads((Path(__file__).resolve().parents[1] /
                          "Notebooks/02_measure_and_crop.ipynb").read_text())
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code"
                and cell.get("metadata", {}).get("spacr", {}).get("feature") == 574)
    source = "".join(cell["source"])
    tree = ast.parse(source)
    namespace = {"settings": {"src": str(archive_run)}}
    exec(compile(tree, "<archive-notebook>", "exec"), namespace)
    assert namespace["build_archive"] is False
    assert "archive_status" not in namespace
    namespace.update(build_archive=True, archive_output=str(tmp_path / "exports"),
                     archive_metadata=_metadata())
    action = next(node for node in tree.body if isinstance(node, ast.If))
    exec(compile(ast.Module(body=[action], type_ignores=[]), "<archive-notebook>", "exec"), namespace)
    assert namespace["archive_status"] == 0
    assert report._validate_archive_package(tmp_path / "exports" / "archive-fixture") == []
