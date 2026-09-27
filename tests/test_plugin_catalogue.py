"""A plugin and recipe catalogue: browse, install, load, update, uninstall.

The catalogue here is a local folder holding ``catalogue.json`` and a real
wheel built on the spot, so pip installs it with no network. The plugin goes
into its own folder under a temporary plugin home, is discovered like any
other plugin, and is gone again after uninstalling.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import zipfile

import pytest

from spacr import cli_plugins, plugins


PLUGIN_SOURCE = '''
"""A catalogue test plugin."""

plugin = {
    "name": "Catalogue probe",
    "version": "{version}",
    "apps": [{
        "key": "catalogue_probe_app",
        "name": "Catalogue probe",
        "description": "Returns its settings.",
        "entrypoint": "catalogue_probe:run",
        "defaults": "catalogue_probe:defaults",
    }],
}


def run(settings):
    return settings


def defaults(settings=None):
    return {"probe": "{version}"}
'''


def _wheel(folder, version):
    """Build a minimal pure-Python wheel for ``catalogue_probe``."""
    dist = f"catalogue_probe-{version}.dist-info"
    files = {
        "catalogue_probe/__init__.py": PLUGIN_SOURCE.replace("{version}", version),
        f"{dist}/METADATA": (
            "Metadata-Version: 2.1\nName: catalogue-probe\n"
            f"Version: {version}\n"),
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"),
    }
    record = []
    for name, text in files.items():
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(text.encode())}")
    record.append(f"{dist}/RECORD,,")
    files[f"{dist}/RECORD"] = "\n".join(record) + "\n"
    path = folder / f"catalogue_probe-{version}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return path


def _catalogue(folder, version="1.0.0", **extra):
    wheel = _wheel(folder, version)
    data = {
        "plugins": [{
            "key": "catalogue_probe", "name": "Catalogue probe",
            "version": version, "author": "Test Lab", "licence": "MIT",
            "summary": "A plugin that returns its settings.",
            "source": wheel.name, "entry": "catalogue_probe:plugin",
            **extra,
        }],
        "recipes": [{
            "key": "toxo_infection", "name": "Toxoplasma infection assay",
            "version": "0.2", "author": "Test Lab", "licence": "CC-BY-4.0",
            "summary": "Masks and measurements for an infection screen.",
            "app": "measure",
            "settings": {"channels": [0, 1, 2], "cell_mask_dim": 4},
        }],
    }
    (folder / "catalogue.json").write_text(json.dumps(data))
    return folder


@pytest.fixture
def home(tmp_path, monkeypatch):
    target = tmp_path / "home"
    monkeypatch.setenv("SPACR_PLUGIN_HOME", str(target))
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    monkeypatch.delenv("SPACR_DISABLE_PLUGINS", raising=False)
    plugins.reload_plugins()
    yield target
    plugins._forget_modules_under(str(target / "site" / "catalogue_probe"))
    monkeypatch.delenv("SPACR_PLUGIN_HOME")
    plugins.reload_plugins()


def test_the_catalogue_lists_plugins_and_recipes_with_their_metadata(
        tmp_path, home):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    rows = plugins._catalogue_rows(source)
    assert [(row["kind"], row["key"], row["status"]) for row in rows] == [
        ("plugin", "catalogue_probe", "available"),
        ("recipe", "toxo_infection", "available"),
    ]
    assert rows[0]["author"] == "Test Lab" and rows[0]["licence"] == "MIT"
    assert rows[1]["app"] == "measure"


def test_a_plugin_installs_loads_updates_and_uninstalls_cleanly(
        tmp_path, home):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    record = plugins._install_from_catalogue("catalogue_probe", source)
    assert record["version"] == "1.0.0"
    assert (home / "site" / "catalogue_probe" / "catalogue_probe").is_dir()
    assert not (home / "site" / ".catalogue_probe.new").exists()
    app = plugins.get_app("catalogue_probe_app")
    assert app is not None
    assert plugins.load_object(app.defaults)() == {"probe": "1.0.0"}
    assert "Catalogue probe" in [p.name for p in plugins.discover_plugins()]
    assert plugins._catalogue_rows(source)[0]["status"] == "installed"

    _catalogue(source, version="1.1.0")
    assert plugins._catalogue_rows(source)[0]["status"] == "update available"
    plugins._install_from_catalogue("catalogue_probe", source)
    app = plugins.get_app("catalogue_probe_app")
    assert plugins.load_object(app.defaults)() == {"probe": "1.1.0"}

    assert plugins._uninstall_from_catalogue("catalogue_probe") is True
    assert plugins.get_app("catalogue_probe_app") is None
    assert "catalogue_probe" not in sys.modules
    assert not (home / "site" / "catalogue_probe").exists()
    assert plugins._catalogue_installed() == {}
    assert plugins.diagnostics() == ()
    assert plugins._uninstall_from_catalogue("catalogue_probe") is False


def test_a_recipe_installs_as_a_settings_file_and_uninstalls(tmp_path, home):
    from spacr.cli import load_settings_file

    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    record = plugins._install_from_catalogue("toxo_infection", source)
    assert load_settings_file(record["path"]) == {
        "channels": [0, 1, 2], "cell_mask_dim": 4}
    assert plugins._catalogue_rows(source)[1]["installed"] == "0.2"
    assert plugins._uninstall_from_catalogue("toxo_infection")
    assert not (home / "recipes" / "toxo_infection.json").exists()


def test_a_failed_update_keeps_the_working_version(tmp_path, home):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    plugins._install_from_catalogue("catalogue_probe", source)
    _catalogue(source, version="2.0.0", entry="catalogue_probe:missing")
    with pytest.raises(AttributeError):
        plugins._install_from_catalogue("catalogue_probe", source)
    assert plugins._catalogue_installed()["catalogue_probe"]["version"] == "1.0.0"
    assert plugins.get_app("catalogue_probe_app") is not None
    assert not (home / "site" / ".catalogue_probe.new").exists()


def test_bad_checksums_and_incompatible_plugins_are_refused(tmp_path, home):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source, sha256="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        plugins._install_from_catalogue("catalogue_probe", source)
    _catalogue(source, api_version="2.0")
    assert plugins._catalogue_rows(source)[0]["status"] == "incompatible"
    with pytest.raises(ValueError, match="SDK"):
        plugins._install_from_catalogue("catalogue_probe", source)
    assert plugins._catalogue_installed() == {}


def test_malformed_catalogues_say_what_is_wrong(tmp_path, monkeypatch):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"plugins": [{"key": "x1", "name": "X"}]}))
    with pytest.raises(ValueError, match="version"):
        plugins._read_catalogue(path)
    path.write_text(json.dumps({"recipes": [
        {"key": "r1", "name": "R", "version": "1", "settings": {"a": 1}},
        {"key": "r1", "name": "S", "version": "1", "settings": {"a": 1}}]}))
    with pytest.raises(ValueError, match="repeats"):
        plugins._read_catalogue(tmp_path)
    monkeypatch.delenv("SPACR_PLUGIN_CATALOGUE", raising=False)
    with pytest.raises(ValueError, match="no catalogue"):
        plugins._read_catalogue(None)


def test_the_command_line_browses_installs_and_uninstalls(
        tmp_path, home, capsys):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    assert cli_plugins.main(["catalogue", "--catalogue", str(source)]) == 0
    listing = capsys.readouterr().out
    assert "Catalogue probe 1.0.0" in listing and "MIT" in listing
    assert "Toxoplasma infection assay 0.2" in listing
    assert cli_plugins.main(
        ["install", "catalogue_probe", "--catalogue", str(source)]) == 0
    assert "Installed Catalogue probe 1.0.0" in capsys.readouterr().out
    assert cli_plugins.main(["list", "--json"]) == 0
    assert "catalogue_probe_app" in json.loads(capsys.readouterr().out)["apps"]
    assert cli_plugins.main(["uninstall", "catalogue_probe"]) == 0
    assert plugins.get_app("catalogue_probe_app") is None
    assert cli_plugins.main(["uninstall", "catalogue_probe"]) == 1
    assert cli_plugins.main(["install", "nope", "--catalogue",
                             str(source)]) == 1
    assert "no entry" in capsys.readouterr().err
