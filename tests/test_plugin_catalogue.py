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


# ---------------------------------------------------------------------------
# Edges the coverage ratchet found unexercised (dispatch 36739819315)
# ---------------------------------------------------------------------------

def test_the_command_line_prints_json_and_refuses_a_keyless_install(
        tmp_path, home, capsys):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    assert cli_plugins.main(
        ["catalogue", "--catalogue", str(source), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["key"] for row in rows] == ["catalogue_probe", "toxo_infection"]
    assert rows[0]["status"] == "available"
    assert cli_plugins.main(["install"]) == 1
    assert "install needs the key" in capsys.readouterr().err


def test_a_catalogue_at_an_address_is_downloaded_and_its_sources_resolved(
        monkeypatch):
    """An http(s) catalogue is read over the network, and a relative source
    in it is taken relative to the catalogue's own address."""
    import urllib.request

    address = "https://plugins.example.invalid/spacr/catalogue.json"
    payload = json.dumps({"recipes": [{
        "key": "remote_recipe", "name": "Remote", "version": "1",
        "source": "recipes/remote.json"}]}).encode()
    opened = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload

    def fake_urlopen(location, timeout):
        opened.append((location, timeout))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    (entry,) = plugins._read_catalogue(address)
    assert opened == [(address, 60)]
    assert entry.source == (
        "https://plugins.example.invalid/spacr/recipes/remote.json")


def test_absolute_and_blank_sources_are_left_as_written(tmp_path):
    absolute = str(tmp_path / "plugin.whl")
    assert plugins._resolve_source(str(tmp_path / "c.json"), absolute) == absolute
    assert plugins._resolve_source(str(tmp_path / "c.json"), "") == ""


@pytest.mark.parametrize("row,error,match", [
    (["not", "a", "mapping"], TypeError, "must be a mapping"),
    ({"kind": "recipe", "key": "k", "name": "N", "version": "1",
      "colour": "red"}, ValueError, "unknown catalogue fields"),
    ({"kind": "recipe", "key": "k", "name": "", "version": "1"},
     ValueError, "missing 'name'"),
    ({"kind": "theme", "key": "a_key", "name": "N", "version": "1"},
     ValueError, "kind must be one of"),
    ({"kind": "recipe", "key": "Not A Key!", "name": "N", "version": "1"},
     ValueError, "invalid catalogue key"),
    ({"kind": "recipe", "key": "a_key", "name": "N", "version": "1",
      "settings": ["a"]}, TypeError, "settings must be a mapping"),
    ({"kind": "plugin", "key": "a_key", "name": "N", "version": "1",
      "entry": "no colon here", "source": "x.whl"},
     ValueError, "needs entry"),
    ({"kind": "plugin", "key": "a_key", "name": "N", "version": "1",
      "entry": "pkg.mod:plugin"}, ValueError, "no source to install"),
    ({"kind": "recipe", "key": "a_key", "name": "N", "version": "1"},
     ValueError, "neither settings nor source"),
])
def test_a_malformed_catalogue_row_says_what_is_wrong(tmp_path, row, error,
                                                      match):
    with pytest.raises(error, match=match):
        plugins._entry_from_mapping(row, str(tmp_path / "catalogue.json"))


@pytest.mark.parametrize("document,match", [
    ([1, 2, 3], "must be a JSON object"),
    ({"plugins": "catalogue_probe"}, "plugins must be a list"),
])
def test_a_malformed_catalogue_file_says_what_is_wrong(tmp_path, document,
                                                       match):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=match):
        plugins._read_catalogue(path)


def test_unparseable_versions_compare_by_difference():
    assert plugins._newer("nightly-b", "nightly-a") is True
    assert plugins._newer("nightly-a", "nightly-a") is False


def test_a_catalogue_plugin_whose_folder_went_missing_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="is missing"):
        plugins._load_catalogue_plugin({"path": str(tmp_path / "gone"),
                                        "entry": "gone:plugin"})


def test_pip_failure_carries_its_output():
    ran = []

    def runner(command, **kwargs):
        ran.append(command)
        return type("Result", (), {"returncode": 1, "stderr": "",
                                   "stdout": "no matching distribution"})()

    with pytest.raises(RuntimeError, match="no matching distribution"):
        plugins._pip(["nothing-here"], runner)
    assert ran[0][-1] == "nothing-here"


def test_a_plugins_requirements_are_installed_beside_it(tmp_path, home,
                                                        monkeypatch):
    """Requirements go into the plugin's own folder in a second pip call."""
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source, requirements=["tinydep==1.0"])
    calls = []
    real_pip = plugins._pip

    def pip(arguments, runner=None):
        calls.append(list(arguments))
        if "tinydep==1.0" in arguments:
            return None
        return real_pip(arguments, runner)

    monkeypatch.setattr(plugins, "_pip", pip)
    record = plugins._install_from_catalogue("catalogue_probe", source)
    assert calls[-1][:2] == ["--target", calls[0][2]]
    assert calls[-1][-1] == "tinydep==1.0"
    assert record["version"] == "1.0.0"
    assert plugins._uninstall_from_catalogue("catalogue_probe")


def test_a_recipe_from_a_file_must_hold_a_settings_object(tmp_path, home):
    source = tmp_path / "cat"
    source.mkdir()
    (source / "listy.json").write_text(json.dumps([1, 2]))
    (source / "good.json").write_text(json.dumps({"channels": [0]}))
    (source / "catalogue.json").write_text(json.dumps({"recipes": [
        {"key": "listy", "name": "Listy", "version": "1",
         "source": "listy.json"},
        {"key": "good", "name": "Good", "version": "1",
         "source": "good.json"}]}))
    with pytest.raises(ValueError, match="is not a settings object"):
        plugins._install_from_catalogue("listy", source)
    record = plugins._install_from_catalogue("good", source)
    assert json.loads(open(record["path"]).read()) == {"channels": [0]}


def test_uninstalling_a_recipe_whose_file_is_gone_still_forgets_it(
        tmp_path, home):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    record = plugins._install_from_catalogue("toxo_infection", source)
    import os
    os.remove(record["path"])
    assert plugins._uninstall_from_catalogue("toxo_infection") is True
    assert "toxo_infection" not in plugins._catalogue_installed()


def test_discovery_reports_an_unreadable_install_record(monkeypatch):
    def unreadable(home=None):
        raise OSError("install record is locked")

    monkeypatch.setattr(plugins, "_catalogue_installed", unreadable)
    sources = dict(plugins._installed_sources())
    with pytest.raises(OSError, match="locked"):
        sources["catalogue installs"]()


def test_discovery_skips_installed_recipes(monkeypatch):
    monkeypatch.setattr(plugins, "_catalogue_installed", lambda home=None: {
        "a_recipe": {"kind": "recipe", "path": "/nowhere.json"}})
    assert "a_recipe" not in dict(plugins._installed_sources())


def test_a_catalogue_listing_leaves_out_an_empty_summary(tmp_path, home,
                                                         capsys):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source, summary="")
    assert cli_plugins.main(["catalogue", "--catalogue", str(source)]) == 0
    listing = capsys.readouterr().out.splitlines()
    probe = listing.index(next(line for line in listing
                               if "Catalogue probe 1.0.0" in line))
    assert listing[probe + 1].startswith("  by Test Lab")
    assert listing[probe + 2].startswith("[recipe]")


def test_a_failed_install_from_a_requirement_leaves_no_staging_folder(
        tmp_path, home, monkeypatch):
    """A source that is a pip requirement, not a file here, goes to pip as
    written, with no local index; when pip fails nothing is left behind."""
    source = tmp_path / "cat"
    source.mkdir()
    (source / "catalogue.json").write_text(json.dumps({"plugins": [{
        "key": "from_pypi", "name": "From PyPI", "version": "1",
        "source": "spacr-probe-plugin>=1", "entry": "probe:plugin"}]}))
    seen = []

    def pip(arguments, runner=None):
        seen.append(list(arguments))
        raise RuntimeError("pip could not install the plugin")

    monkeypatch.setattr(plugins, "_pip", pip)
    with pytest.raises(RuntimeError, match="could not install"):
        plugins._install_from_catalogue("from_pypi", source)
    assert "--no-index" not in seen[0]
    assert seen[0][-1] == "spacr-probe-plugin>=1"
    assert not (home / "site" / ".from_pypi.new").exists()
