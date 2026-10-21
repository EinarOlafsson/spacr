"""Release catalogs omit development provenance without changing runtime data."""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest
import setuptools

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("spacr/qt/i18n_catalogs")
FIELDS = {"MODEL", "LICENSE", "SECONDARY_MODEL", "SECONDARY_LICENSE", "NORMALIZER"}
LOCALES = ("de", "es", "fr", "hi", "is", "ko", "pt", "sv", "zh_CN")


@pytest.fixture
def setup_namespace(monkeypatch):
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **options: captured.update(options))
    monkeypatch.chdir(ROOT)
    namespace = runpy.run_path(str(ROOT / "setup.py"))
    assert captured["cmdclass"]["build_py"] is namespace["RuntimeCatalogBuild"]
    assert captured["cmdclass"]["sdist"] is namespace["RuntimeCatalogSdist"]
    return namespace


def _runtime_ast(source):
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id in FIELDS)]
    return ast.dump(tree, include_attributes=False)


def test_staged_projection_preserves_hardlinked_sources_and_runtime_data(
        tmp_path, setup_namespace):
    destination = tmp_path / CATALOG
    destination.mkdir(parents=True)
    originals = {}
    link_sources = tmp_path / "source-catalogs"
    link_sources.mkdir()
    for language in LOCALES:
        source = ROOT / CATALOG / (language + ".py")
        originals[source] = source.read_bytes()
        local_source = link_sources / source.name
        shutil.copyfile(source, local_source)
        os.link(local_source, destination / source.name)
    notice = destination / "LICENSE.txt"
    notice.write_text("Redistributed library copyright and license notice.\n")
    english = destination / "en.py"
    shutil.copyfile(ROOT / CATALOG / english.name, english)
    english_bytes = english.read_bytes()

    project = setup_namespace["project_runtime_catalogs"]
    project(tmp_path)
    once = {p.name: p.read_bytes() for p in destination.iterdir()}
    project(tmp_path)
    assert once == {p.name: p.read_bytes() for p in destination.iterdir()}
    assert notice.read_text() == "Redistributed library copyright and license notice.\n"
    assert english.read_bytes() == english_bytes
    for source, original in originals.items():
        assert source.read_bytes() == original
        staged = destination / source.name
        local_source = link_sources / source.name
        assert local_source.read_bytes() == original
        assert staged.stat().st_ino != local_source.stat().st_ino
        namespace = runpy.run_path(str(staged))
        original_namespace = runpy.run_path(str(source))
        assert not FIELDS.intersection(namespace)
        assert namespace["SOURCE_HASHES"] == original_namespace["SOURCE_HASHES"]
        assert {k: v for k, v in namespace.items() if k.isupper()} == {
            k: v for k, v in original_namespace.items()
            if k.isupper() and k not in FIELDS}
        assert _runtime_ast(staged.read_text()) == _runtime_ast(original.decode())


@pytest.mark.parametrize("original", [
    'MODEL = compute_something()\nSOURCE_HASHES = {}\n',
    'MODEL = "upstream"; SOURCE_HASHES = {"must": "remain"}\n',
    'MODEL = "upstream"  # Redistributed copyright notice.\n',
])
def test_projection_refuses_to_discard_executable_provenance(
        tmp_path, setup_namespace, original):
    folder = tmp_path / CATALOG
    folder.mkdir(parents=True)
    target = folder / "de.py"
    target.write_text(original)
    with pytest.raises(ValueError, match="Expected literal catalog provenance"):
        setup_namespace["project_runtime_catalogs"](tmp_path)
    assert target.read_text() == original
    assert list(folder.iterdir()) == [target]


def _build(project, command, output):
    result = subprocess.run(
        [sys.executable, "setup.py", command, "--dist-dir", str(output)],
        cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120, check=False)
    assert result.returncode == 0, result.stdout[-12000:]


def test_real_wheel_and_wheel_from_sdist_have_identical_runtime_catalogs(tmp_path):
    """Use the real setup commands and catalogs without unrelated scientific code."""
    project = tmp_path / "project"
    project.mkdir()
    for name in ("setup.py", "_spacr_catalog_build.py", "pyproject.toml", "MANIFEST.in",
                 "README.rst", "LICENSE"):
        shutil.copyfile(ROOT / name, project / name)
    for parent in ("spacr", "spacr/qt"):
        folder = project / parent
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    shutil.copytree(ROOT / CATALOG, project / CATALOG,
                    ignore=shutil.ignore_patterns("__pycache__"))
    font_notice = Path("spacr/resources/font/open_sans/OFL.txt")
    (project / font_notice).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / font_notice, project / font_notice)
    shutil.copytree(ROOT / "packaging/linux", project / "packaging/linux")
    originals = {p.relative_to(project): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in project.rglob("*") if p.is_file()}

    direct = tmp_path / "direct"
    source_dist = tmp_path / "source-dist"
    rebuilt = tmp_path / "rebuilt"
    _build(project, "sdist", source_dist)
    _build(project, "bdist_wheel", direct)
    for relative, digest in originals.items():
        assert hashlib.sha256((project / relative).read_bytes()).hexdigest() == digest
    with tarfile.open(next(source_dist.glob("*.tar.gz"))) as archive:
        archive.extractall(tmp_path / "unpacked", filter="data")
    unpacked = next((tmp_path / "unpacked").iterdir())
    _build(unpacked, "bdist_wheel", rebuilt)

    with zipfile.ZipFile(next(direct.glob("*.whl"))) as first, \
            zipfile.ZipFile(next(rebuilt.glob("*.whl"))) as second:
        assert first.read(str(font_notice)) == second.read(str(font_notice)) == \
            (ROOT / font_notice).read_bytes()
        for wheel in (first, second):
            license_path = next(name for name in wheel.namelist()
                                if name.endswith(".dist-info/licenses/LICENSE"))
            assert wheel.read(license_path) == (ROOT / "LICENSE").read_bytes()
            assert not any("resources/AI/" in name for name in wheel.namelist())
        for name in ("__init__", "en", *LOCALES):
            relative = CATALOG / (name + ".py")
            data = first.read(str(relative))
            assert data == second.read(str(relative)) == (unpacked / relative).read_bytes()
            assert _runtime_ast(data.decode()) == _runtime_ast((ROOT / relative).read_text())
            if name in LOCALES:
                namespace = {}
                exec(compile(data, str(relative), "exec"), namespace)
                assert not FIELDS.intersection(namespace)
                assert namespace["SOURCE_HASHES"]
