"""The native catalog hook projects actual modules and excludes agent resources."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path("spacr/qt/i18n_catalogs")
LOCALES = ("de", "es", "fr", "hi", "is", "ko", "pt", "sv", "zh_CN")
FIELDS = {"MODEL", "LICENSE", "SECONDARY_MODEL", "SECONDARY_LICENSE", "NORMALIZER"}
HOOK = Path("packaging/hooks/pre_find_module_path/hook-spacr.qt.i18n_catalogs.py")


def test_staging_preserves_loader_english_and_every_non_provenance_value(tmp_path):
    helper = runpy.run_path(str(ROOT / "_spacr_catalog_build.py"))
    before = {p.name: p.read_bytes() for p in (ROOT / CATALOG).glob("*.py")}
    search = helper["stage_runtime_catalogs"](ROOT, tmp_path)
    staged = search / "i18n_catalogs"
    assert {p.name: p.read_bytes() for p in (ROOT / CATALOG).glob("*.py")} == before
    for name in ("__init__.py", "en.py"):
        assert (staged / name).read_bytes() == before[name]
    for language in LOCALES:
        original = runpy.run_path(str(ROOT / CATALOG / (language + ".py")))
        projected = runpy.run_path(str(staged / (language + ".py")))
        assert not FIELDS.intersection(projected)
        assert {k: v for k, v in projected.items() if k.isupper()} == {
            k: v for k, v in original.items() if k.isupper() and k not in FIELDS}


def _resources_expression():
    source = (ROOT / "packaging/spacr.spec").read_text()
    tree = ast.parse(source)
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "collect_data_files"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "spacr")
    main_analysis = next(node for node in tree.body if isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "a" for t in node.targets))
    hookspath = next(k.value for k in main_analysis.value.keywords if k.arg == "hookspath")
    assert ast.literal_eval(hookspath.elts[0].args[0].right) == "hooks"
    return ast.get_source_segment(source, call)


def test_actual_frozen_program_uses_projected_catalogs_and_preserves_notices(tmp_path):
    pytest.importorskip("PyInstaller", reason="native build validation needs PyInstaller")
    project = tmp_path / "project"
    project.mkdir()
    for parent in ("spacr", "spacr/qt"):
        folder = project / parent
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "__init__.py").write_text("")
    shutil.copytree(ROOT / CATALOG, project / CATALOG,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(ROOT / "_spacr_catalog_build.py", project / "_spacr_catalog_build.py")
    (project / HOOK).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / HOOK, project / HOOK)
    font = Path("spacr/resources/font/open_sans/OFL.txt")
    (project / font).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / font, project / font)
    agent = project / "spacr/resources/AI/AGENTS.md"
    agent.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "spacr/resources/AI/AGENTS.md", agent)
    (agent.parent / "archive").mkdir()
    (agent.parent / "archive/handoff.md").write_text("Operational handoff.\n")
    originals = {p.relative_to(project): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in project.rglob("*") if p.is_file()}
    expected = {}
    for language in LOCALES:
        namespace = runpy.run_path(str(ROOT / CATALOG / (language + ".py")))
        data = sorted((k, v) for k, v in namespace.items() if k.isupper() and k not in FIELDS)
        expected[language] = hashlib.sha256(repr(data).encode()).hexdigest()
    app = project / "check_catalogs.py"
    app.write_text('''from pathlib import Path
import hashlib
import sys
import spacr.qt.i18n_catalogs as catalogs
expected = ''' + repr(expected) + '''
fields = {"MODEL", "LICENSE", "SECONDARY_MODEL", "SECONDARY_LICENSE", "NORMALIZER"}
for language, digest in expected.items():
    module = catalogs._module(language)
    assert module is not None
    assert not fields.intersection(vars(module))
    data = sorted((k, v) for k, v in vars(module).items() if k.isupper())
    assert hashlib.sha256(repr(data).encode()).hexdigest() == digest
    key = next(iter(module.SETTING_LABELS))
    source = catalogs._english().SETTING_LABELS[key]
    assert catalogs.setting_label(key, source, language) == module.SETTING_LABELS[key]
root = Path(sys._MEIPASS)
assert not (root / "spacr/resources/AI").exists()
assert hashlib.sha256((root / ''' + repr(str(font)) + ''').read_bytes()).hexdigest() == ''' + repr(hashlib.sha256((ROOT / font).read_bytes()).hexdigest()) + '''
print("9 projected runtime catalogs, actual lookups, font license and agent exclusion passed")
''')
    spec = project / "native_boundary.spec"
    expression = _resources_expression()
    spec.write_text('''from PyInstaller.utils.hooks import collect_data_files
from pathlib import Path
ROOT = Path(''' + repr(str(project)) + ''')
datas = ''' + expression + '''
a = Analysis([str(ROOT / "check_catalogs.py")], pathex=[str(ROOT)],
             datas=datas, binaries=[],
             hiddenimports=''' + repr(["spacr.qt.i18n_catalogs." + code for code in ("en", *LOCALES)]) + ''',
             hookspath=[str(ROOT / "packaging/hooks")])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="catalog-boundary", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="catalog-boundary", strip=False, upx=False)
''')
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    env = dict(os.environ, TMPDIR=str(temporary))
    env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
         "--distpath", str(tmp_path / "dist"), "--workpath", str(tmp_path / "build"), str(spec)],
        cwd=project, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=180, check=False)
    assert result.returncode == 0, result.stdout[-14000:]
    binary = tmp_path / "dist/catalog-boundary" / (
        "catalog-boundary.exe" if sys.platform == "win32" else "catalog-boundary")
    execution = subprocess.run([str(binary)], cwd=tmp_path, text=True,
                               capture_output=True, timeout=30, check=False)
    assert execution.returncode == 0, execution.stdout + execution.stderr
    assert "9 projected runtime catalogs, actual lookups" in execution.stdout
    assert not list(temporary.glob("spacr-catalogs-*"))
    for relative, digest in originals.items():
        assert hashlib.sha256((project / relative).read_bytes()).hexdigest() == digest
