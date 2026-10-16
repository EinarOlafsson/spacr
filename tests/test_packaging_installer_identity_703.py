"""The windowed launcher can report its loaded identity without running Qt."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "packaging/spacr_launcher.py"


def _launcher(monkeypatch, tmp_path, *, frozen=True):
    spec = importlib.util.spec_from_file_location("installer_identity_launcher", LAUNCHER)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    root = tmp_path / "staged bundle"
    imports = root / "_internal"
    package = ModuleType("spacr")
    package.__file__ = str(imports / "spacr/__init__.py")
    package.__version__ = "1.5.1.5"
    qt = ModuleType("spacr.qt")
    qt.__file__ = str(imports / "spacr/qt/__init__.py")

    def no_gui():
        pytest.fail("The identity probe started the application")

    qt.run = no_gui
    package.qt = qt
    monkeypatch.setitem(sys.modules, "spacr", package)
    monkeypatch.setitem(sys.modules, "spacr.qt", qt)
    bindings = []

    def import_binding(name):
        bindings.append(name)
        return ModuleType(name)

    monkeypatch.setattr(launcher, "import_module", import_binding)
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(imports), raising=False)
    monkeypatch.setattr(sys, "executable", str(root / "spacr.exe"))
    monkeypatch.setattr(launcher.multiprocessing, "freeze_support", lambda: None)
    return launcher, package, qt, root, bindings


def test_frozen_identity_checks_imports_and_writes_exact_loaded_paths(monkeypatch, tmp_path):
    launcher, package, qt, root, bindings = _launcher(monkeypatch, tmp_path)
    output = tmp_path / "identity.json"
    monkeypatch.setattr(sys, "argv", [str(root / "spacr.exe"), "--installer-identity", str(output)])
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert launcher.main() == 0
    assert json.loads(output.read_text()) == {
        "schema": 1, "version": "1.5.1.5", "module": package.__file__,
        "executable": str(root / "spacr.exe"), "frozen": True,
        "bundle_root": str(root), "import_root": str(root / "_internal"),
    }
    assert bindings == ["PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("foreign", ["package", "qt", "import-root"])
def test_frozen_foreign_origin_refuses_before_creating_receipt(monkeypatch, tmp_path, foreign):
    launcher, package, qt, root, _ = _launcher(monkeypatch, tmp_path)
    if foreign == "package":
        package.__file__ = str(tmp_path / "foreign/spacr/__init__.py")
    elif foreign == "qt":
        qt.__file__ = str(tmp_path / "foreign/qt/__init__.py")
    else:
        monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "foreign"))
    output = tmp_path / "identity.json"
    monkeypatch.setattr(sys, "argv", ["spacr.exe", "--installer-identity", str(output)])
    assert launcher.main() == 2
    assert not output.exists()


@pytest.mark.parametrize("failure", ["gui", "binding"])
def test_missing_import_refuses_without_a_receipt(monkeypatch, tmp_path, failure):
    launcher, package, qt, root, _ = _launcher(monkeypatch, tmp_path)
    if failure == "gui":
        monkeypatch.setitem(sys.modules, "spacr.qt", None)
    else:
        def broken_binding(name):
            raise ImportError(name)
        monkeypatch.setattr(launcher, "import_module", broken_binding)
    output = tmp_path / "identity.json"
    monkeypatch.setattr(sys, "argv", ["spacr.exe", "--installer-identity", str(output)])
    monkeypatch.setattr(sys, "stderr", None)
    assert launcher.main() == 2
    assert not output.exists()


@pytest.mark.parametrize("arguments", [[], ["one", "two"]])
def test_invalid_identity_arguments_do_not_start_qt(monkeypatch, tmp_path, arguments):
    launcher, *_ = _launcher(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", ["spacr.exe", "--installer-identity", *arguments])
    assert launcher.main() == 2


def test_existing_destination_is_preserved(monkeypatch, tmp_path):
    launcher, *_ = _launcher(monkeypatch, tmp_path)
    output = tmp_path / "identity.json"
    output.write_bytes(b"foreign original")
    monkeypatch.setattr(sys, "argv", ["spacr.exe", "--installer-identity", str(output)])
    assert launcher.main() == 2
    assert output.read_bytes() == b"foreign original"


def test_nonfrozen_probe_reports_its_state_honestly(monkeypatch, tmp_path):
    launcher, *_ = _launcher(monkeypatch, tmp_path, frozen=False)
    output = tmp_path / "identity.json"
    monkeypatch.setattr(sys, "argv", ["python", "--installer-identity", str(output)])
    assert launcher.main() == 0
    assert json.loads(output.read_text())["frozen"] is False


def test_partial_write_is_not_left_as_a_receipt(monkeypatch, tmp_path):
    launcher, *_ = _launcher(monkeypatch, tmp_path)
    output = tmp_path / "identity.json"
    monkeypatch.setattr(sys, "argv", ["spacr.exe", "--installer-identity", str(output)])
    real_fdopen = launcher.os.fdopen

    class BrokenOutput:
        def __init__(self, descriptor, *args, **kwargs):
            self.output = real_fdopen(descriptor, *args, **kwargs)

        def __enter__(self):
            return self

        def write(self, value):
            self.output.write(value[:20])
            raise OSError("controlled disk failure")

        def __exit__(self, *args):
            self.output.close()

    monkeypatch.setattr(launcher.os, "fdopen", BrokenOutput)
    assert launcher.main() == 2
    assert not output.exists()


def test_actual_lightweight_package_and_qt_import_without_models(tmp_path):
    import os
    import subprocess

    output = tmp_path / "actual-identity.json"
    program = '''import runpy, sys
class NoModels:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'cellpose', 'tensorflow'}:
            raise RuntimeError('Model import forbidden: ' + fullname)
sys.meta_path.insert(0, NoModels())
sys.argv = [sys.argv[1], '--installer-identity', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    environment = dict(os.environ, PYTHONPATH=str(LAUNCHER.parents[1]),
                       CUDA_VISIBLE_DEVICES="", QT_QPA_PLATFORM="offscreen")
    result = subprocess.run([sys.executable, "-c", program, str(LAUNCHER), str(output)],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(output.read_text())
    assert receipt["schema"] == 1
    assert receipt["frozen"] is False
    assert Path(receipt["module"]) == LAUNCHER.parents[1] / "spacr/__init__.py"
    assert Path(receipt["executable"]) == Path(sys.executable).resolve()
