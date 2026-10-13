"""The shipped example exports atomically through Measure's real setting."""
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6 import QtCore, QtWidgets

from spacr.qt.screens import settings_model as sm

KEY = "cellprofiler_pipeline"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def edit(qtbot):
    """A pipeline editor with a previously configured path."""
    widget = sm._ScalarEdit()
    widget.setText("previous.cppipe")
    qtbot.addWidget(widget)
    return widget


def test_resource_matches_verified_fixture_and_is_in_source_distribution():
    resource = ROOT / "spacr/resources/data/cellprofiler_example.cppipe"
    assert resource.read_bytes() == (ROOT / "tests/data/cellprofiler/example.cppipe").read_bytes()
    assert "include spacr/resources/data/cellprofiler_example.cppipe" in (ROOT / "MANIFEST.in").read_text()


def test_measure_action_exports_and_collects_path(qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences

    preferences._set_show_alpha_features(True)
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    defaults = sm.resolve_default_settings("measure")
    model = sm.SettingsWidgets("measure", parent, skip_keys=set(defaults) - {KEY})
    model.build_sections()
    editor = model._widgets[KEY]
    actions = [action for action in editor.actions()
               if action.objectName() == "CellProfilerExampleExport"]
    assert len(actions) == 1
    assert all(token in actions[0].toolTip() for token in
               ("DNA", "Actin", "_nucleus_mask.tif", "_cell_mask.tif"))
    destination = tmp_path / "chosen.cppipe"
    destination.write_bytes(b"old pipeline")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *args: (str(destination), ""))
    with qtbot.waitSignal(editor.editingFinished):
        actions[0].trigger()
    assert destination.read_bytes() == (ROOT / "tests/data/cellprofiler/example.cppipe").read_bytes()
    assert model.collect()[KEY] == str(destination)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["chosen.cppipe"]


def test_cancel_preserves_setting_and_files(edit, tmp_path, monkeypatch):
    existing = tmp_path / "existing.cppipe"
    existing.write_bytes(b"existing")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: ("", ""))
    monkeypatch.setattr(sm, "_write_cellprofiler_example",
                        lambda path: pytest.fail("cancel must not start a write"))
    assert sm._export_cellprofiler_example(edit) is False
    assert edit.text() == "previous.cppipe"
    assert existing.read_bytes() == b"existing"


@pytest.mark.parametrize("failure", ["resource", "open", "write", "commit"])
def test_failed_save_preserves_existing_destination_and_setting(
        failure, edit, tmp_path, monkeypatch):
    destination = tmp_path / "existing.cppipe"
    destination.write_bytes(b"existing pipeline")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *args: (str(destination), ""))
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))
    real_save = QtCore.QSaveFile
    saves = []

    class FailingSave:
        """Use real temporary writes, injecting only one failed operation."""
        def __init__(self, path):
            self.file = real_save(path)
            self.fallback = None
            self.cancelled = False
            saves.append(self)

        def setDirectWriteFallback(self, enabled):
            self.fallback = enabled
            self.file.setDirectWriteFallback(enabled)

        def open(self, mode):
            return False if failure == "open" else self.file.open(mode)

        def write(self, data):
            return self.file.write(data[:10] if failure == "write" else data)

        def commit(self):
            return False if failure == "commit" else self.file.commit()

        def errorString(self):
            return "simulated save error"

        def cancelWriting(self):
            self.cancelled = True
            self.file.cancelWriting()
            self.file.close()

    if failure == "resource":
        import importlib.resources
        monkeypatch.setattr(importlib.resources, "files", lambda *args: (_ for _ in ()).throw(OSError("missing resource")))
    monkeypatch.setattr(QtCore, "QSaveFile", FailingSave)
    signals = []
    edit.editingFinished.connect(lambda: signals.append(True))
    assert sm._export_cellprofiler_example(edit) is False
    assert edit.text() == "previous.cppipe"
    assert not signals
    assert warnings
    assert destination.read_bytes() == b"existing pipeline"
    assert [path.name for path in tmp_path.iterdir()] == ["existing.cppipe"]
    if failure != "resource":
        assert len(saves) == 1
        assert saves[0].fallback is False
        assert saves[0].cancelled


def test_real_invalid_parent_reports_failure(edit, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *args: (str(tmp_path / "absent" / "example.cppipe"), ""))
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))
    assert sm._export_cellprofiler_example(edit) is False
    assert warnings
    assert edit.text() == "previous.cppipe"
    assert not list(tmp_path.iterdir())


def test_other_apps_do_not_offer_measure_pipeline_action(edit):
    model = sm.SettingsWidgets.__new__(sm.SettingsWidgets)
    model.app_key = "mask"
    model._add_source_actions(KEY, edit)
    assert not edit.actions()
