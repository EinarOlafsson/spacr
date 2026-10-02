"""Committed settings observations preserve edit timing without rehashing models."""

import json
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QWidget

from spacr import run_journal as rj
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import SettingsWidgets, _ControlsBuiltWhenAskedFor, _ListEditor


@pytest.fixture
def journal(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rj, "runs_root", lambda: runs)
    now = {"utc": "2026-10-01T10:00:00.000000Z"}
    monkeypatch.setattr(rj, "_utc_now", lambda: now["utc"])
    source = tmp_path / "plate"
    source.mkdir()
    return source, now


def events(record):
    path = rj._seen_log(record)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def tiny_screen(qtbot, source):
    parent = QWidget()
    qtbot.addWidget(parent)
    model = object.__new__(SettingsWidgets)
    model.app_key = "measure"
    model._parent = parent
    model._defaults = {
        "src": str(source),
        "n_jobs": 2,
        "plot": False,
        "experiment": "a",
        "channels": [0, 1],
    }
    model._controls_arriving = 0
    model._widgets = _ControlsBuiltWhenAskedFor(model)
    model._widgets["src"] = QLineEdit(str(source), parent)
    model._widgets["n_jobs"] = QLineEdit("2", parent)
    model._widgets["plot"] = QCheckBox(parent)
    choice = QComboBox(parent)
    choice.addItem("a", "a")
    choice.addItem("b", "b")
    model._widgets["experiment"] = choice
    model._widgets["channels"] = _ListEditor("channels", [0, 1], element_type=int, parent=parent)
    errors = []
    screen = SimpleNamespace(
        _settings_model=model,
        app_key="measure",
        _console=SimpleNamespace(append_error=errors.append),
    )
    screen.observe = lambda key=None: AppScreen._observe_settings_commit(screen, key)
    model._enable_commit_observation(screen.observe)
    return screen, model, errors


def test_blind_committed_edit_retains_time_and_new_posthoc_edit_is_distinct(qtbot, journal):
    source, now = journal
    screen, model, _ = tiny_screen(qtbot, source)
    blind = rj.start_blinding(["field1"], scope="annotate", src=str(source))
    record = rj.lock_analysis(model.collect(), app_key="measure")
    field = model._widgets["n_jobs"]
    now["utc"] = "2026-10-01T10:01:00.000000Z"
    field.setText("3")
    assert not events(record), "typing a valid draft is not a committed edit"
    field.editingFinished.emit()
    assert [(e["key"], e["utc"]) for e in events(record)] == [("n_jobs", now["utc"])]
    field.editingFinished.emit()
    assert len(events(record)) == 1
    now["utc"] = "2026-10-01T10:02:00.000000Z"
    rj.unblind(blind["key_id"], reason="finished")
    assert rj.check_analysis_lock(model.collect(), app_key="measure")["status"] == "deviation"
    now["utc"] = "2026-10-01T10:03:00.000000Z"
    field.setText("4")
    field.editingFinished.emit()
    verdict = rj.check_analysis_lock(model.collect(), app_key="measure")
    assert verdict["status"] == "post_hoc"
    assert verdict["deviations"][0]["first_seen_utc"] == now["utc"]
    assert "status" not in screen._settings_lock_observation


def test_invalid_committed_text_and_list_drafts_never_record_or_get_rewritten(qtbot, journal):
    source, _ = journal
    _, model, errors = tiny_screen(qtbot, source)
    record = rj.lock_analysis(model.collect(), app_key="measure")
    field = model._widgets["n_jobs"]
    for value in ("bad", "-2"):
        field.setText(value)
        field.editingFinished.emit()
        assert field.text() == value and not events(record)
    field.setText("2")
    editor = model._widgets["channels"]
    editor._strips[0]._entry.setText("bad")
    editor._strips[0]._entry.editingFinished.emit()
    assert not events(record)
    assert editor.get_value() == [0, 1, "bad"]
    assert not errors


def test_checkbox_combo_and_committed_chip_edits_are_observed(qtbot, journal):
    source, _ = journal
    _, model, _ = tiny_screen(qtbot, source)
    record = rj.lock_analysis(model.collect(), app_key="measure")
    model._widgets["plot"].setChecked(True)
    model._widgets["experiment"].setCurrentIndex(1)
    editor = model._widgets["channels"]
    editor._strips[0]._entry.setText("2")
    assert {e["key"] for e in events(record)} == {"plot", "experiment"}
    editor._strips[0]._entry.editingFinished.emit()
    assert {e["key"] for e in events(record)} == {"plot", "experiment", "channels"}


def test_deferred_controls_attach_without_building_other_rows(qtbot, journal):
    source, _ = journal
    _, model, _ = tiny_screen(qtbot, source)
    record = rj.lock_analysis(model.collect(), app_key="measure")
    model._widgets.wait_for("n_jobs", {"control": "int", "value": 2})
    model._widgets.wait_for("untouched", {"control": "text", "value": "stay hidden"})
    field = QLineEdit("2")
    qtbot.addWidget(field)
    model._widgets.settle("n_jobs", field)
    field.setText("3")
    field.editingFinished.emit()
    assert [e["key"] for e in events(record)] == ["n_jobs"]
    assert not model._widgets.is_built("untouched")


def test_source_switch_and_bulk_load_bind_only_completed_new_source(qtbot, journal, tmp_path):
    source, _ = journal
    screen, model, _ = tiny_screen(qtbot, source)
    first = rj.lock_analysis(model.collect(), app_key="measure")
    other = tmp_path / "other"
    other.mkdir()
    second_settings = {**model.collect(), "src": str(other), "n_jobs": 7}
    second = rj.lock_analysis(second_settings, app_key="measure")
    model._applying_settings = True
    model._widgets["n_jobs"].setText("7")
    model._widgets["n_jobs"].editingFinished.emit()
    model._widgets["src"].setText(str(other))
    model._applying_settings = False
    screen.observe()
    assert not events(first) and not events(second)
    model._widgets["n_jobs"].setText("8")
    model._widgets["n_jobs"].editingFinished.emit()
    assert not events(first) and [e["key"] for e in events(second)] == ["n_jobs"]


def test_settings_observation_never_reads_model_gate_files_or_checks_full_verdict(
    journal, monkeypatch
):
    source, _ = journal
    settings = {"src": str(source), "n_jobs": 2}
    record = rj.lock_analysis(settings, app_key="measure")

    def forbidden(*args, **kwargs):
        raise AssertionError("settings observations must not read model/gate payloads")

    monkeypatch.setattr(rj, "hash_file", forbidden)
    monkeypatch.setattr(rj, "_gate_deviations", forbidden)
    monkeypatch.setattr(rj, "_lock_verdict", forbidden)
    result = rj._observe_settings_changes({**settings, "n_jobs": 3}, "measure", ["n_jobs"])
    assert result["scope"] == "committed settings only" and "status" not in result
    assert [e["key"] for e in events(record)] == ["n_jobs"]


def test_no_lock_and_failed_write_preserve_edit_and_never_claim_verified(
    qtbot, journal, monkeypatch
):
    source, _ = journal
    screen, model, errors = tiny_screen(qtbot, source)
    field = model._widgets["n_jobs"]
    field.setText("3")
    field.editingFinished.emit()
    assert screen._settings_lock_observation is None
    rj.lock_analysis(model.collect(), app_key="measure")

    def fail(*args, **kwargs):
        raise PermissionError("journal is read-only")

    monkeypatch.setattr(rj, "_note_first_seen", fail)
    field.setText("4")
    field.editingFinished.emit()
    assert field.text() == "4" and screen._settings_lock_observation is None
    assert errors == ["journal is read-only\n"]


def test_actual_appscreen_bulk_import_and_late_control_commit(qtbot, qt_theme_applied, journal):
    source, _ = journal
    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    screen.apply_settings_dict({"src": str(source), "n_jobs": 2})
    record = rj.lock_analysis(screen._settings_model.collect(), app_key="measure")
    screen.apply_settings_dict({"n_jobs": 3})
    assert "n_jobs" in {e["key"] for e in events(record)}
    widget = screen._settings_model._widgets["test_nr"]
    value = screen._settings_model._read_widget(widget)
    screen._settings_model.set_value_for_key("test_nr", int(value or 1) + 1)
    assert "test_nr" in {e["key"] for e in events(record)}


def test_row_exclusion_checks_and_removal_commit_without_background_load_events(qtbot, journal):
    from spacr.qt.widgets.row_exclusion import RowExclusionEditor

    source, _ = journal
    _, model, _ = tiny_screen(qtbot, source)
    editor = RowExclusionEditor({"drug": ["DMSO"]}, parent=model._parent, threaded=False)
    model._widgets["exclude_rows"] = editor
    record = rj.lock_analysis(model.collect(), app_key="measure")
    row = editor._rows[0]
    row.values.set_options(["DMSO", "drugA"], ["DMSO"])
    assert not events(record)
    row.values._toggle_index(row.values.model().index(1, 0))
    assert [e["key"] for e in events(record)] == ["exclude_rows"]
    editor._remove_row(row)
    assert len(events(record)) == 2
    editor.set_value({"drug": ["silent load"]})
    assert len(events(record)) == 2, "setting loads notify only after the enclosing transaction"


def test_invalid_barcode_pattern_and_tampered_lock_cannot_record_edits(qtbot, journal):
    from spacr.qt.widgets.barcode_regex import BarcodeRegexWidget

    source, _ = journal
    _, model, errors = tiny_screen(qtbot, source)
    field = BarcodeRegexWidget("(?P<columnID>A)(?P<grna>B)(?P<rowID>C)", model._parent)
    model._widgets["regex"] = field
    record = rj.lock_analysis(model.collect(), app_key="measure")
    field._line_edit.setText("[")
    field._line_edit.editingFinished.emit()
    assert not events(record) and field.get_value() == "["
    path = rj._locks_root() / f"{record['lock_id']}.json"
    data = json.loads(path.read_text())
    data["note"] = "altered"
    path.write_text(json.dumps(data))
    model._widgets["n_jobs"].setText("3")
    model._widgets["n_jobs"].editingFinished.emit()
    assert not events(record)
    assert errors and "lock has changed" in errors[-1]


def test_bulk_invalid_snapshot_never_stamps_other_valid_drafts(qtbot, journal):
    source, _ = journal
    screen, model, _ = tiny_screen(qtbot, source)
    record = rj.lock_analysis(model.collect(), app_key="measure")
    model._applying_settings = True
    model._widgets["plot"].setChecked(True)
    model._widgets["n_jobs"].setText("invalid")
    model._applying_settings = False
    screen.observe()
    assert not events(record)
    assert model._widgets["plot"].isChecked()
    assert model._widgets["n_jobs"].text() == "invalid"


def test_editable_combo_text_waits_for_commit(qtbot, journal):
    source, _ = journal
    _, model, _ = tiny_screen(qtbot, source)
    combo = model._widgets["experiment"]
    combo.setEditable(True)
    # The actual builder knows editability before installing its observer.
    combo._spacr_commit_model = None
    model._watch_setting_commit("experiment", combo)
    record = rj.lock_analysis(model.collect(), app_key="measure")
    model._parent.show()
    line = combo.lineEdit()
    line.setFocus()
    line.selectAll()
    qtbot.keyClicks(line, "changed experiment")
    assert not events(record)
    line.editingFinished.emit()
    assert [event["key"] for event in events(record)] == ["experiment"]


def test_spin_typing_waits_for_commit_and_arrow_step_is_immediate(qtbot, journal):
    from PySide6.QtWidgets import QSpinBox

    source, _ = journal
    _, model, _ = tiny_screen(qtbot, source)
    spin = QSpinBox(model._parent)
    spin.setRange(1, 100)
    spin.setValue(2)
    model._widgets["n_jobs"] = spin
    record = rj.lock_analysis(model.collect(), app_key="measure")
    model._parent.show()
    spin.setFocus()
    spin.lineEdit().selectAll()
    qtbot.keyClicks(spin.lineEdit(), "31")
    assert not events(record)
    spin.editingFinished.emit()
    assert len(events(record)) == 1
    spin.stepUp()
    assert len(events(record)) == 2
