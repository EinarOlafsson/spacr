"""Blind scoring on Annotate and Make Masks (544), and Lock analysis (573).

All three controls are ALPHA features, registered in
``spacr.settings.ALPHA_FEATURES`` (``AnnotateBlindToggle`` and
``MakeMasksBlindToggle`` under 544, ``AnalysisLockButton`` under 573), so
they are hidden until Preferences -> "Show alpha features" is on.

Blinded, no line on either screen names the plate, well or file a crop or
field came from, the order is the key's shuffle, and unblinding restores
both and leaves a record in the key's log.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QDialog, QMessageBox  # noqa: E402

from spacr import run_journal as rj  # noqa: E402

_REAL_DIALOG_EXEC = QDialog.exec
_REAL_MESSAGE_QUESTION = QMessageBox.question
_REAL_MESSAGE_WARNING = QMessageBox.warning


@pytest.fixture
def journal(tmp_path, monkeypatch):
    runs = tmp_path / "home" / "runs"
    runs.mkdir(parents=True)
    monkeypatch.setattr(rj, "runs_root", lambda: runs)
    return runs


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": True}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


@pytest.fixture
def plate(tmp_path) -> Path:
    """A plate whose crop names carry their well, as spaCR's do."""
    src = tmp_path / "plate7_drugA"
    (src / "measurements").mkdir(parents=True)
    (src / "data").mkdir()
    rng = np.random.default_rng(1)
    paths = []
    for row in range(1, 4):
        for col in range(1, 5):
            for obj in range(2):
                path = src / "data" / f"plate7_r{row}_c{col}_f1_o{obj}.png"
                Image.fromarray(rng.integers(0, 255, (12, 12, 3),
                                             dtype=np.uint8)).save(path)
                paths.append(str(path))
    with sqlite3.connect(src / "measurements" / "measurements.db") as conn:
        conn.execute('CREATE TABLE "png_list" (png_path TEXT PRIMARY KEY)')
        conn.executemany('INSERT INTO "png_list" (png_path) VALUES (?)',
                         [(p,) for p in paths])
    return src


@pytest.fixture
def annotate(qtbot, qt_theme_applied, plate, journal, alpha):
    from spacr.qt.screens.annotate import AnnotateScreen

    widget = AnnotateScreen()
    qtbot.addWidget(widget)
    widget._settings.image_size = (32, 32)
    widget.resize(900, 700)
    widget._open_source(str(plate))
    qtbot.waitUntil(lambda: bool(widget._page_paths), timeout=10000)
    yield widget
    if widget._worker is not None:
        widget._worker.stop(wait=True)


def _all_paths(plate):
    with sqlite3.connect(plate / "measurements" / "measurements.db") as conn:
        return [row[0] for row in conn.execute(
            'SELECT png_path FROM "png_list"')]


def _visible_text(widget) -> str:
    from PySide6.QtWidgets import QLabel

    return "\n".join(label.text() for label in widget.findChildren(QLabel)
                     if label.isVisibleTo(widget))


def test_annotate_blinds_shuffles_and_unblinds_with_a_record(annotate, plate,
                                                            qtbot):
    screen = annotate
    insertion = _all_paths(plate)
    assert screen._btn_blind.objectName() == "AnnotateBlindToggle"
    screen._btn_blind.setChecked(True)
    blind = screen._blind
    assert blind is not None and screen._btn_blind.isChecked()

    order = [path for path, _value in screen._filtered_rows]
    assert sorted(order) == sorted(insertion) and order != insertion
    stored = rj._read_blinding_key(blind["key_id"])
    assert order == stored["order"]
    shown = _visible_text(screen)
    assert "plate7" not in shown and "drugA" not in shown
    assert "_r1_c" not in shown
    assert screen._page_label.text().startswith("Blinded")
    for button in (screen._btn_coverage, screen._btn_auto,
                   screen._btn_browse_db):
        assert not button.isEnabled()

    labelled = screen._page_paths[0][0]
    assert screen._set_annotation(0, 1)
    screen._flush_pending()
    assert dict(screen._filtered_rows)[labelled] == 1
    screen._load_page()
    assert screen._current_value(0) == 1

    screen._refresh_total(then=screen._load_page)
    qtbot.waitUntil(lambda: not screen._total_jobs.is_busy(), timeout=10000)
    assert [p for p, _ in screen._filtered_rows] == order

    assert screen._end_blind(ask=lambda: False) is False
    assert screen._blind is blind
    assert [e["event"] for e in rj._blinding_events(blind["key_id"])] == [
        "blinded"]

    assert screen._end_blind(ask=lambda: True) is True
    qtbot.waitUntil(lambda: not screen._total_jobs.is_busy(), timeout=10000)
    assert screen._blind is None and not screen._btn_blind.isChecked()
    assert screen._filtered_rows is None
    assert str(plate) in screen._src_label.text()
    assert all(b.isEnabled() for b in (screen._btn_coverage, screen._btn_auto,
                                       screen._btn_browse_db))
    events = rj._blinding_events(blind["key_id"])
    assert [e["event"] for e in events] == ["blinded", "unblinded"]
    assert events[-1]["who"] and events[-1]["utc"]
    assert rj._unblinding_times(plate) == [events[-1]["utc"]]


def test_a_new_source_ends_a_blinded_session_without_unblinding(
        annotate, plate):
    annotate._btn_blind.setChecked(True)
    key_id = annotate._blind["key_id"]
    annotate._open_source(str(plate))
    assert annotate._blind is None and not annotate._btn_blind.isChecked()
    assert [e["event"] for e in rj._blinding_events(key_id)] == [
        "blinded", "closed"]
    assert rj._unblinding_times(plate) == []


def test_blinding_hides_settings_and_console_history_then_restores_them(
        annotate, plate, qtbot, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from spacr.qt.screens.annotate import _SettingsDialog

    screen = annotate
    screen.show()
    screen._console_switch.setChecked(True)
    qtbot.waitUntil(lambda: screen._console_wrap.isVisible())
    history = screen._console.copy_all()
    assert str(plate) in history
    source, database = screen._settings.src, screen._settings.db_path
    screen._btn_blind.setChecked(True)
    assert not screen._btn_settings.isEnabled()
    assert not screen._console_switch.isEnabled()
    assert not screen._console_wrap.isVisible()
    screen._on_console_switch(True)
    screen._on_open_settings()
    assert not screen._console_wrap.isVisible()
    assert getattr(screen, '_settings_dialog', None) is None
    QApplication.clipboard().setText('unchanged while blinded')
    screen._on_copy_console()
    assert QApplication.clipboard().text() == 'unchanged while blinded'
    assert (screen._settings.src, screen._settings.db_path) == (source, database)
    assert screen._end_blind(ask=lambda: False) is False
    assert not screen._console_wrap.isVisible()
    assert screen._end_blind(ask=lambda: True)
    assert screen._btn_settings.isEnabled() and screen._console_switch.isEnabled()
    assert screen._console_wrap.isVisible()
    assert history in screen._console.copy_all()
    displayed = []
    expired = []

    def close_settings():
        dialog = screen._settings_dialog
        if dialog is not None:
            displayed.append(dialog._src_edit.text())
            dialog.reject()

    def abort_modal():
        # The deadline runs from before the dialog is built, so on a slow
        # coverage runner it can come due in the same event pass in which
        # the 20 ms hook already closed it. Only a dialog still open is a
        # hang.
        dialog = QApplication.activeModalWidget()
        if dialog is not None:
            expired.append(True)
            dialog.reject()

    monkeypatch.setattr(_SettingsDialog, 'exec', _REAL_DIALOG_EXEC)
    timer = QTimer(screen)
    timer.timeout.connect(close_settings)
    deadline = QTimer(screen)
    deadline.setSingleShot(True)
    deadline.timeout.connect(abort_modal)
    deadline.start(5000)
    timer.start(20)
    try:
        screen._on_open_settings()
    finally:
        timer.stop()
        timer.deleteLater()
        deadline.stop()
        deadline.deleteLater()
    assert not expired
    assert displayed == [source]


@pytest.fixture
def masks(qtbot, qt_theme_applied, tmp_path, journal, alpha):
    from spacr.qt.screens.make_masks import MakeMasksScreen

    folder = tmp_path / "plate7_drugA_images"
    folder.mkdir()
    rng = np.random.default_rng(2)
    names = [f"plate7_r{r}_c{c}_f1.tif" for r in (1, 2) for c in (1, 2, 3)]
    for name in names:
        imageio.imwrite(folder / name,
                        rng.integers(0, 65535, (32, 32), dtype=np.uint16))
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder))
    return widget, folder, names


def test_make_masks_names_fields_by_code_while_blinded(masks, qtbot):
    screen, folder, names = masks
    listed = list(screen._image_files)
    assert sorted(listed) == sorted(names)
    screen._current_index = 2
    screen._load_current()
    screen._btn_blind.setChecked(True)
    blind = screen._blind
    assert blind is not None
    assert screen._image_files != listed
    assert sorted(screen._image_files) == sorted(names)
    qtbot.waitUntil(lambda: "(1/6)" in screen._status_label.text(),
                    timeout=10000)
    shown = screen._status_label.text() + screen._src_label.text()
    assert shown.split()[0] in blind["codes"].values()
    assert "plate7" not in shown and ".tif" not in shown
    assert screen.recrop(0, 0, 10, 10) is None
    assert "blinded" in screen._status_label.text()
    screen._on_next()
    assert "plate7" not in screen._status_label.text()
    on_screen = screen._image_files[screen._current_index]

    assert screen._end_blind(ask=lambda: True)
    assert screen._blind is None and not screen._btn_blind.isChecked()
    assert screen._image_files == listed
    assert screen._image_files[screen._current_index] == on_screen
    assert str(folder) in screen._src_label.text()
    assert on_screen in screen._status_label.text()
    assert [e["event"] for e in rj._blinding_events(blind["key_id"])] == [
        "blinded", "unblinded"]


def test_make_masks_blinds_raw_status_failures_and_actual_folder_confirmation(
        masks, qtbot, monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from spacr.qt.screens import make_masks as module

    screen, folder, names = masks
    screen.show()
    screen._console_section.show()
    screen._status_label.setText(f'Opened {folder / names[0]}')
    before = screen._masks_console.console.copy_all()
    assert str(folder) in before
    screen._btn_blind.setChecked(True)
    assert not screen._console_section.isVisible()
    screen._console_section.set_folded(True, by_user=False)
    screen._console_section.set_folded(False, by_user=False)
    screen._view_pane.set_collapsed('Shortcuts', True, by_user=False)
    screen._view_pane.set_collapsed('Shortcuts', False, by_user=False)
    assert not screen._console_section.isVisible()
    selected = screen._image_files[screen._current_index]
    path = folder / selected
    code = screen._blind['codes'][str(path)]
    raw = f'Could not read {path}; field {selected} in {folder.name}'
    screen._canvas.status.emit(raw)
    assert code in screen._status_label.text()
    assert 'plate7' not in screen._status_label.text()
    assert 'drugA' not in screen._status_label.toolTip()
    original = path.read_bytes()
    path.write_bytes(b'not a TIFF')
    try:
        screen._load_current()
        assert 'Load failed' in screen._status_label.text()
        assert 'plate7' not in screen._status_label.text()
        assert 'drugA' not in screen._status_label.toolTip()
        assert screen._canvas.image is None
    finally:
        path.write_bytes(original)
    shown = []
    expired = []
    monkeypatch.setattr(module, 'is_headless', lambda: False)
    monkeypatch.setattr(QMessageBox, 'question', _REAL_MESSAGE_QUESTION)
    monkeypatch.setattr(QMessageBox, 'warning', _REAL_MESSAGE_WARNING)

    def refuse_dialog():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            shown.append((dialog.windowTitle(), dialog.text()))
            dialog.done(QMessageBox.No)

    def abort_modal():
        # The deadline runs from before the dialog is built, so on a slow
        # coverage runner it can come due in the same event pass in which
        # the 20 ms hook already closed it. Only a dialog still open is a
        # hang.
        dialog = QApplication.activeModalWidget()
        if dialog is not None:
            expired.append(True)
            dialog.reject()

    timer = QTimer(screen)
    timer.timeout.connect(refuse_dialog)
    deadline = QTimer(screen)
    deadline.setSingleShot(True)
    deadline.timeout.connect(abort_modal)
    deadline.start(5000)
    timer.start(20)
    try:
        assert screen.mask_whole_folder() is False
    finally:
        timer.stop()
        deadline.stop()
    assert not expired
    assert len(shown) == 1
    assert 'Segment all 6 images' in shown[0][1]
    assert 'plate7' not in str(shown) and 'drugA' not in str(shown)
    timer.start(20)
    deadline.start(5000)
    try:
        screen._warn('Read failed', raw)
    finally:
        timer.stop()
        timer.deleteLater()
        deadline.stop()
        deadline.deleteLater()
    assert not expired
    assert len(shown) == 2 and code in shown[1][1]
    assert 'plate7' not in str(shown) and 'drugA' not in str(shown)
    assert screen._folder == str(folder) and sorted(screen._image_files) == sorted(names)
    assert screen._end_blind(ask=lambda: True)
    assert screen._console_section.isVisible()
    assert before in screen._masks_console.console.copy_all()
    screen._canvas.status.emit(raw)
    assert screen._status_label.text() == raw
    assert screen._canvas.image is not None


def test_every_new_control_is_hidden_until_alpha_features_are_shown(
        qtbot, qt_theme_applied, monkeypatch, journal):
    from spacr.qt import preferences
    from spacr.qt.screens.annotate import AnnotateScreen
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.make_masks import MakeMasksScreen
    from spacr.settings import ALPHA_FEATURES

    assert set(ALPHA_FEATURES[544]["widgets"]) == {
        "AnnotateBlindToggle", "MakeMasksBlindToggle"}
    assert ALPHA_FEATURES[573] == {"widgets": ("AnalysisLockButton",)}
    for shown in (False, True):
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: s)
        annotate = AnnotateScreen()
        masks = MakeMasksScreen()
        app = AppScreen("measure")
        for widget in (annotate, masks, app):
            qtbot.addWidget(widget)
        buttons = (annotate._btn_blind, masks._btn_blind,
                   app._btn_analysis_lock)
        assert [b.isHidden() for b in buttons] == [not shown] * 3
        monkeypatch.setattr(preferences, "_get_show_alpha_features",
                            lambda s=shown: not s)
        for widget in (annotate, masks, app):
            preferences._apply_alpha_widgets(widget)
        assert [b.isHidden() for b in buttons] == [shown] * 3
        app.close()


def test_lock_analysis_dialog_locks_then_verifies(qtbot, qt_theme_applied,
                                                  journal, alpha, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    assert screen._settings_model.set_value_for_key("src",
                                                    str(tmp_path / "p"))
    dialog = screen._analysis_lock_dialog()
    qtbot.addWidget(dialog)
    parts = dialog._spacr_lock_parts
    assert "No analysis lock" in parts["status"].text()
    parts["hypotheses"].setPlainText("Drug A lowers infection.")
    parts["thresholds"].setPlainText("infection_fraction < 0.2")
    parts["lock"].click()
    assert "Locked" in parts["status"].text()
    assert not parts["lock"].isEnabled()
    lock = rj._find_lock("measure", tmp_path / "p")
    assert lock["plan"]["hypotheses"] == "Drug A lowers infection."
    assert lock["plan"]["thresholds"] == "infection_fraction < 0.2"
    assert lock["sha256"][:16] in parts["status"].text()

    again = screen._analysis_lock_dialog()
    qtbot.addWidget(again)
    assert "verified" in again._spacr_lock_parts["status"].text()
    settings = dict(screen._settings_model.collect())
    result = rj.check_analysis_lock(settings, app_key="measure")
    assert result["status"] == "verified", result["deviations"]
    screen.close()


def test_the_lock_dialog_locks_gate_files_and_the_gate_editor_says_so(
        qtbot, qt_theme_applied, journal, alpha, tmp_path, monkeypatch):
    import json

    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.gate_editor import GateEditorScreen
    from spacr.qt.widgets.gate_spec import GateSet, gate_from_dict

    def gates(low):
        return GateSet([gate_from_dict({
            "kind": "threshold", "name": "big", "parent": None,
            "column": "cell_area", "low": low, "high": None})])

    strategy = tmp_path / "strategy.json"
    gates(100.0).save(str(strategy))
    model = tmp_path / "cyto.pt"
    model.write_bytes(b"weights")
    src = tmp_path / "p"
    with rj.open_run("measure", {"src": str(src)}) as earlier:
        earlier.record_model("cellpose_cyto", model)

    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    assert screen._settings_model.set_value_for_key("src", str(src))
    dialog = screen._analysis_lock_dialog()
    qtbot.addWidget(dialog)
    parts = dialog._spacr_lock_parts
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        lambda *a, **k: warned.append(a[-1]))
    parts["gate_files"].setText(str(tmp_path / "missing.json"))
    parts["lock"].click()
    assert warned and rj._find_lock("measure", src) is None
    parts["gate_files"].setText(f" {strategy} ; ")
    parts["lock"].click()
    lock = rj._find_lock("measure", src)
    assert list(lock["gates"]) == [str(strategy.resolve())]
    assert lock["models"]["cellpose_cyto"]["path"] == str(model.resolve())

    editor = GateEditorScreen()
    qtbot.addWidget(editor)
    editor.load_gates(str(strategy))
    assert f"match analysis lock {lock['sha256'][:16]}" in (
        editor._source.text())
    editor.gates.set_gates(gates(150.0))
    editor.save_gates(str(strategy))
    text = editor._source.text()
    assert "differ from analysis lock" in text and "changed big" in text
    assert json.loads(strategy.read_text())["gates"][0]["low"] == 150.0
    screen.close()


# ---------------------------------------------------------------------------
# The Blind switch's refusals (coverage ratchet, 288)
# ---------------------------------------------------------------------------

def test_the_switch_goes_back_when_blinding_or_unblinding_is_refused(
        annotate, monkeypatch, qtbot):
    screen = annotate
    answers = iter([QMessageBox.No, QMessageBox.Yes])
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: next(answers)))
    screen._btn_blind.setChecked(True)
    assert screen._blind is not None
    screen._btn_blind.setChecked(False)
    assert screen._blind is not None and screen._btn_blind.isChecked(), (
        "a refused unblind puts the switch back on")
    screen._btn_blind.setChecked(False)
    qtbot.waitUntil(lambda: not screen._total_jobs.is_busy(), timeout=10000)
    assert screen._blind is None and not screen._btn_blind.isChecked()
    assert screen._end_blind() is True, "nothing blinded is nothing to ask"


def test_blinding_without_a_source_is_refused_and_the_switch_goes_back(
        qtbot, qt_theme_applied, journal, alpha, monkeypatch):
    from spacr.qt.screens.annotate import AnnotateScreen

    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: told.append(a[1])))
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    screen._btn_blind.setChecked(True)
    assert screen._blind is None and not screen._btn_blind.isChecked()
    assert told == ["Open a source first"]
    screen.__dict__["_btn_blind"] = None
    screen._set_blind_checked(True)


def test_a_blinded_total_reads_the_whole_population_in_key_order(annotate):
    from spacr.qt.screens.annotate import _blinded_total

    screen = annotate
    outcome = {"total": 3, "filtered_rows": None, "note": "x"}
    assert _blinded_total(outcome, screen._settings, None) is outcome
    paths = _all_paths_of(screen)
    rank = {path: index for index, path in enumerate(reversed(paths))}
    blinded = _blinded_total(outcome, screen._settings, rank)
    assert [row[0] for row in blinded["filtered_rows"]] == list(
        reversed(paths))
    assert blinded["total"] == len(paths) and blinded["note"] == ""
    given = [(path, None) for path in paths[:2]]
    kept = _blinded_total(dict(outcome, filtered_rows=given),
                          screen._settings, rank)
    assert [row[0] for row in kept["filtered_rows"]] == [paths[1], paths[0]]
    screen._on_blind_toggled(False)
    assert screen._blind is None


def _all_paths_of(screen):
    with sqlite3.connect(screen._settings.db_path) as conn:
        return [row[0] for row in conn.execute(
            f'SELECT png_path FROM "{screen._settings.png_table}"')]
