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

from spacr import run_journal as rj  # noqa: E402


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
