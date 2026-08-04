"""Training Runs screen — :mod:`spacr.qt.screens.train_compare`.

Everything here runs offscreen against real temporary run folders built the way
``spacr.io._save_progress`` builds them (see
``tests/test_train_compare.py``, whose builders are reused).

The properties pinned:

* the screen **builds** offscreen and **discovers** runs from a temp root;
* ticking runs and pressing Overlay draws one line per series, each labelled
  run · split · fold, and re-drawing on a metric change keeps that true;
* the **bucketed diff** lands beside the plot — environment drift in its own
  bucket, and two identical runs saying "no differences" in words rather than
  going blank;
* a **broken run folder** is listed and its problem reported inline;
* **no modal dialogs** on any path — a QMessageBox hangs a headless run, so an
  autouse fixture (copied from ``tests/qt/test_db_browser.py``) fails the test
  if one is opened.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from PySide6.QtCore import Qt

from spacr.qt.screens.train_compare import (
    APP_INTRO,
    APP_KEY,
    APP_NAME,
    APP_SECTION,
    FOLD_MODE_LABELS,
    TrainCompareScreen,
)

from tests.test_train_compare import BASE_SETTINGS, make_run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_modal_dialogs(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    ``MakeMasksScreen._load_current`` once hung the whole headless suite on a
    QMessageBox; this fixture makes that failure mode impossible to reintroduce
    here without a red test.
    """
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    def _boom(*_a, **_k):
        raise AssertionError(
            "a modal dialog was opened — errors must be reported inline")

    for name in ("about", "critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("exec", "exec_", "open", "show"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    yield


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A synchronous screen — scans run inline so assertions are exact."""
    w = TrainCompareScreen(threaded=False)
    qtbot.addWidget(w)
    return w


@pytest.fixture
def run_root(tmp_path):
    """Three good runs (one of them 3-fold) plus one broken folder."""
    root = str(tmp_path)
    make_run(root, "dsA", epochs=10,
             train=np.linspace(0.50, 0.80, 10),
             val=np.linspace(0.48, 0.74, 10),
             settings={**BASE_SETTINGS, "epochs": 10, "n_jobs": 30})
    make_run(root, "dsB", epochs=25,
             train=np.linspace(0.50, 0.93, 25),
             val=np.linspace(0.48, 0.86, 25),
             settings={**BASE_SETTINGS, "epochs": 25,
                       "learning_rate": 0.001, "n_jobs": 8})
    make_run(root, "dsCV", model_type="resnet50", epochs=6,
             folds={1: (np.linspace(0.5, 0.8, 6), np.linspace(0.5, 0.7, 6)),
                    2: (np.linspace(0.5, 0.9, 6), np.linspace(0.5, 0.8, 6))},
             settings={**BASE_SETTINGS, "model_type": "resnet50",
                       "cross_validation_folds": 2})
    broken = os.path.join(root, "dsX", "model", "maxvit_t", "rgb", "epochs_3")
    os.makedirs(broken)
    with open(os.path.join(broken, "maxvit_t_epoch_3_channels_rgb.pth"), "wb") as f:
        f.write(b"not really a checkpoint")
    return root


def _id_for(screen, needle):
    """The discovered run id whose folder path contains ``needle``."""
    for run in screen.runs():
        if needle in str(run.path):
            return run.run_id
    raise AssertionError(f"no discovered run under {needle}: "
                         f"{[str(r.path) for r in screen.runs()]}")


def _in_discovery_order(screen, ids):
    """``ids`` in the order the screen lists them (newest folder first)."""
    wanted = set(ids)
    return [rid for rid in screen.run_ids() if rid in wanted]


# ---------------------------------------------------------------------------
# Construction and registration constants
# ---------------------------------------------------------------------------

def test_screen_builds_offscreen_without_raising(qtbot, qt_theme_applied):
    w = TrainCompareScreen()
    qtbot.addWidget(w)
    assert w.status_text()
    assert w.run_rows() == []
    assert w.comparison() is None


def test_registration_constants_put_it_in_results_and_qc():
    """``APP_SECTION`` is the SUBJECT, not the staging bucket.

    #16i staged this screen into Alpha modules, so for a while this
    asserted ``"Alpha modules"``. That conflates two different axes:
    staging says how finished the app is and moves the day it is signed
    off, while the subject says what it does and does not. Comparing
    training runs is reading a result, and this constant says so."""
    assert APP_KEY == "train_compare"
    assert APP_NAME == "Training Runs"
    assert APP_SECTION == "Results & QC"
    assert APP_INTRO.strip()


def test_registration_matches_the_app_registry_when_it_is_wired_up():
    """The APPS table is owned by ``spacr.qt.app``; check consistency if present.

    The screen ships before the registry entry, so this asserts agreement
    rather than existence — it turns into a real check the moment the entry
    lands, and never a false red before then.
    """
    from spacr.qt.app import APPS, app_stage
    entry = next((a for a in APPS if a[0] == APP_KEY), None)
    if entry is None:
        pytest.skip("train_compare not yet registered in spacr.qt.app.APPS")
    key, name, desc, section = entry
    assert name == APP_NAME
    # ``section == APP_SECTION`` again. #16i filed this app under its
    # staging bucket instead; maturity is a stage rather than a section
    # now, so both axes are asserted and neither can drift unnoticed.
    assert section == APP_SECTION
    assert app_stage(key) == "alpha"
    assert desc and desc.strip()


def test_fold_modes_are_the_ones_the_module_accepts():
    from spacr.train_compare import FOLD_MODES
    assert tuple(value for _label, value in FOLD_MODE_LABELS) == FOLD_MODES


def test_empty_canvas_uses_the_active_dark_surface_not_white(screen):
    """The large pre-overlay canvas must not flash as a white rectangle.

    The dark fill moved: it used to be the figure's own ``facecolor``,
    which is opaque by construction and made this the one flat slab on a
    page of translucent panels (``Z9``). The figure patch is transparent
    now and the canvas paints the page panel underneath it in
    ``paintEvent`` — so the canvas is still never white, and the
    page-opacity slider reaches it.
    """
    from spacr.qt.theme import panel_qcolor

    assert screen.figure().get_facecolor()[3] == pytest.approx(0.0), (
        "the figure patch must be transparent, or it covers the panel")
    painted = _canvas_pixel(screen._canvas)
    assert painted.lightnessF() < 0.5, (
        f"the empty canvas painted {painted.name()}, not a dark surface")
    expected = panel_qcolor("surface")
    assert abs(painted.red() - expected.red()) < 40, (
        f"the empty canvas painted {painted.name()}, not the page panel "
        f"{expected.name()}")


def _canvas_pixel(canvas):
    """Render the canvas over black and read a pixel of its panel."""
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter
    canvas.resize(200, 150)
    image = QImage(200, 150, QImage.Format_ARGB32)
    image.fill(0xFF000000)
    painter = QPainter(image)
    canvas.render(painter, QPoint(0, 0))
    painter.end()
    return image.pixelColor(100, 75)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_scan_discovers_runs_and_lists_them_with_their_key_settings(
        screen, run_root):
    assert screen.scan(run_root) is True
    rows = screen.run_rows()
    assert len(rows) == 4                    # three good + one broken
    joined = "\n".join(rows)
    assert "25 epochs" in joined
    assert "6 epochs x 2 folds" in joined
    assert "model_type=maxvit_t" in joined
    assert "learning_rate=0.0001" in joined
    assert "Found 4 runs" in screen.status_text()


def test_scan_emits_the_run_count(screen, run_root, qtbot):
    with qtbot.waitSignal(screen.runs_discovered, timeout=2000) as blocker:
        screen.scan(run_root)
    assert blocker.args == [4]


def test_a_folder_with_no_runs_is_reported_inline_not_in_a_dialog(
        screen, tmp_path):
    assert screen.scan(str(tmp_path)) is True
    assert screen.run_rows() == []
    assert "No training runs under" in screen.status_text()
    assert "train.csv" in screen.status_text()


def test_a_path_that_is_not_a_folder_is_refused_inline(screen, tmp_path):
    missing = str(tmp_path / "nope")
    assert screen.scan(missing) is False
    assert "Not a folder" in screen.last_error
    assert screen.status_text().startswith("Not a folder")


def test_an_empty_path_is_refused_inline(screen):
    assert screen.scan("") is False
    assert "Type or choose a folder" in screen.last_error


# ---------------------------------------------------------------------------
# Broken folders
# ---------------------------------------------------------------------------

def test_a_broken_run_folder_is_listed_and_its_problem_reported_inline(
        screen, run_root):
    screen.scan(run_root)
    bad_id = _id_for(screen, "dsX")
    assert bad_id in screen.run_ids()
    problems = screen.problem_text()
    assert "no per-epoch curves" in problems
    assert "no settings found" in problems
    assert bad_id in problems
    assert "1 with no curves" in screen.status_text()


def test_a_broken_run_does_not_stop_the_others_being_overlaid(
        screen, run_root):
    screen.scan(run_root)
    good = _id_for(screen, "dsA")
    bad = _id_for(screen, "dsX")
    assert screen.select_runs([good, bad]) is True
    assert screen.overlay() is True
    labels = screen.series_labels()
    assert labels == [f"{good} · train", f"{good} · val"]
    assert f"no curves in {bad}" in screen.status_text()


def test_selecting_an_unknown_run_is_reported_inline(screen, run_root):
    screen.scan(run_root)
    assert screen.select_runs(["not-a-run"]) is False
    assert "No such run: not-a-run" in screen.last_error


def test_overlay_without_a_selection_is_refused_inline(screen, run_root):
    screen.scan(run_root)
    assert screen.overlay() is False
    assert "Tick at least one run" in screen.last_error


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def test_overlay_draws_one_labelled_line_per_series(screen, run_root):
    screen.scan(run_root)
    a, b = _id_for(screen, "dsA"), _id_for(screen, "dsB")
    screen.select_runs([a, b])
    assert screen.overlay() is True

    expected = [f"{rid} · {split}"
                for rid in _in_discovery_order(screen, [a, b])
                for split in ("train", "val")]
    assert screen.series_labels() == expected
    ax = screen.figure().axes[0]
    drawn = {line.get_label(): len(line.get_xdata()) for line in ax.lines}
    # Their own lengths — the 10-epoch run is not stretched to 25.
    assert drawn == {f"{a} · train": 10, f"{a} · val": 10,
                     f"{b} · train": 25, f"{b} · val": 25}
    assert "different epoch counts" in screen.status_text()


def test_metric_picker_offers_what_the_runs_logged_and_redraws(
        screen, run_root):
    screen.scan(run_root)
    a = _id_for(screen, "dsA")
    screen.select_runs([a])
    screen.overlay()
    assert screen.selected_metric() == "accuracy"
    assert "loss" in screen.available_metrics()
    assert screen.figure().axes[0].get_ylabel() == "accuracy"

    assert screen.set_metric("loss") is True
    assert screen.figure().axes[0].get_ylabel() == "loss"
    assert screen.series_labels() == [f"{a} · train", f"{a} · val"]

    assert screen.set_metric("f1_macro") is False
    assert "No run logged 'f1_macro'" in screen.last_error


def test_fold_mode_switches_between_per_fold_and_the_labelled_mean(
        screen, run_root):
    screen.scan(run_root)
    cv = _id_for(screen, "dsCV")
    screen.select_runs([cv])
    screen.overlay()
    assert screen.series_labels() == [
        f"{cv} · train · fold 1", f"{cv} · train · fold 2",
        f"{cv} · val · fold 1", f"{cv} · val · fold 2"]

    assert screen.set_fold_mode("mean") is True
    labels = screen.series_labels()
    assert labels == [f"{cv} · train · mean of 2 folds ±sd",
                      f"{cv} · val · mean of 2 folds ±sd"]
    assert all("mean of 2 folds" in label for label in labels)

    assert screen.set_fold_mode("nonsense") is False
    assert "Unknown fold mode" in screen.last_error


def test_clicking_a_series_identifies_its_run(screen, run_root):
    screen.scan(run_root)
    b = _id_for(screen, "dsB")
    screen.select_runs([b])
    screen.overlay()

    label = f"{b} · val"
    line = next(l for l in screen.figure().axes[0].lines
                if l.get_label() == label)

    class _PickEvent:
        artist = line

    screen._on_pick(_PickEvent())
    picked = screen.picked_text()
    assert label in picked
    assert "epochs 1–25" in picked
    assert "last accuracy" in picked and "best accuracy" in picked
    assert "optimistic" in picked
    assert "dsB" in picked                      # the folder it came from


def test_clicking_something_that_is_not_a_series_is_a_no_op(screen, run_root):
    screen.scan(run_root)
    screen.select_runs([_id_for(screen, "dsA")])
    screen.overlay()

    class _Nothing:
        artist = None

    screen._on_pick(_Nothing())
    assert screen.picked_text() == ""


def test_series_clicked_signal_carries_the_label(screen, run_root, qtbot):
    screen.scan(run_root)
    a = _id_for(screen, "dsA")
    screen.select_runs([a])
    screen.overlay()
    with qtbot.waitSignal(screen.series_clicked, timeout=2000) as blocker:
        screen.identify_series(f"{a} · train")
    assert blocker.args == [f"{a} · train"]


# ---------------------------------------------------------------------------
# The settings diff beside the plot
# ---------------------------------------------------------------------------

def test_the_diff_table_buckets_changes_and_environment_drift(
        screen, run_root):
    screen.scan(run_root)
    a, b = _id_for(screen, "dsA"), _id_for(screen, "dsB")
    screen.select_runs([a, b])
    screen.overlay()

    order = _in_discovery_order(screen, [a, b])
    assert screen.diff_headers() == ["bucket", "setting"] + order
    rows = screen.diff_rows()
    changed = {r[1]: r for r in rows if r[0] == "changed"}
    env = {r[1]: r for r in rows if r[0] == "env"}
    assert set(changed) == {"epochs", "learning_rate"}
    lr = dict(zip(order, changed["learning_rate"][2:]))
    assert lr == {a: "0.0001", b: "0.001"}
    # n_jobs differs too, and it is machine drift, not a modelling decision.
    assert "n_jobs" in env
    assert "n_jobs" not in changed
    assert "2 setting(s) changed" in screen.summary_text()
    assert "1 environment drift" in screen.summary_text()


def test_schema_drift_is_its_own_bucket(screen, run_root):
    screen.scan(run_root)
    a, cv = _id_for(screen, "dsA"), _id_for(screen, "dsCV")
    screen.select_runs([a, cv])
    screen.overlay()
    rows = screen.diff_rows()
    drift = {r[1]: r for r in rows if r[0] == "drift"}
    # dsCV records cross_validation_folds and dsA does not.
    assert "cross_validation_folds" in drift
    order = _in_discovery_order(screen, [a, cv])
    assert dict(zip(order, drift["cross_validation_folds"][2:])) == {
        a: "not recorded", cv: "recorded"}
    assert "schema" in screen.summary_text()


def test_identical_settings_say_no_differences_instead_of_going_blank(
        screen, tmp_path):
    root = str(tmp_path)
    same = dict(BASE_SETTINGS)
    make_run(root, "dsA", epochs=10, train=[0.5] * 10, settings=same)
    make_run(root, "dsB", epochs=10, train=[0.6] * 10, settings=same)
    screen.scan(root)
    screen.select_runs(screen.run_ids())
    assert screen.overlay() is True

    rows = screen.diff_rows()
    assert rows, "an empty table reads as a failure"
    assert len(rows) == 1
    assert "No differences" in rows[0][1]
    assert "identical settings" in rows[0][1]
    assert "No differences" in screen.summary_text()


def test_a_run_without_settings_makes_the_diff_say_so(screen, run_root):
    screen.scan(run_root)
    screen.select_runs([_id_for(screen, "dsA"), _id_for(screen, "dsX")])
    screen.overlay()
    assert "not comparable" in screen.summary_text()
    assert screen.diff_rows() == [["—", "no settings to compare"]]


def test_rescanning_clears_the_previous_plot_and_diff(screen, run_root,
                                                      tmp_path):
    screen.scan(run_root)
    screen.select_runs([_id_for(screen, "dsA")])
    screen.overlay()
    assert screen.series_labels()

    empty = tmp_path / "elsewhere"
    empty.mkdir()
    screen.scan(str(empty))
    assert screen.series_labels() == []
    assert screen.diff_rows() == []
    assert screen.comparison() is None
    assert screen.figure().axes == []


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

def test_a_threaded_scan_settles_and_lists_the_same_runs(qtbot,
                                                         qt_theme_applied,
                                                         run_root):
    w = TrainCompareScreen(threaded=True)
    qtbot.addWidget(w)
    with qtbot.waitSignal(w.job_finished, timeout=15000) as blocker:
        assert w.scan(run_root) is True
    assert blocker.args == [True]
    assert len(w.run_rows()) == 4
    qtbot.waitUntil(lambda: w.active_jobs() == 0, timeout=15000)
    assert not w.is_busy()
    w.close()


def test_a_threaded_scan_applies_results_on_the_gui_thread(
        qtbot, qt_theme_applied, run_root, monkeypatch):
    """Worker completion must never mutate Qt widgets from its QThread."""
    w = TrainCompareScreen(threaded=True)
    qtbot.addWidget(w)
    applied_on = []
    original = w._apply_runs

    def _record(result):
        applied_on.append(threading.current_thread())
        return original(result)

    monkeypatch.setattr(w, "_apply_runs", _record)
    with qtbot.waitSignal(w.job_finished, timeout=15000):
        assert w.scan(run_root) is True
    assert applied_on == [threading.main_thread()]
    w.close()


# ---------------------------------------------------------------------------
# Wiring, palettes and failure paths
# ---------------------------------------------------------------------------

def test_the_choose_folder_button_scans_what_the_dialog_returns(
        screen, run_root, monkeypatch):
    """The file dialog is stubbed, not opened — a real one blocks headlessly.

    The autouse fixture above makes ``getExistingDirectory`` raise; this test
    replaces it with a stub so the *wiring* is still exercised without any
    dialog ever appearing.
    """
    from PySide6.QtWidgets import QFileDialog
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *_a, **_k: run_root))
    screen._pick_folder()
    assert screen.root() == run_root
    assert len(screen.run_rows()) == 4

    # Cancelling returns "" and must change nothing.
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *_a, **_k: ""))
    screen._pick_folder()
    assert screen.root() == run_root


def test_the_scan_button_scans_the_typed_path(screen, run_root, qtbot):
    screen._path_edit.setText(run_root)
    assert screen._btn_scan.isEnabled()
    qtbot.mouseClick(screen._btn_scan, Qt.LeftButton)
    assert screen.root() == run_root
    assert len(screen.run_rows()) == 4


def test_the_overlay_button_is_only_enabled_with_a_selection(screen, run_root,
                                                             qtbot):
    screen.scan(run_root)
    assert not screen._btn_overlay.isEnabled()
    screen.select_runs([_id_for(screen, "dsA")])
    assert screen._btn_overlay.isEnabled()
    qtbot.mouseClick(screen._btn_overlay, Qt.LeftButton)
    assert screen.series_labels()


def test_a_failure_while_comparing_lands_in_the_status_line(
        screen, run_root, monkeypatch):
    screen.scan(run_root)
    screen.select_runs([_id_for(screen, "dsA")])
    import spacr.qt.screens.train_compare as mod

    def boom(*_a, **_k):
        raise RuntimeError("compare exploded")

    monkeypatch.setattr(mod.tc, "compare_runs", boom)
    assert screen.overlay() is False
    assert "RuntimeError: compare exploded" in screen.last_error


def test_a_failure_while_applying_a_scan_lands_in_the_status_line(
        screen, run_root, monkeypatch):
    def boom(_runs):
        raise RuntimeError("apply exploded")

    monkeypatch.setattr(screen, "_apply_runs", boom)
    assert screen.scan(run_root) is False
    assert "RuntimeError: apply exploded" in screen.last_error


def test_a_worker_traceback_is_reduced_to_its_last_line(screen):
    screen._on_worker_error_text(
        "Traceback (most recent call last):\n  ...\nValueError: boom\n")
    assert screen.status_text() == "ValueError: boom"
    assert screen.last_error == "ValueError: boom"
    screen._on_worker_error_text("   \n  ")
    assert screen.status_text() == "Scan failed."


def test_drawing_before_a_comparison_exists_is_a_no_op(screen):
    screen._draw()
    assert screen.figure().axes == []


def test_the_plot_follows_the_light_theme_when_that_is_the_preference(
        screen, run_root, monkeypatch):
    import spacr.qt.preferences as prefs
    from spacr.qt.theme import LIGHT_PALETTE
    monkeypatch.setattr(prefs, "get_theme", lambda: "light")
    screen.scan(run_root)
    screen.select_runs([_id_for(screen, "dsA")])
    screen.overlay()
    ax = screen.figure().axes[0]
    # `surface_alt`, not `surface`: the plotting area is a panel within
    # the panel now, and it carries the page opacity like every other one.
    assert ax.get_facecolor()[:3] == pytest.approx(
        _rgb(LIGHT_PALETTE["surface_alt"]), abs=1e-3)


def test_an_unreadable_theme_preference_falls_back_to_dark(screen, run_root,
                                                           monkeypatch):
    import spacr.qt.preferences as prefs
    from spacr.qt.theme import DARK_PALETTE

    def boom():
        raise RuntimeError("no QSettings here")

    monkeypatch.setattr(prefs, "get_theme", boom)
    screen.scan(run_root)
    screen.select_runs([_id_for(screen, "dsA")])
    screen.overlay()
    ax = screen.figure().axes[0]
    assert ax.get_facecolor()[:3] == pytest.approx(
        _rgb(DARK_PALETTE["surface_alt"]), abs=1e-3)


def _rgb(hex_colour):
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def test_closing_the_screen_winds_down_pending_jobs(screen):
    class _Thread:
        def __init__(self, explode=False):
            self.explode = explode
            self.quit_called = False

        def quit(self):
            self.quit_called = True
            if self.explode:
                raise RuntimeError("already gone")

        def wait(self, _ms):
            return True

    good, bad = _Thread(), _Thread(explode=True)
    screen._jobs = [(good, None), (bad, None)]
    screen.close()
    # Both were asked to stop, and the one that refused did not take the
    # close down with it.
    assert good.quit_called and bad.quit_called
    assert screen.active_jobs() == 0
