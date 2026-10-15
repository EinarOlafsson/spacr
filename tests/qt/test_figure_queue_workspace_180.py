"""Session-created figures survive the real screen's saved workspace."""
import copy
import json
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QImage

from spacr import workspace
from spacr.qt import preferences
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets.figure_queue import FigureQueue


@pytest.fixture(autouse=True)
def private_preferences(monkeypatch, tmp_path):
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings", lambda: QSettings(
        str(path), QSettings.Format.IniFormat))
    preferences.set_figure_format("png")
    workspace.clear_providers()
    yield
    workspace.clear_providers()


def append_raster(queue, directory, name, color):
    path = directory / f"{name}.png"
    image = QImage(24, 16, QImage.Format.Format_RGB32)
    image.fill(QColor(color))
    assert image.save(str(path))
    figure = Figure(label=name)
    queue.set_live_canvas_enabled(False)
    queue.add_figure(figure, str(path))
    return Path(queue._png_paths[queue.count() - 1])


@pytest.mark.parametrize("mode,app_key", [("reference", "mask"), ("copy", "mask"),
                                         ("reference", "regression")])
def test_actual_screen_carries_session_figures_after_original_queue_closes(
    qtbot, qt_theme_applied, tmp_path, mode, app_key,
):
    first = AppScreen(app_key)
    qtbot.addWidget(first)
    second = None
    try:
        queue = first._figure_queue
        queue.mark_run("session figures")
        originals = [append_raster(queue, tmp_path, "first", "red"),
                     append_raster(queue, tmp_path, "second", "blue")]
        original_bytes = [path.read_bytes() for path in originals]
        queue.show_index(0)
        document = workspace.collect(workspace.providers(), app_key=app_key)
        section = f"{app_key}:figure_queue"
        assert section in document["sections"]
        run = tmp_path / "saved-run"
        workspace.save(run, document, mode=mode)
        first.close()
        assert all(not path.exists() for path in originals)
        loaded = workspace.load(run)
        before = copy.deepcopy(loaded)
        second = AppScreen(app_key)
        qtbot.addWidget(second)
        restored = second._figure_queue
        grid = second._figure_grid if app_key == "regression" else None
        report = workspace.restore(workspace.providers(), loaded, run_dir=run)
        assert section in report["restored"]
        assert restored.count() == 2
        assert restored.figure_titles() == ["first", "second"]
        assert restored.run_sections() == [("session figures", 0, 2)]
        assert restored._current == 0
        assert [Path(restored._png_paths[i]).read_bytes() for i in range(2)] == original_bytes
        assert [p.toImage().pixelColor(0, 0).name() for p in restored.all_pixmaps()] == ["#ff0000", "#0000ff"]
        assert not restored.has_live_figure(0)
        assert not restored.is_restorable(0)
        assert all(not restored._list.item(i).icon().isNull() for i in range(2))
        assert not second._figures_card.isHidden()
        if grid is not None:
            assert len(grid._cells) == 2
        assert loaded == before
        carried = {path: path.read_bytes() for path in (run / workspace.FILES_DIR).iterdir()}
        second.close()
        assert all(path.read_bytes() == content for path, content in carried.items())
    finally:
        first.close()
        if second is not None:
            second.close()


def test_only_queue_owned_small_generated_assets_are_carried(qtbot, tmp_path):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    own = append_raster(queue, tmp_path, "owned", "red")
    own.with_suffix(".pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    own.with_suffix(".pkl").write_bytes(b"never carry live Python objects")
    external = tmp_path / "external.png"
    external.write_bytes(own.read_bytes())
    queue._png_paths[1] = str(external)
    queue._titles[1] = "external"
    queue._count = 2
    state = queue.workspace_state()
    carried = {row["path"] for row in state[workspace.CARRY_KEY]}
    assert carried == {str(own), str(own.with_suffix(".pdf"))}
    assert state["figures"][1]["png"] == str(external)
    workspace.set_default_mode("reference", copy_limit_mb=0)
    assert queue.workspace_state()[workspace.CARRY_KEY] == []
    queue.close()
    assert external.is_file()


def test_optional_pdf_is_paired_after_distinct_bundle_digest_names(qtbot, tmp_path):
    first, second = FigureQueue(), FigureQueue()
    for queue in (first, second):
        qtbot.addWidget(queue)
    png = append_raster(first, tmp_path, "vector", "green")
    pdf = png.with_suffix(".pdf")
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    run = tmp_path / "run"
    workspace.save(run, workspace.collect({"queue": first}), mode="reference")
    first.close()
    document = workspace.load(run)
    carried = {path: path.read_bytes() for path in (run / workspace.FILES_DIR).iterdir()}
    assert len(carried) == 2
    assert workspace.restore({"queue": second}, document, run_dir=run)["restored"] == ["queue"]
    target = Path(second._png_paths[0])
    assert target.with_suffix(".pdf").read_bytes() == b"%PDF-1.4\n%%EOF\n"
    new_run = tmp_path / "resaved"
    workspace.save(new_run, workspace.collect({"queue": second}), mode="reference")
    assert len(list((new_run / workspace.FILES_DIR).iterdir())) == 2
    second.close()
    assert all(path.read_bytes() == content for path, content in carried.items())


@pytest.mark.parametrize("change", [
    {"version": 2}, {"current": True}, {"current": 99}, {"figures": "invalid"},
    {"figures": [{"png": "/tmp/data.pkl", "title": "unsafe"}]},
    {"runs": [{"label": "invalid", "start": -1}]},
])
def test_invalid_state_leaves_existing_queue_untouched(qtbot, tmp_path, change):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    path = append_raster(queue, tmp_path, "before", "red")
    state = queue.workspace_state()
    state.update(change)
    assert queue.apply_workspace_state(state) is False
    assert queue.count() == 1
    assert Path(queue._png_paths[0]) == path
    assert path.is_file()
    queue.close()


def test_non_png_and_over_budget_restore_leave_existing_queue_untouched(qtbot, tmp_path):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    path = append_raster(queue, tmp_path, "before", "red")
    bad = tmp_path / "disguised.png"
    bad.write_text("not a PNG")
    state = {"version": 1, "figures": [{"png": str(bad), "title": "bad"}],
             "runs": [], "current": 0}
    assert queue.apply_workspace_state(state) is False
    assert Path(queue._png_paths[0]) == path
    workspace.set_default_mode("reference", copy_limit_mb=0)
    state["figures"][0]["png"] = str(path)
    with pytest.raises(ValueError, match="copy limit"):
        queue.apply_workspace_state(state)
    assert Path(queue._png_paths[0]) == path
    assert path.is_file()
    queue.close()


def test_missing_raster_retains_slot_and_normal_restore_reports_missing(qtbot, tmp_path):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    missing = tmp_path / "missing.png"
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    assert image.save(str(missing))
    state = {"version": 1, "figures": [{"png": str(missing), "title": "missing"}],
             "runs": [], "current": 0}
    document = workspace.collect({"queue": state})
    missing.unlink()
    report = workspace.restore({"queue": queue}, document)
    assert report["restored"] == ["queue"]
    assert report["files"][0]["state"] == "missing"
    assert queue.count() == 1
    assert queue.figure_titles() == ["missing"]
    assert queue.all_pixmaps() == [None]
    queue.close()


def test_off_writes_nothing_and_never_removes_original_queue_assets(qtbot, tmp_path):
    queue = FigureQueue()
    qtbot.addWidget(queue)
    path = append_raster(queue, tmp_path, "off", "red")
    run = tmp_path / "off-run"
    assert workspace.save(run, workspace.collect({"queue": queue}), mode="off") is None
    assert not run.exists()
    assert path.is_file()
    queue.close()


def test_registry_does_not_construct_unvisited_results_for_restore(qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.app_screen import _REGRESSION_RESULTS

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    queue = screen._figure_queue
    path = append_raster(queue, tmp_path, "deferred", "red")
    state = json.loads(json.dumps(queue.workspace_state()))
    assert screen._part_is_owed(_REGRESSION_RESULTS)
    document = {"sections": {"regression:figure_queue": state,
                              "regression:figures": {"cell_width": 400, "collapsed": []}}}
    report = workspace.restore(workspace.providers(), document)
    assert "regression:figure_queue" in report["restored"]
    assert any(row["section"] == "regression:figures" for row in report["skipped"])
    assert screen._part_is_owed(_REGRESSION_RESULTS)
    assert screen._if_built("_figure_grid") is None
    assert not path.exists()
    assert Path(queue._png_paths[0]).is_file()
    screen.close()
