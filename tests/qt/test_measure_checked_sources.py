"""Checked fields retain identity, bounded work, and single-field navigation."""
import numpy as np

from spacr.qt.widgets.measure_preview import MeasurePreviewPanel


def _field(path, label=1):
    data = np.zeros((20, 20, 5), dtype=np.uint16)
    data[..., :3] = 20
    if label:
        data[2:12, 2:12, 4] = label
    np.save(path, data)
    return str(path.resolve())


def _panel(qtbot, path):
    panel = MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel._mask_dim.setValue(4)
    assert panel.load_array(path)
    return panel


def test_repeated_labels_uncheck_empty_and_filters(qtbot, tmp_path):
    first = _field(tmp_path / "one.npy")
    second = _field(tmp_path / "two.npy")
    empty = _field(tmp_path / "empty.npy", 0)
    panel = _panel(qtbot, first)
    before = panel.settings_for_propagation()
    panel._set_source_checked(second, True)
    panel._set_source_checked(empty, True)
    assert len(panel._crops) == 2
    assert {e["source_path"] for e in panel._crops} == {first, second}
    assert len({e["object_key"] for e in panel._crops}) == 2
    assert {e["object_key"][1] for e in panel._crops} == {"cell"}
    assert panel.settings_for_propagation() == before
    panel._set_source_checked(first, False)
    assert [e["source_path"] for e in panel._crops] == [second]
    panel._min_area.setValue(101)
    assert not panel._crops
    panel._min_area.setValue(0)
    assert len(panel._crops) == 1
    panel._uncheck_all_sources()
    assert not panel._crops and not panel._checked_sources
    assert "No images checked" in panel._status.text()


def test_crop_navigation_uses_exact_source_and_preserves_checks(qtbot, tmp_path):
    first = _field(tmp_path / "one.npy")
    second = _field(tmp_path / "two.npy")
    panel = _panel(qtbot, first)
    panel._set_source_checked(second, True)
    index = next(i for i, e in enumerate(panel._crops)
                 if e["source_path"] == second)
    panel._open_crop_source(index)
    assert panel._data_path == second
    assert panel._checked_sources == {first, second}
    assert len(panel._crops) == 2


def test_budget_and_read_error_do_not_lose_other_sources(qtbot, tmp_path):
    first = _field(tmp_path / "one.npy")
    second = _field(tmp_path / "two.npy")
    panel = _panel(qtbot, first)
    panel._set_source_checked(second, True)
    panel._max_crops.setValue(1)
    assert len(panel._crops) <= 1
    assert "Increase Maximum" in panel._status.text()
    panel._max_crops.setValue(60)
    panel._set_source_checked(str(tmp_path / "missing.npy"), True)
    assert len(panel._crops) == 2
    assert "missing.npy" in panel._status.text()


def test_latest_selection_replaces_inflight_request(qtbot, tmp_path, monkeypatch):
    first = _field(tmp_path / "one.npy")
    second = _field(tmp_path / "two.npy")
    panel = _panel(qtbot, first)
    queued = []
    monkeypatch.setattr(panel._jobs, "submit",
                        lambda fn, done: queued.append((fn, done)))
    panel.refresh()
    panel._set_source_checked(second, True)
    panel._set_source_checked(first, False)
    assert len(queued) == 1
    fn, done = queued.pop(0)
    done(fn())
    assert len(queued) == 1 and not panel._crops
    fn, done = queued.pop(0)
    done(fn())
    assert [e["source_path"] for e in panel._crops] == [second]
    assert not panel._crop_running


def test_menu_checkbox_and_old_thumbnail_signal(qtbot, tmp_path):
    first = _field(tmp_path / "one.npy")
    second = _field(tmp_path / "two.npy")
    panel = _panel(qtbot, first)
    old_token = panel._crop_token
    action = next(a for a in panel._checked_menu.actions()
                  if a.text() == "two.npy")
    action.setChecked(True)
    assert panel._checked_sources == {first, second}
    assert len(panel._crops) == 2
    panel._on_current_thumb_clicked(old_token, 0)
    assert not panel._selected


def test_single_fov_navigation_and_new_folder_reset(qtbot, tmp_path):
    first = _field(tmp_path / "one.npy")
    second = _field(tmp_path / "two.npy")
    panel = _panel(qtbot, first)
    index = panel._fov_box.findData(second)
    assert index >= 0
    panel._fov_box.setCurrentIndex(index)
    assert panel._checked_sources == {second}
    other = tmp_path / "other"
    other.mkdir()
    third = _field(other / "three.npy")
    panel.load_array(third)
    assert panel._checked_sources == {third}
