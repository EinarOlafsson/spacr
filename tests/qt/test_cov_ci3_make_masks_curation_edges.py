"""Make Masks' N685 curation controls at their edges.

The vvvv export with no folder, a refused folder, a blinded session or a
failing writer; the queue tally and First unreviewed when the screen is
half built, on the field already shown, mid-recrop or unreadable; the
magnifier's whole-image auto-accept and right-click removal; and the
small guards around Save on navigation, blinding, the prompt buttons and
the folded screens.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import cpu_modes
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen

pytestmark = pytest.mark.qt

IMG_N = 64


def field_image():
    img = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    img[8:56, 8:56] = 20000
    return img


def objects_mask():
    mask = np.zeros((IMG_N, IMG_N), dtype=np.uint16)
    mask[4:10, 4:10] = 3
    mask[20:40, 20:40] = 7
    mask[54:60, 44:54] = 9
    return mask


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    root = tmp_path / "field"
    root.mkdir()
    for index in range(3):
        imageio.imwrite(root / f"img_{index:02d}.tif", field_image())
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    assert widget._open_folder(str(folder))
    widget._canvas.set_image_and_mask(field_image(), objects_mask())
    widget._history.clear()
    widget._history.push(widget._canvas.mask)
    widget._magnifier.segment = lambda request: np.zeros(
        request.crop.shape[:2] if request.crop is not None
        else request.shape, dtype=np.int32)
    yield widget
    widget._magnifier.set_enabled(False)
    widget._magnifier.close()


@pytest.fixture
def warnings(screen, monkeypatch):
    said = []
    monkeypatch.setattr(screen, "_warn",
                        lambda title, text: said.append((title, text)))
    return said


@pytest.fixture
def alpha(monkeypatch):
    from spacr.qt import preferences

    state = {"on": True}
    monkeypatch.setattr(preferences, "_get_show_alpha_features",
                        lambda: state["on"])
    return state


def _dialog(monkeypatch, answer):
    asked = []

    def pick(parent, title, start):
        asked.append(start)
        return answer

    monkeypatch.setattr(mm.QFileDialog, "getExistingDirectory",
                        staticmethod(pick))
    return asked


# vvvv export


def test_choosing_an_export_folder_remembers_it(screen, monkeypatch, tmp_path):
    chosen = str(tmp_path / "patch")
    asked = _dialog(monkeypatch, chosen)
    assert screen._vvvv_export_root() == ""
    assert screen._ask_vvvv_export_root() == chosen
    assert asked == [screen._folder]
    assert screen._vvvv_export_root() == chosen
    asked = _dialog(monkeypatch, "")
    assert screen._ask_vvvv_export_root() == ""
    assert asked == [chosen]
    assert screen._vvvv_export_root() == chosen


def test_export_on_save_without_a_folder_switches_itself_off(
        screen, monkeypatch, tmp_path):
    from spacr.qt.prefs import _s

    _dialog(monkeypatch, "")
    screen._btn_vvvv_on_save.setChecked(True)
    assert not screen._btn_vvvv_on_save.isChecked()
    assert _s().value("make_masks/vvvv_export_on_save", True,
                      type=bool) is False
    _dialog(monkeypatch, str(tmp_path / "out"))
    screen._btn_vvvv_on_save.setChecked(True)
    assert screen._btn_vvvv_on_save.isChecked()
    assert _s().value("make_masks/vvvv_export_on_save", False,
                      type=bool) is True
    assert screen._vvvv_export_root() == str(tmp_path / "out")


def test_export_asks_for_a_folder_and_says_where_it_wrote(
        screen, monkeypatch, tmp_path):
    out = tmp_path / "vvvv"
    _dialog(monkeypatch, str(out))
    written = screen._on_export_vvvv()
    assert written == str(out / "img_00")
    np.testing.assert_array_equal(
        imageio.imread(out / "img_00" / "img_00_labels.png"),
        screen._canvas.mask)
    assert str(out / "img_00") in screen._status_label.text()


def test_export_cancelled_at_the_folder_writes_nothing(
        screen, monkeypatch, tmp_path):
    _dialog(monkeypatch, "")
    screen._status_label.setText("before")
    assert screen._on_export_vvvv() is None
    assert screen._status_label.text() == "before"
    assert not any(tmp_path.glob("**/manifest.json"))


def test_export_without_a_field_writes_nothing(screen, tmp_path):
    screen._image_files = []
    assert screen._on_export_vvvv(str(tmp_path / "out")) is None
    assert not (tmp_path / "out").exists()


def test_export_while_blinded_warns_and_writes_nothing(
        screen, warnings, tmp_path):
    screen._blind = {"codes": {}}
    try:
        assert screen._on_export_vvvv(str(tmp_path / "out")) is None
    finally:
        screen._blind = None
    assert [title for title, _text in warnings] == [
        "vvvv export is off while blinded"]
    assert not (tmp_path / "out").exists()


def test_a_failing_export_warns_with_the_reason(screen, warnings, tmp_path):
    bad = np.zeros((2, IMG_N, IMG_N), dtype=np.uint16)
    assert screen._on_export_vvvv(str(tmp_path / "out"), labels=bad) is None
    assert warnings and warnings[0][0] == "vvvv export failed"
    assert "2-D" in warnings[0][1]


def test_export_on_save_while_blinded_exports_nothing(
        screen, alpha, monkeypatch, tmp_path):
    out = tmp_path / "vvvv"
    monkeypatch.setattr(type(screen), "_vvvv_export_root",
                        staticmethod(lambda: str(out)))
    screen._btn_vvvv_on_save.setChecked(True)
    assert screen._export_vvvv_on_save() == str(out / "img_00")
    alpha["on"] = False
    assert screen._export_vvvv_on_save() is None
    screen._btn_vvvv_on_save.setChecked(False)


# queue tally, Go to and First unreviewed


def test_the_tally_waits_for_its_box():
    half_built = SimpleNamespace()
    assert MakeMasksScreen._sync_queue_tally(half_built) is None
    assert vars(half_built) == {}


def test_the_tally_counts_without_a_first_unreviewed_button(screen):
    del screen._btn_first_unreviewed
    screen._current_index = 2
    screen._sync_queue_tally()
    assert screen._tally_index.text() == "3"
    assert screen._tally_total.text() == "/3"


def test_going_to_the_field_on_screen_reloads_nothing(screen):
    shown = screen._canvas.image
    screen._tally_index.setText("9")
    assert screen._go_to_index(0) is False
    assert screen._canvas.image is shown
    assert screen._tally_index.text() == "1"


def test_a_recropped_field_is_retired_instead_of_left(screen, monkeypatch):
    retired = []
    monkeypatch.setattr(screen, "finish_recrop",
                        lambda: retired.append(screen._current_index) or True)
    assert screen._go_to_index(2) is False
    assert retired == [0]
    assert screen._current_index == 0
    assert screen._tally_index.text() == "1"


def test_unreadable_review_status_warns_and_stays(screen, warnings,
                                                  monkeypatch):
    def unreadable(fields):
        list(fields)
        raise OSError("permission denied")

    monkeypatch.setattr(engine, "_first_unreviewed", unreadable)
    assert screen._on_first_unreviewed() is False
    assert warnings == [("Cannot read review status", "permission denied")]
    assert screen._current_index == 0


# magnifier: whole-image auto-accept and right-click removal


def _whole_image_result(magnifier, labels):
    labels = np.asarray(labels, dtype=np.int32)
    key = magnifier._image_key_now()
    request = mm._MagnifierRequest(
        key=key, crop=None, box=(0, 0, IMG_N, IMG_N), shape=(IMG_N, IMG_N),
        mode="otsu", sensitivity=0.0, bright=True, min_area=0,
        model_name="cpsam", diameter=0, colour=(255, 0, 0), scope="image")
    return mm._MagnifierResult(request, labels, "otsu", "", None,
                               mm._object_count(labels),
                               extents=mm._object_extents(labels))


@pytest.fixture
def accepting(screen):
    magnifier = screen._magnifier
    magnifier.enabled = True
    magnifier.scope = "image"
    magnifier.auto_accept = True
    magnifier._stroke = None
    magnifier._waiting.clear()
    labels = np.zeros((IMG_N, IMG_N), dtype=np.int32)
    labels[44:50, 10:18] = 5
    magnifier._image_result = _whole_image_result(magnifier, labels)
    emitted = []
    magnifier.commit_ready.disconnect()
    magnifier.commit_ready.connect(emitted.append)
    yield magnifier, emitted
    magnifier.auto_accept = False
    magnifier.enabled = False
    magnifier._image_result = None


def test_whole_image_auto_accept_takes_the_object_under_the_mouse(accepting):
    magnifier, emitted = accepting
    magnifier._cursor = (12, 46)
    assert magnifier._accept_proposed() is True
    assert len(emitted) == 1
    accepted = emitted[0]
    assert accepted.request.box == (10, 44, 18, 50)
    assert accepted.labels.shape == (6, 8)
    assert (accepted.labels == 5).all() and accepted.count == 1
    assert magnifier._accept_proposed() is False
    assert len(emitted) == 1


def test_whole_image_auto_accept_skips_background(accepting):
    magnifier, emitted = accepting
    magnifier._cursor = (30, 30)
    assert magnifier._accept_proposed() is False
    assert emitted == []


def test_whole_image_auto_accept_skips_a_mouse_past_the_labels(accepting):
    magnifier, emitted = accepting
    magnifier._cursor = (IMG_N + 3, 5)
    assert magnifier._accept_proposed() is False
    magnifier._cursor = (5, -2)
    assert magnifier._accept_proposed() is False
    assert emitted == []


def test_remove_deletes_the_object_under_the_lens(screen):
    magnifier = screen._magnifier
    assert magnifier.remove() is False
    magnifier.enabled = True
    magnifier._cursor = (6, 6)
    try:
        assert magnifier.remove() is True
    finally:
        magnifier.enabled = False
    assert not np.any(screen._canvas.mask == 3)
    assert {int(v) for v in np.unique(screen._canvas.mask)} == {0, 7, 9}
    assert screen._log.edits[-1].kind == "delete"


def test_remove_at_answers_no_for_a_click_off_the_image(screen):
    asked = []
    magnifier = screen._magnifier
    magnifier.remove_requested.connect(lambda x, y: asked.append((x, y)))
    magnifier.enabled = True
    try:
        assert magnifier.remove_at(None) is False
    finally:
        magnifier.enabled = False
    assert magnifier.remove_at((6, 6)) is False
    assert asked == []
    assert int(screen._canvas.mask[6, 6]) == 3


# Save on navigation


def test_save_on_navigation_with_no_field_lets_navigation_go(screen):
    screen._btn_save_on_navigation.setChecked(True)
    screen._image_files = []
    assert screen._save_mask_if_needed() is True
    screen._btn_save_on_navigation.setChecked(False)


def test_save_on_navigation_holds_navigation_while_a_field_loads(
        screen, folder):
    screen._btn_save_on_navigation.setChecked(True)
    screen._loading = True
    try:
        assert screen._save_mask_if_needed() is False
    finally:
        screen._loading = False
    screen._last_saved_mask = None
    assert screen._save_mask_if_needed() is False
    assert not os.path.exists(engine.mask_save_path(str(folder),
                                                    "img_00.tif"))
    screen._btn_save_on_navigation.setChecked(False)


# small guards


def test_prompt_buttons_sync_without_their_buttons(screen, monkeypatch):
    prompter = screen._prompter
    discard = screen._btn_prompt_discard
    monkeypatch.setattr(prompter, "_enabled", True)
    monkeypatch.setattr(prompter, "_gate", lambda: True)
    monkeypatch.setattr(prompter, "points", [(1, 1, True)])
    assert prompter.enabled
    screen._btn_prompt_accept = None
    screen._sync_prompt_buttons()
    assert discard.isEnabled()
    screen._btn_prompt_discard = None
    prompter.points = []
    screen._sync_prompt_buttons()
    assert discard.isEnabled()
    screen._btn_prompt_discard = discard
    screen._sync_prompt_buttons()
    assert not discard.isEnabled()


def test_uncertainty_buttons_switch_together_when_one_is_missing(screen):
    button = screen._btn_uncertainty
    screen._btn_uncertainty_setting = None
    screen._set_uncertainty_enabled(False)
    assert not button.isEnabled()
    screen._set_uncertainty_enabled(True)
    assert button.isEnabled()


def test_a_second_threaded_uncertainty_run_reuses_the_worker(screen):
    submitted = []
    screen._uncertainty_worker = SimpleNamespace(submit=submitted.append)
    request = {"kind": "field"}
    try:
        assert screen._start_uncertainty(request, threaded=True) is True
        assert submitted == [request]
        assert screen._uncertainty_request is request
        assert not screen._btn_uncertainty.isEnabled()
        assert screen._start_uncertainty({"kind": "field"}, True) is False
        assert submitted == [request]
    finally:
        screen._uncertainty_worker = None
        screen._uncertainty_request = None


def test_unticking_blind_when_not_blinded_changes_nothing(screen):
    order = list(screen._image_files)
    screen._on_blind_toggled(False)
    assert screen._blind is None
    assert screen._image_files == order


def test_unlocking_rois_that_were_never_locked_leaves_them(screen):
    button = screen._btn_rois
    button.setEnabled(False)
    screen._blind_lock_rois(False)
    assert not button.isEnabled()
    button.setEnabled(True)
    screen._blind_lock_rois(False)
    assert button.isEnabled()


def test_blinding_hides_a_field_at_the_root_of_the_disk(screen):
    root_file = os.path.join(os.path.sep, "field.tif")
    screen._blind = {"codes": {root_file: "F-07"}}
    try:
        assert screen._blind_text(f"saved {root_file}") == "saved F-07"
        assert screen._blind_text("field") == "F-07"
    finally:
        screen._blind = None


def test_unblinding_with_no_field_on_screen_and_no_folder(screen):
    names = list(screen._image_files)
    folder = screen._folder
    original = [os.path.abspath(os.path.join(folder, name))
                for name in reversed(names)]
    screen._image_files = list(names)
    screen._blind = {"original": original, "codes": {}}
    screen._console_section.setProperty("_spacr_blind_visible", True)
    screen._current_index = len(names)
    screen._src_label.setText("unchanged")
    screen._folder = ""
    screen._field_folders = [folder] * len(names)
    screen._queue = None
    try:
        screen._restore_blind_order()
    finally:
        screen._folder = folder
        screen._field_folders = None
    assert screen._blind is None
    assert screen._image_files == list(reversed(names))
    assert screen._current_index == len(names)
    assert screen._src_label.text() == "unchanged"
    assert screen._console_section.property("_spacr_blind_visible") is None


def test_unreadable_puncta_scales_become_nan(screen, monkeypatch):
    monkeypatch.setattr(screen._mag_mode, "currentData",
                        lambda *args: cpu_modes.PUNCTA)
    screen._puncta_widgets["puncta_sigmas"].setText("2, wide")
    params = screen._cpu_params()
    assert len(params.puncta_sigmas) == 1
    assert math.isnan(params.puncta_sigmas[0])
    screen._puncta_widgets["puncta_sigmas"].setText("1.5 3")
    assert screen._cpu_params().puncta_sigmas == (1.5, 3.0)


def test_a_stroke_on_an_empty_canvas_records_no_policy(screen):
    before = dict(screen._manual_id_history)
    screen._canvas.mask = None
    screen._on_stroke_started()
    assert screen._manual_id_history == before
    screen._canvas.mask = objects_mask()
    screen._on_stroke_started()
    key = screen._manual_mask_key(screen._canvas.mask)
    assert screen._manual_id_history[key] == screen._canvas.manual_ids


def test_a_folded_screen_that_fails_the_alpha_gate_is_logged(
        screen, alpha, caplog):
    refreshed = []

    def broken():
        refreshed.append(True)
        raise RuntimeError("not ready")

    screen._uncertainty_ensemble = None
    screen._fold_screens["broken"] = SimpleNamespace(
        _refresh_alpha_visibility=broken)
    try:
        with caplog.at_level(logging.DEBUG, logger="spacr.qt.make_masks"):
            screen._refresh_alpha_visibility()
    finally:
        screen._fold_screens.pop("broken", None)
    assert refreshed == [True]
    assert any("folded screen" in record.getMessage()
               for record in caplog.records)


def test_a_sort_without_a_first_channel_opens_nothing(screen, monkeypatch,
                                                      tmp_path):
    from spacr.qt import prefs

    posted, opened = [], []
    monkeypatch.setattr(screen._masks_console, "post",
                        lambda text, *args: posted.append(text))
    monkeypatch.setattr(screen, "_open_folder",
                        lambda path, *a, **k: opened.append(path) or True)
    dest = tmp_path / "sorted"
    dest.mkdir()
    worker = SimpleNamespace(error=None, result=SimpleNamespace(
        moved=4, stacks=["s"], merged=[], dest=str(dest),
        manifest=str(dest / "manifest.csv")))
    screen._on_channels_sorted(worker)
    assert opened == []
    assert len(posted) == 2 and "Moved 4 file(s)" in posted[0]
    assert prefs.get_recent_sources("measure")[0] == str(dest)
    (dest / "C01").mkdir()
    screen._on_channels_sorted(worker)
    assert opened == [str(dest / "C01")]


def test_dropped_masks_already_beside_their_images_are_not_copied(
        screen, monkeypatch, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    (first / "masks").mkdir(parents=True)
    second.mkdir()
    image_a, image_b = first / "one.tif", second / "two.tif"
    for image in (image_a, image_b):
        imageio.imwrite(image, field_image())
    mask_a = first / "masks" / "one.tif"
    imageio.imwrite(mask_a, objects_mask())
    asked, queued = [], []
    monkeypatch.setattr(screen, "_confirm",
                        lambda *args: asked.append(args) or True)
    monkeypatch.setattr(screen, "_open_queue",
                        lambda paths: queued.append(list(paths)) or True)
    found = SimpleNamespace(unpaired_masks=[], images=[str(image_a),
                                                       str(image_b)],
                            masks={str(image_a): str(mask_a)})
    assert screen._open_with_masks(found) is True
    assert asked == []
    assert queued == [[str(image_a), str(image_b)]]
    assert sorted(p.name for p in (first / "masks").iterdir()) == ["one.tif"]


def test_the_compare_window_zooms_both_views_and_nothing_once_emptied(qtbot):
    picture = np.arange(64.0).reshape(8, 8)
    window = mm._ComparePreview(picture, picture, "steps")
    qtbot.addWidget(window)
    first, second = window.views
    window.zoom(2.0)
    assert first._user_zoomed
    assert second.zoom_factor() == pytest.approx(first.zoom_factor())
    window.fit()
    assert not first._user_zoomed
    window.views = []
    window.zoom(2.0)
    window.fit()
    assert not first._user_zoomed
    window.views = [first, second]
    window.close()
