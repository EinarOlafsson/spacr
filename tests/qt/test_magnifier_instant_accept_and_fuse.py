"""Current magnifier proposals use normal edit, undo and durable save paths."""
import threading

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from tests.qt.test_the_live_magnifier_segments_under_the_mouse import (
    CodedStub, fields, hover, rect_mask, screen, switch_on, wait_for_result,
)


def choose_overlap(screen, rule):
    screen._mag_overlap.setCurrentIndex(screen._mag_overlap.findData(rule))
    assert screen._magnifier.overlap == rule


def test_merge_retains_whole_old_objects_and_joins_transitive_proposals():
    mask = rect_mask((12, 14), {8: (0, 1, 7, 5), 3: (9, 1, 14, 5),
                               21: (0, 9, 2, 12)})
    incoming = rect_mask((6, 10), {1: (0, 0, 4, 2), 2: (3, 0, 8, 2),
                                  4: (5, 4, 7, 6)}, dtype=np.int32)
    original = mask.copy()
    out, ids = engine._paste_region_objects(mask, incoming, (2, 2), overlap='merge')
    assert ids == [3, 22]
    assert set(np.unique(out)) == {0, 3, 21, 22}
    assert np.all(out[original == 8] == 3)
    assert np.all(out[original == 3] == 3)
    assert np.all(out[original == 21] == 21)
    assert np.all(out[2:4, 2:10] == 3)
    assert np.all(out[6:8, 7:9] == 22)
    np.testing.assert_array_equal(mask, original)


def test_small_proposals_do_not_merge_or_erase_old_objects():
    mask = rect_mask((8, 8), {7: (0, 0, 5, 5)})
    labels = np.ones((2, 2), np.int32)
    for overlap in ('merge', 'replace'):
        out, ids = engine._paste_region_objects(
            mask, labels, (3, 3), overlap=overlap, min_area=5,
            replace_whole=True)
        np.testing.assert_array_equal(out, mask)
        assert ids == []


def test_replacement_removes_the_old_object_outside_the_crop():
    mask = rect_mask((12, 12), {7: (0, 0, 6, 6), 11: (9, 9, 12, 12)})
    labels = np.ones((3, 3), np.int32)
    legacy, _ = engine._paste_region_objects(mask, labels, (4, 4), overlap='replace')
    assert legacy[0, 0] == 7
    out, ids = engine._paste_region_objects(
        mask, labels, (4, 4), overlap='replace', replace_whole=True)
    assert ids == [12]
    assert not np.any(out == 7)
    assert np.all(out[4:7, 4:7] == 12)
    assert np.all(out[9:12, 9:12] == 11)


def test_new_controls_are_nested_under_object_detection(screen):
    categories = dict(screen._settings_categories)
    card = categories['Magnification settings']
    assert categories['Object detection'].isAncestorOf(card)
    assert card.isAncestorOf(screen._mag_auto_accept)
    assert card.isAncestorOf(screen._mag_overlap)
    assert not screen._mag_auto_accept.isChecked()
    assert screen._mag_overlap.currentData() == 'clip'
    assert {screen._mag_overlap.itemData(i)
            for i in range(screen._mag_overlap.count())} == {'clip', 'merge', 'replace', 'skip'}


@pytest.mark.parametrize('mode', ['otsu', 'cellpose'])
@pytest.mark.parametrize('overlap', ['clip', 'merge', 'replace'])
def test_auto_accept_can_undo_and_survives_save_next_and_reopen(
        qtbot, screen, fields, mode, overlap):
    initial = rect_mask((64, 64), {7: (18, 18, 23, 25), 11: (52, 52, 56, 56)})
    screen._canvas.mask = initial.copy()
    screen._canvas.refresh()
    screen._history.push(initial)
    choose_overlap(screen, overlap)
    switch_on(screen, CodedStub({2: (20, 20, 26, 25)}))
    screen._magnifier.set_mode(mode)
    screen._mag_auto_accept.setChecked(True)
    hover(screen, 28, 28)
    qtbot.waitUntil(lambda: not np.array_equal(screen._canvas.mask, initial), timeout=10000)
    wait_for_result(qtbot, screen)
    saved = screen._canvas.mask.copy()
    assert saved[53, 53] == 11
    if overlap == 'merge':
        assert saved[20, 25] == 7 and saved[18, 18] == 7
    elif overlap == 'replace':
        assert saved[18, 18] == 0 and not np.any(saved == 7)
    else:
        assert saved[18, 18] == 7 and saved[20, 25] > 11
    screen._on_undo()
    screen._magnifier.refresh()
    qtbot.wait(150)
    np.testing.assert_array_equal(screen._canvas.mask, initial)
    screen._on_redo()
    np.testing.assert_array_equal(screen._canvas.mask, saved)
    screen._on_save()
    screen._on_next()
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    try:
        assert made._open_folder(str(fields))
        np.testing.assert_array_equal(made._canvas.mask, saved)
        np.testing.assert_array_equal(imageio.imread(fields / 'masks' / 'a.tif'), saved)
    finally:
        made._magnifier.close()
        made.close_folded()


def test_enabling_accepts_the_already_visible_proposal(qtbot, screen):
    switch_on(screen, CodedStub({1: (20, 20, 26, 25)}))
    hover(screen, 28, 28)
    wait_for_result(qtbot, screen)
    assert not screen._canvas.mask.any()
    screen._mag_auto_accept.setChecked(True)
    assert screen._canvas.mask[20, 20] > 0


def test_changed_settings_reject_the_old_proposal_while_auto_accept_stays_on(qtbot, screen):
    switch_on(screen, CodedStub({1: (20, 20, 26, 25)}))
    hover(screen, 28, 28)
    wait_for_result(qtbot, screen)
    previous = screen._magnifier._shown
    screen._mag_sensitivity.setValue(1)
    assert previous.request.key != screen._magnifier._requested_key
    screen._mag_auto_accept.setChecked(True)
    assert screen._magnifier.auto_accept
    assert not screen._canvas.mask.any()
    qtbot.waitUntil(lambda: screen._canvas.mask.any(), timeout=10000)
    assert screen._magnifier._shown.request.key != previous.request.key


def test_late_endpoint_preview_is_not_auto_accepted_after_an_explicit_edit(qtbot, screen):
    choose_overlap(screen, 'replace')
    switch_on(screen, CodedStub({1: (20, 20, 26, 25)}))
    hover(screen, 28, 28)
    wait_for_result(qtbot, screen)
    controller = screen._magnifier
    result = controller._shown
    controller.auto_accept = True
    controller._shown = None
    controller._accept_proposed(commit=False)
    controller._shown = result
    assert not controller._accept_proposed()
    assert not screen._canvas.mask.any()
    hover(screen, 30, 30)
    qtbot.waitUntil(lambda: screen._canvas.mask.any(), timeout=10000)


def test_touching_selection_keeps_region_coordinates(qtbot, screen):
    screen._mag_save.setCurrentIndex(screen._mag_save.findData('touching'))
    switch_on(screen, CodedStub({1: (20, 20, 26, 25), 2: (31, 31, 35, 35)}))
    screen._mag_auto_accept.setChecked(True)
    hover(screen, 23, 23)
    qtbot.waitUntil(lambda: screen._canvas.mask[20, 20] > 0, timeout=10000)
    assert screen._canvas.mask[32, 32] == 0
    assert screen._canvas.mask[8, 8] == 0


@pytest.mark.parametrize('cancel', ['off', 'disabled', 'field', 'settings'])
def test_late_proposal_cannot_auto_accept_after_context_changes(qtbot, screen, cancel):
    stub = CodedStub({1: (20, 20, 26, 25)})
    stub.gate = threading.Event()
    switch_on(screen, stub)
    screen._mag_auto_accept.setChecked(True)
    hover(screen, 28, 28)
    qtbot.waitUntil(lambda: bool(stub.calls), timeout=10000)
    if cancel == 'off':
        screen._magnifier.hover(None)
    elif cancel == 'disabled':
        screen._btn_magnifier.setChecked(False)
    elif cancel == 'field':
        screen._on_next()
    else:
        screen._mag_sensitivity.setValue(1)
        screen._mag_auto_accept.setChecked(False)
    stub.gate.set()
    qtbot.wait(250)
    assert not screen._canvas.mask.any()
