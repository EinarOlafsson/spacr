"""Local Easy mode input, native layout, drop and explicit module handoff."""
import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _image(path):
    pixels = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    Image.fromarray(pixels).save(path)
    return pixels


def _bytes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_local_headers_identify_real_mixed_contents_without_decoding_or_writing(
        tmp_path, private_ai_profile, monkeypatch):
    from spacr.qt.widgets.home import _easy_inspect_path

    preferences, _, _ = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    image = tmp_path / 'field.tif'
    _image(image)
    table = tmp_path / 'objects.csv'
    table.write_text('label,area,intensity\n1,12,4\n2,9,-2\n')
    database = tmp_path / 'measurements.db'
    with sqlite3.connect(database) as connection:
        connection.execute('create table cells(label integer, area real)')
        connection.execute('insert into cells values(7, 12)')
    before = _bytes([image, table, database])
    entries = set(tmp_path.iterdir())

    def no_pixel_decode(*args, **kwargs):
        raise AssertionError('header inspection must not decode pixels')

    from PIL import TiffImagePlugin

    monkeypatch.setattr(TiffImagePlugin.TiffImageFile, 'load', no_pixel_decode)
    result = _easy_inspect_path(str(tmp_path))
    actions = {action[0]: action for action in result['actions']}
    assert set(actions) == {'make_masks', 'mask', 'db_browser', 'foreign'}
    assert actions['make_masks'][3] == str(tmp_path)
    assert actions['foreign'][3] == str(table)
    assert actions['db_browser'][3] == str(database)
    assert 'several content types' in result['message']
    assert 'nothing has been processed or sent to AI' in result['message']
    assert _easy_inspect_path(str(image))['actions'][0][3] == str(image)
    assert _bytes([image, table, database]) == before
    assert set(tmp_path.iterdir()) == entries
    renamed = tmp_path / 'image-without-format.bin'
    renamed.write_bytes(image.read_bytes())
    mismatch = _easy_inspect_path(str(renamed))
    assert mismatch['actions'] == []
    assert 'filename extension' in mismatch['message']
    assert renamed.read_bytes() == image.read_bytes()
    assert preferences.get_preferred_provider() == 'none'


def test_invalid_and_calibrated_collection_inputs_have_honest_local_guidance(
        tmp_path, private_ai_profile):
    from spacr.qt.widgets.home import _easy_inspect_path

    assert _easy_inspect_path(str(tmp_path / 'missing'))['actions'] == []
    unsupported = tmp_path / 'notes.txt'
    unsupported.write_text('not an image or a table')
    assert 'No supported' in _easy_inspect_path(str(unsupported))['message']
    merged = tmp_path / 'merged'
    merged.mkdir()
    np.save(merged / 'stack_1.npy', np.zeros((3, 4, 2), dtype=np.float32))
    ordinary = _easy_inspect_path(str(tmp_path))
    assert [a[0] for a in ordinary['actions']] == ['measure']
    marker = tmp_path / 'plate_batch.json'
    marker.write_text('{incomplete')
    result = _easy_inspect_path(str(tmp_path))
    assert not result['actions']
    assert 'ordinary Measure is not suggested' in result['message']
    assert marker.read_text() == '{incomplete'


@pytest.mark.parametrize('theme', ['light', 'dark'])
@pytest.mark.parametrize('size', [(640, 650), (1280, 850)])
def test_real_typed_inspection_keeps_cards_readable_disjoint_and_actionable(
        qtbot, qapp, private_ai_profile, tmp_path, theme, size):
    from spacr.qt import theme as styling
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_easy_mode(True)
    styling.apply_qpalette(qapp, theme)
    image = tmp_path / 'field.tif'
    _image(image)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(*size)
    page.show()
    selected = []
    page.module_requested.connect(lambda key, path: selected.append((key, path)))
    try:
        page.path_edit.setFocus()
        qtbot.keyClicks(page.path_edit, str(image))
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: len(page.field.cards) == 2, timeout=5000)
        qtbot.waitUntil(lambda: all(not card._growing for card in page.field.cards), timeout=7000)
        assert page.path_edit.text() == str(image)
        assert page.path_edit.placeholderText() == 'Type a path or Drop some images here ...'
        assert page.path_edit.rect().contains(page.path_edit.browse.geometry())
        assert page.path_edit.isVisible()
        panel = page.field.panel.geometry()
        rectangles = [card.geometry() for card in page.field.cards]
        assert all(page.field.rect().contains(rect) for rect in rectangles)
        assert all(not rect.intersects(panel) for rect in rectangles)
        assert not rectangles[0].intersects(rectangles[1])
        for card in page.field.cards:
            assert card.button.text() and card.button.accessibleName()
            assert card.width() >= 220 and card.height() >= card.sizeHint().height()
        card = next(card for card in page.field.cards if card._key == 'make_masks')
        page.layout().itemAt(0).widget().ensureWidgetVisible(card)
        qtbot.mouseClick(card.button, Qt.LeftButton)
        assert selected == [('make_masks', str(image))]
        assert preferences.get_preferred_provider() == 'none'
        page.hide()
        assert not page.field._animations
    finally:
        page.close()
        assert not page._runner.active_jobs()


@pytest.mark.parametrize('reject_pending', [False, True])
def test_latest_typed_path_wins_after_real_worker_and_gui_stays_responsive(
        qtbot, private_ai_profile, tmp_path, monkeypatch, reject_pending):
    from spacr.qt.widgets import home

    preferences, _, _ = private_ai_profile
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_features(True)
    first = tmp_path / 'first.tif'
    second = tmp_path / 'second.csv'
    _image(first)
    second.write_text('label,area\n1,12\n2,4\n')
    entered, release = threading.Event(), threading.Event()
    original = home._easy_inspect_path

    def inspect(value):
        if value == str(first):
            entered.set()
            assert release.wait(5)
        return original(value)

    monkeypatch.setattr(home, '_easy_inspect_path', inspect)
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page.show()
    try:
        page.accept_paths([str(first)])
        qtbot.waitUntil(entered.is_set, timeout=3000)
        page.path_edit.setText(str(second))
        page._schedule_inspection(str(second))
        assert page.path_edit.isEnabled()
        if reject_pending:
            page.accept_paths([str(first), str(second)])
            assert 'one local file or folder' in page.status.text()
        release.set()
        if reject_pending:
            qtbot.waitUntil(lambda: not page._runner.is_busy(), timeout=3000)
            assert not page.field.cards
            assert 'one local file or folder' in page.status.text()
            qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: not page._runner.is_busy() and page._requested == str(second), timeout=5000)
        assert len(page.field.cards) == 3
        assert {card._key for card in page.field.cards} == {'foreign', 'make_masks', 'mask'}
        assert page.field.cards[0]._key == 'foreign'
        assert page.field.cards[0]._path == str(second)
        assert all(card._path == str(tmp_path) for card in page.field.cards[1:])
        assert all(card._path != str(first) for card in page.field.cards)
        assert 'first.tif' not in page.status.text()
        page.accept_paths([str(first), str(second)])
        assert 'one local file or folder' in page.status.text()
        assert not page.field.cards
    finally:
        release.set()
        page.close()
        assert not page._runner.active_jobs()


def test_actual_window_drop_handoff_and_easy_dock_preserve_ordinary_preferences(
        qtbot, private_ai_profile, tmp_path):
    from spacr.qt.app import MainWindow, make_home_page
    from spacr.qt.widgets.home import HomePage, _EasyHomePage
    from spacr.qt.walkthrough import mark_seen

    preferences, _, _ = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_easy_mode(False)
    ordinary = make_home_page()
    qtbot.addWidget(ordinary)
    assert isinstance(ordinary, HomePage)
    ordinary.close()
    preferences.set_dock_mode('hidden')
    preferences.set_easy_mode(True)
    image = tmp_path / 'field.tif'
    pixels = _image(image)
    before = image.read_bytes()
    mark_seen('make_masks')
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1100, 800)
    window.show()
    page = window._startup
    assert isinstance(page, _EasyHomePage)
    try:
        assert window._dock_mode == 'hidden'
        preferences.set_easy_mode_dock(True)
        window.refresh_after_preferences({"easy_mode_dock"})
        assert window._dock_mode == 'locked'
        assert preferences.get_dock_mode() == 'hidden'
        preferences.set_easy_mode_dock(False)
        window.refresh_after_preferences({"easy_mode_dock"})
        assert window._dock_mode == 'hidden'
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(image))])
        target = window.centralWidget()
        enter = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(target, enter)
        assert enter.isAccepted()
        drop = QDropEvent(QPointF(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(target, drop)
        assert drop.isAccepted()
        qtbot.waitUntil(lambda: len(page.field.cards) == 2, timeout=5000)
        assert page.path_edit.text() == str(image)
        card = next(card for card in page.field.cards if card._key == 'make_masks')
        qtbot.waitUntil(lambda: not card._growing and card.button.isEnabled(), timeout=7000)
        card.button.click()
        qtbot.waitUntil(lambda: 'make_masks' in window._screens, timeout=10000)
        screen = window._screens['make_masks']
        qtbot.waitUntil(lambda: bool(screen._image_files), timeout=5000)
        assert (Path(screen._folder) / screen._image_files[0]).resolve() == image
        assert window._stack.currentWidget() is screen
        np.testing.assert_array_equal(np.asarray(Image.open(image)), pixels)
        assert image.read_bytes() == before
        preferences.set_easy_mode(False)
        window.apply_dock_mode()
        window._rebuild_startup_page()
        assert isinstance(window._startup, HomePage)
        assert preferences.get_dock_mode() == window._dock_mode == 'hidden'
    finally:
        from shiboken6 import isValid

        if isValid(page):
            page.close()
        window.close()


def test_reduce_motion_and_reinspection_retire_owned_animation_objects(
        qtbot, qapp, private_ai_profile, tmp_path):
    from PySide6.QtCore import QEvent, QPropertyAnimation

    from spacr.qt.widgets.home import _EasyHomePage, _easy_inspect_path

    preferences, _, _ = private_ai_profile
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(True)
    image = tmp_path / 'field.tif'
    _image(image)
    actions = _easy_inspect_path(str(image))['actions']
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.show()
    try:
        page.field.set_cards(actions)
        assert not page.field.findChildren(QPropertyAnimation)
        assert all(not card._growing and card._content_opacity.opacity() == 1.0
                   for card in page.field.cards)
        preferences.set_reduce_motion(False)
        preferences._hold_the_backdrops_still(qapp)
        for _ in range(3):
            page.field.set_cards(actions)
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            assert len(page.field.findChildren(QPropertyAnimation)) == len(actions)
        first = page.field.cards[0]
        assert first.geometry().width() == 1
        qtbot.waitUntil(lambda: 10 < first.width() < first._final_rect.width(), timeout=3000)
        assert first._growing
        assert not first.button.isEnabled()
        assert first._content_opacity.opacity() < 1.0
        preferences.set_reduce_motion(True)
        preferences._hold_the_backdrops_still(qapp)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not page.field._animations
        assert not page.field.findChildren(QPropertyAnimation)
        assert all(not card._growing and card.geometry() == card._final_rect
                   and card.button.isEnabled() and card._content_opacity.opacity() == 1.0
                   for card in page.field.cards)
        preferences.set_reduce_motion(False)
        preferences.set_ambient_enabled(False)
        page.field.set_cards(actions)
        assert not page.field._animations
        assert all(card.geometry() == card._final_rect and card._content_opacity.opacity() == 1.0
                   for card in page.field.cards)
        page.hide()
        assert not page.field._animations
    finally:
        page.close()


def test_growing_card_footprints_part_existing_dots_and_release_without_lattice_mutation(qapp):
    from spacr.qt.widgets import ambient

    engine = ambient.make_engine('data_art_impulse_lens', 'spacr', '#101418', seed=42)
    transport = ambient._QueuedArtInput()
    x = np.array([0.45, 0.50, 0.55, 0.85], dtype=np.float32)
    y = np.array([0.50, 0.50, 0.50, 0.85], dtype=np.float32)
    original_x, original_y = x.copy(), y.copy()
    transport._growing_regions = ((0.48, 0.48, 0.52, 0.52),)
    transport._consume(engine)
    small_x, small_y = engine._part_growing_regions(x, y, 1000, 800)
    transport._growing_regions = ((0.35, 0.35, 0.65, 0.65),)
    transport._consume(engine)
    large_x, large_y = engine._part_growing_regions(x, y, 1000, 800)
    assert large_x[0] < small_x[0] and large_x[2] > small_x[2]
    assert np.all((large_x[:3] < 0.35) | (large_x[:3] > 0.65)
                  | (large_y[:3] < 0.35) | (large_y[:3] > 0.65))
    assert (large_x[-1], large_y[-1]) == (x[-1], y[-1])
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    transport._growing_regions = ()
    transport._consume(engine)
    restored_x, restored_y = engine._part_growing_regions(x, y, 1000, 800)
    np.testing.assert_array_equal(restored_x, x)
    np.testing.assert_array_equal(restored_y, y)
    transport._growing_regions = ((0.35, 0.35, 0.65, 0.65),)
    transport._consume(engine, discard_clicks=True)
    assert engine._growing_regions == ()
