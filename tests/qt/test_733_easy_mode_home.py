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
from PySide6.QtWidgets import QApplication, QPushButton

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _image(path):
    pixels = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    Image.fromarray(pixels).save(path)
    return pixels


def _bytes(paths):
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


@pytest.fixture
def isolated_input_root(tmp_path):
    root = tmp_path / 'input'
    root.mkdir()
    return root


def test_master_context_stops_at_the_intentional_parent_but_keeps_its_siblings(
        tmp_path, private_ai_profile, monkeypatch):
    from PIL import TiffImagePlugin
    from spacr.qt.widgets.home import _easy_inspect_path

    master = tmp_path / 'project'
    source = master / 'images'
    source.mkdir(parents=True)
    image = source / 'field.tif'
    _image(image)
    outside = tmp_path / 'another_project'
    outside.mkdir()
    outside_container = outside / 'acquisition.czi'
    outside_container.write_bytes(b'ZISRAWFILE' + bytes(54))
    originals = _bytes([image, outside_container])

    def no_pixels(*args, **kwargs):
        raise AssertionError('context inspection must not decode pixels')

    monkeypatch.setattr(TiffImagePlugin.TiffImageFile, 'load', no_pixels)
    first = _easy_inspect_path(str(source))
    assert [action[0] for action in first['actions']] == ['make_masks', 'mask']
    assert first['context']['reader_candidates'] == []
    assert str(outside) not in first['context']['image_folders']
    neighbor = master / 'acquisition'
    neighbor.mkdir()
    inside_container = neighbor / 'acquisition.czi'
    inside_container.write_bytes(outside_container.read_bytes())
    originals.update(_bytes([inside_container]))
    second = _easy_inspect_path(str(source))
    assert [action[0] for action in second['actions']] == ['make_masks', 'mask', 'convert']
    assert second['actions'][2][3] == str(neighbor)
    assert [row['path'] for row in second['context']['reader_candidates']] == [str(inside_container)]
    assert second['context']['reader_candidates'][0]['metadata_inspected'] is False
    assert _bytes([image, outside_container, inside_container]) == originals


def test_local_headers_identify_real_mixed_contents_without_decoding_or_writing(
        isolated_input_root, private_ai_profile, monkeypatch):
    from spacr.qt.widgets.home import _easy_inspect_path

    tmp_path = isolated_input_root
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
        isolated_input_root, private_ai_profile):
    from spacr.qt.widgets.home import _easy_inspect_path

    tmp_path = isolated_input_root
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
        qtbot, qapp, private_ai_profile, isolated_input_root, theme, size):
    from spacr.qt import theme as styling
    from spacr.qt.widgets.home import _EasyHomePage

    tmp_path = isolated_input_root
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
        assert not page.status.isVisible()
        assert not page.path_edit.hasFrame()
        assert page.field.rim.isVisible()
        panel = page.field.panel.geometry()
        rectangles = [card.geometry() for card in page.field.cards]
        assert all(page.field.rect().contains(rect) for rect in rectangles)
        assert all(not rect.intersects(panel) for rect in rectangles)
        assert not rectangles[0].intersects(rectangles[1])
        for card in page.field.cards:
            assert card.accessibleName()
            assert not card.findChildren(QPushButton)
            assert card.width() >= 220 and card.height() >= card.sizeHint().height()
        card = next(card for card in page.field.cards if card._key == 'make_masks')
        page.layout().itemAt(0).widget().ensureWidgetVisible(card)
        qtbot.mouseClick(card, Qt.LeftButton)
        assert selected == []
        assert page._workflow is not None and not page.path_edit.isVisible()
        assert page._workflow.central._key == 'make_masks'
        qtbot.mouseClick(page._workflow.central, Qt.LeftButton)
        assert selected == [('make_masks', str(image))]
        qtbot.mouseClick(page._workflow.back, Qt.LeftButton)
        assert page._workflow is None and page.path_edit.isVisible()
        assert page.path_edit.text() == str(image)
        assert preferences.get_preferred_provider() == 'none'
        if theme == 'light' and size == (1280, 850):
            from spacr.qt.widgets.home import _easy_inspect_path

            table = tmp_path / 'objects.csv'
            table.write_text('label,area\n1,12\n')
            with sqlite3.connect(tmp_path / 'measurements.db') as connection:
                connection.execute('create table cells(label integer)')
            actions = _easy_inspect_path(str(tmp_path))['actions']
            assert len(actions) == 4
            preferences.set_reduce_motion(True)
            page.field.set_cards(actions)
            original = [card.geometry() for card in page.field.cards]
            for resized in (size, (640, 650), (1100, 780), size):
                page.resize(*resized)
                qapp.processEvents()
                assert [card._key for card in page.field.cards] == [action[0] for action in actions]
                rectangles = [card.geometry() for card in page.field.cards]
                assert all(page.field.rect().contains(rect) for rect in rectangles)
                assert all(rect.top() >= page.path_edit.geometry().bottom() + 24
                           for rect in rectangles)
                assert rectangles == sorted(rectangles, key=lambda rect: (rect.y(), rect.x()))
                assert all(not first.intersects(second)
                           for index, first in enumerate(rectangles)
                           for second in rectangles[index + 1:])
            assert [card.geometry() for card in page.field.cards] == original
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
            assert [card._key for card in page.field.cards] == ['warning']
            assert 'one local file or folder' in page.field.cards[0].description.text()
            assert not page.status.isVisible()
            assert 'one local file or folder' in page.status.text()
            qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: not page._runner.is_busy() and page._requested == str(second), timeout=5000)
        assert len(page.field.cards) == 3
        assert [card._key for card in page.field.cards] == ['foreign', 'regex', 'make_masks']
        assert page.field.cards[0]._key == 'foreign'
        assert page.field.cards[0]._path == str(second)
        assert page.field.cards[1]._proposal['example_path'] == str(first)
        assert page.field.cards[2]._path == str(tmp_path)
        assert page._last_result['path'] == str(second)
        assert all(card._path != str(first) for card in page.field.cards if card._key != 'regex')
        assert 'first.tif' not in page.status.text()
        page.accept_paths([str(first), str(second)])
        assert 'one local file or folder' in page.status.text()
        assert [card._key for card in page.field.cards] == ['warning']
        assert 'one local file or folder' in page.field.cards[0].description.text()
    finally:
        release.set()
        page.close()
        assert not page._runner.active_jobs()


def test_actual_window_drop_handoff_and_easy_dock_preserve_ordinary_preferences(
        qtbot, qapp, request, private_ai_profile, tmp_path, monkeypatch):
    from spacr.qt.app import MainWindow, make_home_page
    from spacr.qt.widgets.home import HomePage, _EasyHomePage
    from spacr.qt.walkthrough import mark_seen

    from spacr.qt import theme
    from spacr.qt.widgets.ambient import _preferred_background

    from functools import partial

    old_palette = qapp.palette()
    request.addfinalizer(partial(qapp.setPalette, old_palette))
    preferences, _, _ = private_ai_profile
    preferences.set_theme('light')
    theme.apply_qpalette(qapp, 'light')
    normal_background = _preferred_background()
    palette_choice = preferences.get_ambient_palette()
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
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        backdrop = window.window_backdrop()
        assert backdrop.background_color().getRgb() == (25, 25, 25, 255)
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QColor, QImage, QPainter

        base = QImage(12, 12, QImage.Format_ARGB32)
        base.fill(QColor('magenta'))
        painter = QPainter(base)
        try:
            backdrop._paint_base(painter, QRect(0, 0, 12, 12))
        finally:
            painter.end()
        assert all(base.pixelColor(x, y).getRgb() == (25, 25, 25, 255)
                   for x in range(12) for y in range(12))
        assert preferences.get_theme() == 'light'
        assert preferences.get_ambient_palette() == palette_choice
        old_page = page
        window.refresh_theme()
        page = window._startup
        assert page is not old_page and isinstance(page, _EasyHomePage)
        assert window._stack.currentWidget() is page
        assert window.window_backdrop().background_color().getRgb() == (25, 25, 25, 255)
        window._retire_the_dock_backdrop()
        assert window.window_backdrop() is None
        window._backdrop_the_dock_column()
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        assert window.window_backdrop() is not backdrop
        assert window.window_backdrop().background_color().getRgb() == (25, 25, 25, 255)
        preferences.set_easy_mode_dock(True)
        window.refresh_after_preferences({"easy_mode_dock"})
        assert window._dock_mode == 'locked'
        assert preferences.get_dock_mode() == 'hidden'
        preferences.set_easy_mode_dock(False)
        window.refresh_after_preferences({"easy_mode_dock"})
        assert window._dock_mode == 'hidden'
        from PySide6.QtWidgets import QDialog
        from spacr.qt.widgets import ambient

        waves = []
        original_publish = ambient.AmbientWidget._publish_boundary_ripples

        def publish(backdrop, segments, strength, **kwargs):
            waves.append((segments, strength))
            return original_publish(backdrop, segments, strength, **kwargs)

        monkeypatch.setattr(ambient.AmbientWidget, '_publish_boundary_ripples', publish)
        assert window.property('spacrEasySearchRippleOnly')
        popup = QDialog(window)
        popup.setModal(True)
        popup.show()
        qtbot.waitUntil(lambda: QApplication.activeModalWidget() is popup)
        assert window.window_backdrop()._popup_wave_origin_for_tick() is None
        ambient._field_ripple_for_popup(popup)
        assert waves == []
        popup.close()
        popup.deleteLater()
        assert preferences.get_ambient_enabled() and preferences._field_ripples_enabled()
        assert window.window_backdrop().theme() in ambient._FIELD_THEMES
        qtbot.waitUntil(lambda: window.window_backdrop()._should_run(), timeout=5000)
        qtbot.mouseMove(page.path_edit, QPoint(12, 12))
        page.path_edit.setFocus()
        qapp.processEvents()
        assert waves == []
        qtbot.mouseClick(page.path_edit, Qt.LeftButton)
        assert len(waves) == 1 and waves[0][0] and waves[0][1] == 0.12
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(image))])
        target = window.centralWidget()
        enter = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(target, enter)
        assert enter.isAccepted()
        drop = QDropEvent(QPointF(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(target, drop)
        assert drop.isAccepted()
        assert window._stack.currentWidget() is page
        assert 'home' not in window._screens
        qtbot.waitUntil(lambda: len(page.field.cards) == 2, timeout=5000)
        assert page.path_edit.text() == str(image)
        assert window._stack.currentWidget() is page
        assert 'home' not in window._screens
        card = next(card for card in page.field.cards if card._key == 'make_masks')
        qtbot.waitUntil(lambda: not card._growing and card.isEnabled(), timeout=7000)
        assert len(waves) == 1
        card.click()
        assert window._stack.currentWidget() is page
        assert page._workflow.central._key == 'make_masks'
        page._workflow.central.click()
        qtbot.waitUntil(lambda: 'make_masks' in window._screens, timeout=10000)
        screen = window._screens['make_masks']
        qtbot.waitUntil(lambda: bool(screen._image_files), timeout=5000)
        assert (Path(screen._folder) / screen._image_files[0]).resolve() == image
        assert window._stack.currentWidget() is screen
        assert not window.property('spacrEasySearchRippleOnly')
        before_normal = len(waves)
        ambient.field_ripple_for_widget(window)
        assert len(waves) == before_normal + 1
        assert window.window_backdrop().background_color() == normal_background
        assert not window.window_backdrop()._background_explicit
        window._stack.setCurrentWidget(page)
        assert window.window_backdrop().background_color().getRgb() == (25, 25, 25, 255)
        window._stack.setCurrentWidget(screen)
        assert window.window_backdrop().background_color() == normal_background
        assert preferences.get_theme() == 'light'
        assert preferences.get_ambient_palette() == palette_choice
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
        assert not first.isEnabled()
        assert first._content_opacity.opacity() < 1.0
        preferences.set_reduce_motion(True)
        preferences._hold_the_backdrops_still(qapp)
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert not page.field._animations
        assert not page.field.findChildren(QPropertyAnimation)
        assert all(not card._growing and card.geometry() == card._final_rect
                   and card.isEnabled() and card._content_opacity.opacity() == 1.0
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


@pytest.mark.parametrize('valid', [False, True])
def test_editable_regex_card_colors_captures_and_requires_explicit_validation(
        qtbot, qapp, private_ai_profile, tmp_path, valid):
    from spacr.import_plan import for_get_regex
    from spacr.qt.widgets.home import _EasyHomePage, _EasyRegexCard

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    images = [tmp_path / name for name in ('scan_101_C1.tif', 'scan_101_C2.tif')]
    for image in images:
        _image(image)
    original = _bytes(images)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 850)
    page.show()
    try:
        page.accept_paths([str(images[0])])
        qtbot.waitUntil(lambda: bool(page.field.cards)
                       and isinstance(page.field.cards[0], _EasyRegexCard), timeout=5000)
        card = page.field.cards[0]
        assert card.pattern_edit.toPlainText()
        for capture in card._proposal.get('groups', ()):
            assert 'color:' + card._highlight.colors[capture['name']] in card.description.text()
        assert not any(item._key == 'mask' for item in page.field.cards)
        assert not page._settings_overrides
        pattern = (r'scan_(?P<fieldID>[0-9]+)_C(?P<chanID>[12])\.tif$'
                   if valid else r'(?P<fieldID>missing)')
        card.pattern_edit.setFocus()
        qtbot.keyClick(card.pattern_edit, Qt.Key_A, Qt.ControlModifier)
        qtbot.keyClicks(card.pattern_edit, pattern)
        assert card.pattern_edit.toPlainText() == pattern
        qapp.processEvents()
        if valid:
            formats = card.pattern_edit.document().firstBlock().layout().formats()
            colors = {item.format.foreground().color().name() for item in formats}
            assert set(card._highlight.colors.values()) <= colors
            assert len(colors) >= 2
            assert '<span' not in card.description.text()
        qtbot.mouseClick(card, Qt.LeftButton, pos=QPoint(card.width() // 2, card.height() - 6))
        qtbot.waitUntil(lambda: not page._runner.is_busy(), timeout=5000)
        if valid:
            confirmed = page.field.cards[0]
            assert isinstance(confirmed, _EasyRegexCard)
            assert not confirmed._proposal['required']
            assert confirmed.pattern_edit.toPlainText() == pattern
            assert all('color:' + color in confirmed.description.text()
                       for color in confirmed._highlight.colors.values())
            assert '>101</span>' in confirmed.description.text()
            assert any(item._key == 'mask' for item in page.field.cards)
            assert page._settings_overrides == {'mask': {
                'metadata_type': 'custom', 'custom_regex': for_get_regex(pattern)}}
        else:
            assert isinstance(page.field.cards[0], _EasyRegexCard)
            assert page.field.cards[0].error.text()
            assert not any(item._key == 'mask' for item in page.field.cards)
            assert not page._settings_overrides
        assert _bytes(images) == original
    finally:
        page.close()


def test_confirmed_regex_card_handoff_seeds_native_mask_without_reconfiguration(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    from spacr.import_plan import for_get_regex
    from spacr.regex_infer import _metadata_parse_report
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets.home import _EasyRegexCard

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    images = [tmp_path / name for name in ('scan_101_C1.tif', 'scan_101_C2.tif')]
    for image in images:
        _image(image)
    original = _bytes(images)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1280, 850)
    window.show()
    page = window._startup
    try:
        page.accept_paths([str(images[0])])
        qtbot.waitUntil(lambda: bool(page.field.cards)
                       and isinstance(page.field.cards[0], _EasyRegexCard), timeout=5000)
        card = page.field.cards[0]
        pattern = r'scan_(?P<fieldID>[0-9]+)_C(?P<chanID>[12])\.tif$'
        card.pattern_edit.setPlainText(pattern)
        card.click()
        qtbot.waitUntil(lambda: not page._runner.is_busy()
                       and any(item._key == 'mask' for item in page.field.cards), timeout=5000)
        assert window._stack.currentWidget() is page
        assert 'home' not in window._screens and 'mask' not in window._screens
        card = next(item for item in page.field.cards if item._key == 'mask')
        qtbot.mouseClick(card, Qt.LeftButton, pos=QPoint(18, 18))
        assert window._stack.currentWidget() is page
        assert 'mask' not in window._screens
        page._workflow.central.click()
        qtbot.waitUntil(lambda: 'mask' in window._screens
                       and window._stack.currentWidget() is window._screens['mask'], timeout=10000)
        values = window._screens['mask']._settings_model.collect()
        assert values['metadata_type'] == 'custom'
        assert values['custom_regex'] == for_get_regex(pattern)
        assert 'img_format' not in values
        assert _metadata_parse_report([image.name for image in images], values['metadata_type'],
                                      images[0].suffix.lstrip('.'), values['custom_regex'])[:2] == (2, 2)
        assert values['src'] == str(tmp_path)
        assert _bytes(images) == original
        from spacr.qt.dnd_handlers import MaskDropHandler, scan_is_busy
        from spacr.qt import multi_format

        container = tmp_path / 'acquisition.npz'
        np.savez_compressed(container, pixels=np.arange(96, dtype=np.uint16).reshape(8, 12))
        container_bytes = container.read_bytes()
        source_entries = set(tmp_path.iterdir())
        delegated = []
        loads = []
        native_apply = MaskDropHandler.apply
        native_load = np.load
        native_describe = multi_format.describe_file

        def load_container(path, *args, **kwargs):
            if str(path) == str(container):
                assert delegated, 'Easy inspection decoded the archive before an explicit choice'
                loads.append(path)
            return native_load(path, *args, **kwargs)

        def describe_container(path):
            if str(path) == str(container):
                assert delegated, 'Easy inspection invoked the native container reader'
            return native_describe(path)

        def apply_container(handler, path, screen):
            delegated.append((path, screen))
            return native_apply(handler, path, screen)

        monkeypatch.setattr(MaskDropHandler, 'apply', apply_container)
        monkeypatch.setattr(np, 'load', load_container)
        monkeypatch.setattr(multi_format, 'describe_file', describe_container)
        window._easy_mode_drop([str(container)])
        qtbot.waitUntil(lambda: not page._runner.is_busy()
                       and page._last_result['path'] == str(container), timeout=5000)
        assert window._stack.currentWidget() is page
        assert not delegated and not loads
        assert page._last_result['context']['native_metadata_inspected'] is False
        card = next(item for item in page.field.cards if item._key == 'mask')
        assert card._path == str(container)
        card.click()
        assert not delegated
        assert window._stack.currentWidget() is page
        page._workflow.central.click()
        qtbot.waitUntil(lambda: bool(delegated), timeout=5000)
        target = window._screens['mask']
        assert delegated == [(container, target)]
        qtbot.waitUntil(lambda: not scan_is_busy(target)
                       and target._settings_model.collect()['metadata_type'] == 'auto', timeout=5000)
        assert target._settings_model.collect()['src'] == str(tmp_path)
        assert window._stack.currentWidget() is target
        assert loads and set(tmp_path.iterdir()) == source_entries
        assert container.read_bytes() == container_bytes
        from spacr.qt.widgets import qc_field_browser

        opened = []
        open_overview = qc_field_browser._open_overview_module

        def overview(owner, settings):
            opened.append((owner, dict(settings)))
            return open_overview(owner, settings)

        monkeypatch.setattr(qc_field_browser, '_open_overview_module', overview)
        window._on_nav_selected('__home__')
        workflow = page._workflow
        neighbor = workflow.view.nodes['overview']
        workflow.view.ensureVisible(neighbor)
        qtbot.mouseClick(workflow.view.viewport(), Qt.LeftButton,
                         pos=workflow.view.mapFromScene(neighbor.sceneBoundingRect().center()))
        assert workflow.central._key == 'overview' and opened == []
        assert window._stack.currentWidget() is page
        workflow.central.click()
        qtbot.waitUntil(lambda: bool(opened) and getattr(target, '_overview_tab', None) is not None, timeout=10000)
        assert opened[0][0] is target and opened[0][1]['src'] == str(tmp_path)
        assert 'overview' not in window._screens and 'home' not in window._screens
        assert _bytes(images) == original and container.read_bytes() == container_bytes
    finally:
        page.close()
        window.close()


def test_workflow_selection_hover_and_escape_preserve_the_verified_input(
        qtbot, private_ai_profile, tmp_path):
    from spacr.qt.app import APPS, app_is_visible, folded_children
    from spacr.qt.widgets.fold_strip import fold_label
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    images = [tmp_path / name for name in ('W0025F0001T0001Z000C1.tif', 'W0025F0001T0001Z000C2.tif')]
    for image in images:
        _image(image)
    before = _bytes(images)
    chosen = []
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 850)
    page.show()
    page.module_requested.connect(lambda key, path: chosen.append((key, path)))
    try:
        page.accept_paths([str(images[0])])
        qtbot.waitUntil(lambda: any(card._key == 'mask' for card in page.field.cards), timeout=5000)
        cards = tuple(page.field.cards)
        card = next(card for card in cards if card._key == 'mask')
        card.click()
        workflow = page._workflow
        assert workflow is not None and chosen == []
        assert not page.path_edit.isVisible() and not page._suggestions_scroll.isVisible()
        assert workflow.central._key == 'mask' and workflow.central._path == str(tmp_path)
        qtbot.waitUntil(lambda: (workflow.central.geometry().center()
                                - workflow.view.viewport().mapTo(workflow, workflow.view.viewport().rect().center()))
                       .manhattanLength() <= 2)
        assert workflow.view.transform().m11() == 1.0
        assert workflow.central.width() >= 260 and workflow.central.height() >= 180
        heading = workflow.central.content.layout().itemAt(0).layout()
        title = heading.itemAt(1).widget()
        assert title.font().pixelSize() >= 18
        assert title.alignment() & Qt.AlignHCenter
        from spacr.qt.hidpi import device_ratio, logical_size

        assert logical_size(workflow.central.icon.pixmap()).width() == 64
        assert workflow.central.icon.pixmap().devicePixelRatioF() == device_ratio(workflow.central.icon)
        documented = [edge for edge in workflow.view.edges if edge.edge['kind'] != 'compatible']
        assert documented and all(edge.isVisible() for edge in documented)
        unrelated = [edge for edge in workflow.view.edges if edge.edge['kind'] == 'compatible'
                     and 'mask' not in (edge.edge['from'], edge.edge['to'])]
        assert unrelated and all(not edge.isVisible() for edge in unrelated)
        incoming = [workflow.view.nodes[edge.edge['from']] for edge in documented
                    if edge.edge['to'] == 'mask' and edge.edge['from'] != 'mask']
        outgoing = [workflow.view.nodes[edge.edge['to']] for edge in documented
                    if edge.edge['from'] == 'mask' and edge.edge['to'] != 'mask']
        selected_node = workflow.view.nodes['mask']
        assert incoming and outgoing
        assert all(node.pos().x() < selected_node.pos().x() for node in incoming)
        assert all(node.pos().x() > selected_node.pos().x() for node in outgoing)
        folds = folded_children()
        parents = {child: host for host, children in folds.items() for child in children}
        registered = {row[0] for row in APPS if app_is_visible(row[0])
                      and (row[0] not in parents or app_is_visible(parents[row[0]]))}
        assert registered <= set(workflow.data['modules'])
        for host, children in folds.items():
            for child in children:
                if (app_is_visible(host) and app_is_visible(child)
                        and preferences.maturity_is_visible(fold_label(child)[2])):
                    assert child in workflow.data['modules']
        assert 'overview' in workflow.data['modules']
        assert not {'3D', 'Time', 'concatenate'} & workflow.data['modules'].keys()
        owner = [edge for edge in workflow.view.edges if edge.edge['to'] == 'overview'
                 and edge.edge['kind'] == 'containment']
        assert len(owner) == 1 and owner[0].edge['from'] == 'mask'
        for size in ((640, 650), (1280, 850)):
            page.resize(*size)
            qtbot.waitUntil(lambda: workflow.view.size() == workflow.size())
            assert workflow.rect().contains(workflow.central.geometry())
            assert workflow.rect().contains(workflow.details.geometry())
            assert workflow.details.height() == 128
            assert not workflow.central.geometry().intersects(workflow.details.geometry())
        qtbot.waitUntil(lambda: workflow.details.verticalScrollBar().maximum() > 0)
        neighbor = workflow.view.nodes['make_masks']
        workflow.view.ensureVisible(neighbor)
        assert workflow.rect().contains(workflow.details.geometry())
        selected = workflow.view.nodes[workflow.central._key]
        projected = workflow.view.mapFromScene(selected.sceneBoundingRect()).boundingRect()
        projected.moveTopLeft(workflow.view.viewport().mapTo(workflow, projected.topLeft()))
        assert workflow.central.geometry() == projected
        details_geometry = workflow.details.geometry()
        qtbot.mouseMove(workflow.view.viewport(), workflow.view.mapFromScene(neighbor.sceneBoundingRect().center()))
        qtbot.waitUntil(lambda: 'Make Masks' in workflow.details.toPlainText())
        assert workflow.details.geometry() == details_geometry
        assert workflow.central._key == 'mask' and chosen == []
        qtbot.mouseClick(workflow.view.viewport(), Qt.LeftButton,
                         pos=workflow.view.mapFromScene(neighbor.sceneBoundingRect().center()))
        assert workflow.central._key == 'make_masks' and chosen == []
        assert workflow.central._path == str(images[0])
        workflow.central.click()
        assert chosen == [('make_masks', str(images[0]))]
        qtbot.keyClick(workflow.central, Qt.Key_Escape)
        assert page._workflow is None and page.path_edit.isVisible()
        assert page.path_edit.text() == str(images[0])
        assert tuple(page.field.cards) == cards
        assert _bytes(images) == before
    finally:
        page.close()
