"""Local Easy mode input, native layout, drop and explicit module handoff."""
import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QCoreApplication, QMimeData, QPoint, QPointF, QRect, QRectF, Qt, QUrl
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
    assert all(action[3] == str(source) for action in first['actions'])
    assert first['context']['reader_candidates'] == []
    assert str(outside) not in first['context']['image_folders']
    neighbor = master / 'acquisition'
    neighbor.mkdir()
    inside_container = neighbor / 'acquisition.czi'
    inside_container.write_bytes(outside_container.read_bytes())
    originals.update(_bytes([inside_container]))
    second = _easy_inspect_path(str(source))
    assert [action[0] for action in second['actions']] == ['make_masks', 'mask']
    assert all(action[3] == str(source) for action in second['actions'])
    assert second['context']['reader_candidates'] == []
    explicit_master = _easy_inspect_path(str(master))
    assert [row['path'] for row in explicit_master['context']['reader_candidates']] == [str(inside_container)]
    assert explicit_master['context']['reader_candidates'][0]['metadata_inspected'] is False
    assert next(action[3] for action in explicit_master['actions'] if action[0] == 'convert') == str(neighbor)
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
    assert set(actions) == {'make_masks', 'mask', 'db_browser', 'foreign', 'graph_builder', 'gate_editor'}
    assert actions['make_masks'][3] == str(tmp_path)
    assert actions['foreign'][3] == str(table)
    assert actions['db_browser'][3] == str(database)
    assert 'convert' not in actions
    assert all(actions[key][3] == str(table) for key in ('graph_builder', 'gate_editor'))
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
    context = ordinary['context']
    routes = context['generated_outputs']['routes']
    assert ordinary['actions'][0][0] == 'measure'
    assert {key for key, route in routes.items() if route['ready']} == {'measure', 'make_masks'}
    assert set(context['analysis_inputs']) == {'measure', 'make_masks'}
    assert context['analysis_inputs']['make_masks'] == {'source': str(merged)}
    assert routes['make_masks']['source'] == str(merged)
    assert routes['make_masks']['prerequisite'] == ''
    assert routes['measure']['source'] == str(tmp_path)
    assert routes['measure']['missing_inputs'] == []
    assert {'annotate', 'db_browser'} <= set(routes)
    assert 'crops' in routes['annotate']['missing_inputs']
    assert 'measurements' in routes['db_browser']['missing_inputs']
    assert {action[0] for action in ordinary['actions']} == set(routes)
    for key, title, reason, source in ordinary['actions']:
        assert (title, reason, source) == tuple(routes[key][field]
                                               for field in ('title', 'reason', 'source'))
        if key not in ('measure', 'make_masks'):
            assert routes[key]['prerequisite']
            assert source == str(tmp_path)
    marker = tmp_path / 'plate_batch.json'
    marker.write_text('{incomplete')
    result = _easy_inspect_path(str(tmp_path))
    assert [action[0] for action in result['actions']] == ['make_masks']
    assert result['actions'][0][3] == str(merged)
    assert result['context']['analysis_inputs'] == {'make_masks': {'source': str(merged)}}
    assert result['context']['generated_outputs']['routes'] == {}
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
        qtbot.wait(650)
        assert not page._runner.is_busy() and not page.field.cards
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: {'make_masks', 'mask'} <=
                       {card._key for card in page.field.cards}, timeout=5000)
        qtbot.waitUntil(lambda: all(not card._growing for card in page.field.cards), timeout=7000)
        assert page.path_edit.text() == str(image)
        assert page.path_edit.placeholderText() == 'Drop a file or folder | Type a Path | Ask a Question ◉'
        assert page.path_edit.rect().contains(page.path_edit.browse.geometry())
        assert page.path_edit.isVisible()
        assert not page.status.isVisible()
        assert not page.path_edit.hasFrame()
        assert page.field.rim.isVisible()
        panel = page.field.panel.geometry()
        rectangles = [card.geometry() for card in page.field.cards]
        assert abs(panel.center().y() - page.field.rect().center().y()) <= 1
        assert all(rect.bottom() <= panel.top() - 24 for rect in rectangles)
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
        assert page._workflow is not None and page.path_edit.isVisible()
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
            assert {action[0] for action in actions} == {'make_masks', 'mask', 'db_browser', 'foreign', 'graph_builder', 'gate_editor'}
            preferences.set_reduce_motion(False)
            page.field.set_cards(actions)
            assert len(page.field._animations) == len(actions)
            for index, animation in enumerate(page.field._animations):
                assert animation.animationAt(0).duration() == index * 200
                assert animation.animationAt(1).duration() == 500
            page.field._finish_growth()
            preferences.set_reduce_motion(True)
            original = [card.geometry() for card in page.field.cards]
            for resized in (size, (640, 650), (1100, 780), size):
                page.resize(*resized)
                qapp.processEvents()
                assert [card._key for card in page.field.cards] == [action[0] for action in actions]
                rectangles = [card.geometry() for card in page.field.cards]
                assert len({rect.top() for rect in rectangles}) == 1
                assert all(rect.bottom() <= page.path_edit.geometry().top() - 24
                           for rect in rectangles)
                assert page.field._scrollbar.maximum() > 0
                assert abs(page.path_edit.geometry().center().y() - page.field.rect().center().y()) <= 1
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
        assert page._requested == str(first) and not page._queued_inspection
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        assert page._queued_inspection or page._requested == str(second)
        assert page.path_edit.isEnabled()
        if reject_pending:
            page.accept_paths([str(first), str(second)])
            assert 'one local file or folder' in page.status.text()
        release.set()
        if reject_pending:
            qtbot.waitUntil(lambda: not page._runner.is_busy(), timeout=3000)
            assert not page.field.cards
            assert 'one local file or folder' in page.field.details.toPlainText()
            assert not page.status.isVisible()
            assert 'one local file or folder' in page.status.text()
            qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: not page._runner.is_busy() and page._requested == str(second)
                       and bool(page._last_result) and page._last_result['path'] == str(second), timeout=5000)
        assert [card._key for card in page.field.cards] == ['foreign', 'graph_builder', 'gate_editor']
        assert page.field.cards[0]._key == 'foreign'
        assert page.field.cards[0]._path == str(second)
        assert all(card._key not in ('regex', 'make_masks', 'mask') for card in page.field.cards)
        assert all(card._key != 'convert' for card in page.field.cards)
        assert all(card._path == str(second) for card in page.field.cards
                   if card._key in ('graph_builder', 'gate_editor'))
        assert page._last_result['path'] == str(second)
        assert all(card._path != str(first) for card in page.field.cards if card._key != 'regex')
        assert 'first.tif' not in page.status.text()
        page.accept_paths([str(first), str(second)])
        assert 'one local file or folder' in page.status.text()
        assert not page.field.cards
        assert 'one local file or folder' in page.field.details.toPlainText()
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
        assert window._dock_mode == 'locked'
        assert window._dock_edge.isVisible() and window._dock_edge.is_collapsed()
        assert not window._dock_slot.isVisible()
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        backdrop = window.window_backdrop()
        assert backdrop.background_color() == normal_background
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QColor, QImage, QPainter

        base = QImage(12, 12, QImage.Format_ARGB32)
        base.fill(QColor('magenta'))
        painter = QPainter(base)
        try:
            backdrop._paint_base(painter, QRect(0, 0, 12, 12))
        finally:
            painter.end()
        assert all(base.pixelColor(x, y) == normal_background
                   for x in range(12) for y in range(12))
        assert preferences.get_theme() == 'light'
        assert preferences.get_ambient_palette() == palette_choice
        old_page = page
        window.refresh_theme()
        page = window._startup
        assert page is not old_page and isinstance(page, _EasyHomePage)
        assert window._stack.currentWidget() is page
        assert window.window_backdrop().background_color() == normal_background
        window._retire_the_dock_backdrop()
        assert window.window_backdrop() is None
        window._backdrop_the_dock_column()
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        assert window.window_backdrop() is not backdrop
        assert window.window_backdrop().background_color() == normal_background
        preferences.set_easy_mode_dock(True)
        window.refresh_after_preferences({"easy_mode_dock"})
        assert window._dock_mode == 'locked'
        assert preferences.get_dock_mode() == 'hidden'
        preferences.set_easy_mode_dock(False)
        window.refresh_after_preferences({"easy_mode_dock"})
        assert window._dock_mode == 'locked'
        assert window._dock_edge.isVisible() and window._dock_edge.is_collapsed()
        assert preferences.get_easy_mode_dock() is False
        assert preferences.get_dock_mode() == 'hidden'
        from PySide6.QtWidgets import QDialog
        from spacr.qt.widgets import ambient

        waves = []
        original_publish = ambient.AmbientWidget._publish_boundary_ripples

        def publish(backdrop, segments, strength, **kwargs):
            waves.append((segments, strength))
            return original_publish(backdrop, segments, strength, **kwargs)

        monkeypatch.setattr(ambient.AmbientWidget, '_publish_boundary_ripples', publish)
        assert not window.property('spacrEasySearchRippleOnly')
        popup = QDialog(window)
        popup.setModal(True)
        popup.show()
        qtbot.waitUntil(lambda: QApplication.activeModalWidget() is popup)
        ambient._field_ripple_for_popup(popup)
        popup.close()
        popup.deleteLater()
        assert preferences.get_ambient_enabled() and preferences._field_ripples_enabled()
        assert window.window_backdrop().theme() in ambient._FIELD_THEMES
        qtbot.waitUntil(lambda: window.window_backdrop()._should_run(), timeout=5000)
        qtbot.mouseMove(page.path_edit, QPoint(12, 12))
        page.path_edit.setFocus()
        qapp.processEvents()
        before_click = len(waves)
        qtbot.mouseClick(page.path_edit, Qt.LeftButton)
        assert all(strength != 0.12 for _segments, strength in waves[before_click:])
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
        qtbot.waitUntil(lambda: {'make_masks', 'mask'} <=
                       {card._key for card in page.field.cards}, timeout=5000)
        assert page.path_edit.text() == str(image)
        assert window._stack.currentWidget() is page
        assert 'home' not in window._screens
        card = next(card for card in page.field.cards if card._key == 'make_masks')
        qtbot.waitUntil(lambda: not card._growing and card.isEnabled(), timeout=7000)
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
        assert window.window_backdrop().background_color() == normal_background
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
        assert preferences.get_dock_mode() == 'hidden'
        assert window._dock_mode == 'locked' and window._dock_edge.isVisible()
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
        assert {'mask', 'make_masks'} <= {item._key for item in page.field.cards}
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
            assert {'mask', 'make_masks'} <= {item._key for item in page.field.cards}
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
        assert page._workflow is None and page.path_edit.text() == '' and not page.field.cards
        before_loads = len(loads)
        page.accept_paths([str(container)])
        qtbot.waitUntil(lambda: not page._runner.is_busy() and any(
            item._key == 'mask' for item in page.field.cards), timeout=5000)
        assert len(loads) == before_loads
        next(item for item in page.field.cards if item._key == 'mask').click()
        workflow = page._workflow
        neighbor = workflow.view.nodes['overview']
        workflow.view.ensureVisible(neighbor)
        qtbot.mouseClick(workflow.view.viewport(), Qt.LeftButton,
                         pos=workflow.view.mapFromScene(neighbor.sceneBoundingRect().center()))
        assert workflow.central._key == 'mask' and workflow._chosen_key == 'overview' and opened == []
        assert window._stack.currentWidget() is page
        qtbot.mouseClick(workflow.view.viewport(), Qt.LeftButton,
                         pos=workflow.view.mapFromScene(neighbor.sceneBoundingRect().center()))
        qtbot.waitUntil(lambda: bool(opened) and getattr(target, '_overview_tab', None) is not None, timeout=10000)
        assert opened[0][0] is target and opened[0][1]['src'] == str(tmp_path)
        assert 'overview' not in window._screens and 'home' not in window._screens
        assert _bytes(images) == original and container.read_bytes() == container_bytes
    finally:
        page.close()
        window.close()


@pytest.mark.parametrize('global_theme', ['light', 'dark'])
def test_workflow_selection_hover_and_escape_preserve_the_verified_input(
        qtbot, qapp, private_ai_profile, tmp_path, monkeypatch, global_theme):
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QComboBox
    from spacr.qt import theme
    from spacr.qt.widgets.field_fade import fades
    from spacr.qt.app import APPS, app_is_visible, folded_children
    from spacr.qt.widgets.fold_strip import fold_label
    from spacr.qt.widgets.home import _EasyHomePage, _EASY_WORKFLOW_PORT_RADIUS
    from spacr.qt.hidpi import device_ratio, logical_size

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_show_alpha(True)
    preferences._set_show_alpha_features(True)
    images = [tmp_path / name for name in ('W0025F0001T0001Z000C1.tif', 'W0025F0001T0001Z000C2.tif')]
    for image in images:
        _image(image)
    before = _bytes(images)
    chosen = []
    original_sheet, original_palette = qapp.styleSheet(), qapp.palette()
    theme.apply_qpalette(qapp, global_theme)
    qapp.setStyleSheet(theme.stylesheet(global_theme, preferences.get_font_scale()))
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 850)
    page.show()
    page.module_requested.connect(lambda key, path: chosen.append((key, path)))
    try:
        page.accept_paths([str(images[0])])
        qtbot.waitUntil(lambda: any(card._key == 'mask' for card in page.field.cards), timeout=5000)
        cards = tuple(page.field.cards)
        selected_card = next(card for card in cards if card._key == 'mask')
        original_parent = selected_card.parentWidget()
        original_rect = QRect(selected_card.mapToGlobal(QPoint()), selected_card.size())
        selected_card.click()
        workflow = page._workflow
        assert workflow is not None and chosen == []
        assert not page.path_edit.isVisible()
        assert page._suggestions_scroll.isVisible() and page.field.isVisible()
        assert workflow.central is selected_card
        assert QRect(selected_card.mapToGlobal(QPoint()), selected_card.size()) == original_rect
        assert selected_card.parentWidget() is workflow
        assert workflow._progress == 1.0
        page.field._finish_growth()
        page.field._arrange()
        assert QRect(selected_card.mapToGlobal(QPoint()), selected_card.size()) == original_rect
        assert workflow.central._key == 'mask' and workflow.central._path == str(tmp_path)
        assert workflow.central.width() == workflow.central.height()
        assert workflow.central.width() >= 192
        assert workflow.view.transform().m11() == 1.0
        title = workflow.central.title
        from spacr.qt.theme import font_px
        assert title.font().pixelSize() == font_px(18)
        assert title.alignment() & Qt.AlignHCenter
        assert logical_size(workflow.central.icon.pixmap()).width() == 96
        assert workflow.central.icon.pixmap().devicePixelRatioF() == device_ratio(workflow.central.icon)
        visible = {key for key, node in workflow.view.nodes.items() if node.isVisible()}
        assert {'measure', 'overview'} <= visible and len(visible) <= 4
        assert 'mask' not in visible
        assert all(edge.edge['from'] == 'mask' and edge.edge['to'] in visible
                   and edge.edge['kind'] in ('documented', 'containment') for edge in workflow.view.edges)
        assert len(workflow.view.edges) == len(visible)
        assert ' '.join(workflow.data['modules']['mask']['documentation_note'].split()) in ' '.join(
            workflow.details.toPlainText().split())
        for edge in workflow.view.edges:
            if edge.edge.get('mode_labels'):
                assert edge.boundingRect().contains(edge._badge_rect())
                assert edge._badge_rect().bottom() == workflow.view.nodes[edge.edge['to']].pos().y() - 14
            assert edge.path().elementCount() >= 7
            assert any(edge.path().elementAt(index).isCurveTo()
                       for index in range(edge.path().elementCount()))
            path = edge.path()
            midpoint = path.pointAtPercent(path.percentAtLength(path.length() / 2))
            assert (edge._arrow_center - midpoint).manhattanLength() < 0.0001
            centroid = (edge.arrowhead[0] + edge.arrowhead[1] + edge.arrowhead[2]) / 3
            assert (centroid - midpoint).manhattanLength() < 0.0001
            before_mid = path.pointAtPercent(path.percentAtLength(path.length() / 2 - 0.1))
            after_mid = path.pointAtPercent(path.percentAtLength(path.length() / 2 + 0.1))
            tangent = after_mid - before_mid
            arrow = edge.arrowhead[0] - (edge.arrowhead[1] + edge.arrowhead[2]) / 2
            assert tangent.x() * arrow.x() + tangent.y() * arrow.y() > 0
            if edge.edge.get('mode_labels'):
                assert not edge.arrowhead.boundingRect().intersects(edge._badge_rect())
            assert all(not edge.arrowhead.boundingRect().intersects(node.sceneBoundingRect())
                       for node in workflow.view.nodes.values() if node.isVisible())
            target = workflow.view.nodes[edge.edge['to']]
            assert len(target._input_ports) == 1
            assert target._input_ports[0].x() == _EASY_WORKFLOW_PORT_RADIUS + 1
            assert target._input_ports[0].y() >= target.boundingRect().height() * 0.65
            assert edge.end == target.pos() + target._input_ports[0]
        ports = workflow.central._workflow_ports
        assert len(ports) == len(workflow.view.edges) and _EASY_WORKFLOW_PORT_RADIUS == 4
        assert len(set(point.y() for point in ports)) == len(ports)
        assert all(point.x() == workflow.central.width() - _EASY_WORKFLOW_PORT_RADIUS - 1
                   and point.y() <= workflow.central.height() / 3 for point in ports)
        gaps = [second.y() - first.y() for first, second in zip(ports, ports[1:])]
        assert max(gaps) - min(gaps) <= 1 and min(gaps) > 2 * _EASY_WORKFLOW_PORT_RADIUS
        owner = [edge for edge in workflow.view.edges if edge.edge['to'] == 'overview']
        assert len(owner) == 1 and owner[0].edge['kind'] == 'containment'
        folds = folded_children()
        parents = {child: host for host, children in folds.items() for child in children}
        registered = {row[0] for row in APPS if app_is_visible(row[0])
                      and (row[0] not in parents or app_is_visible(parents[row[0]]))}
        assert registered <= set(workflow.data['modules'])
        picker_keys = {workflow.tools.itemData(index) for index in range(workflow.tools.count())}
        assert picker_keys == set(workflow.data['modules'])
        for host, children in folds.items():
            for child in children:
                if (app_is_visible(host) and app_is_visible(child)
                        and preferences.maturity_is_visible(fold_label(child)[2])):
                    assert child in picker_keys
        assert not {'3D', 'Time', 'concatenate'} & workflow.data['modules'].keys()
        neighbor = workflow.view.nodes['measure']
        workflow.view.ensureVisible(neighbor)
        geometry = workflow.details.geometry()
        qtbot.mouseMove(workflow.view.viewport(), workflow.view.mapFromScene(neighbor.sceneBoundingRect().center()))
        qtbot.waitUntil(lambda: 'Measure' in workflow.details.toPlainText())
        assert workflow.details.geometry() == geometry
        assert workflow.central._key == 'mask' and chosen == []
        original_nodes = {key: node.pos() for key, node in workflow.view.nodes.items() if node.isVisible()}
        original_edges = {(edge.edge['from'], edge.edge['to']) for edge in workflow.view.edges}
        qtbot.mouseClick(workflow.view.viewport(), Qt.LeftButton,
                         pos=workflow.view.mapFromScene(neighbor.sceneBoundingRect().center()))
        assert workflow.central is selected_card and workflow._chosen_key == 'measure' and chosen == []
        assert all(workflow.view.nodes[key].pos() == position for key, position in original_nodes.items())
        assert original_edges <= {(edge.edge['from'], edge.edge['to']) for edge in workflow.view.edges}
        assert neighbor.highlighted
        assert QRect(workflow.central.mapToGlobal(QPoint()), workflow.central.size()) == original_rect
        assert workflow.central.icon.pixmap().devicePixelRatioF() == device_ratio(workflow.central.icon)
        expected = workflow.central._module_icon.pixmap(round(96 * device_ratio(workflow.central.icon)),
                                                       round(96 * device_ratio(workflow.central.icon)))
        expected.setDevicePixelRatio(device_ratio(workflow.central.icon))
        assert workflow.central.icon.pixmap().toImage() == expected.toImage()
        from spacr.qt.iconset import app_icon
        assert neighbor._glyph.pixmap(96).toImage() == app_icon('measure', theme='dark').pixmap(96).toImage()
        assert workflow.central._path == str(tmp_path)
        assert len(workflow.view.branch_records) == 4
        assert {record['to'] for record in workflow.view.branch_records} == {
            'annotate', 'classify_merged', 'umap', 'graph_builder'}
        assert workflow.more.isVisible() and workflow.more.count() > 0
        assert {workflow.more.itemData(index) for index in range(workflow.more.count())} == {
            record['to'] for record in workflow.view.remaining_records}
        assert not {record['to'] for record in workflow.view.branch_records} & {
            workflow.more.itemData(index) for index in range(workflow.more.count())}
        dark = theme.palette_for('dark')
        ordinary = QComboBox()
        qtbot.addWidget(ordinary)
        assert fades(ordinary)
        for selector in (workflow.tools, workflow.more):
            assert selector.currentIndex() == -1 and selector.placeholderText()
            assert not fades(selector)
            assert selector.palette().color(QPalette.Text) == QColor(dark['fg'])
            assert selector.palette().color(QPalette.PlaceholderText) == QColor(dark['fg_muted'])
            raster = selector.grab().toImage()
            background = raster.pixelColor(raster.width() // 2, raster.height() - 6)
            background_sum = background.red() + background.green() + background.blue()
            matches = sum(raster.pixelColor(x, y).red() + raster.pixelColor(x, y).green()
                          + raster.pixelColor(x, y).blue() >= background_sum + 36
                          for y in range(6, raster.height() - 6)
                          for x in range(8, raster.width() - 28))
            assert matches > 5
        for size in ((500, 650), (640, 650), (1080, 601), (1280, 850)):
            page.resize(*size)
            qtbot.waitUntil(lambda: workflow.size() == page.size())
            assert not workflow.tools.geometry().intersects(workflow.more.geometry())
            assert not workflow.back.geometry().intersects(workflow.tools.geometry())
            assert workflow.central.width() == workflow.central.height()
            assert workflow.rect().contains(workflow.central.geometry())
            assert workflow.rect().contains(workflow.details.geometry())
            assert not workflow.central.geometry().intersects(workflow.details.geometry())
            assert not workflow.view.geometry().intersects(workflow.central.geometry())
            assert not workflow.view.geometry().intersects(workflow.details.geometry())
            first_row = min(node.pos().y() for node in workflow.view.nodes.values() if node.isVisible())
            row_nodes = [node for node in workflow.view.nodes.values() if node.isVisible() and node.pos().y() == first_row]
            assert row_nodes and all(workflow.view.viewport().height() >= node.boundingRect().height() + first_row
                                     for node in row_nodes)
            fixed = (workflow.central.geometry(), workflow.details.geometry(), workflow.back.geometry())
            if size == (640, 650):
                qtbot.waitUntil(lambda: workflow.view.verticalScrollBar().maximum() > 0)
                workflow.view.verticalScrollBar().setValue(workflow.view.verticalScrollBar().maximum())
                assert workflow.view.verticalScrollBar().value() > 0
                assert (workflow.central.geometry(), workflow.details.geometry(), workflow.back.geometry()) == fixed
                from spacr.qt.widgets.home import _EasyWorkflowEdge
                drawn = []
                original_draw = _EasyWorkflowEdge.draw_connection

                def draw_connection(edge, painter):
                    drawn.append(edge)
                    return original_draw(edge, painter)

                with monkeypatch.context() as observer:
                    observer.setattr(_EasyWorkflowEdge, 'draw_connection', draw_connection)
                    assert not workflow.grab().isNull()
                expected_edges = {edge for edge in workflow.view.edges
                                  if workflow.view.mapFromScene(workflow.view.nodes[edge.edge['to']]
                                      .sceneBoundingRect()).boundingRect().intersects(workflow.view.viewport().rect())}
                assert expected_edges and len(expected_edges) < len(workflow.view.edges)
                assert set(drawn) == expected_edges
        additional = workflow.more.itemData(0)
        workflow.more.showPopup()
        next_index = workflow.more.model().index(0, 0)
        next_popup = workflow.more.view()
        assert next_popup.palette().color(QPalette.Text) == QColor(dark['fg'])
        assert next_popup.palette().color(QPalette.HighlightedText) == QColor(dark['bg'])
        next_popup.scrollTo(next_index)
        qtbot.waitUntil(lambda: next_popup.visualRect(next_index).intersects(next_popup.viewport().rect()))
        qtbot.mouseClick(next_popup.viewport(), Qt.LeftButton, pos=next_popup.visualRect(next_index).center())
        assert workflow.central is selected_card and workflow._chosen_key == additional and chosen == []
        workflow.tools.showPopup()
        index = workflow.tools.model().index(workflow.tools.findData('make_masks'), 0)
        popup = workflow.tools.view()
        assert popup.palette().color(QPalette.Text) == QColor(dark['fg'])
        assert popup.palette().color(QPalette.HighlightedText) == QColor(dark['bg'])
        popup.scrollTo(index)
        qtbot.waitUntil(lambda: popup.visualRect(index).intersects(popup.viewport().rect()))
        qtbot.mouseClick(popup.viewport(), Qt.LeftButton, pos=popup.visualRect(index).center())
        assert workflow.central is selected_card and workflow._chosen_key == 'make_masks' and chosen == []
        assert workflow.routes['make_masks'][3] == str(images[0])
        child = workflow.view.nodes['make_masks']
        workflow.view.ensureVisible(child)
        qtbot.mouseClick(workflow.view.viewport(), Qt.LeftButton,
                         pos=workflow.view.mapFromScene(child.sceneBoundingRect().center()))
        assert chosen == [('make_masks', str(images[0]))]
        qtbot.keyClick(workflow.central, Qt.Key_Escape)
        assert page._workflow is None and page.path_edit.isVisible()
        assert page.path_edit.text() == str(images[0])
        assert tuple(page.field.cards) == cards
        assert selected_card.parentWidget() is original_parent and selected_card.isVisible()
        assert _bytes(images) == before
    finally:
        page.close()
        qapp.setStyleSheet(original_sheet)
        qapp.setPalette(original_palette)


def test_network_emerges_around_the_same_card_and_live_reduce_motion_finishes_it(
        qtbot, qapp, private_ai_profile, isolated_input_root):
    from PySide6.QtCore import QAbstractAnimation
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(True)
    image = isolated_input_root / 'W0025F0001T0001Z000C1.tif'
    _image(image)
    original = image.read_bytes()
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 850)
    page.show()
    chosen = []
    page.module_requested.connect(lambda key, path: chosen.append((key, path)))
    try:
        page.accept_paths([str(image)])
        qtbot.waitUntil(lambda: any(card._key == 'mask' for card in page.field.cards), timeout=5000)
        source = next(card for card in page.field.cards if card._key == 'mask')
        parent = source.parentWidget()
        rectangle = QRect(source.mapToGlobal(QPoint()), source.size())
        field = page.field
        preferences.set_reduce_motion(False)
        source.click()
        workflow = page._workflow
        assert workflow.central is source and page.field is field and field.isVisible()
        assert page._suggestions_scroll.isVisible() and workflow._transition.duration() == 500
        assert workflow.property('spacrAmbientTransparentOverlay') is True
        assert not workflow.autoFillBackground() and not workflow.testAttribute(Qt.WA_OpaquePaintEvent)
        assert workflow._progress == 0.0
        assert QRect(source.mapToGlobal(QPoint()), source.size()) == rectangle
        qtbot.waitUntil(lambda: 0.2 <= workflow._progress < 0.8, timeout=1800)
        middle = workflow._progress
        assert field.isVisible() and QRect(source.mapToGlobal(QPoint()), source.size()) == rectangle
        qtbot.waitUntil(lambda: workflow._progress > middle + 0.1, timeout=1000)
        assert chosen == []
        preferences.set_reduce_motion(True)
        preferences._hold_the_backdrops_still(qapp)
        assert workflow._progress == 1.0
        assert workflow._transition.state() == QAbstractAnimation.Stopped
        assert QRect(source.mapToGlobal(QPoint()), source.size()) == rectangle
        qtbot.keyClick(source, Qt.Key_Escape)
        assert page._workflow is None and source.parentWidget() is parent and source.isVisible()
        assert source.geometry() == source._final_rect and not source._workflow_ports
        assert page.path_edit.text() == str(image) and image.read_bytes() == original
    finally:
        page.close()


@pytest.mark.parametrize('width', [340, 500, 640])
def test_single_action_keeps_its_anchor_and_complete_children_in_narrow_windows(
        qtbot, private_ai_profile, isolated_input_root, width):
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    path = isolated_input_root / 'objects.csv'
    path.write_text('label,area\n1,12\n2,9\n')
    original = path.read_bytes()
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(width, 700)
    page.show()
    try:
        page.accept_paths([str(path)])
        qtbot.waitUntil(lambda: {'graph_builder', 'gate_editor'} <=
                        {card._key for card in page.field.cards}, timeout=5000)
        source = next(card for card in page.field.cards if card._key == 'graph_builder')
        page._suggestions_scroll.ensureWidgetVisible(source)
        rectangle = QRect(source.mapToGlobal(QPoint()), source.size())
        source.click()
        workflow = page._workflow
        assert workflow.central is source
        assert QRect(source.mapToGlobal(QPoint()), source.size()) == rectangle
        visible = [node for node in workflow.view.nodes.values() if node.isVisible()]
        assert visible
        first = min(node.pos().y() for node in visible)
        for node in visible:
            assert node.boundingRect().width() == node.boundingRect().height()
            assert 0 <= node.pos().x() and node.sceneBoundingRect().right() <= workflow.view.viewport().width()
            if node.pos().y() == first:
                assert node.sceneBoundingRect().bottom() <= workflow.view.viewport().height()
        assert not workflow.view.geometry().intersects(workflow.central.geometry())
        root_top = workflow.view.mapToScene(workflow.view.viewport().mapFromGlobal(
            source.mapToGlobal(QPoint())))
        root_rect = QRectF(root_top.x(), root_top.y(), source.width(), source.height())
        for edge in workflow.view.edges:
            direction = edge.path().pointAtPercent(edge.path().percentAtLength(2)) - edge.start
            assert direction.x() > 0
            target = workflow.view.nodes[edge.edge['to']]
            if not workflow.view.mapFromScene(target.sceneBoundingRect()).boundingRect().intersects(
                    workflow.view.viewport().rect()):
                continue
            path_length = edge.path().length()
            for step in range(1, 80):
                distance = 8 + (path_length - 16) * step / 80
                point = edge.path().pointAtPercent(edge.path().percentAtLength(distance))
                assert not root_rect.adjusted(1, 1, -1, -1).contains(point)
        qtbot.keyClick(source, Qt.Key_Escape)
        assert page._workflow is None and source.parentWidget() is page.field
        assert path.read_bytes() == original
    finally:
        page.close()


def _choose_home_browse_action(qtbot, choice):
    from PySide6.QtWidgets import QMenu

    menu = QApplication.activePopupWidget()
    assert isinstance(menu, QMenu)
    try:
        if choice == 'cancel':
            qtbot.keyClick(menu, Qt.Key_Escape)
        else:
            action = menu.actions()[0 if choice == 'file' else 1]
            qtbot.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(action).center())
    finally:
        menu.close()


class _HomeBrowseDialogs:
    """Record only the standard file-dialog boundary after a real menu choice."""

    def __init__(self, path):
        self.path, self.calls = path, []

    def file(self, *args):
        self.calls.append('file')
        return self.path, ''

    def folder(self, *args):
        self.calls.append('folder')
        return self.path


def _fail_owned_home_inspection():
    raise PermissionError('owned unreadable input')


def _wait_owned_home_inspection(entered, release):
    entered.set()
    assert release.wait(5)


@pytest.mark.parametrize('choice,accepted', [
    ('file', True), ('folder', True), ('cancel', False),
    ('file', False), ('folder', False),
])
def test_browse_uses_real_menu_choice_and_only_accepts_confirmed_paths(
        qtbot, private_ai_profile, isolated_input_root, monkeypatch, choice, accepted):
    from functools import partial
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QFileDialog
    from spacr.qt.widgets.home import _EasyHomePage

    image = isolated_input_root / 'field.tif'
    _image(image)
    path = str(image if choice == 'file' else isolated_input_root)
    dialogs = _HomeBrowseDialogs(path if accepted else '')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', dialogs.file)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', dialogs.folder)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.show()
    before = image.read_bytes()
    timer = QTimer(page)
    timer.setSingleShot(True)
    timer.timeout.connect(partial(_choose_home_browse_action, qtbot, choice))
    try:
        timer.start(0)
        qtbot.mouseClick(page.path_edit.browse, Qt.LeftButton)
        assert dialogs.calls == ([] if choice == 'cancel' else [choice])
        if accepted:
            qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
            assert page.path_edit.text() == path and page._requested == path
            assert page._last_result['path'] == path
            assert {'make_masks', 'mask'} <= {card._key for card in page.field.cards}
        else:
            assert page.path_edit.text() == '' and page._last_result is None
            assert not page._runner.active_jobs() and not page.field.cards
        assert image.read_bytes() == before
    finally:
        timer.stop()
        page.close()
        assert not page._runner.active_jobs()


@pytest.mark.parametrize('blank', [False, True])
def test_owned_inspection_failure_retires_stale_actions_without_reviving_blank_input(
        qtbot, private_ai_profile, isolated_input_root, blank):
    from spacr.qt.widgets.home import _EasyHomePage

    path = isolated_input_root / 'field.tif'
    _image(path)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    if not blank:
        page.resize(640, 650)
    page.show()
    try:
        page.accept_paths([str(path)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
        assert any(card._key == 'mask' for card in page.field.cards)
        if blank:
            page.path_edit.clear()
            page._clear_empty_input()
        with qtbot.waitSignal(page._runner.job_finished, timeout=5000):
            page._runner.submit(_fail_owned_home_inspection)
        qtbot.waitUntil(lambda: not page._runner.active_jobs(), timeout=5000)
        if blank:
            assert not page.field.cards and page.path_edit.text() == ''
            assert not page.status.text()
        else:
            assert not page.field.cards
            assert 'Check permissions' in page.status.text()
            assert page.field.details.toPlainText() == page.status.text()
            assert page.field.details.isVisible()
            assert not page.field.details.visibleRegion().isEmpty()
            assert page.path_edit.accessibleDescription() == page.status.text()
            assert page.path_edit.toolTip() == page.status.text()
        assert path.exists()
    finally:
        page.close()


def test_regex_refusal_preserves_owned_busy_job_and_missing_result(
        qtbot, private_ai_profile, isolated_input_root):
    from functools import partial
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_reduce_motion(True)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.show()
    entered, release = threading.Event(), threading.Event()
    try:
        generation = page._input_generation
        page._apply_regex('(?P<channel>.*)')
        assert page._input_generation == generation and page._regex_cancel is None
        assert not page._runner.active_jobs()
        path = isolated_input_root / 'field.tif'
        _image(path)
        page.accept_paths([str(path)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
        cards, result, generation = tuple(page.field.cards), page._last_result, page._input_generation
        page._runner.submit(partial(_wait_owned_home_inspection, entered, release))
        qtbot.waitUntil(entered.is_set, timeout=3000)
        jobs = page._runner.active_jobs()
        page._apply_regex('(?P<channel>.*)')
        assert page._runner.active_jobs() == jobs and page._runner.is_busy()
        assert page._last_result is result and tuple(page.field.cards) == cards
        assert page._input_generation == generation and page._regex_cancel is None
        assert all(card.isEnabled() for card in cards)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not page._runner.active_jobs(), timeout=5000)
        page.close()


def test_suggestion_refusals_and_host_commands_do_not_create_a_workflow(
        qtbot, private_ai_profile):
    from PySide6.QtTest import QSignalSpy
    from spacr.qt.widgets.home import _EasyHomePage

    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.show()
    commands = QSignalSpy(page.command_requested)
    modules = QSignalSpy(page.module_requested)
    try:
        page._select_suggestion('missing-tool', '/missing')
        page._select_suggestion('mask', '/missing')
        assert page._workflow is None and not page._runner.active_jobs()
        page._select_suggestion('__easy_command__:preferences', '')
        assert commands.count() == 1 and commands.at(0) == ['preferences']
        assert modules.count() == 0 and page._workflow is None
        assert page.path_edit.text() == '' and page._last_result is None
    finally:
        page.close()


def test_graph_keyboard_and_field_wheel_preserve_the_real_borrowed_anchor(
        qtbot, private_ai_profile, isolated_input_root):
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent, QWheelEvent
    from PySide6.QtTest import QSignalSpy
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    path = isolated_input_root / 'objects.csv'
    path.write_text('label,area\n1,12\n2,9\n')
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(340, 650)
    page.show()
    before = path.read_bytes()
    try:
        page.accept_paths([str(path)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
        field = page.field
        source = next(card for card in field.cards if card._key == 'graph_builder')
        field._scrollbar.setValue(0)
        source.click()
        workflow = page._workflow
        rectangle = QRect(source.mapToGlobal(QPoint()), source.size())
        chosen = QSignalSpy(workflow.view.chosen)
        node = next(node for node in workflow.view.nodes.values() if node.isVisible())
        for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            workflow.view.scene().sendEvent(node, QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
        assert chosen.count() == 3 and all(chosen.at(index) == [node.key] for index in range(3))
        workflow.view.scene().sendEvent(node, QKeyEvent(QEvent.KeyPress, Qt.Key_A, Qt.NoModifier))
        assert chosen.count() == 3
        selected = workflow._chosen_key
        workflow.select('missing-tool')
        assert workflow._chosen_key == selected and workflow.central is source
        panel, details = QRect(field.panel.geometry()), QRect(field.details.geometry())
        assert field._scrollbar.maximum() > 0
        position = QPointF(field._row_rect.center())
        event = QWheelEvent(position, QPointF(field.mapToGlobal(position.toPoint())),
                            QPoint(-24, 0), QPoint(), Qt.NoButton, Qt.NoModifier,
                            Qt.ScrollUpdate, False)
        QApplication.sendEvent(field, event)
        assert event.isAccepted() and field._scrollbar.value() == min(24, field._scrollbar.maximum())
        event = QWheelEvent(position, QPointF(field.mapToGlobal(position.toPoint())),
                            QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                            Qt.ScrollUpdate, False)
        QApplication.sendEvent(field, event)
        assert event.isAccepted() and field._scrollbar.value() == min(84, field._scrollbar.maximum())
        value = field._scrollbar.value()
        outside = QWheelEvent(QPointF(), QPointF(field.mapToGlobal(QPoint())),
                              QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                              Qt.ScrollUpdate, False)
        QApplication.sendEvent(field, outside)
        assert field._scrollbar.value() == value
        assert field.panel.geometry() == panel and field.details.geometry() == details
        assert QRect(source.mapToGlobal(QPoint()), source.size()) == rectangle
        assert workflow.central is source and source.parentWidget() is workflow
        assert workflow.view.width() > 0 and workflow.view.height() > 0
        assert not workflow.view.geometry().intersects(source.geometry())
        qtbot.keyClick(source, Qt.Key_Escape)
        assert page._workflow is None and source.parentWidget() is field
        assert page.path_edit.text() == str(path) and path.read_bytes() == before
    finally:
        page.close()


def test_workflow_without_a_valid_selection_can_arrange_then_recover_its_real_card(
        qtbot, private_ai_profile, isolated_input_root):
    from PySide6.QtTest import QSignalSpy
    from spacr.qt.widgets.home import _EasyHomePage, _EasyWorkflowPage

    preferences, _, _ = private_ai_profile
    preferences.set_reduce_motion(True)
    path = isolated_input_root / 'objects.csv'
    path.write_text('label,area\n1,12\n')
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(640, 650)
    page.show()
    try:
        page.accept_paths([str(path)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
        source = next(card for card in page.field.cards if card._key == 'graph_builder')
        source.click()
        data = page._workflow.data
        actions = list(page._last_result['actions'])
        page._show_suggestions()
        anchor = QRect(source.mapTo(page, QPoint()), source.size())
        source.activated.disconnect(page._select_suggestion)
        workflow = _EasyWorkflowPage(data, actions, 'missing-tool', source, anchor, page)
        page._workflow = workflow
        workflow.setGeometry(page.rect())
        opened = QSignalSpy(workflow.open_requested)
        rectangle = QRect(source.geometry())
        assert workflow.central is None and workflow._chosen_key is None
        workflow._arrange()
        assert source.geometry() == rectangle and opened.count() == 0
        workflow.select('missing-tool')
        assert workflow.central is None and opened.count() == 0
        workflow.select(source._key)
        assert workflow.central is source and workflow._chosen_key == source._key
        assert source.geometry() == anchor and opened.count() == 0
        assert workflow.view.width() > 0 and workflow.view.height() > 0
        assert not workflow.view.geometry().intersects(source.geometry())
        page._show_suggestions()
        assert page._workflow is None and source.parentWidget() is page.field
        assert source.geometry() == source._final_rect and page.path_edit.text() == str(path)
    finally:
        page.close()


def test_compact_workflow_uses_real_space_below_its_unchanged_anchor(
        qtbot, qapp, private_ai_profile, isolated_input_root):
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_reduce_motion(True)
    path = isolated_input_root / 'objects.csv'
    path.write_text('label,area\n1,12\n')
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(340, 650)
    page.show()
    try:
        page.accept_paths([str(path)])
        qtbot.waitUntil(lambda: page._last_result is not None and not page._runner.is_busy(), timeout=5000)
        source = next(card for card in page.field.cards if card._key == 'graph_builder')
        compact_height = source.height() + 2 * page.path_edit.height()
        page.resize(page.width(), compact_height)
        qapp.processEvents()
        rectangle = QRect(source.mapToGlobal(QPoint()), source.size())
        source.click()
        workflow = page._workflow
        assert workflow.central is source and workflow.view._node_side == 144
        assert workflow.view._root_y == 0
        assert workflow.view.y() == source.geometry().bottom() + 1
        assert workflow.view.width() > 0 and workflow.view.height() > 0
        assert page.rect().contains(workflow.view.geometry())
        assert not workflow.view.geometry().intersects(source.geometry())
        assert QRect(source.mapToGlobal(QPoint()), source.size()) == rectangle
        assert page.path_edit.text() == str(path)
        qtbot.keyClick(source, Qt.Key_Escape)
        assert page._workflow is None and source.parentWidget() is page.field
        assert source.geometry() == source._final_rect
    finally:
        page.close()
