"""Easy input work leaves the live field visible and empty input truly empty."""
import threading

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.mark.parametrize(('theme', 'pending'), [('dark', 'inspection'), ('light', 'regex')])
def test_clearing_easy_input_retires_pending_work_without_guidance(
        qtbot, qapp, private_ai_profile, tmp_path, monkeypatch, theme, pending):
    from spacr.qt import theme as styling
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets import home

    preferences, _, _ = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    preferences.set_theme(theme)
    previous_sheet = qapp.styleSheet()
    qapp.setStyleSheet(styling.stylesheet(theme))
    image = tmp_path / 'scan_101_C1.tif'
    tifffile.imwrite(image, np.arange(1200, dtype=np.uint16).reshape(30, 40))
    original_bytes = image.read_bytes()
    entered, release = threading.Event(), threading.Event()
    blocked = []
    original_inspect = home._easy_inspect_path
    original_regex = home._easy_validate_regex

    def inspect(value):
        if blocked and pending == 'inspection':
            entered.set()
            assert release.wait(5)
        return original_inspect(value)

    def validate(result, pattern, cancelled=None):
        entered.set()
        assert release.wait(5)
        assert cancelled is not None and cancelled()
        return original_regex(result, pattern, cancelled=cancelled)

    monkeypatch.setattr(home, '_easy_inspect_path', inspect)
    monkeypatch.setattr(home, '_easy_validate_regex', validate)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1280, 900)
    window.show()
    page = window._startup
    try:
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        page.path_edit.setFocus()
        qtbot.keyClicks(page.path_edit, str(image))
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        qtbot.waitUntil(lambda: not page._runner.is_busy() and bool(page.field.cards), timeout=10000)
        assert window._stack.currentWidget() is page
        assert page._last_result['path'] == str(image)
        assert any(isinstance(card, home._EasyRegexCard) for card in page.field.cards)
        if pending == 'inspection':
            blocked.append(True)
            qtbot.keyClick(page.path_edit, Qt.Key_Return)
        else:
            card = next(card for card in page.field.cards if isinstance(card, home._EasyRegexCard))
            qtbot.waitUntil(lambda: not card._growing, timeout=7000)
            card.pattern_edit.setPlainText(r'scan_(?P<fieldID>[0-9]+)_C(?P<chanID>[0-9]+)\.tif$')
            card.click()
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert page._runner.is_busy()
        page.path_edit.setFocus()
        page.path_edit.selectAll()
        qtbot.keyClick(page.path_edit, Qt.Key_Backspace)
        assert page.path_edit.text() == ''
        assert not page._runner.is_busy() and not page._debounce.isActive()
        assert not page.field.cards and not page.field._animations
        assert page._last_result is None and page._settings_overrides == {}
        assert page._regex_cancel is None
        assert page.status.text() == '' and not page.status.isVisible()
        assert page.path_edit.accessibleDescription() == '' and page.path_edit.toolTip() == ''
        assert page.path_edit.isVisible() and page._workflow is None
        release.set()
        qtbot.waitUntil(lambda: page._runner.active_jobs() == 0, timeout=10000)
        qtbot.keyClick(page.path_edit, Qt.Key_Return)
        assert not page._debounce.isActive() and not page._runner.active_jobs()
        assert not page.field.cards and page.status.text() == ''
        assert window._stack.currentWidget() is page
        assert image.read_bytes() == original_bytes
    finally:
        release.set()
        window.close()
        qapp.setStyleSheet(previous_sheet)


@pytest.mark.parametrize(('theme', 'input_kind', 'restore'), [
    ('dark', 'file', 'escape'), ('light', 'folder', 'back')])
def test_first_suggestion_keeps_rendered_field_and_restores_anchored_card(
        qtbot, qapp, private_ai_profile, tmp_path, theme, input_kind, restore):
    from PySide6.QtCore import QPoint, QRect
    from spacr.qt import theme as styling
    from spacr.qt.app import MainWindow, install_the_dialog_filters

    preferences, _, _ = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    preferences.set_ambient_theme('data_art_impulse_lens')
    preferences.set_ambient_density(.25)
    preferences.set_theme(theme)
    previous_sheet = qapp.styleSheet()
    qapp.setStyleSheet(styling.stylesheet(theme))
    source = tmp_path / 'images'
    source.mkdir()
    image = source / 'W0025F0001T0001Z000C1.tif'
    tifffile.imwrite(image, np.arange(1200, dtype=np.uint16).reshape(30, 40))
    original = image.read_bytes()
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1280, 900)
    window.show()
    install_the_dialog_filters(qapp)
    page = window._startup

    def field_pixels():
        band = QRect(page.width() - 96, 160, 64, min(400, page.height() - 176))
        corner = page.mapTo(window, band.topLeft())
        rendered = window.grab(QRect(corner, band.size())).toImage()
        colors = np.array([[rendered.pixelColor(x, y).getRgb()[:3]
                            for x in range(rendered.width())]
                           for y in range(rendered.height())], dtype=np.int16)
        return int(((colors[..., 2] > colors[..., 0] + 8)
                    & (colors[..., 2] > colors[..., 1] + 3)).sum())

    try:
        qtbot.waitUntil(lambda: window.window_backdrop() is not None, timeout=5000)
        page.accept_paths([str(image if input_kind == 'file' else source)])
        qtbot.waitUntil(lambda: any(card._key == 'mask' and not card._growing
                                   for card in page.field.cards), timeout=10000)
        card = next(card for card in page.field.cards if card._key == 'mask')
        anchor = QRect(card.mapToGlobal(QPoint()), card.size())
        search_anchor = QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size())
        details_anchor = QRect(page.field.details.mapToGlobal(QPoint()), page.field.details.size())
        backdrop = window.window_backdrop()
        qtbot.waitUntil(lambda: field_pixels() > 20, timeout=5000)
        qtbot.mouseClick(card, Qt.LeftButton)
        workflow = page._workflow
        assert workflow is not None and workflow.central is card
        assert QRect(card.mapToGlobal(QPoint()), card.size()) == anchor
        assert window._stack.currentWidget() is page and page.field.isVisible()
        assert backdrop is window.window_backdrop() and backdrop.isVisible() and backdrop.is_running()
        assert backdrop._engine._node_visit is None
        viewport = workflow.view.viewport()
        free = next(QPoint(x, y) for y in range(24, viewport.height() - 24, 24)
                    for x in range(24, viewport.width() - 24, 24)
                    if workflow.view.itemAt(QPoint(x, y)) is None
                    and workflow.childAt(workflow.mapFromGlobal(viewport.mapToGlobal(QPoint(x, y)))) is viewport
                    and not any(part.isVisible() and QRect(part.mapToGlobal(QPoint()), part.size()).contains(
                        viewport.mapToGlobal(QPoint(x, y))) for part in workflow._departing))
        field_point = backdrop.mapFromGlobal(viewport.mapToGlobal(free))
        assert not backdrop._held_rows._covered().contains(field_point)
        assert page.path_edit.isVisible()
        assert QRect(page.path_edit.mapToGlobal(QPoint()), page.path_edit.size()) == search_anchor
        assert workflow.details is page.field.details
        assert QRect(workflow.details.mapToGlobal(QPoint()), workflow.details.size()) == details_anchor
        assert field_pixels() > 20
        qtbot.waitUntil(lambda: workflow._progress == 1., timeout=5000)
        assert not page.path_edit.isVisible()
        assert backdrop._engine._node_visit is None
        assert field_pixels() > 20
        assert workflow.central.isVisible() and workflow.tools.isVisible()
        assert any(node.isVisible() for node in workflow.view.nodes.values())
        if restore == 'escape':
            qtbot.keyClick(workflow.central, Qt.Key_Escape)
        else:
            qtbot.mouseClick(workflow.back, Qt.LeftButton)
        assert page._workflow is None and page.path_edit.isVisible()
        qtbot.waitUntil(lambda: backdrop._engine._node_visit is None, timeout=5000)
        assert card.parentWidget() is page.field and card.isVisible()
        assert QRect(card.mapToGlobal(QPoint()), card.size()) == anchor
        assert field_pixels() > 20
        assert image.read_bytes() == original
    finally:
        window.close()
        qapp.setStyleSheet(previous_sheet)


def test_grouped_choices_keep_all_tools_readable_and_preserve_first_click(qtbot, private_ai_profile, tmp_path):
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(False)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1080, 850)
    page.show()
    paths = {'images': str(tmp_path / 'images'), 'measurements': str(tmp_path / 'measurements.db')}
    keys = ('mask', 'graph_builder', 'make_masks', 'db_browser', 'convert', 'gate_editor', 'measure', 'regression')
    groups = {key: dict(kind='images' if index % 2 == 0 else 'measurements',
        group='Images' if index % 2 == 0 else 'Measurements',
        reason='Detected microscopy images.' if index % 2 == 0 else 'Detected measurements database.',
        source=paths['images' if index % 2 == 0 else 'measurements'])
        for index, key in enumerate(keys)}
    actions = [(key, key, groups[key]['reason'], groups[key]['source']) for key in keys]
    other_images = str(tmp_path / 'second-plate-images')
    actions.append(('mask', 'Mask', groups['mask']['reason'], other_images))
    page.path_edit.setText(str(tmp_path))
    page._requested = str(tmp_path)
    page._show_result(page._input_generation, dict(path=str(tmp_path), message='', actions=actions,
        context={'action_groups': groups}))
    assert len(page.field._animations) == len(actions)
    assert sum(card._growing for card in page.field.cards) == len(actions)
    page.field._finish_growth()
    assert len(page.field.cards) == len(actions)
    assert [groups[card._key]['kind'] for card in page.field.cards] == ['images'] * 5 + ['measurements'] * 4
    assert len({card._final_rect.y() for card in page.field.cards}) == 1
    assert page.field._scrollbar.orientation() == Qt.Horizontal
    assert page.field._scrollbar.maximum() > 0
    for card in page.field.cards:
        assert card.description.isHidden()
        assert groups[card._key]['reason'] in card.accessibleDescription()
        assert groups[card._key]['group'] in card.accessibleDescription()
        page.field.explain_card(card)
        assert groups[card._key]['reason'] in page.field.details.toPlainText()
        assert groups[card._key]['group'] in page.field.details.toPlainText()
        assert (card._key, card._path) in {(action[0], action[3]) for action in actions}
        assert card.width() == card.height()
    last = page.field.cards[-1]
    assert not last.isVisible()
    page.field._scrollbar.setValue(page.field._scrollbar.maximum())
    qtbot.waitUntil(last.isVisible)
    assert page.field._row_rect.intersects(last.geometry())
    emitted = []
    page.module_requested.connect(lambda key, path: emitted.append((key, path)))
    qtbot.mouseClick(last, Qt.LeftButton)
    assert page._workflow.central is last and not emitted
    qtbot.mouseClick(last, Qt.LeftButton)
    assert emitted == [(last._key, groups[last._key]['source'])]
    page._show_suggestions()
    assert len(page.field.cards) == len(actions) and last.parentWidget() is page.field
    second_plate = next(card for card in page.field.cards if card._path == other_images)
    page.field._scrollbar.setValue(max(0, page.field._scrollbar.value() + second_plate.x() - page.field._row_rect.left()))
    qtbot.waitUntil(second_plate.isVisible)
    qtbot.mouseClick(second_plate, Qt.LeftButton)
    assert page._workflow.central is second_plate
    qtbot.mouseClick(second_plate, Qt.LeftButton)
    assert emitted[-1] == ('mask', other_images)
    page.close()


def test_whole_window_drop_preserves_every_frame_during_inspection(
        qtbot, qapp, private_ai_profile, tmp_path, monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent, QMimeData, QObject, QPoint, QPointF, QRect, QTimer, QUrl
    from PySide6.QtWidgets import QGraphicsOpacityEffect
    from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage
    from spacr.qt import theme as styling
    from spacr.qt.app import MainWindow, install_the_dialog_filters
    from spacr.qt.widgets import home

    preferences, _, _ = private_ai_profile
    preferences._set_spacr_ai_enabled(False)
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    preferences.set_ambient_theme('data_art_impulse_lens')
    preferences.set_theme('dark')
    previous_sheet = qapp.styleSheet()
    qapp.setStyleSheet(styling.stylesheet('dark'))
    tifffile.imwrite(tmp_path / 'W0025F0001T0001Z000C1.tif', np.zeros((10, 12), np.uint16))
    entered, release = threading.Event(), threading.Event()
    original = home._easy_inspect_path
    inspected = []
    def held(value):
        inspected.append(value)
        entered.set()
        assert release.wait(5)
        return original(value)
    monkeypatch.setattr(home, '_easy_inspect_path', held)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1280, 900)
    window.show()
    install_the_dialog_filters(qapp)
    page = window._startup
    def visible_field_pixels():
        area = QRect(page.width() - 96, 160, 64, 160)
        image = window.grab(QRect(page.mapTo(window, area.topLeft()), area.size())).toImage()
        image = image.convertToFormat(QImage.Format_RGBA8888)
        colors = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
        colors = colors[:, :image.width() * 4].reshape(image.height(), image.width(), 4).astype(np.int16)
        return int(((colors[..., 2] > colors[..., 0] + 8)
                    & (colors[..., 2] > colors[..., 1] + 3)).sum())
    try:
        qtbot.waitUntil(lambda: window.window_backdrop() is not None and visible_field_pixels() > 20,
                        timeout=5000)
        backdrop = window.window_backdrop()
        restyles = []
        class Restyles(QObject):
            def eventFilter(self, watched, event):
                if event.type() == QEvent.StyleChange:
                    restyles.append(event.type())
                return False
        observer = Restyles(page)
        page.installEventFilter(observer)
        def drop_path(path):
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(path))])
            enter = QDragEnterEvent(QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            qapp.sendEvent(window, enter)
            drop = QDropEvent(QPointF(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            qapp.sendEvent(window, drop)
            assert enter.isAccepted() and drop.isAccepted()
        drop_path(tmp_path)
        qtbot.waitUntil(entered.is_set)
        frames = []
        for _ in range(24):
            qtbot.wait(25)
            frames.append(visible_field_pixels())
        assert min(frames) > 20, frames
        assert not restyles, "Dropping data must not repolish the unchanged live Home surface"
        assert backdrop is window.window_backdrop() and backdrop.isVisible() and backdrop.is_running()
        assert page.path_edit.text() == str(tmp_path) and page._runner.is_busy()
        for number in range(3):
            latest = tmp_path / f'replacement-{number}'
            latest.mkdir()
            tifffile.imwrite(latest / 'W0025F0001T0001Z000C1.tif', np.zeros((10, 12), np.uint16))
            drop_path(latest)
            assert page._last_result is None and not page.field.cards
            assert page._runner.active_jobs() <= 1
            for _ in range(4):
                qtbot.wait(25)
                frames.append(visible_field_pixels())
        assert min(frames) > 20
        assert page.path_edit.text() == str(latest)
        release.set()
        qtbot.waitUntil(lambda: not page._runner.is_busy() and bool(page.field.cards), timeout=5000)
        assert visible_field_pixels() > 20
        assert inspected == [str(tmp_path), str(latest)]
        assert page._last_result['path'] == str(latest)
        timers = len(page.findChildren(QTimer))
        for _ in range(3):
            previous = tuple(page.field.cards)
            drop_path(latest)
            qtbot.waitUntil(lambda: not page._runner.is_busy() and bool(page.field.cards)
                            and tuple(page.field.cards) != previous, timeout=5000)
            for _ in range(4):
                qtbot.wait(25)
                assert visible_field_pixels() > 20
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            assert len(page.field.findChildren(home._EasySuggestionCard)) == len(page.field.cards)
            assert len(page.field.findChildren(QGraphicsOpacityEffect)) == len(page.field.cards) + 1
            assert len(page.field._animations) <= len(page.field.cards)
            assert len(page.findChildren(QTimer)) == timers
            assert page._runner.active_jobs() <= 1
        assert not restyles
    finally:
        release.set()
        qtbot.waitUntil(lambda: page._runner.active_jobs() == 0, timeout=5000)
        window.close()
        qapp.setStyleSheet(previous_sheet)


def test_selected_node_lens_visits_brightens_and_resumes_existing_walk(qapp):
    from PySide6.QtGui import QImage
    from spacr.qt.widgets import ambient
    options = dict(seed=684, density=1, blink_percent=0, popup_wave_frequency=0)
    engine = ambient.make_engine('data_art_impulse_lens', 'spacr', '#191919', **options)
    control = ambient.make_engine('data_art_impulse_lens', 'spacr', '#191919', **options)
    target = (.17, .78)
    engine._node_visit = (0., target)
    def pixels(image):
        image = image.convertToFormat(QImage.Format_RGBA8888)
        return np.frombuffer(image.constBits(), np.uint8).copy().reshape(image.height(), image.bytesPerLine())
    assert engine._field_center()[2] == 0.
    np.testing.assert_array_equal(pixels(engine.shade(320, 200)), pixels(control.shade(320, 200)))
    engine.time = control.time = 3.
    assert engine._field_center() == pytest.approx((*target, 1.))
    visited = pixels(engine.shade(320, 200))
    wandering = pixels(control.shade(320, 200))
    assert visited.mean() > wandering.mean()
    assert not np.array_equal(visited, wandering)
    engine.time = control.time = 6.
    assert engine._field_center() == control._field_center()
    assert engine._node_visit is None
    np.testing.assert_array_equal(pixels(engine.shade(320, 200)), pixels(control.shade(320, 200)))


def test_network_hover_paints_immediately_without_icon_reads_or_restyles(
        qtbot, private_ai_profile, tmp_path, monkeypatch):
    import time
    from spacr.qt.widgets import home
    from spacr.qt import iconset
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_show_alpha(True)
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 900)
    page.show()
    page.field.set_cards([('mask', 'Mask', 'Detected image.', str(tmp_path))])
    page.field.cards[0].activated.connect(page._select_suggestion)
    page.field.cards[0].click()
    workflow = page._workflow
    paints = []
    original = home._EasyWorkflowNode.paint
    def paint(node, *args):
        paints.append((node.key, node._hovered, time.perf_counter()))
        return original(node, *args)
    monkeypatch.setattr(home._EasyWorkflowNode, 'paint', paint)
    def forbidden(*args, **kwargs):
        raise AssertionError('Hover painting must reuse the resolved node icon')
    monkeypatch.setattr(iconset, 'app_icon', forbidden)
    latencies = []
    qtbot.wait(50)
    before_sheet = page.styleSheet()
    for node in (node for node in workflow.view.nodes.values() if node.isVisible()):
        point = workflow.view.mapFromScene(node.sceneBoundingRect().center())
        if not workflow.view.viewport().rect().contains(point):
            continue
        paints.clear()
        started = time.perf_counter()
        qtbot.mouseMove(workflow.view.viewport(), point)
        qtbot.waitUntil(lambda: any(key == node.key and hovered for key, hovered, _when in paints), timeout=500)
        painted = next(when for key, hovered, when in paints if key == node.key and hovered)
        latencies.append(painted - started)
        assert node._hovered
    assert latencies and max(latencies) < .5
    assert page.styleSheet() == before_sheet
    assert workflow.details is page.field.details
    assert workflow.details.width() == min(780, page.field.width() - 32)
    assert workflow.details.height() == 88 + 2 * workflow.details.fontMetrics().lineSpacing()
    print('hover-to-paint seconds:', latencies)
    page.close()


def test_alpha_cards_and_nodes_share_registered_stage_and_old_home_feedback(
        qtbot, qapp, private_ai_profile):
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QImage, QPainter
    from spacr.qt import theme
    from spacr.qt.widgets import home
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_show_alpha(True)
    previous = qapp.styleSheet()
    qapp.setStyleSheet(theme.stylesheet('dark'))
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 900)
    page.show()
    try:
        page.field.set_cards([('convert', 'Import', 'Detected image.', '')])
        card = page.field.cards[0]
        assert card.property('stage') == 'alpha' and card.alpha.isVisible()
        assert card.alpha.text() == theme.ALPHA_MARK
        assert card.alpha.width() == 40 and card.alpha.height() == 40
        assert card.alpha.font().pixelSize() == theme.font_px(28)
        assert card.alpha.geometry().right() < card.width()
        assert theme._stage_button_rules('QPushButton#EasySuggestionCard') in page._easy_stylesheet
        assert theme._stage_button_rules('QPushButton#AppTile') in theme.stylesheet('dark')
        qtbot.mouseMove(page, QPoint(0, 0))
        icon_origin = card.icon.mapTo(card, QPoint())
        composite = card.grab().toImage()
        assert composite.pixelColor(icon_origin).getRgb()[:3] == (25, 25, 25)
        card.activated.connect(page._select_suggestion)
        card.click()
        workflow = page._workflow
        node = workflow.view.nodes['convert']
        assert node._stage == 'alpha'
        rendered = QImage(192, 192, QImage.Format_ARGB32_Premultiplied)
        rendered.fill(0)
        painter = QPainter(rendered)
        node.paint(painter, None)
        painter.end()
        teal_rows = {y for x in range(142, 184) for y in range(8, 48)
                     for color in (rendered.pixelColor(x, y),)
                     if color.green() > color.red() + 20 and color.blue() > color.red() + 20
                     and abs(color.green() - color.blue()) < 10}
        assert len(teal_rows) > 14
        assert rendered.pixelColor(56, 28).getRgb()[:3] == (25, 25, 25)
    finally:
        page.close()
        qapp.setStyleSheet(previous)


def test_explicit_stack_dimension_card_preserves_input_through_two_clicks(qtbot, private_ai_profile, tmp_path):
    from spacr.qt.widgets import home
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    path = tmp_path / 'W0025F0001.tif'
    tifffile.imwrite(path, np.zeros((2, 3, 8, 10), np.uint16),
                     metadata={'axes': 'TZYX'}, photometric='minisblack')
    original = path.read_bytes()
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 900)
    page.show()
    page.accept_paths([str(path)])
    qtbot.waitUntil(lambda: any(card._key == '3d' for card in page.field.cards), timeout=5000)
    card = next(card for card in page.field.cards if card._key == '3d')
    page.field._scrollbar.setValue(max(0, page.field._scrollbar.value() + card.x() - page.field._row_rect.left()))
    qtbot.waitUntil(card.isVisible)
    chosen = []
    page.module_requested.connect(lambda key, source: chosen.append((key, source)))
    qtbot.mouseClick(card, Qt.LeftButton)
    assert page._workflow is not None and page._workflow.central is card and not chosen
    assert page._workflow.data['modules']['3d']['route'] == 'dimension'
    qtbot.mouseClick(card, Qt.LeftButton)
    assert chosen == [('3d', str(path))]
    assert path.read_bytes() == original
    page.close()


def test_help_network_links_are_navigation_without_invented_inputs(qtbot, private_ai_profile, tmp_path):
    from spacr.qt.app import _HELP_MODULES, app_is_visible
    from spacr.qt.widgets import home
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_show_alpha(True)
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1280, 900)
    page.show()
    page.field.set_cards([('mask', 'Mask', 'Detected image.', str(tmp_path))])
    page.field.cards[0].activated.connect(page._select_suggestion)
    page.field.cards[0].click()
    workflow = page._workflow
    available = {key for key, _title, _tip in _HELP_MODULES if app_is_visible(key)}
    records = workflow.view.branch_records + workflow.view.remaining_records
    assert available <= {record['to'] for record in records}
    assert all(record['kind'] != 'utility' for record in workflow.view.branch_records)
    utilities = [record for record in records if record['kind'] == 'utility']
    assert utilities and all(not record['artifacts'] and not record['handoff'] for record in utilities)
    assert all(record['mode_labels'] == ['Tools'] for record in utilities)
    assert 'run_history' not in workflow.routes
    workflow.select('run_history')
    assert workflow._chosen_key == 'run_history'
    assert workflow.central._key == 'mask' and workflow.central._path == str(tmp_path)
    assert not any(record['kind'] == 'utility' and record['from'] in available
                   for record in workflow.data['easy_connections'])
    opened = []
    page.module_requested.connect(lambda key, path: opened.append((key, path)))
    child = workflow.view.nodes['run_history']
    workflow.view.ensureVisible(child)
    qtbot.mouseClick(workflow.view.viewport(), Qt.LeftButton,
                     pos=workflow.view.mapFromScene(child.sceneBoundingRect().center()))
    assert opened == [('run_history', '')]
    page.close()


@pytest.mark.parametrize('dock_choice', [None, False, True])
def test_navigator_default_dock_is_collapsed_present_and_saved_choices_win(
        qtbot, private_ai_profile, dock_choice):
    from spacr.qt.app import MainWindow
    preferences, _, _ = private_ai_profile
    settings = preferences._settings()
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    assert not settings.contains('prefs/easy_mode_dock')
    if dock_choice is not None:
        preferences.set_easy_mode_dock(dock_choice)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    edge = window._dock_edge
    try:
        assert edge.isVisible()
        assert edge.is_collapsed()
        assert not window._dock_slot.isVisible()
        assert settings.contains('prefs/easy_mode_dock') == (dock_choice is not None)
        qtbot.mouseClick(edge, Qt.LeftButton)
        assert window._dock_slot.isVisible()
        assert settings.value('prefs/navigator_dock_collapsed', type=bool) is False
        window.apply_dock_mode()
        assert not edge.is_collapsed() and window._dock_slot.isVisible()
        preferences.set_easy_mode(False)
        window.apply_dock_mode('locked')
        assert not edge.is_collapsed() and window._dock_slot.isVisible()
    finally:
        window.close()


def test_navigator_native_runtime_defaults_preserve_headings_user_expansion_and_normal_mode(
        qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow
    preferences, _, _ = private_ai_profile
    settings = preferences._settings()
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1280, 900)
    window.show()
    try:
        window._open_easy_named_module('mask')
        screen = window._screens['mask']
        split = screen._runtime_splitter
        panes = [split.pane(name) for name in ('Console', 'Actions', 'System')]
        assert all(pane is not None for pane in panes)
        assert all(pane.widget.isVisible() and pane.folder.heading.isVisible()
                   and pane.folder.shut and not pane.folder.body.isVisible() for pane in panes)
        assert screen._console is not None
        untouched = {pane.folder: screen._navigator_pane_states[pane.folder]['original']
                     for pane in panes if pane.name != 'Console'}
        console = split.pane('Console')
        qtbot.mouseClick(console.folder.heading, Qt.LeftButton)
        assert not console.folder.shut and console.folder.body.isVisible()
        saved = 'ui/navigator/panes/mask::runtime/Console'
        assert settings.contains(saved) and settings.value(saved, type=bool) is False
        window._on_nav_selected('__home__')
        window._open_easy_named_module('mask')
        assert not console.folder.shut
        preferences.set_easy_mode(False)
        window.apply_dock_mode()
        assert not console.folder.shut
        assert all(folder.shut == original for folder, original in untouched.items())
        preferences.set_easy_mode(True)
        window.apply_dock_mode()
        assert not console.folder.shut
        assert all(pane.folder.shut for pane in panes if pane.name != 'Console')
    finally:
        window.close()


@pytest.mark.parametrize('theme_name', ['dark', 'light'])
def test_navigator_description_uses_home_panel_opacity_with_rgb25(
        qtbot, private_ai_profile, theme_name):
    from spacr.qt import theme
    from spacr.qt.widgets import home
    preferences, _, _ = private_ai_profile
    preferences.set_theme(theme_name)
    preferences.set_pane_opacity(.60)
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    try:
        alpha = theme.panel_alpha(theme_name, 'surface_alt', preferences.get_pane_opacity())
        expected = theme.css_color('#191919', alpha)
        assert expected.startswith('rgba(25, 25, 25, ')
        assert 'background: ' + expected + ';' in page._easy_stylesheet
        same_alpha = theme.pane_surface('surface_alt', theme_name).rsplit(',', 1)[1]
        assert expected.rsplit(',', 1)[1] == same_alpha
    finally:
        page.close()


def test_navigator_preserves_saved_native_runtime_expansions(qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets.collapsible_splitter import set_pane_extents
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    preferences.set_folded_panel('mask/Actions', False, default_shut=True)
    set_pane_extents('mask::runtime', {'Console': 300})
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.show()
    try:
        window._open_easy_named_module('mask')
        split = window._screens['mask']._runtime_splitter
        assert not split.pane('Actions').folder.shut
        assert not split.pane('Console').folder.shut
        assert split.pane('System').folder.shut
    finally:
        window.close()
