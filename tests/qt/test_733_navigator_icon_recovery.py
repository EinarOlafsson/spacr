"""A nullable registered icon does not disable native Navigator choices."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtTest import QSignalSpy

from tests.qt.test_733_easy_mode_home import (
    _bytes, _image, isolated_input_root as isolated_input_root,
    trace_navigator_native_input_jobs as trace_navigator_native_input_jobs,
)
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile

pytestmark = pytest.mark.qt


def test_a_nullable_icon_keeps_native_graph_labels_ports_and_explicit_module_choice(
        qtbot, qapp, private_ai_profile, isolated_input_root, monkeypatch):
    from spacr.qt import app
    from spacr.qt.walkthrough import mark_seen
    from spacr.qt.i18n import tr

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(False)
    preferences.set_reduce_motion(True)
    mark_seen('make_masks')
    image = isolated_input_root / 'field.tif'
    pixels = _image(image)
    before = _bytes([image])
    entries = set(isolated_input_root.iterdir())
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1200, 900)
    window.show()
    page = window._startup
    selected = QSignalSpy(page.module_requested)
    try:
        page.accept_paths([str(image)])
        qtbot.waitUntil(lambda: not page._runner.is_busy() and any(
            card._key == 'make_masks' for card in page.field.cards), timeout=5000)
        source = next(card for card in page.field.cards if card._key == 'make_masks')
        resolved = []

        def absent_icon(key, *, theme=None):
            resolved.append(key)
            return None

        with monkeypatch.context() as nullable:
            nullable.setattr(app, '_icon_for_app', absent_icon)
            source.click()
        workflow = page._workflow
        assert workflow is not None and workflow.central is source
        assert source.accessibleName() and set(workflow.view.nodes) <= set(resolved)
        assert selected.count() == 0
        node = next(node for node in workflow.view.nodes.values() if node.isVisible())
        workflow.view.ensureVisible(node)
        qapp.processEvents()
        assert node._input_ports and node.boundingRect().contains(node._input_ports[0])
        frame = workflow.view.viewport().grab().toImage()
        point = workflow.view.mapFromScene(node.mapToScene(node._input_ports[0]))
        nearby = workflow.view.mapFromScene(node.mapToScene(node._input_ports[0].x() + 12,
                                                           node._input_ports[0].y() - 16))
        ratio = frame.devicePixelRatio()
        point, nearby = point * ratio, nearby * ratio
        assert not frame.isNull() and frame.rect().contains(point) and frame.rect().contains(nearby)
        assert frame.pixelColor(point) != frame.pixelColor(nearby)
        qtbot.mouseMove(workflow.view.viewport(),
                        workflow.view.mapFromScene(node.sceneBoundingRect().center()))
        assert tr(workflow.data['modules'][node.key]['name']) in page.field.details.toPlainText()
        qtbot.mouseMove(workflow.back, workflow.back.rect().center())
        assert workflow._chosen_key == 'make_masks' and selected.count() == 0
        window.resize(1000, 780)
        qapp.processEvents()
        assert not workflow.view.viewport().grab().isNull()
        assert workflow.central is source and page.path_edit.text() == str(image)
        assert selected.count() == 0 and _bytes([image]) == before
        workflow.central.click()
        qtbot.waitUntil(lambda: 'make_masks' in window._screens, timeout=5000)
        receiver = window._screens['make_masks']
        assert selected.count() == 1 and selected.at(0) == ['make_masks', str(image)]
        assert window._stack.currentWidget() is receiver
        qtbot.waitUntil(lambda: bool(receiver._image_files), timeout=5000)
        assert (Path(receiver._folder) / receiver._image_files[0]).resolve() == image
        assert _bytes([image]) == before and set(isolated_input_root.iterdir()) == entries
        assert np.array_equal(np.asarray(Image.open(image)), pixels)
    finally:
        window.close()
