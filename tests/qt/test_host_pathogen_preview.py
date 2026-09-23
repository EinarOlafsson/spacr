"""Real preview loading, selection, current form settings and stale-work safety."""

import threading

import pytest

pytest.importorskip('PySide6')
pytestmark = pytest.mark.qt

from spacr.host_pathogen_example import build_example
from spacr.host_pathogen import default_settings
from spacr.qt.settings_pack import settings_from_pack
from spacr.qt.widgets.host_pathogen_preview import HostPathogenPreviewPanel


@pytest.fixture
def config(tmp_path):
    folder = build_example(tmp_path / 'sample')
    settings, _ = settings_from_pack('host_pathogen', folder / 'settings', defaults=default_settings())
    return settings


def test_preview_loads_selects_vacuoles_and_preserves_zoom(qtbot, config):
    panel = HostPathogenPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.resize(850, 650)
    panel.show()
    panel.apply_settings(config)
    assert panel.run_preview()
    assert panel._table.rowCount() == 7
    assert panel._field.count() == 4
    assert '6 hosts' in panel._summary.text()
    assert 'API</a>' in panel._details.text()
    panel._view.scale(2, 2)
    old = panel._view.transform()
    panel._table.selectRow(3)
    assert panel._selected == 4
    assert 'parasites 8' in panel._details.text()
    assert panel._view.transform() == old
    panel._hover_pixel(73, 75)
    panel._image_clicked()
    assert panel._selected == 1
    panel._field.setCurrentIndex(2)
    assert panel._result['field']['identity']['columnID'] == 'c2'
    assert 'positive' in panel._details.text()


def test_registered_preview_is_lazy_and_uses_current_form(qtbot, qt_theme_applied, config):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window._on_nav_selected('host_pathogen')
    screen = window._screens['host_pathogen']
    qtbot.waitUntil(lambda: getattr(screen, '_registry_preview', None) is not None, timeout=5000)
    screen.apply_settings_dict(config)
    host = screen._registry_preview
    assert not host.panel_is_built()
    host.toggle.setChecked(True)
    panel = host.panel
    assert panel.run_preview()
    qtbot.waitUntil(lambda: panel._result is not None, timeout=10000)
    assert panel._table.rowCount() == 7
    screen._settings_model.set_value_for_key('hp_marker_thresholds', {0: .5, 1: 2})
    qtbot.waitUntil(lambda: panel._jobs.active_jobs() == 0, timeout=5000)
    assert panel.run_preview()
    qtbot.waitUntil(lambda: panel._result is not None, timeout=10000)
    assert panel._result['results']['vacuoles'].channel_0_state.iloc[0] == 'positive'
    panel.shutdown()


@pytest.mark.parametrize('failure', [False, True])
def test_cancel_discards_late_success_and_error(qtbot, monkeypatch, config, failure):
    import spacr.host_pathogen_preview as engine

    panel = HostPathogenPreviewPanel(threaded=True)
    qtbot.addWidget(panel)
    panel.apply_settings(config)
    entered, release = threading.Event(), threading.Event()
    original = engine.preview_field

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if failure:
            raise ValueError('obsolete failure')
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, 'preview_field', blocked)
    try:
        assert panel.run_preview()
        qtbot.waitUntil(entered.is_set, timeout=5000)
        assert panel.cancel_preview()
        release.set()
        qtbot.waitUntil(lambda: panel._jobs.active_jobs() == 0, timeout=5000)
        assert panel._result is None
        assert 'cancelled' in panel.preview_status().lower()
        assert panel._run_btn.isEnabled()
    finally:
        release.set()
        panel.shutdown()


def test_visible_preview_updates_after_form_changes(qtbot, config):
    panel = HostPathogenPreviewPanel(threaded=False, settings_reader=lambda: config)
    qtbot.addWidget(panel)
    panel.show()
    assert panel.run_preview()
    config['hp_marker_thresholds'] = {0: .5, 1: 2}
    qtbot.waitUntil(lambda: panel._result is not None and
                    panel._result['results']['vacuoles'].channel_0_state.iloc[0] == 'positive', timeout=3000)
    panel.shutdown()


def test_missing_source_and_database_errors_recover_on_load(qtbot, config, tmp_path):
    panel = HostPathogenPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel._image_clicked()
    assert not panel.run_preview()
    assert 'source' in panel.preview_status()
    panel.apply_settings(dict(config, src=str(tmp_path / 'absent')))
    assert panel.run_preview()
    assert 'Preview failed' in panel.preview_status()
    assert panel._result is None and panel._run_btn.isEnabled()
    assert panel.load_source_async(config['src'])
    assert panel._table.rowCount() == 7
    panel.close()
    assert not panel._settings_timer.isActive()


def test_display_controls_and_missing_images_leave_table_usable(qtbot, config):
    from pathlib import Path

    panel = HostPathogenPreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.apply_settings(config)
    panel.run_preview()
    panel._overlays['host'].setChecked(False)
    panel._hover_pixel(-1, -1)
    panel._image_clicked()
    assert panel._selected == 1
    panel._hover_pixel(0, 0)
    panel._image_clicked()
    assert panel._selected == 1
    panel._channel.setValue(1)
    assert panel._result is None and 'Display plane changed' in panel.preview_status()
    panel._planes['vacuole'].setValue(-1)
    panel.run_preview()
    assert 'vacuole' not in panel._result['masks']
    panel._image_clicked()
    panel._table.selectRow(4)
    assert 'unknown' in panel._details.text()
    for image in (Path(config['src']) / 'merged').glob('*.npy'):
        image.unlink()
    panel.run_preview()
    assert panel._result['image'] is None
    assert panel._table.rowCount() == 7
    assert 'No matching merged image' in panel.preview_status()
    panel._table.selectRow(3)
    assert 'parasites 8' in panel._details.text()
    panel.shutdown()
