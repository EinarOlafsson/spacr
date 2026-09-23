"""Host–Pathogen's Test data action prepares and applies its real sample offline."""

import pytest

pytest.importorskip('PySide6')
pytestmark = pytest.mark.qt


def test_button_prepares_sample_offline_and_fills_runnable_settings(qtbot, qt_theme_applied, tmp_path, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt import assay_examples
    from spacr.host_pathogen_example import is_present

    def refuse_network(*args, **kwargs):
        raise AssertionError('The generated example must never use the network')

    monkeypatch.setattr('requests.get', refuse_network)
    screen = AppScreen('host_pathogen')
    qtbot.addWidget(screen)
    button = screen._assay_example_button
    assert button.text() == 'Load test data…'
    assert 'SYNTHETIC' in button.toolTip()
    assert 'offline' in button.toolTip()
    destination = tmp_path / 'sample'
    assay_examples.load_the_assay_example(screen, folder=destination)
    qtbot.waitUntil(lambda: button.isEnabled(), timeout=15000)
    qtbot.waitUntil(lambda: screen._settings_model.collect()['src'] == str(destination), timeout=5000)
    settings = screen._settings_model.collect()
    assert settings['hp_marker_channels'] == [0, 1]
    assert settings['hp_marker_thresholds'] == {0: 2., 1: 2.}
    assert settings['hp_parasite_table'] == 'organelle'
    assert settings['hp_count_column'] in ('', None)
    assert is_present(destination)
    from spacr.host_pathogen import analyze_host_pathogen

    results = analyze_host_pathogen(settings)
    assert len(results['vacuoles']) == 28
    assert results['cells'].infected.sum() == 20
    assert len(results['orphan_parasites']) == 4
    assert assay_examples.load_the_assay_example(screen, folder=destination,
                                                  ask=refuse_network) == {'src': str(destination)}


def test_worker_cancellation_reports_failure_without_partial_data(qapp, tmp_path):
    from spacr.qt.assay_examples import _HostPathogenExampleWorker

    worker = _HostPathogenExampleWorker(tmp_path / 'sample')
    results = []
    worker.finished.connect(lambda *args: results.append(args))
    worker.cancel()
    worker.run()
    assert len(results) == 1 and results[0][0] is False
    assert 'cancelled' in results[0][-1]
    assert not (tmp_path / 'sample').exists()


@pytest.mark.parametrize('value,expected', [
    ('{0: 2.0, 1: 1.5}', {0: 2., 1: 1.5}),
    ('{}', {}),
    ('[]', '[]'),
    ('{broken', '{broken'),
])
def test_threshold_dictionary_is_typed_without_reinterpreting_invalid_text(value, expected):
    from spacr.qt.screens.settings_model import SettingsWidgets

    assert SettingsWidgets._coerce_to_expected_type('hp_marker_thresholds', value) == expected
