"""Host–Pathogen's Test data action downloads and applies its real microscopy sample."""

import pytest

pytest.importorskip('PySide6')
pytestmark = pytest.mark.qt


def test_real_sample_uses_separate_cache_and_shared_download_worker(qapp, tmp_path, monkeypatch):
    from spacr.qt import assay_examples
    from spacr.example_archives import example_set, example_set_folder
    from spacr.host_pathogen_example import example_folder

    chosen = example_set('host_pathogen')
    assert chosen.repo == 'einarolafsson/spacr-example-host-pathogen'
    assert example_set_folder('host_pathogen') != example_folder()
    assert 'real THP-1' in assay_examples._tooltip('host_pathogen')
    assert 'replication remains unknown' in assay_examples._tooltip('host_pathogen')
    seen = {}

    def dialog(parent, dest, on_done, *, worker_factory, title):
        seen['worker'] = worker_factory(dest)
        seen['title'] = title

    monkeypatch.setattr(assay_examples, 'download_toxo_mito_demo', dialog)
    assay_examples.download_assay_example(None, 'host_pathogen', tmp_path, None)
    assert seen['worker'].repo == chosen.repo
    assert 'synthetic' not in seen['title'].lower()


def test_partial_real_sample_requires_download_and_complete_sample_is_cached(
        qtbot, qt_theme_applied, tmp_path):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt import assay_examples
    from spacr.example_archives import example_set
    from tests.qt.test_the_assay_modules_offer_test_data import _unpack_a_published_copy

    screen = AppScreen('host_pathogen')
    qtbot.addWidget(screen)
    folder = _unpack_a_published_copy(tmp_path / 'sample', 'host_pathogen')
    chosen = example_set('host_pathogen')
    assert chosen.is_present(folder)

    def refuse_download(*args):
        raise AssertionError('A complete cached example must not download again')

    result = assay_examples.load_the_assay_example(screen, folder=folder, ask=refuse_download)
    assert result == {'src': str(folder)}
    assert screen._settings_model.collect()['hp_marker_channels'] == [1]
    (folder / 'merged/PLATE1_E02_1_1.npy').unlink()
    assert not chosen.is_present(folder)
    calls = []
    assay_examples.load_the_assay_example(screen, folder=folder,
        ask=lambda *args: calls.append(args))
    assert len(calls) == 1


@pytest.mark.parametrize('value,expected', [
    ('{0: 2.0, 1: 1.5}', {0: 2., 1: 1.5}),
    ('{}', {}),
    ('[]', '[]'),
    ('{broken', '{broken'),
])
def test_threshold_dictionary_is_typed_without_reinterpreting_invalid_text(value, expected):
    from spacr.qt.screens.settings_model import SettingsWidgets

    assert SettingsWidgets._coerce_to_expected_type('hp_marker_thresholds', value) == expected
