"""Missing Plaque checkpoints need consent and never substitute stock weights."""
from types import SimpleNamespace

import pytest
from PIL import Image
from PySide6.QtWidgets import QMessageBox, QWidget

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widgets import plaque_preview as ppv


@pytest.mark.parametrize('accept', [False, True])
def test_missing_preview_model_asks_before_download_and_does_not_segment(
        qtbot, monkeypatch, tmp_path, accept):

    monkeypatch.setattr(ppv, 'missing_papers_packages', lambda: [])
    panel = ppv.PlaquePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    source = tmp_path / 'well.tif'
    Image.new('RGB', (40, 50), 'white').save(source)
    assert panel.load_source_async(source)
    entry = SimpleNamespace(key='toxoplasma_plaque_v6', name='v6.CP_model')
    monkeypatch.setattr(ppv, 'resolve_plaque_model', lambda settings: ('', 'missing v6', entry))
    monkeypatch.setattr(ppv, '_cellpose_model', lambda *args: pytest.fail('missing model reached Cellpose'))
    asked, started = [], []

    def answer(parent, title, text, buttons, default):
        assert parent is panel and 'toxoplasma_plaque_v6' in text
        assert buttons == QMessageBox.Yes | QMessageBox.No
        assert default == QMessageBox.No
        asked.append(text)
        return QMessageBox.Yes if accept else QMessageBox.No

    monkeypatch.setattr(QMessageBox, 'question', answer)
    monkeypatch.setattr(panel, '_start_download', lambda: started.append(entry))
    try:
        assert panel.run_preview()
        assert asked and started == ([entry] if accept else [])
        assert panel._missing_entry is entry and not panel._download_btn.isHidden()
        assert panel._plaque_result is None
        assert panel._run_btn.isEnabled()
    finally:
        panel.close()


@pytest.mark.parametrize('accept', [False, True])
def test_batch_missing_model_preflight_is_read_only_and_requires_consent(
        qtbot, monkeypatch, tmp_path, accept):
    from spacr import submodules
    from spacr.qt import model_install

    parent = QWidget()
    qtbot.addWidget(parent)
    requests = []

    def missing(settings, fetch=True):
        requests.append(fetch)
        assert fetch is False
        raise submodules.ModelZooMissing(str(tmp_path / 'v6.CP_model'))

    monkeypatch.setattr(submodules, '_resolve_plaque_model', missing)
    monkeypatch.setattr(model_install, '_confirm_checkpoint_download',
                        lambda owner, name: accept if name == 'toxoplasma_plaque_v6' else pytest.fail(name))
    assert AppScreen._confirm_plaque_model_download(parent, {}) is accept
    assert requests == [False] and not (tmp_path / 'v6.CP_model').exists()


def test_cancelled_batch_consent_stops_before_project_claim_or_launch(qtbot, monkeypatch):
    from spacr import submodules
    from spacr.qt import model_install
    from spacr.qt.screens import app_screen

    parent = QWidget()
    qtbot.addWidget(parent)
    parent.app_key = 'analyze_plaques'
    parent._settings_model = SimpleNamespace(collect=lambda: {})
    parent._wait_for_the_entry_import = lambda override: False
    parent._confirm_plaque_model_download = lambda settings: AppScreen._confirm_plaque_model_download(parent, settings)
    parent._claim_the_project = lambda settings: pytest.fail('cancelled download claimed a project')
    monkeypatch.setattr(app_screen, 'resolve_pipeline_entry', lambda key: lambda settings: pytest.fail('cancelled run started'))
    monkeypatch.setattr(submodules, '_resolve_plaque_model', lambda *args, **kwargs:
                        (_ for _ in ()).throw(submodules.ModelZooMissing('v6')))
    monkeypatch.setattr(model_install, '_confirm_checkpoint_download', lambda *args: False)
    AppScreen._on_run(parent)


def test_cached_model_needs_no_confirmation_and_unknown_model_stays_blocked(
        qtbot, monkeypatch, tmp_path):
    from spacr.qt import model_install

    parent = QWidget()
    qtbot.addWidget(parent)
    model = tmp_path / 'custom.CP_model'
    model.write_bytes(b'existing model')
    monkeypatch.setattr(model_install, '_confirm_checkpoint_download', lambda *args: pytest.fail('cached model prompted'))
    assert AppScreen._confirm_plaque_model_download(parent, {'plaque_model': str(model)})
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args))
    assert not AppScreen._confirm_plaque_model_download(parent, {'plaque_model': 'not-a-catalogue-model'})
    assert warnings and 'not-a-catalogue-model' in warnings[0][-1]
    assert AppScreen._confirm_plaque_model_download(parent, {'colony_counting': True})
