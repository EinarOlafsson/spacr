"""Backend prose translates at the Qt boundary; package output stays verbatim."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QTextBrowser

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ('de', 'sv', 'es', 'fr', 'pt', 'zh_CN', 'hi', 'ko', 'is')


@pytest.mark.parametrize('language', LANGUAGES)
def test_installer_and_backend_cards_use_current_reviews(
        qtbot, qt_theme_applied, monkeypatch, tmp_path, language):
    import spacr._segmentation_backends as backends
    from spacr.qt import i18n
    from spacr.qt.widgets import model_zoo_picker as picker

    payload = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language /
                          '2026-09-23-backend-installer.json').read_text())
    targets = {row['source']: row['translation'] for row in payload['records']}
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    environment = str(tmp_path / 'backend <isolated>')
    monkeypatch.setattr(backends, '_backend_state', lambda name:
                        backends._BackendState(name, backends._INSTALLABLE,
                                               'fixture diagnostic', environment))

    def forbidden_job(**kwargs):
        raise AssertionError('Rendering translations must not start installation')

    dialog = picker.BackendInstallDialog('spotnet', job=forbidden_job)
    qtbot.addWidget(dialog)
    spec = backends._SPECS['spotnet']
    name = targets['SpotNet (DeepCell)']
    assert dialog.windowTitle() == targets['Install {name}'].format(name=name)
    assert dialog.start_button.text() == targets['Install']
    assert dialog.cancel_button.text() == i18n.tr('Cancel')
    notice = next(s for s in targets if s.startswith('It installs into an environment'))
    assert dialog.blurb.text() == '\n\n'.join((
        targets[spec.blurb], targets[notice].format(
            environment=environment, packages=', '.join(spec.torch + spec.requirements),
            size_gb=spec.size_gb),
        targets['Licence: {licence}'].format(licence=targets[spec.licence_note])))
    raw_output = 'Downloading pip-25.1-py3-none-any.whl'
    dialog._on_progress(1, 5, 'Update pip: ' + raw_output)
    assert dialog.status.text() == targets['Update pip'] + ': ' + raw_output
    assert dialog.progress.format() == targets['step {step} of {steps}'].format(step=2, steps=5)
    dialog._on_progress(2, 5, 'Install SpotNet (DeepCell): ' + raw_output)
    assert dialog.status.text() == targets['Install {name}'].format(name=name) + ': ' + raw_output
    diagnostic = 'ERROR: package <probe> returned status 17; /tmp/env/lib'
    dialog._on_failed(diagnostic)
    assert dialog.details.toPlainText() == diagnostic
    assert dialog.status.text() == targets[
        'Installing {name} failed. Nothing was left half-built.'].format(name=name)
    assert dialog.start_button.text() == targets['Try again']
    dialog._on_cancelled()
    assert dialog.status.text() == targets['Cancelled. Nothing was left behind.']
    dialog.close()

    uninstall = picker.BackendInstallDialog('spotnet', uninstall=True, job=forbidden_job)
    qtbot.addWidget(uninstall)
    assert uninstall.windowTitle() == targets['Uninstall {name}'].format(name=name)
    notice = next(s for s in targets if s.startswith('Uninstalling {name} deletes'))
    assert uninstall.blurb.text() == targets[notice].format(name=name, environment=environment)
    assert uninstall.start_button.text() == targets['Uninstall']
    uninstall.close()

    browser = QTextBrowser()
    qtbot.addWidget(browser)
    host = SimpleNamespace(card=browser)
    for key, spec in backends._SPECS.items():
        entry = SimpleNamespace(name=spec.label, kind='backend', source='installable',
                                uri='backend:' + key, trained_on=spec.blurb,
                                licence=spec.licence, metrics={},
                                notes=('installable: fixture diagnostic',
                                       spec.licence_note, spec.published))
        picker.ModelZooPicker._show_card(host, entry)
        rendered = browser.toPlainText()
        for source in (spec.blurb, spec.licence_note, spec.published):
            assert targets[source] in rendered
        assert entry.trained_on == spec.blurb
        assert 'fixture diagnostic' in rendered


def test_backend_card_escapes_prose_and_preserves_a_supplied_scorecard(qtbot):
    from spacr.qt.widgets import model_zoo_picker as picker

    browser = QTextBrowser()
    qtbot.addWidget(browser)
    entry = SimpleNamespace(name='DINOCell', kind='backend', source='installed',
                            uri='backend:dinocell', trained_on='x < y & y > z',
                            licence='MIT', metrics={'f1': 0.8123},
                            notes=('installed: <local path>', 'MIT', 'A < B'))
    picker.ModelZooPicker._show_card(SimpleNamespace(card=browser), entry)
    rendered = browser.toPlainText()
    assert '0.8123' in rendered
    assert 'x < y & y > z' in rendered
    assert 'installed: <local path>' in rendered
    assert 'A < B' in rendered


def test_registry_prose_is_extracted_without_package_requirements(monkeypatch):
    from dataclasses import replace

    import spacr._segmentation_backends as backends

    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    import build_i18n_catalogs as builder

    spec = replace(backends._SPECS['spotnet'],
                   blurb='A future backend description supplied through its registry.',
                   licence_note='A future licence note supplied through its registry.',
                   published='A future results note supplied through its registry.')
    monkeypatch.setitem(backends._SPECS, 'spotnet', spec)
    sources = set(builder.canonical_sources()['ui'])
    assert {spec.blurb, spec.licence_note, spec.published} <= sources
    assert not set(spec.requirements) & sources
    assert spec.homepage not in sources
