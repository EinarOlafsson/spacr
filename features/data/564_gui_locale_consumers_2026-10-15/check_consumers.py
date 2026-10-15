"""Actual generated-catalog consumers for two Activation Maps controls."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(sys.argv[1]).resolve()
EXPECTED_HEAD = sys.argv[2]
OUT = Path(sys.argv[3]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == EXPECTED_HEAD
LANGUAGES = ('sv', 'de', 'es', 'zh_CN', 'pt', 'hi', 'ko', 'is', 'fr')
KEYS = ('counterfactual_strength', 'counterfactual_structure_weight')
DEFAULTS = (0.95, 0.0)
NAMES = ('ActivationCounterfactualStrength', 'ActivationCounterfactualStructureWeight')
REVIEW = '2026-10-15-f564-diffusion-controls-home-b.json'
FILES = [
    'spacr/settings.py', 'spacr/attribution.py', 'spacr/deep_spacr.py',
    'spacr/qt/screens/settings_model.py', 'spacr/qt/screens/app_screen.py',
    'spacr/qt/screens/settings_flow_index.py', 'spacr/qt/i18n.py',
    'spacr/qt/i18n_catalogs/__init__.py',
    'docs/source/_generated/settings_flow.rst',
] + [f'spacr/qt/i18n_catalogs/{lang}.py' for lang in ('en', *LANGUAGES)]
FILES += [f'docs/i18n/reviewed/runtime/{lang}/{REVIEW}' for lang in LANGUAGES]
sha = lambda value: hashlib.sha256(value).hexdigest()
before = {path: sha((ROOT / path).read_bytes()) for path in FILES}
network_attempts = []

def audit(event, args):
    if event == 'socket.connect':
        network_attempts.append(repr(args))
        raise AssertionError('GUI consumer must not request network work')

sys.addaudithook(audit)
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QLabel
from spacr.qt import preferences
from spacr.qt.i18n import current_language, retranslate_widget_tree, tr
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import _setting_label_for_field, api_docs_url
from spacr.qt.screens.settings_flow_index import SETTINGS_WITH_A_FLOW_SECTION
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

app = QApplication.instance() or QApplication([])
private_ini = OUT / 'preferences.ini'
assert not private_ini.exists(), 'Use a new isolated settings file for each attempt'
os.environ.pop('SPACR_LANGUAGE', None)
flow = (ROOT / 'docs/source/_generated/settings_flow.rst').read_text()
rows = []
diagnostics = []
start = time.monotonic()
with patch.object(preferences, '_settings', lambda: QSettings(str(private_ini), QSettings.IniFormat)):
    preferences.set_language('en')
    preferences._set_show_alpha_features(True)
    screen = AppScreen('activation')
    try:
        screen._open_the_heading_of('counterfactuals')
        screen._refresh_alpha_visibility()
        model = screen._settings_model
        for language in LANGUAGES:
            reviewed = json.loads((ROOT / f'docs/i18n/reviewed/runtime/{language}/{REVIEW}').read_text())
            records = {(r['table'], r['key']): r for r in reviewed['records']}
            assert len(records) == 7
            preferences.set_language(language)
            assert current_language() == language
            retranslate_widget_tree(screen, language)
            for key, expected_default, name in zip(KEYS, DEFAULTS, NAMES):
                editor = model._widgets[key]
                label = _setting_label_for_field(screen, editor)
                assert isinstance(editor, QDoubleSpinBox), (language, key, type(editor))
                assert editor.objectName() == name
                assert isinstance(label, QLabel), (language, key, type(label))
                assert screen.setting_row_is_visible(key), (language, key, 'hidden row')
                label_record = records['setting_labels', key]
                body_record = records['setting_tooltips', key]
                assert label.text() == label_record['translation'], (language, key, label.text())
                tooltip = label.toolTip()
                expected_header = f'<b>{html.escape(label_record["translation"])}</b> <i>({html.escape(tr("float", language))})</i>'
                assert expected_header in tooltip, (language, key, 'typed header', tooltip)
                assert html.escape(body_record['translation']) in tooltip, (language, key, 'body', tooltip)
                assert key in SETTINGS_WITH_A_FLOW_SECTION, (key, 'flow index')
                assert f'.. _setting-flow-{key}:' in flow, (key, 'flow RST anchor')
                url = api_docs_url('activation', key, language)
                assert url.endswith(f'/settings_flow.html?lang={language}#setting-flow-{key}'), url
                assert f'href="{html.escape(url, quote=True)}"' in tooltip
                assert html.escape(tr('Open spaCR API documentation', language)) in tooltip
                canonical = editor.property('apiTooltipDescriptionSource')
                assert canonical.endswith(body_record['source']), (language, key, canonical)
                assert editor.property('apiTooltipDescription') == canonical
                assert editor.value() == expected_default, (language, key, editor.value())
                rows.append({
                    'language': language, 'key': key, 'label': label.text(),
                    'tooltip': tooltip, 'canonical_metadata': canonical,
                    'default': editor.value(), 'url': url,
                    'label_target_sha256': sha(label_record['translation'].encode()),
                    'body_target_sha256': sha(body_record['translation'].encode()),
                })
            for (table, key), record in records.items():
                if table == 'ui':
                    rendered = tr(key, language)
                    assert rendered == record['translation'], (language, key, rendered)
                    diagnostics.append({'language': language, 'source': key, 'rendered': rendered})
            print(f'{language}: 2 actual labels, typed bodies, flow links and defaults; 3 diagnostic lookups PASS', flush=True)
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()
        screen.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
assert not network_attempts, network_attempts
after = {path: sha((ROOT / path).read_bytes()) for path in FILES}
assert before == after, 'Source/catalog bytes changed during acceptance'
result = {
    'status': 'PASS', 'tested_commit': EXPECTED_HEAD,
    'languages': list(LANGUAGES), 'actual_setting_consumers': rows,
    'diagnostic_catalog_lookups': diagnostics, 'source_sha256': before,
    'private_settings': str(private_ini), 'network_attempts': network_attempts,
    'elapsed_seconds': time.monotonic() - start,
    'scope': 'Real AppScreen lazy Activation form; nine normal language passes; 36 setting targets and 27 diagnostic lookups. No backend execution, training, models, screenshots, native platform or scientific acceptance.',
}
(OUT / 'consumer.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': 'PASS', 'control_language_cases': len(rows), 'reviewed_targets_consumed': len(rows) * 2 + len(diagnostics)}, ensure_ascii=False), flush=True)
