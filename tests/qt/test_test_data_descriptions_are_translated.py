"""The chooser's actual hover prose is translated and fits its reserved pane."""
import hashlib
import json
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtGui import QTextDocument

from spacr.qt.widgets.test_data_chooser import TestDataChooser as Chooser


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ('de', 'es', 'fr', 'hi', 'is', 'ko', 'pt', 'sv', 'zh_CN')


def test_class_owned_hover_prose_enters_the_english_inventory(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'tools'))
    import build_i18n_catalogs as builder

    # UI state must not turn the source inventory into German text.
    monkeypatch.setenv('SPACR_LANGUAGE', 'de')
    sources = builder._indirect_runtime_ui_sources()
    assert Chooser.RESTING_TEXT in sources
    assert all(description in sources for _key, _label, description in Chooser.ROUTES)
    from spacr.import_examples import IMPORT_VARIANTS
    from spacr.qt.import_demo import ImportTestDataChooser

    assert ImportTestDataChooser.RESTING_TEXT in sources
    missing = [variant.key for variant in IMPORT_VARIANTS if variant.description not in sources]
    assert not missing, missing
    assert all(variant.label in sources for variant in IMPORT_VARIANTS)


@pytest.mark.parametrize('language', LANGUAGES)
def test_rest_hover_tooltip_and_pane_use_the_reviewed_translation(
        qtbot, qt_theme_applied, monkeypatch, language):
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    reviewed = json.loads((ROOT/f'docs/i18n/reviewed/runtime/{language}/2026-09-21-test-data-chooser.json').read_text())
    targets = {}
    for record in reviewed['records']:
        assert record['source_sha256'] == hashlib.sha256(record['source'].encode()).hexdigest()
        targets[record['source']] = record['translation']
    dialog = Chooser()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(lambda: dialog._laid_out and dialog._description.width() > 200)
    QCoreApplication.sendEvent(dialog._buttons['load'], QEvent(QEvent.Leave))
    assert dialog.description_text() == targets[Chooser.RESTING_TEXT]
    expected = [targets[Chooser.RESTING_TEXT]]
    for key, _label, source in Chooser.ROUTES:
        button = dialog._buttons[key]
        assert button.toolTip() == targets[source]
        QCoreApplication.sendEvent(button, QEvent(QEvent.Enter))
        actual = dialog.description_text()
        assert actual == targets[source] and actual != source
        document = QTextDocument()
        document.setDefaultFont(dialog._description.font())
        document.setDocumentMargin(0)
        document.setPlainText(actual)
        document.setTextWidth(dialog._description.contentsRect().width())
        assert document.size().height() <= dialog._description.contentsRect().height(), language
        expected.append(actual)
        QCoreApplication.sendEvent(button, QEvent(QEvent.Leave))
        assert dialog.description_text() == targets[Chooser.RESTING_TEXT]
    assert dialog.every_description() == tuple(expected)
    assert dialog.chosen == ''  # inspecting a route must not choose/download it
