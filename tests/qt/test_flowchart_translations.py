"""Translated flowchart controls keep candidate links distinct from file handoffs."""
import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialogButtonBox, QLabel


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ('de', 'es', 'fr', 'sv', 'pt', 'is', 'zh_CN', 'ko', 'hi')


@pytest.mark.parametrize('language', LANGUAGES)
def test_flowchart_controls_and_candidate_explanations(qtbot, qt_theme_applied, monkeypatch, language):
    from spacr.qt import i18n
    from spacr.qt.widgets.sample_project import SampleProjectDialog
    from spacr.qt.widgets.workflow_diagram import SpacrFlowchartDialog

    records = json.loads((ROOT / 'docs/i18n/reviewed/runtime' / language /
                          '2026-09-23-flowchart-chrome.json').read_text())['records']
    targets = {row['source']: row['translation'] for row in records}
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    dialog = SpacrFlowchartDialog()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(10)
    assert dialog.windowTitle() == targets['spaCR flowchart']
    assert dialog.module_picker.itemText(0) == targets['Select a module…']
    assert dialog.edge_picker.itemText(0) == targets['Select a connection…']
    assert dialog.edge_picker.accessibleName() == targets['Connection']
    assert dialog.compatible.text() == targets['Show matching data types']
    hint = 'Hover or select a module or connection to read its inputs, outputs and explanation here.'
    assert dialog.details.toPlainText() == targets[hint]
    dialog.view.describe_node('measure')
    assert targets['{module} API'].format(module=i18n.tr('Measure', language)) in dialog.details.toPlainText()
    assert 'lang=' + language in dialog.details.toHtml()
    candidate = next(edge for edge in dialog.view.edges if edge.edge['kind'] == 'compatible')
    dialog.compatible.setChecked(True)
    dialog.view.describe_edge(candidate)
    warning = ("Matching data types. Check the destination's required fields, "
               'object identities and settings; this is not an automatic transfer.')
    assert targets[warning] in dialog.details.toPlainText()
    assert candidate.isVisible()
    assert dialog.details.openExternalLinks()

    overview = SampleProjectDialog()
    qtbot.addWidget(overview)
    overview.show()
    qtbot.wait(10)
    assert overview.windowTitle() == targets['Pipeline overviews']
    intro = ('Explore the modules, inputs and outputs in each pipeline. Hover a module or arrow for details. '
             'Select a pipeline and choose Start example to open its first module with sample data. '
             'Ctrl+wheel zooms a diagram; drag to pan.')
    assert targets[intro] in {label.text() for label in overview.findChildren(QLabel)}
    buttons = overview.findChild(QDialogButtonBox)
    assert buttons.button(QDialogButtonBox.Ok).text() == targets['Start example']
    assert overview.list.count() == len(overview._entries) == 9
