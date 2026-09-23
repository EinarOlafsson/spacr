"""Published section headings survive real fold controls and language changes."""
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = ("de", "es", "fr", "sv", "pt", "is", "zh_CN", "ko", "hi")


@pytest.mark.parametrize("language", LANGUAGES)
def test_reviewed_headings_render_and_preserve_fold_names(qtbot, language):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget
    from spacr.qt.i18n import retranslate_widget_tree
    from spacr.qt.widgets.collapsible_splitter import CollapsibleSplitter

    review = json.loads((ROOT / "docs/i18n/reviewed/runtime" / language /
                         "2026-09-22-section-headings.json").read_text())
    inventory = json.loads((ROOT / "features/data/288_section_title_inventory_2026-09-22.json").read_text())
    records = {row["source"]: row["translation"] for row in review["records"]}
    assert set(records) == set(inventory["added"])
    splitter = CollapsibleSplitter(Qt.Vertical)
    qtbot.addWidget(splitter)
    sections = {source: splitter.add_section(QWidget(), source)
                for source in records}

    retranslate_widget_tree(splitter, language)
    for source, section in sections.items():
        assert section.heading.text().endswith(" " + records[source])
        assert section.folder.name == source
        assert splitter.pane(source).name == source

    retranslate_widget_tree(splitter, "en")
    for source, section in sections.items():
        assert section.heading.text().endswith(" " + source)
        assert section.folder.name == source
