"""Every word the Dose-Response screen shows can reach a translator.

An audit of the screen in all nine catalogs found its results-grid headers,
its status words and several messages in no catalog at all, so every language
showed them in English. These tests pin both halves of the fix: each string
goes through `tr` at the point it is shown, and each is a source the runtime
catalog generator will actually collect.

Translations themselves are the catalog pass's; nothing here depends on one.
A stand-in `tr` that visibly marks its input proves the call is made.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens import dose_response as screen_module
from spacr.qt.screens.dose_response import (
    TABLE_COLUMNS, DoseResponseScreen, _DOSE_RESPONSE_UI_SOURCES,
)

from tests.qt.test_dose_response_screen import frame, screen  # noqa: F401

pytestmark = pytest.mark.qt

ROOT = Path(__file__).resolve().parents[2]


def _marked(text, *args, **values):
    """A stand-in translation that cannot be mistaken for the English."""
    rendered = str(text).format(**values) if values else str(text)
    return f"⟦{rendered}⟧"


@pytest.fixture
def marked(monkeypatch):
    monkeypatch.setattr(screen_module, "tr", _marked)
    monkeypatch.setattr("spacr.qt.i18n.tr", _marked)


def test_the_grid_headers_and_status_words_go_through_tr(qtbot, frame, marked):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="synthetic")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")

    headers = [widget.table.horizontalHeaderItem(i).text()
               for i in range(widget.table.columnCount())]
    assert headers == [f"⟦{header}⟧" for _key, header in TABLE_COLUMNS]

    widget.fit()
    status_column = [k for k, _h in TABLE_COLUMNS].index("status")
    statuses = {widget.table.item(row, status_column).text()
                for row in range(widget.table.rowCount())}
    assert statuses and all(s.startswith("⟦") for s in statuses), statuses


def test_the_status_messages_go_through_tr(qtbot, marked, tmp_path):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_frame(pd.DataFrame({"a": ["x", "y"], "b": [1, 1]}))
    assert widget._source.text().startswith("⟦")
    assert widget.report.toPlainText().startswith("⟦No column")

    broken = tmp_path / "not_a_database.db"
    broken.write_text("not sqlite")
    widget.load_path(str(broken))
    assert widget._source.text().startswith("⟦could not read")


def test_the_button_says_fit_curve_not_the_zoom_word(screen):
    """"Fit" is zoom-to-fit in three viewers; this button fits a curve."""
    assert screen.fit_button.text() == "Fit curve"


def test_every_wrapped_string_is_a_source_the_generator_collects():
    """Wrapping is half the fix; the catalog pass has to see the string too."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import build_i18n_catalogs as generator
    finally:
        sys.path.remove(str(ROOT / "tools"))

    literal = set(generator.extract_static_ui_sources())
    for source in ("Fit curve", "all rows", "could not read {name}: {reason}",
                   "loading {name}…", "{rows} rows × {columns} columns",
                   "{name} · {rows} rows × {columns} columns"):
        assert source in literal, source
    assert any(s.startswith("No column of this table has at least four")
               for s in literal)
    text = (ROOT / "tools" / "build_i18n_catalogs.py").read_text(encoding="utf-8")
    assert "ui_sources.update(_DOSE_RESPONSE_UI_SOURCES)" in text
    assert {"Group", "Status", "refused"} <= _DOSE_RESPONSE_UI_SOURCES
