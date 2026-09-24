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

import json
import subprocess
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


@pytest.fixture(scope='module')
def literal_sources():
    """Extract static sources without collecting thousands of prior Qt objects.

    CI's long Qt shard crashed in native garbage collection during ast.parse,
    before asserting anything about these strings. Static extraction needs no
    QApplication; a fresh interpreter keeps that contract independent of the
    GUI tests' accumulated native objects. Extraction errors remain failures.
    """
    code = (
        "import contextlib, json, sys\n"
        "sys.path.insert(0, 'tools')\n"
        "with contextlib.redirect_stdout(sys.stderr):\n"
        "    import build_i18n_catalogs as generator\n"
        "    sources = generator.extract_static_ui_sources()\n"
        "print(json.dumps(sources))\n"
    )
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr[-12000:]
    return set(json.loads(result.stdout))


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


def test_every_wrapped_string_is_a_source_the_generator_collects(literal_sources):
    """Wrapping is half the fix; the catalog pass has to see the string too."""
    literal = literal_sources
    for source in ("Fit curve", "all rows", "could not read {name}: {reason}",
                   "loading {name}…", "{rows} rows × {columns} columns",
                   "{name} · {rows} rows × {columns} columns"):
        assert source in literal, source
    assert any(s.startswith("No column of this table has at least four")
               for s in literal)
    text = (ROOT / "tools" / "build_i18n_catalogs.py").read_text(encoding="utf-8")
    assert "ui_sources.update(_DOSE_RESPONSE_UI_SOURCES)" in text
    assert {"Group", "Status", "refused"} <= _DOSE_RESPONSE_UI_SOURCES


# -- the report pane, the plot and the whole-table item ----------------------
#
# Found by the home session's catalog pass: these reached the screen as
# f-strings and a plain constant, so every language showed them in English.

REPORT_TEMPLATES = (
    "{plate}: usable, Z′ {zprime}",
    "{plate}: refused, Z′ {zprime}",
    "{name}: not pooled — {reason}",
    "{name}: pooled EC50 {ec50}{unit} ({low}–{high}) across {used} of "
    "{plates} plates, I² {spread}",
    "{line}, plates disagree",
    "{name}: no selectivity index — {reason}",
    "{name}: selectivity index {index} ({low}–{high}), host EC50 {host} "
    "over response EC50 {response}",
    "{name}: no synergy surface — {reason}",
    "{name}: no combination well could be scored against {model}",
    "{name}: {model} excess over {cells} combination wells, max {max} at "
    "{dose_a} + {dose_b}, min {min}; {synergistic} synergistic, "
    "{antagonistic} antagonistic",
    "{name}: REFUSED",
)


def _lines(widget):
    return widget.report.toPlainText().splitlines()


def test_the_plate_and_pooled_lines_go_through_tr(qtbot, marked):
    import numpy as np
    from tests.qt.test_dose_response_normalises_to_each_plate import _plate

    rng = np.random.default_rng(20260915)
    table = pd.concat([_plate("P1", 1000.0, 100.0, rng),
                       _plate("P2", 2000.0, 200.0, rng),
                       _plate("P3", 1500.0, 150.0, rng, with_negative=False)],
                      ignore_index=True)
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(table, label="three plates")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")
    widget.plate_picker.setCurrentText("plate")
    widget.control_picker.setCurrentText("role")

    widget.fit()

    lines = _lines(widget)
    assert any(line.startswith("⟦P1: usable, Z′ ") for line in lines), lines
    assert any(line.startswith("⟦P3: refused, Z′ ") for line in lines), lines
    assert any(line.startswith("⟦geneA: pooled EC50 ") for line in lines), lines


def test_the_selectivity_lines_go_through_tr(qtbot, marked):
    import numpy as np
    from tests.qt.test_dose_response_selectivity_index import (
        DOSES, REPLICATES, _kill,
    )

    rng = np.random.default_rng(20260915)
    dose = np.repeat(DOSES, REPLICATES)
    table = pd.concat([pd.DataFrame({
        "gene": gene, "conc_uM": dose,
        "parasite_killed": _kill(1.0, rng, dose),
        "host_killed": _kill(host_ec50, rng, dose)})
        for gene, host_ec50 in (("geneA", 10.0), ("geneB", None))],
        ignore_index=True)
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(table, label="two compounds")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("parasite_killed")
    widget.group_picker.setCurrentText("gene")
    widget.host_picker.setCurrentText("host_killed")

    widget.fit()

    lines = _lines(widget)
    assert any(line.startswith("⟦geneA: selectivity index ")
               for line in lines), lines
    assert any(line.startswith("⟦geneB: no selectivity index — ")
               for line in lines), lines


def test_the_synergy_line_goes_through_tr(qtbot, marked):
    from tests.qt.test_dose_response_scores_a_checkerboard import (
        _board, _screen,
    )

    widget = _screen(qtbot, _board())
    widget.second_dose_picker.setCurrentText("cmpd_b_uM")

    widget.fit()

    assert any(line.startswith("⟦⟦all rows⟧: Bliss excess over ")
               for line in _lines(widget)), _lines(widget)


def test_a_refusal_the_plot_and_the_whole_table_item_go_through_tr(
        qtbot, frame, marked):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="synthetic")

    axes = widget._figure.axes[0]
    assert axes.get_xlabel() == "⟦concentration⟧"
    assert axes.get_ylabel() == "⟦response⟧"
    assert widget.group_picker.itemText(0) == f"⟦{screen_module.NO_GROUP}⟧"
    assert widget.group_picker.itemData(0) == screen_module.NO_GROUP

    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")
    widget.fit()
    widget.show_group(2)
    assert widget.report.toPlainText().startswith("⟦geneC: REFUSED⟧")

    widget.group_picker.setCurrentIndex(0)
    widget.fit()
    labels = [line.get_label() for line in widget._figure.axes[0].get_lines()]
    assert "⟦all rows⟧" in labels, labels


def test_the_report_and_plot_templates_are_sources_the_generator_collects(literal_sources):
    literal = literal_sources
    for source in (*REPORT_TEMPLATES, "concentration", "response"):
        assert source in literal, source
    assert screen_module.NO_GROUP in _DOSE_RESPONSE_UI_SOURCES
