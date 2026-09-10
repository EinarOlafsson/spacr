"""The formula editor before there is anything for it to work with.

Two states the panel reaches before its ordinary one:

*Its module body running twice in one process.* The panel registers its QSS
block at import, and :func:`spacr.qt.theme.register_widget_qss` refuses a
name that is already taken so two widgets cannot quietly claim one. A second
execution — a reload, or an importer that runs the source into a fresh module
object — therefore meets its own registration coming back, and must leave the
block already in the stylesheet alone rather than raise out of an import that
nothing is in a position to catch.

*A formula defined before a table is loaded.* Nothing can be validated yet, so
the formula is kept rather than refused: definitions first and table second is
the order a saved analysis is restored in.
"""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from spacr.qt import theme                                       # noqa: E402
from spacr.qt.widgets.formula import ColumnFormula                # noqa: E402
from spacr.qt.widgets import formula_editor                      # noqa: E402


def test_the_panels_stylesheet_block_is_registered_once_and_is_taken():
    """The name is claimed, and claiming it again is an error by design."""
    assert "FormulaPanel" in theme.widget_qss_names()

    with pytest.raises(ValueError):
        theme.register_widget_qss("FormulaPanel", lambda palette, opacity: "")


def test_running_the_module_body_again_keeps_the_first_registration():
    """A second execution is absorbed; the block already in place survives.

    The re-run must not raise, and must not swap the registered callable
    for its own copy: the stylesheet the running application is wearing was
    built from the first one.
    """
    incumbent = theme._WIDGET_QSS["FormulaPanel"]
    names_before = theme.widget_qss_names()

    spec = importlib.util.find_spec(formula_editor.__name__)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)

    assert theme._WIDGET_QSS["FormulaPanel"] is incumbent
    assert theme.widget_qss_names() == names_before
    assert fresh.PREVIEW_ROWS == formula_editor.PREVIEW_ROWS
    assert fresh.FormulaPanel is not formula_editor.FormulaPanel


def test_the_registered_block_still_styles_the_panel_after_the_second_run():
    """The surviving block is a live one, not a stale closure."""
    palette = dict(theme.palette_for("dark"))
    palette.setdefault("theme", "dark")

    qss = theme._WIDGET_QSS["FormulaPanel"](palette, None)

    assert "QLabel#FormulaStatus" in qss
    assert "QListWidget#FormulaList" in qss


def test_a_formula_can_be_defined_before_a_table_is_loaded(qtbot):
    """With no table there is nothing to validate against, so it is kept.

    Defining the columns first and loading the table second is the order a
    saved analysis is restored in, and a panel that refused every formula
    until a frame arrived would drop the lot.
    """
    panel = formula_editor.FormulaPanel()
    qtbot.addWidget(panel)
    assert panel.frame() is None

    with qtbot.waitSignal(panel.formulas_changed, timeout=1000):
        added = panel.add_formula(ColumnFormula("half", "area / 2"))

    assert added is True
    assert panel.formulas().names == ("half",)
    assert panel.computed_frame() is None
    assert panel.results() == []


def test_the_formula_kept_before_the_table_is_applied_when_the_table_arrives(
        qtbot):
    """The column appears on the frame, computed, once there is one."""
    panel = formula_editor.FormulaPanel()
    qtbot.addWidget(panel)
    panel.add_formula(ColumnFormula("half", "area / 2"))

    panel.set_frame(pd.DataFrame({"area": [2.0, 4.0, 6.0]}))

    computed = panel.computed_frame()
    assert list(computed["half"]) == [1.0, 2.0, 3.0]
    assert [r.formula.name for r in panel.results()] == ["half"]
