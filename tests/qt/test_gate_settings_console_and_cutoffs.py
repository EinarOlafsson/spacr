"""Gate Settings' mode dropdown and note, the gating console, and axis cutoffs.

* :class:`GateSettingsDialog` shows the editor's mode without echoing it back,
  ignores modes it does not carry, and survives its selection note being gone.
* :func:`evaluate` / :class:`GateConsole` answer an expression or a question,
  and treat a blank box as no question at all.
* :class:`AxisCutoffs` stores a cutoff only while it cuts something.

One general test per unit, each walking a case table with ``subtests``.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.gate_canvas import AxisCutoffs  # noqa: E402
from spacr.qt.widgets.gate_console import GateConsole, evaluate  # noqa: E402
from spacr.qt.widgets.gate_settings import (  # noqa: E402
    GATE_MODES, GateEditorSettings, GateSettingsDialog,
)

pytestmark = pytest.mark.qt

_COLUMNS = ("cell_area", "cell_channel_1_mean_intensity", "nucleus_area", "prcfo")
_FRAME = pd.DataFrame({"area": [1.0, 2.0, 3.0, 4.0],
                       "keep": [True, True, False, False]})


@pytest.fixture
def make_dialog(qtbot):
    def build():
        dialog = GateSettingsDialog(GateEditorSettings(), columns=_COLUMNS)
        qtbot.addWidget(dialog)
        seen: list = []
        dialog.settings_changed.connect(seen.append)
        return dialog, seen
    return build


@pytest.fixture
def make_console(qtbot):
    def build(frame=_FRAME):
        console = GateConsole()
        qtbot.addWidget(console)
        if frame is not None:
            console.set_frame(frame)
        return console
    return build


def test_the_settings_dialog_mirrors_the_editor_mode_without_echoing(make_dialog, subtests):
    """``set_mode`` shows real modes silently; unknown modes and a missing note change nothing."""
    for mode in GATE_MODES:
        with subtests.test(msg=f"shown_without_echo_{mode}"):
            dialog, seen = make_dialog()
            dialog.set_mode(mode)
            assert dialog._mode.currentText() == mode
            assert seen == []
            settings = dialog.settings()
            # xD is the legacy spelling of 3D with a projected Z.
            assert (settings.gate_mode, settings.xd_projection) == (
                ("3D", True) if mode == "xD" else (mode, False))
    for case_id, mode in (("a_mode_the_dropdown_does_not_carry", "7D"),
                          ("an_empty_mode", "")):
        with subtests.test(msg=case_id):
            dialog, seen = make_dialog()
            before = dialog._mode.currentText()
            dialog.set_mode(mode)
            assert dialog._mode.currentText() == before
            assert dialog.settings().gate_mode == before
            dialog._mode.setCurrentText("3D")
            assert [s.gate_mode for s in seen] == ["3D"]
    with subtests.test(msg="refreshing_an_absent_note"):
        dialog, _ = make_dialog()
        before = dialog.settings()
        dialog._selection_note = None
        assert dialog._refresh_selection_note() is None
        assert dialog._selection_note is None
        assert dialog.settings() == before
    with subtests.test(msg="an_explicit_selection_still_changes"):
        dialog, _ = make_dialog()
        dialog._selection_note = None
        dialog._explicit.setText("cell_area, nucleus_area")
        dialog._on_explicit_changed()
        assert dialog.settings().reduction_columns == ("cell_area", "nucleus_area")


def test_evaluate_answers_in_a_sentence_scoped_to_safe_names(subtests):
    """Masks are counts, numbers are described, and only identifiers become names."""
    frame = pd.DataFrame({"area": [1.0, 3.0], "mean area": [10.0, 20.0],
                          "df": ["x", "y"]})
    cases = [
        ("a_boolean_column_is_a_count", "keep", _FRAME, "2 of 4 objects"),
        ("a_numeric_column_is_its_distribution", "area", _FRAME,
         str(_FRAME["area"].describe())),
        ("an_identifier_column_is_injected", "area.mean()", frame, "2.0"),
        ("df_still_names_the_frame", "df", frame, "2 rows × 3 columns"),
        ("reachable_through_df", "df['mean area'].sum()", frame, "30.0"),
    ]
    for case_id, expression, data, expected in cases:
        with subtests.test(msg=case_id):
            assert evaluate(expression, data) == expected
    with subtests.test(msg="a_non_identifier_column_has_no_bare_name"):
        assert evaluate("mean area", frame).startswith("SyntaxError")


def test_the_console_treats_a_blank_box_as_no_question(make_console, subtests):
    """run / ask / run_input / send_chat write and clear only for a real entry."""
    for case_id, call, frame in (("an_empty_expression", lambda c: c.run("   "), _FRAME),
                                 ("an_empty_question", lambda c: c.ask(""), None)):
        with subtests.test(msg=case_id):
            console = make_console(frame)
            assert call(console) == ""
            assert console.transcript() == ""
    with subtests.test(msg="an_answered_input_box_is_cleared"):
        console = make_console()
        console.input.setText("keep")
        console.run_input()
        assert console.input.text() == ""
        assert "› keep" in console.transcript()
        assert "2 of 4 objects" in console.transcript()
    with subtests.test(msg="a_blank_input_box_is_kept_and_unrecorded"):
        console = make_console()
        console.input.setText("  ")
        console.run_input()
        assert (console.input.text(), console.transcript()) == ("  ", "")
    with subtests.test(msg="the_chat_box_clears_only_after_a_real_question"):
        console = make_console(None)
        console.set_responder(lambda text: f"answered {text}")
        console.chat.setPlainText("how many objects?")
        console.send_chat()
        assert console.chat.toPlainText() == ""
        assert "answered how many objects?" in console.transcript()
        before = console.transcript()
        console.chat.setPlainText("   \n  ")
        console.send_chat()
        assert console.chat.toPlainText() == "   \n  "
        assert console.transcript() == before


def test_axis_cutoffs_hold_only_cutoffs_that_cut(subtests):
    """Insertion order, clear_all's count, and an empty pair meaning absent."""
    with subtests.test(msg="listed_in_the_order_they_were_cut"):
        cutoffs = AxisCutoffs()
        cutoffs.set("area", low=10)
        cutoffs.set("intensity", high=500)
        cutoffs.set("perimeter", low=1, high=2)
        assert cutoffs.columns() == ("area", "intensity", "perimeter")
        assert list(cutoffs) == ["area", "intensity", "perimeter"]
        assert len(cutoffs) == 3
    with subtests.test(msg="clearing_everything_reports_how_many_went"):
        cutoffs = AxisCutoffs()
        cutoffs.set("area", low=10)
        cutoffs.set("intensity", high=500)
        assert cutoffs.clear_all() == 2
        assert (len(cutoffs), cutoffs.columns()) == (0, ())
        assert cutoffs.clear_all() == 0
    with subtests.test(msg="both_ends_none_forgets_the_measurement"):
        cutoffs = AxisCutoffs()
        cutoffs.set("area", low=10)
        stored = cutoffs.set("area", low=None, high=None)
        assert stored.is_set is False
        assert cutoffs.columns() == ()
        assert cutoffs.get("area").describe() == "none"
