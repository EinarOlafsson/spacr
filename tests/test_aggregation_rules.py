"""The per-column merge rules, shown and overridable."""
import numpy as np
import pandas as pd
import pytest

from spacr.merge_tables import MAX, MEAN, MIN, SUM


@pytest.fixture
def frame():
    return pd.DataFrame({"area": [1.0], "min_intensity": [1.0],
                         "max_intensity": [1.0], "mean_intensity": [1.0],
                         "label": ["a"]})


def test_every_column_gets_a_row_and_its_rule(qtbot, frame):
    from spacr.qt.widgets.aggregation_rules import AggregationRulesDialog

    dialog = AggregationRulesDialog(frame)
    qtbot.addWidget(dialog)
    rows = {dialog.tree.topLevelItem(i).text(0): dialog._boxes[
        dialog.tree.topLevelItem(i).text(0)].currentText()
        for i in range(dialog.tree.topLevelItemCount())}

    assert rows["area"] == SUM
    assert rows["min_intensity"] == MIN
    assert rows["max_intensity"] == MAX
    assert rows["mean_intensity"] == MEAN


def test_changing_a_row_records_an_override(qtbot, frame):
    from spacr.qt.widgets.aggregation_rules import AggregationRulesDialog

    dialog = AggregationRulesDialog(frame)
    qtbot.addWidget(dialog)
    seen = []
    dialog.rules_changed.connect(seen.append)

    dialog._boxes["area"].setCurrentText(MEAN)
    assert dialog.overrides() == {"area": MEAN}
    assert seen[-1] == {"area": MEAN}


def test_setting_a_row_back_to_its_default_is_not_an_override(qtbot, frame):
    """Recording it would pin today's default forever: a later improvement to
    the rules would never reach a user who once opened this window."""
    from spacr.qt.widgets.aggregation_rules import AggregationRulesDialog

    dialog = AggregationRulesDialog(frame)
    qtbot.addWidget(dialog)
    dialog._boxes["area"].setCurrentText(MEAN)
    dialog._boxes["area"].setCurrentText(SUM)
    assert dialog.overrides() == {}


def test_existing_overrides_are_shown(qtbot, frame):
    from spacr.qt.widgets.aggregation_rules import AggregationRulesDialog

    dialog = AggregationRulesDialog(frame, overrides={"area": MEAN})
    qtbot.addWidget(dialog)
    assert dialog._boxes["area"].currentText() == MEAN


def test_the_list_can_be_filtered(qtbot, frame):
    """A real measurement table is hundreds of columns wide."""
    from spacr.qt.widgets.aggregation_rules import AggregationRulesDialog

    dialog = AggregationRulesDialog(frame)
    qtbot.addWidget(dialog)
    dialog.search.setText("intensity")

    hidden = [dialog.tree.topLevelItem(i).text(0)
              for i in range(dialog.tree.topLevelItemCount())
              if dialog.tree.topLevelItem(i).isHidden()]
    assert "area" in hidden and "min_intensity" not in hidden


def test_an_override_reaches_the_merge_policy(qtbot):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen._on_aggregation_rules_changed({"area": MEAN})
    assert screen.settings().merge_overrides == {"area": MEAN}
    assert screen._merge_policy().overrides == {"area": MEAN}


def test_the_primary_object_reaches_the_merge_policy(qtbot):
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(screen)
    screen.apply_settings(screen.settings().replaced(merge_primary="pathogen"))
    assert screen._merge_policy().primary == "pathogen"
