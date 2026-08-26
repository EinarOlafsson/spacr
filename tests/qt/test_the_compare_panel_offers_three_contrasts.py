"""187 A and B, at the panel: the three contrasts and the join offer.

The engine's arithmetic is covered by
`tests/test_the_three_contrasts_are_three_questions.py` and
`tests/test_every_measurement_in_the_database_is_offered.py`. What is tested
here is that a user can REACH it: that the contrast is on the panel, that the
controls field appears only where it means something, that a well can be left
out, and that the panel says why the measurement list is short instead of
just being short.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from spacr.gene_measurement_compare import REST, well_labels
from spacr.qt.widgets.measurement_compare_dialog import (MeasurementComparePanel,
                                                         _WellChoice)


def _objects() -> pd.DataFrame:
    rows = []
    offsets = {"1_A_01": 0.0, "1_A_02": 5.0, "1_A_03": 20.0,
               "1_B_01": 40.0}
    for well, offset in offsets.items():
        plate, row, column = well.split("_")
        for cell in range(8):
            rows.append({"plateID": plate, "rowID": row, "columnID": column,
                         "cell_area": offset + cell,
                         "cell_perimeter": offset / 2 + cell})
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture
def objects():
    return _objects()


@pytest.fixture
def groups(objects):
    where = well_labels(objects)
    picked = []
    for well in ("1_A_01", "1_A_02"):
        picked.extend(objects.index[where == well][:3].tolist())
    return {"the gene": picked}


@pytest.fixture
def counts():
    return pd.DataFrame({
        "plateID": ["1", "1", "1"],
        "rowID": ["A", "A", "B"],
        "columnID": ["01", "03", "01"],
        "grna": ["000123_1", "000456_1", "000000_1"],
        "gene": ["000123", "000456", "000000"],
    })


@pytest.fixture
def panel(qtbot, objects, groups, counts):
    widget = MeasurementComparePanel(objects, groups, counts=counts)
    qtbot.addWidget(widget)
    return widget


class TestTheContrastIsOnThePanel:

    def test_all_four_are_offered(self, panel):
        assert panel.contrast.count() == 4

    def test_everything_else_is_the_default(self, panel):
        assert panel.contrast.currentData() == ""

    def test_each_one_names_what_it_removes(self, panel):
        from PySide6.QtCore import Qt

        for index in range(panel.contrast.count()):
            why = panel.contrast.itemData(index, Qt.ToolTipRole)
            assert why and len(why) > 40, panel.contrast.itemText(index)

    def test_choosing_one_rebuilds_the_comparison(self, panel):
        before = panel.comparison().counts()[REST]

        panel.contrast.setCurrentIndex(
            panel.contrast.findData("within_well"))

        assert panel.comparison().counts()[REST] < before

    def test_the_report_names_the_contrast(self, panel):
        panel.contrast.setCurrentIndex(
            panel.contrast.findData("within_well"))

        assert "within the well" in panel.report.toPlainText()


class TestTheControlsFieldAppearsWhereItMeansSomething:

    def test_it_starts_disabled(self, panel):
        assert not panel.controls.isEnabled()

    def test_choosing_the_control_contrast_enables_it(self, panel):
        panel.contrast.setCurrentIndex(
            panel.contrast.findData("against_controls"))

        assert panel.controls.isEnabled()

    def test_leaving_the_control_contrast_disables_it_again(self, panel):
        panel.contrast.setCurrentIndex(
            panel.contrast.findData("against_controls"))
        panel.contrast.setCurrentIndex(panel.contrast.findData("within_well"))

        assert not panel.controls.isEnabled()

    def test_no_control_named_is_a_sentence(self, panel):
        panel.contrast.setCurrentIndex(
            panel.contrast.findData("against_controls"))

        assert "needs the controls named" in panel.report.toPlainText()

    def test_a_named_control_resolves_to_its_wells(self, panel):
        panel.contrast.setCurrentIndex(
            panel.contrast.findData("against_controls"))
        panel.controls.setText("000000")
        panel.refresh()

        # ONE control well, at the panel's default WELL level.
        assert panel.comparison().counts().get(REST) == 1

    def test_a_control_named_as_a_guide_works_too(self, panel):
        """184: gene or guide, either spelling."""
        panel.contrast.setCurrentIndex(
            panel.contrast.findData("against_controls"))
        panel.controls.setText("000000_1")
        panel.refresh()

        assert panel.comparison().counts().get(REST) == 1

    def test_two_controls_are_comma_separated(self, panel):
        panel.contrast.setCurrentIndex(
            panel.contrast.findData("against_controls"))
        panel.controls.setText("000000, 000456")
        panel.refresh()

        assert panel.comparison().counts().get(REST) == 2


class TestTheWellsAreChosen:

    def test_every_annotated_well_is_offered(self, panel):
        assert panel.wells_on_offer() == ("1_A_01", "1_A_02")

    def test_all_of_them_by_default(self, panel):
        assert panel.chosen_wells() is None

    def test_leaving_one_out_shrinks_the_annotation(self, panel):
        panel._chosen_wells = {"1_A_01"}
        panel.refresh()

        assert panel.comparison().counts()["the gene"] == 1

    def test_a_choice_cannot_name_a_well_that_is_gone(self, panel, objects,
                                                      groups):
        """A choice made before a re-run must not name a vanished well."""
        panel._chosen_wells = {"1_A_01", "1_Z_99"}

        assert panel.chosen_wells() == ["1_A_01"]

    def test_the_checklist_starts_with_everything_ticked(self, qtbot):
        """Nothing ticked would read as 'nothing is being compared'."""
        dialog = _WellChoice(("1_A_01", "1_A_02"), None)
        qtbot.addWidget(dialog)

        assert dialog.chosen() == {"1_A_01", "1_A_02"}

    def test_the_checklist_remembers_a_previous_choice(self, qtbot):
        dialog = _WellChoice(("1_A_01", "1_A_02"), {"1_A_02"})
        qtbot.addWidget(dialog)

        assert dialog.chosen() == {"1_A_02"}

    def test_no_wells_at_all_is_a_sentence_not_a_dialog(self, qtbot):
        frame = pd.DataFrame({"cell_area": np.arange(6, dtype=float)})
        panel = MeasurementComparePanel(frame, {"g": [0, 1]})
        qtbot.addWidget(panel)

        assert panel.choose_wells() is False
        assert "do not say which well" in panel.report.toPlainText()


class TestThePanelSaysWhyTheMeasurementListIsShort:
    """`cell_area` IS a joined column, so these use png_list's own spelling."""

    @pytest.fixture
    def crops(self):
        """What comes off `png_list`: the score, and no measurement table."""
        frame = _objects().rename(columns={"cell_area": "pred",
                                           "cell_perimeter": "area"})
        # png_list's own identity columns, so the join gets as far as
        # opening the database rather than stopping at "no object identity".
        return frame.assign(
            fieldID="1",
            prcf=frame["plateID"] + "_" + frame["rowID"] + "_"
                 + frame["columnID"] + "_1",
            object_label=[str(i + 1) for i in range(len(frame))])

    @pytest.fixture
    def unjoined(self, qtbot, crops):
        panel = MeasurementComparePanel(crops, {"the gene": [0, 1, 2]})
        qtbot.addWidget(panel)
        return panel

    def test_joined_rows_say_so(self, panel):
        assert "from the joined measurement tables" in panel.join_note.text()

    def test_the_button_is_gone_once_they_are_joined(self, panel):
        assert not panel.join_button.isVisible()

    def test_unjoined_rows_say_the_join_is_missing(self, qtbot, crops):
        panel = MeasurementComparePanel(crops, {"the gene": [0, 1, 2]},
                                        databases=["/nowhere/x.db"])
        qtbot.addWidget(panel)

        assert "needs the join" in panel.join_note.text()

    def test_with_no_database_it_says_that_instead(self, unjoined):
        assert not unjoined.join_button.isVisible()
        assert "no measurements database is attached" in \
            unjoined.join_note.text()

    def test_the_button_appears_once_a_database_is_attached(
            self, qtbot, crops):
        panel = MeasurementComparePanel(crops, {"the gene": [0, 1, 2]},
                                        databases=["/nowhere/x.db"])
        qtbot.addWidget(panel)
        panel.show()
        qtbot.waitExposed(panel)

        assert panel.join_button.isVisible()

    def test_joining_with_no_database_is_refused_in_words(self, unjoined):
        assert unjoined.join_the_tables() == \
            "no measurements database is attached"

    def test_an_unreadable_database_reaches_the_note(self, qtbot, crops):
        panel = MeasurementComparePanel(crops, {"the gene": [0, 1, 2]},
                                        databases=["/nowhere/x.db"])
        qtbot.addWidget(panel)

        # The join runs off the GUI thread, so the note arrives when the job
        # finishes rather than when the handler returns. Waiting for the job
        # is what tests the path a user takes; asserting straight after the
        # call would only test that the click does not block.
        panel.join_the_tables()
        qtbot.waitUntil(lambda: not panel._joining, timeout=30000)

        assert "no measurement table could be read" in panel.join_note.text()
        assert panel.join_button.isEnabled(), "the button locked itself out"
