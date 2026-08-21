"""Every annotation method says exactly how it chooses cells.

Instruction 208 C: "The API should be verry specific and evplain exactly how
cells are cjhoosen for annotation."

FIVE METHODS CANNOT SHARE ONE TOOLTIP. The shape test holds a setting's help
to 600 characters, and one honest paragraph per method is already more than
that. Per-ENTRY help is the only place with room, and it is where the choice
is actually made.

ONE SOURCE. The dropdown and any written explanation read the same dict,
because a method that decides what every downstream number means cannot have
two descriptions that drift apart.
"""
from __future__ import annotations

import pytest

from spacr.cell_montage import PICKING_MODES
from spacr.picture_settings import PICKING_HELP


class TestEveryMethodIsCovered:

    def test_no_method_is_undocumented(self):
        assert set(PICKING_MODES) <= set(PICKING_HELP)

    def test_nothing_is_documented_that_is_not_offered(self):
        """A described method that cannot be chosen is a promise the panel
        does not keep."""
        assert set(PICKING_HELP) <= set(PICKING_MODES)


class TestEachAnswersTheSameFiveQuestions:
    """What it is given, what it computes, which cells end up annotated,
    which do NOT and why, and the assumption that makes it wrong."""

    @pytest.mark.parametrize("method", sorted(PICKING_HELP))
    def test_it_says_what_it_is_given(self, method):
        assert "Given:" in PICKING_HELP[method]

    @pytest.mark.parametrize("method", sorted(PICKING_HELP))
    def test_it_says_what_it_computes(self, method):
        assert "Computes:" in PICKING_HELP[method]

    @pytest.mark.parametrize("method", sorted(PICKING_HELP))
    def test_it_says_which_cells_are_annotated(self, method):
        assert "Annotated:" in PICKING_HELP[method]

    @pytest.mark.parametrize("method", sorted(PICKING_HELP))
    def test_it_says_which_are_not_and_why(self, method):
        """The question 207 turned on: a method that cannot say which cells
        it declined to annotate cannot be checked."""
        assert "Not annotated:" in PICKING_HELP[method]

    @pytest.mark.parametrize("method", sorted(PICKING_HELP))
    def test_it_names_the_assumption_that_breaks_it(self, method):
        assert "Wrong when:" in PICKING_HELP[method]


class TestTheAnswersAreTheRightOnes:
    """Spot-checks against what the code actually does, so the help cannot
    drift into describing a method nobody wrote."""

    def test_assigned_annotates_everything(self):
        """`assign_well` gives every cell exactly one guide by
        construction."""
        said = PICKING_HELP["assigned"]
        assert "EVERY cell" in said
        assert "Not annotated: none" in said

    def test_rank_computes_no_probability(self):
        assert "No probability" in PICKING_HELP["rank"]

    def test_attributed_names_its_threshold(self):
        assert "0.55" in PICKING_HELP["attributed"]

    def test_multivariate_says_it_falls_back_loudly(self):
        said = PICKING_HELP["multivariate"]
        assert "falls back" in said
        assert "SAYS" in said or "silently" in said

    def test_sudoku_names_the_circularity_it_avoids(self):
        said = PICKING_HELP["sudoku"]
        assert "anchors" in said.lower()
        assert "left out of the graph" in said or "score" in said


@pytest.mark.qt
class TestItReachesTheDropdown:

    def test_each_entry_carries_its_own_help(self, qtbot):
        pytest.importorskip("PySide6")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QComboBox

        from spacr.qt.widgets.picture_settings_dialog import (
            PictureSettingsDialog)

        dialog = PictureSettingsDialog({})
        qtbot.addWidget(dialog)
        combo = dialog._editors["cell_picking"]
        assert isinstance(combo, QComboBox)

        for index in range(combo.count()):
            value = str(combo.itemData(index))
            said = combo.itemData(index, Qt.ToolTipRole)
            assert said, value
            assert said == PICKING_HELP[value]

    def test_other_settings_are_left_alone(self, qtbot):
        """The helper must be a no-op everywhere else."""
        pytest.importorskip("PySide6")
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QComboBox

        from spacr.qt.widgets.picture_settings_dialog import (
            PictureSettingsDialog)

        dialog = PictureSettingsDialog({})
        qtbot.addWidget(dialog)
        for key, editor in dialog._editors.items():
            if key == "cell_picking" or not isinstance(editor, QComboBox):
                continue
            for index in range(editor.count()):
                assert not editor.itemData(index, Qt.ToolTipRole), key
