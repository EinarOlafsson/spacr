"""Choosing the Walk's search space from the panel.

The engine's side of this is `tests/test_hyperparam_walk.py`. These cover
the half that only the panel can get wrong: turning a set of checkboxes
into a starting POINT, and refusing a space that is a grid in disguise.
"""

import pytest

from spacr.hyperparam import MAX_WALK_CANDIDATES_PER_ROUND, UMAP_WALK_PARAMETERS
from spacr.qt.screens.hyperparam import HyperparamPanel, WalkAxesDialog


@pytest.fixture
def panel(qt_theme_applied, qtbot):
    widget = HyperparamPanel("umap")
    qtbot.addWidget(widget)
    widget.apply_settings({"n_neighbors": 15, "min_dist": 0.1,
                           "metric": "euclidean"})
    return widget


class TestPanelAxes:

    def test_an_untouched_panel_walks_the_two_it_always_walked(self, panel):
        """No axes chosen must not mean no axes searched."""
        assert panel.walk_axes() == {}

    def test_a_parameter_umap_does_not_have_is_refused(self, panel):
        with pytest.raises(ValueError, match="Not a searchable"):
            panel.set_walk_axes({"nonsense": {"start": "1"}})

    def test_the_starting_value_comes_from_the_field_when_it_holds_one(
            self, panel):
        panel._value_edits["n_neighbors"].setText("42")
        assert panel.walk_start_for("n_neighbors") == "42"

    def test_a_grid_in_a_field_is_not_a_starting_value(self, panel):
        """`5, 15, 50` is three points. A walk starts at one."""
        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        assert panel.walk_start_for("n_neighbors") == "15"  # from settings

    def test_an_axis_with_no_field_starts_at_umaps_own_default(self, panel):
        assert panel.walk_start_for("spread") == "1.0"
        assert panel.walk_start_for("init") == "spectral"

    def test_a_chosen_axis_enters_the_space_as_one_value(self, panel):
        panel._adaptive.setChecked(True)
        panel.set_walk_axes({"spread": {"start": "2.0", "resolution": 3}})
        space = panel.current_space()
        assert list(space.params["spread"]) == [2.0]

    def test_the_axes_are_ignored_while_the_walk_is_off(self, panel):
        """A grid search over the same fields must not silently pick up a
        starting point that only means something to a walk."""
        panel._adaptive.setChecked(False)
        panel.set_walk_axes({"spread": {"start": "2.0"}})
        assert "spread" not in panel.current_space().params

    def test_a_walk_over_a_multi_valued_field_is_refused_by_name(self, panel):
        panel._adaptive.setChecked(True)
        panel._value_edits["n_neighbors"].setText("5, 15, 50")
        with pytest.raises(ValueError, match="n_neighbors"):
            panel.current_space()

    def test_the_axis_button_follows_the_walk_toggle(self, panel):
        panel._adaptive.setChecked(False)
        assert not panel._walk_axes_button.isEnabled()
        panel._adaptive.setChecked(True)
        assert panel._walk_axes_button.isEnabled()


class TestDialog:

    def test_every_structural_parameter_gets_a_row(self, panel, qtbot):
        dialog = WalkAxesDialog(panel)
        qtbot.addWidget(dialog)
        assert set(dialog._rows) == set(UMAP_WALK_PARAMETERS)

    def test_the_round_cost_is_shown_before_it_is_paid(self, panel, qtbot):
        dialog = WalkAxesDialog(panel)
        qtbot.addWidget(dialog)
        for name in ("n_neighbors", "min_dist"):
            dialog._rows[name][0].setChecked(True)
        # Two axes at resolution 2 is the four diagonal corners: an even
        # resolution leaves the centre OUT of the axis, so nothing is
        # subtracted for it. The dialog gets this number from the engine.
        assert "4 fits per round" in dialog._cost.text()

    def test_the_fallback_is_announced_in_the_dialog(self, panel, qtbot):
        """A user should learn that ten axes stops being a neighbourhood
        HERE, not by waiting seventeen hours for one step."""
        dialog = WalkAxesDialog(panel)
        qtbot.addWidget(dialog)
        for enable, _start, _res in dialog._rows.values():
            enable.setChecked(True)
        dialog._update_cost()
        assert str(MAX_WALK_CANDIDATES_PER_ROUND) in dialog._cost.text()
        assert "ONE axis at a time" in dialog._cost.text()

    def test_no_selection_says_what_will_happen_anyway(self, panel, qtbot):
        dialog = WalkAxesDialog(panel)
        qtbot.addWidget(dialog)
        for enable, _s, _r in dialog._rows.values():
            enable.setChecked(False)
        dialog._update_cost()
        assert "original two" in dialog._cost.text()

    def test_categorical_axes_get_a_list_not_a_free_text_field(
            self, panel, qtbot):
        from PySide6.QtWidgets import QComboBox
        dialog = WalkAxesDialog(panel)
        qtbot.addWidget(dialog)
        assert isinstance(dialog._rows["metric"][1], QComboBox)
        assert isinstance(dialog._rows["init"][1], QComboBox)

    def test_the_dialog_paints_its_own_containers(self, panel, qtbot):
        """An anonymous QWidget in a scroll area is a black rectangle
        without this -- see INVARIANTS 1 and 3."""
        dialog = WalkAxesDialog(panel)
        qtbot.addWidget(dialog)
        assert "WalkAxesPage" in dialog.styleSheet()

    def test_the_selection_round_trips_through_the_panel(self, panel, qtbot):
        dialog = WalkAxesDialog(panel)
        qtbot.addWidget(dialog)
        dialog._rows["spread"][0].setChecked(True)
        dialog._rows["spread"][2].setValue(5)
        panel.set_walk_axes(dialog.selection())
        assert panel.walk_axes()["spread"]["resolution"] == 5
