"""Instruction 186 A: multivariate picking asks rather than substituting.

"if the user has chosen multivariat cell picking and presses ok and the
measurements havnt been swept and or merged. they should be prompted with a
popup asking if they want to merge or sweep or both depending on what has
already been done."

WHAT THE AUDIT FOUND, which is larger than the report. `select_montage` has
taken an `effects_grid` argument since option C shipped and NO CALLER EVER
SET ONE -- so the multivariate picker could not run from the GUI at any
point. It found `None` every time, fell back to the single-score
attribution, and said so in the caption. The fallback worked exactly as
designed and hid the fact that what it fell back FROM was unreachable.

`SweepResult.effects` lived only in the sweep panel's memory and was never
written down, so there was nothing to read even in the session that made it.
It is now written beside the run, which is also what lets a LATER session use
it -- a panel-to-panel handover would work exactly once.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import pandas as pd


class TestTheGridIsWrittenAndRead:

    def test_a_run_with_no_sweep_has_no_grid(self, tmp_path):
        from spacr.cell_montage import effects_grid_from_results

        assert effects_grid_from_results(str(tmp_path)) is None

    def test_what_the_sweep_writes_is_what_the_montage_reads(self, tmp_path):
        from spacr.cell_montage import (effects_grid_from_results,
                                        write_effects_grid)

        effects = pd.DataFrame(
            {"cell_area": [0.4, -0.2], "pathogen_area": [0.1, 0.9]},
            index=["225160_1", "233460_4"])

        written = write_effects_grid(effects, str(tmp_path))
        back = effects_grid_from_results(str(tmp_path))

        assert written
        assert back is not None
        assert list(back.columns) == ["cell_area", "pathogen_area"]
        assert set(back.index) == {"225160_1", "233460_4"}
        assert back.loc["233460_4", "pathogen_area"] == pytest.approx(0.9)

    def test_a_results_csv_finds_the_grid_beside_it(self, tmp_path):
        """`results_path` may be the file, not the folder."""
        from spacr.cell_montage import (effects_grid_from_results,
                                        write_effects_grid)

        write_effects_grid(
            pd.DataFrame({"cell_area": [0.4]}, index=["g1"]), str(tmp_path))
        results = tmp_path / "results.csv"
        results.write_text("feature,coefficient\ng1,0.4\n")

        assert effects_grid_from_results(str(results)) is not None

    def test_an_empty_grid_is_no_grid(self, tmp_path):
        from spacr.cell_montage import (EFFECTS_GRID_FILE,
                                        effects_grid_from_results)

        (tmp_path / EFFECTS_GRID_FILE).write_text("")

        assert effects_grid_from_results(str(tmp_path)) is None

    def test_writing_nothing_writes_nothing(self, tmp_path):
        from spacr.cell_montage import write_effects_grid

        assert write_effects_grid(None, str(tmp_path)) == ""
        assert write_effects_grid(pd.DataFrame(), str(tmp_path)) == ""


class TestItAsksInsteadOfSubstituting:

    @pytest.fixture
    def view(self, qtbot, tmp_path):
        from spacr.qt.widgets.cell_montage_view import CellMontageView

        widget = CellMontageView(results_provider=lambda: str(tmp_path))
        qtbot.addWidget(widget)
        widget._picture_settings = {"cell_picking": "multivariate"}
        return widget

    def test_it_says_what_is_missing_and_where_to_do_it(self, view):
        shortfall = view.multivariate_shortfall()

        assert "sweep" in shortfall
        assert "Measurements tab" in shortfall, "name where the work happens"
        assert "rank" in shortfall, "and the way forward that is not Cancel"

    def test_a_run_that_has_been_swept_says_nothing(self, view, tmp_path):
        from spacr.cell_montage import write_effects_grid

        write_effects_grid(
            pd.DataFrame({"cell_area": [0.4]}, index=["g1"]), str(tmp_path))

        assert view.multivariate_shortfall() == ""

    def test_another_picker_is_never_asked_about(self, view):
        view._picture_settings = {"cell_picking": "rank"}

        assert view.multivariate_shortfall() == ""

    def test_choosing_rank_uses_rank_and_says_so(self, view):
        view._ask_about_multivariate = lambda _text: "rank"

        assert view._multivariate_is_ready(None) is True
        assert view.picture_settings()["cell_picking"] == "rank", (
            "the caption has to say rank because it IS rank")
        assert "rank" in view.status_text()

    def test_cancelling_does_not_build(self, view):
        view._ask_about_multivariate = lambda _text: "cancel"

        assert view._multivariate_is_ready(None) is False

    def test_the_saved_choice_is_not_rewritten(self, view):
        """Their multivariate is still what they want once a sweep exists."""
        view._ask_about_multivariate = lambda _text: "rank"
        view._multivariate_is_ready(None)

        assert view._picture_settings["cell_picking"] == "multivariate"

    def test_just_this_once_does_not_become_permanent(self, view):
        view._ask_about_multivariate = lambda _text: "rank"
        view._multivariate_is_ready(None)

        view.clear_picking_override()

        assert view.picture_settings()["cell_picking"] == "multivariate"


class TestThePrefixIsMeasuredNotHardCoded:
    """The same assumption instruction 184 is about, in the effects join."""

    def test_the_matching_rule_does_not_name_one_organism(self):
        import pathlib

        source = pathlib.Path(
            "spacr/qt/widgets/cell_montage_view.py").read_text()
        # The comment explaining the fix may name it; the CODE must not.
        code = [line for line in source.splitlines()
                if 'TGGT1_' in line and not line.strip().startswith("#")]

        assert not code, (
            f"one organism's name written into the matching rule: {code}")
