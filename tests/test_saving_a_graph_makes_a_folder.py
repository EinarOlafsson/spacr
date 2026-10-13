"""Saving a graph writes a folder, not a file (instruction 223).

"whenever i save a graph i want to generate a folder not a file. the folder
should contain a pdf and a png version of the graph, and a csv with the data
the graph is based on, and another csv which should show in a standardised
way, statistics on the data".

A PDF ON ITS OWN CANNOT BE CHECKED. Six months later the question is always
what the numbers were and whether the difference was tested, and a figure
file answers neither.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr.figures.bundle import NOTHING_TO_COMPARE, save, statistics_frame


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(0)
    return {"nc": rng.normal(0.0, 1.0, 40), "pc": rng.normal(1.0, 1.0, 40)}


def _drawn(path):
    with open(path, "wb") as handle:
        handle.write(b"%PDF-1.4\n")


class TestTheFolderHasEverything:

    def test_all_four_files(self, tmp_path, two_groups):
        out = save(str(tmp_path), "volcano", render=_drawn,
                   data=pd.DataFrame({"a": [1, 2]}), groups=two_groups)
        got = set(os.listdir(out))
        assert {"volcano.pdf", "volcano.png", "data.csv", "statistics.csv",
                "settings.json"} <= got

    def test_the_folder_is_named_for_the_graph(self, tmp_path):
        out = save(str(tmp_path), "volcano", render=_drawn)
        assert os.path.basename(out) == "volcano"

    def test_the_data_is_the_rows_it_was_drawn_from(self, tmp_path):
        frame = pd.DataFrame({"gene": ["a", "b"], "effect": [1.0, -2.0]})
        out = save(str(tmp_path), "g", render=_drawn, data=frame)
        back = pd.read_csv(os.path.join(out, "data.csv"))
        assert list(back["gene"]) == ["a", "b"]

    def test_the_settings_are_beside_the_data(self, tmp_path):
        """Without the filters recorded the numbers cannot be reproduced."""
        out = save(str(tmp_path), "g", render=_drawn,
                   settings={"fraction_threshold": 0.02})
        with open(os.path.join(out, "settings.json")) as handle:
            assert json.load(handle)["fraction_threshold"] == 0.02


class TestTheStatisticsAreTheSharedOnes:

    def test_they_come_from_figures_stats_compare(self, two_groups,
                                                  monkeypatch):
        """Not a second implementation: a figure whose saved statistics
        disagree with the same comparison on screen is worse than one with
        no statistics at all."""
        called = {}
        from spacr.figures import stats as real

        original = real.compare

        def spy(groups, **kwargs):
            called["yes"] = True
            return original(groups, **kwargs)

        monkeypatch.setattr(real, "compare", spy)
        statistics_frame(two_groups, unit="well")
        assert called.get("yes")

    def test_it_names_the_unit(self, two_groups):
        frame = statistics_frame(two_groups, unit="well")
        assert (frame["item"] == "unit").any()
        assert frame.loc[frame["item"] == "unit", "value"].iloc[0] == "well"

    def test_it_gives_n_per_group(self, two_groups):
        frame = statistics_frame(two_groups, unit="well")
        assert (frame["item"] == "n [nc]").any()
        assert (frame["item"] == "n [pc]").any()

    def test_it_records_both_assumption_checks(self, two_groups):
        frame = statistics_frame(two_groups, unit="well")
        items = " ".join(map(str, frame["item"]))
        # Whatever the checks are called, both a normality and a variance
        # check must be recorded -- asserted through the Comparison rather
        # than by naming the tests, which can change.
        from spacr.figures.stats import compare

        names = [a.name for a in compare(two_groups, unit="well").assumptions]
        assert len(names) >= 2
        for name in names:
            assert name in items

    def test_it_names_the_test_and_the_reason(self, two_groups):
        frame = statistics_frame(two_groups, unit="well")
        row = frame.loc[frame["item"] == "test"]
        assert not row.empty
        assert str(row["note"].iloc[0]).strip(), (
            "the reason it was chosen over the alternative")

    def test_it_gives_an_effect_size_with_an_interval(self, two_groups):
        frame = statistics_frame(two_groups, unit="well")
        items = list(frame["item"])
        assert "effect_size" in items
        assert "effect_ci_low" in items and "effect_ci_high" in items

    def test_a_check_that_could_not_see_says_so(self):
        """A check that could not see is not a check that passed, and a file
        recording it as "holds" would be the more misleading of the two."""
        rng = np.random.default_rng(1)
        small = {"a": rng.normal(0, 1, 4), "b": rng.normal(0, 1, 4)}
        frame = statistics_frame(small, unit="well")
        text = " ".join(map(str, frame["value"]))
        assert "could not tell" in text or "refused" in text


class TestNothingToCompareIsAnAnswer:

    def test_a_graph_with_no_groups_still_gets_every_file(self, tmp_path):
        out = save(str(tmp_path), "scatter", render=_drawn,
                   data=pd.DataFrame({"x": [1, 2, 3]}))
        assert os.path.isfile(os.path.join(out, "statistics.csv"))

    def test_and_the_file_says_why(self, tmp_path):
        frame = statistics_frame(None)
        assert frame["note"].iloc[0] == NOTHING_TO_COMPARE

    def test_one_group_is_not_a_comparison(self):
        frame = statistics_frame({"only": [1, 2, 3]})
        assert frame["value"].iloc[0] == "none"

    def test_a_refused_comparison_says_refused_not_nan(self):
        """`compare` raises rather than returning NaN: a comparison that
        could not be made is not one with an unknown answer."""
        frame = statistics_frame({"a": [1.0], "b": [2.0]})
        assert frame["value"].iloc[0] == "refused"
        assert str(frame["note"].iloc[0]).strip()


class TestOverwritingIsNotSilent:

    def test_a_second_save_sits_beside_the_first(self, tmp_path):
        """Overwriting a folder is a bigger act than overwriting a file: a
        folder replaced can take the data and statistics of an earlier save
        with it."""
        first = save(str(tmp_path), "g", render=_drawn)
        second = save(str(tmp_path), "g", render=_drawn)
        assert first != second
        assert os.path.isdir(first) and os.path.isdir(second)

    def test_an_unsafe_name_cannot_escape_the_folder(self, tmp_path):
        out = save(str(tmp_path), "../../etc/passwd", render=_drawn)
        assert os.path.dirname(os.path.abspath(out)) == \
            os.path.abspath(str(tmp_path))


class TestThroughThePlot:

    def test_the_plot_writes_a_bundle(self, tmp_path):
        pytest.importorskip("PySide6")
        pytest.importorskip("pyqtgraph")
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.fast_plots import FastPlot

        QApplication.instance() or QApplication([])
        plot = FastPlot(title="My Graph")
        out = plot.export_bundle(str(tmp_path))
        assert os.path.basename(out) == "My Graph"
        assert os.path.isfile(os.path.join(out, "My Graph.pdf"))
        assert os.path.isfile(os.path.join(out, "My Graph.png"))

    def test_a_plain_plot_has_no_groups_to_invent(self, tmp_path):
        """Inventing groups would put a p-value in the folder for a
        comparison nobody made."""
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.fast_plots import FastPlot

        QApplication.instance() or QApplication([])
        assert FastPlot(title="x").comparison_groups() is None
