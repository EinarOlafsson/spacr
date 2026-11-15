"""Feature Explorer: loading a table, filtering and computing, exporting.

A computed column is a property of an object and is computed over the whole
table, while the ranking is a statement about a population and is computed
over the filtered rows. Loading routes text files past the table listing and
databases through it; the export writes every statistic, in rank order.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.screens import feature_explorer as fe
from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
from spacr.qt.widgets.formula import ColumnFormula
from spacr.selection import CategoryFilter, DataFilter

pytestmark = pytest.mark.qt

STATISTICS = ["feature", "rank", "separation", "statistic", "auc", "cohen_d",
              "ks", "mutual_info", "higher_in", "against", "min_n",
              "shape_not_shift"]


@pytest.fixture
def measurements():
    """Two classes that a couple of the columns separate cleanly."""
    n = 24
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": [f"r{1 + i % 4}" for i in range(n)],
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "condition": ["treated" if i % 2 else "control" for i in range(n)],
        "cell_area": [100.0 + 40.0 * (i % 2) + i * 0.1 for i in range(n)],
        "cell_perimeter": [40.0 + i * 0.01 for i in range(n)],
        "cell_intensity": [5.0 + (i % 3) for i in range(n)],
    })


@pytest.fixture
def screen(qtbot):
    """A Feature Explorer on a PRIVATE link, reading inline."""
    widget = FeatureExplorerScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    return widget


def _picker(screen):
    return [screen._table_picker.itemText(i)
            for i in range(screen._table_picker.count())]


def test_loading_lists_database_tables_and_skips_text_files(
        screen, measurements, tmp_path, monkeypatch, subtests):
    """Databases fill the picker; text suffixes never ask for table names."""
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as db:
        measurements.to_sql("cell", db, index=False)
        measurements.head(10).to_sql("nucleus", db, index=False)

    with subtests.test(msg="a database offers its tables and loads the first"):
        screen.load_path(str(database))
        assert _picker(screen) == ["cell", "nucleus"]
        assert screen._table_picker.isHidden() is False
        assert screen._source.text() == (
            "measurements.db · cell · 24 rows × 9 columns")
        assert (screen.is_busy(), screen.active_jobs()) == (False, 0)
        assert len(screen.explorer._frame) == 24

    with subtests.test(msg="picking another table reloads the same file"):
        screen._on_table_picked("nucleus")
        assert screen._source.text() == (
            "measurements.db · nucleus · 10 rows × 9 columns")
        assert len(screen._frame) == 10

    with subtests.test(msg="a blank pick reloads nothing"):
        screen._on_table_picked("")
        assert len(screen._frame) == 10

    with subtests.test(msg="a file whose tables cannot be listed says so"):
        broken = tmp_path / "not-a-database.db"
        broken.write_bytes(b"this is not sqlite")
        fresh = FeatureExplorerScreen(link=LinkedSelection(), threaded=False)
        try:
            fresh.load_path(str(broken))
            assert fresh._source.text().startswith(
                "could not read not-a-database.db: ")
            assert fresh._frame is None
        finally:
            fresh.close()
            fresh.deleteLater()

    asked = []

    def watched(path):
        asked.append(path)
        raise AssertionError("a text file was asked for its tables")

    monkeypatch.setattr(fe, "table_names", watched)
    for suffix, sep in [(".csv", ","), (".tsv", "\t"), (".txt", ","),
                        (".CSV", ",")]:
        with subtests.test(msg=f"{suffix} is read without a table listing"):
            target = tmp_path / f"features{suffix}"
            target.write_text(f"condition{sep}cell_area\na{sep}10\nb{sep}12\n",
                              encoding="utf-8")
            screen.load_path(str(target))
            assert asked == []
            assert _picker(screen) == []
            assert screen._table_picker.isHidden() is True
            assert screen._source.text() == (
                f"features{suffix} · 2 rows × 2 columns")


def test_filters_and_formulas_reach_the_panel_over_the_right_rows(
        screen, measurements, subtests):
    """Ranking follows the filter; computed columns cover the whole table."""
    with subtests.test(msg="a loaded table reaches panel, filter and formulas"):
        screen.set_frame(measurements)
        assert screen._source.text() == "24 rows × 9 columns"
        assert len(screen.explorer._frame) == 24
        assert len(screen.filters._frame) == 24
        assert screen.formulas.computed_frame().shape == (24, 9)
        labels = [screen.explorer._label.itemText(i)
                  for i in range(screen.explorer._label.count())]
        assert "condition" in labels

    with subtests.test(msg="the ranking is over the filtered rows"):
        screen._link.set_filter(DataFilter([CategoryFilter("rowID", ("r1",))]))
        screen._on_filter_changed()
        assert len(screen.explorer._frame) == 6
        assert set(screen.explorer._frame["rowID"]) == {"r1"}

    with subtests.test(msg="a computed column is over the whole table"):
        assert screen.formulas.add_formula(
            ColumnFormula("roundness", "cell_area / cell_perimeter"))
        screen._on_formulas_changed()
        computed = screen.formulas.computed_frame()
        assert len(computed) == 24
        assert computed["roundness"].round(6).tolist() == (
            measurements["cell_area"] / measurements["cell_perimeter"]
        ).round(6).tolist()
        assert "roundness" in screen.explorer._frame.columns
        assert len(screen.explorer._frame) == 6

    with subtests.test(msg="a filter on a missing column leaves the rows"):
        screen._link.set_filter(
            DataFilter([CategoryFilter("nothing_here", ("x",))]))
        assert len(screen._visible(measurements)) == 24
        screen._on_filter_changed()
        assert len(screen.explorer._frame) == 24


def test_the_ranking_exports_every_statistic_in_rank_order(
        screen, measurements, tmp_path, monkeypatch, subtests):
    """Before ranking nothing is written; after it, every statistic is."""
    with subtests.test(msg="nothing ranked yet"):
        assert screen.ranking_frame() is None
        assert screen.export_ranking(str(tmp_path / "early.csv")) is None
        assert screen._source.text() == "Nothing ranked yet."
        assert not (tmp_path / "early.csv").exists()

    screen.set_frame(measurements)
    screen.explorer._label.setCurrentText("condition")
    with subtests.test(msg="the spec is what the screen would save"):
        assert screen.spec.label == "condition"

    screen.explorer.rank_now()
    with subtests.test(msg="export carries every statistic"):
        out = screen.export_ranking(str(tmp_path / "ranking.csv"))
        assert out == str(tmp_path / "ranking.csv")
        written = pd.read_csv(out)
        assert list(written.columns) == STATISTICS
        assert written["rank"].tolist() == list(range(1, len(written) + 1))
        assert written["feature"].iloc[0] == "cell_area"
        assert screen._source.text() == "Ranking written to ranking.csv"

    for case_id, answer, exists in [("cancelled", "", False),
                                    ("chosen", "chosen.csv", True)]:
        with subtests.test(msg=f"a {case_id} export dialog"):
            asked = []
            target = str(tmp_path / answer) if answer else ""
            monkeypatch.setattr(
                fe.QFileDialog, "getSaveFileName",
                staticmethod(lambda *a, t=target, **k: asked.append(a[1:3])
                             or (t, "")))
            screen.choose_export()
            assert asked == [("Export the ranking", "feature_ranking.csv")]
            assert (tmp_path / "chosen.csv").exists() is exists
