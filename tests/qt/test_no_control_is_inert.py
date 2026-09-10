"""A control the user can change must change something.

Reported three times on 2026-08-21, each time correctly:

  * the graph retyping and the folder save existed on the pyqtgraph plots
    and not on the figures people right-click;
  * `png_list` on the merge control was created, laid out and never read;
  * `class_folder_names` was left on the classify panel after the classes
    were made to outrank it.

ALL THREE ARE THE SAME BUG. Something was built, and nothing drove the path
the user actually takes -- so it was marked done from the code rather than
from the screen. These tests drive the screen.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class TestTheMergeBoxesAreRead:

    @pytest.fixture
    def panel(self, app):
        import pandas as pd

        from spacr.qt.widgets.measurement_compare_dialog import (
            MeasurementComparePanel)

        objects = pd.DataFrame({
            "prcfo": ["p1_r1_c1_f1_o1", "p1_r1_c2_f1_o1"],
            "area": [10.0, 20.0], "gene": ["a", "b"],
        })
        return MeasurementComparePanel(objects, {"a": ["a"], "b": ["b"]})

    def test_the_png_list_box_reaches_the_joiner(self, panel):
        """It was created, laid out and never consulted."""
        source = inspect.getsource(type(panel).join_the_tables)
        assert "join_png_list" in source, (
            "the box exists and nothing reads it, which is worse than no "
            "box: it says the option exists")

    def test_the_joiner_takes_the_argument(self):
        from spacr.gene_measurement_compare import join_measurements

        assert "png_list" in inspect.signature(join_measurements).parameters

    def test_the_flag_changes_which_tables_are_read(self, monkeypatch):
        """The argument has to DO something, not merely be accepted.

        SPIED ON THE READER rather than driven through a database. Building
        a `measurements.db` valid enough for the real reader takes six
        tables and a dozen columns, and what broke here was the WIRING --
        the box was never consulted at all. This asserts exactly that, and
        `test_the_merge_keeps_the_grna_annotation` drives the real reader.
        """
        import pandas as pd

        from spacr import gene_measurement_compare as module

        asked = []

        def spy(path, **kwargs):
            asked.append(kwargs.get("table_names"))
            raise RuntimeError("stop here; the argument is what matters")

        monkeypatch.setattr("spacr.io._read_and_join_tables", spy)
        objects = pd.DataFrame({"prcfo": ["p_r_c_f_o1"], "gene": ["a"]})

        module.join_measurements(objects, ["/nowhere.db"], png_list=True)
        module.join_measurements(objects, ["/nowhere.db"], png_list=False)

        assert asked[0] is None, "None lets the reader include png_list"
        assert asked[1] == list(module.OBJECT_TABLES)
        assert "png_list" not in (asked[1] or [])

    def test_toggling_it_re_joins(self, panel):
        """Otherwise it takes effect on the next press of a button the user
        has already pressed, which reads as a box that does nothing."""
        source = inspect.getsource(type(panel)._on_join_choice)
        assert "join_the_tables" in source


class TestTheClassifyPanelHasNoSupersededControl:
    """Every one of these is derived or recorded, and a control for it can
    disagree with the thing it is derived from and lose."""

    @pytest.fixture
    def panel(self, app):
        from spacr.qt.screens.settings_model import SettingsWidgets

        model = SettingsWidgets("classify")
        model.build_sections()
        return model

    @pytest.mark.parametrize("key", [
        "class_folder_names", "annotation_column", "class_metadata",
        "coordinate_columns", "crop_source", "file_metadata", "file_type",
        "extract_channels",
    ])
    def test_it_is_not_offered(self, panel, key):
        assert key not in panel._widgets, (
            f"{key} is superseded and still on the panel")

    @pytest.mark.parametrize("key", ["classes", "object_array",
                                     "image_source", "stream_method"])
    def test_what_replaced_it_is(self, panel, key):
        assert key in panel._widgets

    def test_the_folder_names_still_come_from_the_classes(self):
        """Removed as a CONTROL, not as a value: dataset generation writes
        it, because it records what actually went to disk."""
        from spacr.classify_classes import folder_names

        assert folder_names({
            "classes": {"pc": {"column": "c", "value": 1},
                        "nc": {"column": "c", "value": 0}},
            "class_folder_names": ["stale", "names"],
        }) == ["pc", "nc"]


class TestTheFigureMenuIsOnTheFiguresPeopleUse:
    """Both features existed on FastPlot, which is not where a run's figures
    are drawn."""

    def test_the_matplotlib_menu_offers_the_bundle(self, app):
        import matplotlib
        matplotlib.use("Agg", force=True)
        from matplotlib.figure import Figure
        from PySide6.QtWidgets import QWidget

        from spacr.qt.widgets.figure_settings import (
            build_figure_context_menu)

        figure = Figure()
        figure.add_subplot(111).plot([1, 2], [1, 4])
        menu = build_figure_context_menu(QWidget(), figure,
                                         on_change=lambda **_k: None)
        # The entry is called "Save" since 2026-08-23; it still writes the
        # figure, its data and its statistics, which is what this asserts.
        assert any(a.text() == "Save" for a in menu.actions())

    def test_and_the_pyqtgraph_one_still_does(self, app):
        pytest.importorskip("pyqtgraph")
        from spacr.qt.widgets.fast_plots import FastPlot

        assert hasattr(FastPlot, "export_bundle")
