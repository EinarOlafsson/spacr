"""What the volcano explorer does with clicks, drops, styles, menus and exports.

Every case drives a real :class:`VolcanoExplorer` with a real results frame and
the real renderer; only the file and message dialogs -- which cannot be
answered offscreen -- are replaced.  One general test per behaviour family,
each walking a case table with pytest's ``subtests`` fixture and asserting the
exact value, column, path or message the widget produces.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDropEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QCheckBox, QDoubleSpinBox, QFileDialog, QInputDialog, QMessageBox, QWidget,
)

from spacr.qt.widgets.volcano_explorer import (  # noqa: E402
    VolcanoExplorer, _setting_named_in,
)
from spacr.volcano_style import VolcanoStyle  # noqa: E402

pytestmark = pytest.mark.qt

_COMPARTMENTS = ["dense granules", "rhoptries 1"]


def _results(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "guide": [f"G{i:03d}" for i in range(n)],
        "gene": [f"TGGT1_{i // 3:06d}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(size=n),
        "adjusted_p_value": rng.random(n) * 0.5 + 1e-6,
        "p_value": rng.random(n),
    })


def _build(qtbot, frame=None) -> VolcanoExplorer:
    widget = VolcanoExplorer(_results() if frame is None else frame)
    qtbot.addWidget(widget)
    return widget


def _localised_frame() -> pd.DataFrame:
    """Ten genes each of two compartments the bundled LOPIT table knows."""
    from spacr.localisation import table

    lookup = table()
    if not lookup:
        pytest.skip("no bundled localisation table to colour by")
    genes = []
    for compartment in _COMPARTMENTS:
        genes += [f"TGGT1_{key}" for key, place in lookup.items()
                  if place == compartment][:10]
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "guide": [f"{gene}_1" for gene in genes],
        "gene": genes,
        "standardized_marginal_effect": rng.normal(size=len(genes)),
        "adjusted_p_value": rng.random(len(genes)) * 0.5 + 1e-6,
    })


def _annotations(path, rows=8, **columns):
    frame = pd.DataFrame({"gene": [f"TGGT1_{i:06d}" for i in range(rows)],
                          **columns})
    if str(path).endswith(".xlsx"):
        frame.to_excel(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def _two_column_annotations(tmp_path):
    return _annotations(tmp_path / "annotations.csv",
                        localisation=["rhoptry", "micronemes"] * 4,
                        essentiality=np.linspace(0.0, 1.0, 8))


def _combo_data(combo) -> list:
    return [combo.itemData(i) for i in range(combo.count())]


class _Click:
    def __init__(self, xdata, ydata, inaxes):
        self.xdata, self.ydata, self.inaxes = xdata, ydata, inaxes


def _urls(paths):
    """A live QMimeData of local file URLs; the caller must hold it."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def _menu_action(menu, prefix: str):
    from spacr.qt.widgets.fast_plots import menu_entries

    for action in menu_entries(menu):
        if action.text().startswith(prefix):
            return action
    raise AssertionError(f"nothing on the menu starts with {prefix!r}")


# --------------------------------------------------------------------------
# which setting a renderer complaint blames
# --------------------------------------------------------------------------

def test_a_renderer_complaint_blames_the_setting_it_names(subtests):
    """The ``name=`` form wins, then the longest bare mention, else nothing."""
    cases = [
        ("spelled with equals", "x_column='foo' is not a column of the results",
         "x_column"),
        ("equals form on its own", "x_column='nope' is not a column",
         "x_column"),
        ("longer name beats its prefix", "colormap='nope' is not a colormap",
         "colormap"),
        ("bare mention", "the colormap could not be resolved", "colormap"),
        ("equals beats a suggested other", "x_column='foo' is not a column; "
         "try colormap", "x_column"),
        ("bare mention inside a sentence",
         "cannot open the font_family you asked for", "font_family"),
        ("nothing named", "the results table is empty", ""),
        ("nothing the panel holds", "cannot open the file you asked for", ""),
        ("empty message", "", ""),
        ("an exception, not a string", ValueError("colormap='nope' is unknown"),
         "colormap"),
    ]
    for case_id, message, expected in cases:
        with subtests.test(msg=case_id):
            assert _setting_named_in(message) == expected


# --------------------------------------------------------------------------
# results in, plot out, and clicking a point
# --------------------------------------------------------------------------

def test_new_results_replace_the_plot_and_a_click_selects_its_row(qtbot,
                                                                 subtests):
    """A click is a lookup in axis-normalised space; new results clear it."""
    explorer = _build(qtbot)
    frame = explorer.results()
    axes = explorer._panels[0]

    with subtests.test(msg="a click on a point selects that row"):
        row = 5
        x = float(frame["standardized_marginal_effect"][row])
        y = float(-np.log10(frame["adjusted_p_value"][row]))
        seen = []
        explorer.point_selected.connect(seen.append)
        explorer._on_click(_Click(x, y, axes))
        assert explorer.selected_index() == row
        assert seen[0]["guide"] == "G005"
        assert explorer._detail_hint.text() == "Selected G005"

    for case_id, click in [("off the axes", _Click(0.0, 0.0, None)),
                           ("empty space", _Click(1e6, 1e6, axes))]:
        with subtests.test(msg=f"a click {case_id} selects nothing"):
            explorer.set_results(_results())
            explorer._on_click(click)
            assert explorer.selected_index() is None

    with subtests.test(msg="the results handed back are a copy"):
        taken = explorer.results()
        taken.loc[0, "guide"] = "EDITED"
        assert explorer.results().loc[0, "guide"] == "G000"

    with subtests.test(msg="new results clear the selection"):
        explorer.select_point(3)
        assert explorer.selected_index() == 3
        explorer.set_results(_results(10))
        assert explorer.selected_index() is None
        assert len(explorer.results()) == 10
        assert _combo_data(explorer._controls["x_column"]) == [
            "standardized_marginal_effect", "adjusted_p_value", "p_value"]

    with subtests.test(msg="an empty result set draws and exports nothing"):
        empty = _build(qtbot, pd.DataFrame())
        empty.refresh()
        assert empty.nearest_point(0.0, 0.0) is None
        assert empty.export("pdf") is None
        assert empty.compartments() == []


def test_the_y_axis_split_and_a_broken_setting_keep_the_figure(qtbot,
                                                               subtests):
    """A split is derived, not refused; a bad setting keeps the plot."""
    with subtests.test(msg="a split axis gets limits derived"):
        explorer = _build(qtbot)
        explorer._style.split_axis = True
        explorer._style.split_y_lims = None
        explorer.refresh()
        y = explorer._plotted_y()
        cut, top = float(np.nanquantile(y, 0.99)), float(np.nanmax(y))
        pad = (top - cut) * 0.1
        np.testing.assert_allclose(
            np.asarray(explorer._style.split_y_lims),
            [[0.0, cut], [cut + pad * 0.5, top + pad]])

    with subtests.test(msg="a split that cannot be sized is left unset"):
        explorer = _build(qtbot)
        explorer.set_style(dataclasses.replace(
            explorer.style(), split_axis=True, split_y_lims=None,
            y_column="not_a_column"))
        assert explorer.style().split_y_lims is None
        assert list(explorer.problems()) == ["y_column"]

    with subtests.test(msg="the plotted y is the raw column with no log"):
        explorer = _build(qtbot)
        explorer._style.y_neg_log10 = False
        np.testing.assert_array_equal(
            explorer._plotted_y(),
            explorer.results()["adjusted_p_value"].to_numpy(float))

    with subtests.test(msg="a bad style keeps the figure"):
        explorer = _build(qtbot)
        explorer._style.x_column = "no_such_column"
        explorer.refresh()
        assert len(explorer._panels) == 1
        texts = [t.get_text() for t in explorer._panels[0].texts]
        assert not any("Cannot draw this plot" in t for t in texts)
        assert list(explorer.problems()) == ["x_column"]

    with subtests.test(msg="the fallback styles, best first"):
        fresh = _build(qtbot, pd.DataFrame())
        offered = list(fresh._drawable_styles({}))
        assert [style is fresh.style() for style in offered] == [True, False]
        assert offered[1] == VolcanoStyle()
        drawn = _build(qtbot)
        offered = list(drawn._drawable_styles({}))
        assert len(offered) == 3
        assert offered[0] is drawn.style()
        assert offered[1] == drawn._last_good_style
        assert offered[2] == VolcanoStyle()


# --------------------------------------------------------------------------
# annotation tables: merged, picked, dropped
# --------------------------------------------------------------------------

def test_an_annotation_table_merges_on_the_best_shared_column(qtbot, tmp_path,
                                                              subtests):
    """The number of new columns comes back; a file with no join raises."""
    both = tmp_path / "both.csv"
    pd.DataFrame({"guide": ["nothing_matches"] * 8,
                  "gene": [f"TGGT1_{i:06d}" for i in range(8)],
                  "localisation": ["rhoptry"] * 8}).to_csv(both, index=False)
    empty = tmp_path / "empty.csv"
    pd.DataFrame({"gene": []}).to_csv(empty, index=False)
    same = tmp_path / "same.csv"
    _results()[["gene"]].to_csv(same, index=False)
    excel = _annotations(tmp_path / "annotations.xlsx",
                         localisation=["rhoptry"] * 8)
    cases = [
        ("a two-column table", _two_column_annotations(tmp_path), {}, 2),
        ("the join that matches most rows", both, {}, 1),
        ("a caller-named join column", _two_column_annotations(tmp_path),
         {"on": "gene"}, 2),
        ("an empty file", empty, {}, 0),
        ("a file bringing no new column", same, {}, 0),
        ("an Excel table", excel, {}, 1),
    ]
    for case_id, path, kwargs, added in cases:
        with subtests.test(msg=case_id):
            explorer = _build(qtbot)
            assert explorer.merge_annotation_file(path, **kwargs) == added
            if added:
                merged = explorer.results()
                # Row 23 carries gene 7, the last row of every table.
                last = "micronemes" if added == 2 else "rhoptry"
                assert merged.loc[0, "localisation"] == "rhoptry"
                assert merged.loc[23, "localisation"] == last
                assert merged["localisation"].notna().all()
                assert "localisation" in _combo_data(
                    explorer._controls["color_by"])

    with subtests.test(msg="a file sharing no column says so"):
        explorer = _build(qtbot)
        unrelated = tmp_path / "unrelated.csv"
        pd.DataFrame({"orf": ["a"], "note": ["b"]}).to_csv(unrelated,
                                                           index=False)
        with pytest.raises(ValueError,
                           match=r"^unrelated\.csv shares no column with the "
                                 r"results"):
            explorer.merge_annotation_file(unrelated)


def test_annotation_and_style_files_arrive_by_dialog_and_by_drop(
        qtbot, tmp_path, monkeypatch, subtests):
    """Pickers, drops and the style save/load round trip; cancels do nothing."""
    annotation = _two_column_annotations(tmp_path)
    style_path = str(tmp_path / "style.json")

    with subtests.test(msg="a style saves and loads back"):
        explorer = _build(qtbot)
        explorer._style.title = "Saved title"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (style_path, "")))
        assert explorer._save_style() == style_path
        explorer._style.title = "Something else"
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (style_path, "")))
        assert explorer._load_style() == style_path
        assert explorer.style().title == "Saved title"

    with subtests.test(msg="cancelled dialogs change nothing"):
        explorer = _build(qtbot)
        for name in ("getSaveFileName", "getOpenFileName"):
            monkeypatch.setattr(QFileDialog, name,
                                staticmethod(lambda *a, **k: ("", "")))
        before = list(explorer.results().columns)
        title = explorer.style().title
        assert explorer._save_style() is None
        assert explorer._load_style() is None
        explorer._pick_annotation_file()
        assert explorer.style().title == title
        assert list(explorer.results().columns) == before

    for case_id, paths in [("one annotation table", [annotation]),
                           ("an unmergeable file before a good one",
                            ["unrelated", annotation])]:
        with subtests.test(msg=f"dropping {case_id} merges it"):
            explorer = _build(qtbot)
            bad = tmp_path / "unrelated.csv"
            pd.DataFrame({"orf": ["a"]}).to_csv(bad, index=False)
            mime = _urls([bad if p == "unrelated" else p for p in paths])
            event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime,
                               Qt.LeftButton, Qt.NoModifier)
            explorer.dropEvent(event)
            assert event.isAccepted()
            assert list(explorer.results().columns)[-2:] == [
                "localisation", "essentiality"]


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_an_export_re_renders_or_says_why_it_wrote_nothing(
        qtbot, tmp_path, monkeypatch, subtests):
    """The path written comes back; every failure is a warning and ``None``."""
    import spacr.plot as plot
    from spacr.qt.widgets import figure_settings

    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args: warned.append(args[1:3])))

    with subtests.test(msg="a PDF is rendered at print size"):
        explorer = _build(qtbot)
        path = tmp_path / "volcano.pdf"
        assert explorer.export("pdf", str(path)) == str(path)
        assert path.read_bytes().startswith(b"%PDF")

    with subtests.test(msg="the suffix the format names is added"):
        explorer = _build(qtbot)
        written = explorer.export("png", str(tmp_path / "volcano"))
        assert written == str(tmp_path / "volcano.png")
        assert os.path.exists(written)

    with subtests.test(msg="EPS goes through the general figure writer"):
        explorer = _build(qtbot)
        path = tmp_path / "volcano.eps"
        assert explorer.export("eps", str(path)) == str(path)
        assert path.read_bytes().startswith(b"%!PS")

    with subtests.test(msg="the save dialog is asked when no path is given"):
        explorer = _build(qtbot)
        target = str(tmp_path / "chosen.pdf")
        asked = []

        def _ask(parent, caption, name, filters):
            asked.append((caption, name, filters))
            return target, filters

        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(_ask))
        assert explorer.export("pdf") == target
        assert asked == [("Export volcano as PDF", "volcano.pdf",
                          "PDF (*.pdf)")]

    with subtests.test(msg="a cancelled dialog writes nothing"):
        explorer = _build(qtbot)
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        assert explorer.export("pdf") is None

    with subtests.test(msg="a render that fails is a warning"):
        warned.clear()
        explorer = _build(qtbot)
        explorer._style.x_column = "no_such_column"
        assert explorer.export("pdf", str(tmp_path / "x.pdf")) is None
        assert [title for title, _text in warned] == ["Export failed"]

    for case_id, fmt, patch in [
            ("an SVG writer that wrote nothing", "svg",
             (plot, "save_figure", lambda *a, **k: "")),
            ("an EPS writer that wrote nothing", "eps",
             (figure_settings, "save_figure_as", lambda *_a: ""))]:
        with subtests.test(msg=case_id):
            warned.clear()
            explorer = _build(qtbot)
            monkeypatch.setattr(*patch)
            path = str(tmp_path / f"missing.{fmt}")
            assert explorer.export(fmt, path) is None
            assert warned == [("Export failed",
                               f"Nothing could be written to {path}.")]

    with subtests.test(msg="an SVG writer that names its file"):
        warned.clear()
        explorer = _build(qtbot)
        target = str(tmp_path / "volcano.svg")
        monkeypatch.setattr(plot, "save_figure", lambda *a, **k: target)
        assert explorer.export("svg", target) == target
        assert warned == []


# --------------------------------------------------------------------------
# the right-click menu and the side panel are two routes to one style
# --------------------------------------------------------------------------

def test_the_menu_and_the_panel_offer_the_same_closed_sets(qtbot, subtests):
    """Choices are read off the panel's own controls, so they cannot drift."""
    fields = {f.name for f in dataclasses.fields(VolcanoStyle)}

    with subtests.test(msg="both routes offer every style field"):
        explorer = _build(qtbot)
        assert explorer.menu_settings() == explorer.panel_settings() == fields

    with subtests.test(msg="the closed sets are real fields the panel closes"):
        explorer = _build(qtbot)
        choices = explorer._style_choices()
        assert set(choices) <= fields
        assert choices["x_column"] == [
            "standardized_marginal_effect", "adjusted_p_value", "p_value"]
        assert choices["color_by"] == [None, "guide", "gene"]
        assert choices["label_column"] == [
            "guide", "gene", "standardized_marginal_effect",
            "adjusted_p_value", "p_value"]
        assert choices["marker"][0] == "o"
        for closed in ("marker", "colormap", "line_style", "font_family"):
            assert closed in choices, closed

    with subtests.test(msg="a column with no name is offered by its text"):
        frame = _results()
        frame[None] = np.arange(len(frame), dtype=float)
        choices = _build(qtbot, frame)._style_choices()
        assert "None" in choices["x_column"]
        assert None in choices["color_by"]

    with subtests.test(msg="an empty picker is not offered at all"):
        choices = _build(qtbot, pd.DataFrame())._style_choices()
        assert "x_column" not in choices
        assert choices["marker"][0] == "o"

    with subtests.test(msg="dropping one control makes the sets differ"):
        explorer = _build(qtbot)
        del explorer._controls["marker_size"]
        assert explorer.menu_settings() - explorer.panel_settings() == {
            "marker_size"}


def test_a_limit_or_label_typed_into_the_menu_reaches_figure_panel_and_file(
        qtbot, tmp_path, monkeypatch, subtests):
    """Measured off the axes, the side panel and the written SVG."""
    with subtests.test(msg="an x limit"):
        explorer = _build(qtbot)
        answers = iter([(-8.0, True), (8.0, True)])
        monkeypatch.setattr(QInputDialog, "getDouble",
                            staticmethod(lambda *a, **k: next(answers)))
        _menu_action(explorer.build_style_menu(), "X lim").trigger()
        assert explorer.style().x_lim == (-8.0, 8.0)
        explorer.refresh()
        assert explorer._panels[0].get_xlim() == (-8.0, 8.0)
        assert explorer._controls["x_lim"].value() == (-8.0, 8.0)
        path = tmp_path / "limited.svg"
        assert explorer.export("svg", str(path)) == str(path)
        written = path.read_text(encoding="utf-8")
        ticks = [t.get_text() for t in explorer._panels[0].get_xticklabels()]
        assert "−8" in ticks
        assert [tick for tick in ticks if tick not in written] == []

    with subtests.test(msg="an x label"):
        explorer = _build(qtbot)
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("Effect, typed",
                                                          True)))
        _menu_action(explorer.build_style_menu(), "X label").trigger()
        assert explorer.style().x_label == "Effect, typed"
        explorer.refresh()
        assert explorer._panels[0].get_xlabel() == "Effect, typed"
        assert explorer._controls["x_label"].text() == "Effect, typed"
        path = tmp_path / "labelled.svg"
        explorer.export("svg", str(path))
        assert "Effect, typed" in path.read_text(encoding="utf-8")


def test_the_panel_controls_write_exactly_what_they_show(qtbot, subtests):
    """Optional numbers, spin boxes and unknown controls round-trip."""
    with subtests.test(msg="unticking auto writes the spin boxes' numbers"):
        explorer = _build(qtbot)
        control = explorer._controls["x_lim"]
        assert explorer.style().x_lim is None
        spins = control.findChildren(QDoubleSpinBox)
        spins[0].setValue(-3.0)
        spins[1].setValue(3.0)
        told = []
        explorer.style_changed.connect(lambda: told.append(1))
        auto = control.findChild(QCheckBox)
        auto.setChecked(False)
        assert told == [1]
        assert explorer.style().x_lim == (-3.0, 3.0)
        assert [spin.isEnabled() for spin in spins] == [True, True]
        auto.setChecked(True)
        assert explorer.style().x_lim is None
        assert [spin.isEnabled() for spin in spins] == [False, False]

    with subtests.test(msg="a limit can be taken back off from the panel"):
        explorer = _build(qtbot)
        explorer._controls["x_lim"].setValue((-3.0, 3.0))
        explorer._on_control_changed()
        assert explorer.style().x_lim == (-3.0, 3.0)
        explorer._controls["x_lim"].setValue(None)
        explorer._on_control_changed()
        assert explorer.style().x_lim is None
        assert explorer._panels[0].get_xlim() != (-3.0, 3.0)

    with subtests.test(msg="touching another control leaves the threshold"):
        explorer = _build(qtbot)
        explorer._controls["marker_size"].setValue(31.0)
        assert explorer.style().marker_size == 31.0
        assert explorer.style().effect_threshold is None

    with subtests.test(msg="a numeric setting of None leaves the spin box"):
        explorer = _build(qtbot)
        explorer.set_style(dataclasses.replace(explorer.style(), dpi=600))
        assert explorer._controls["dpi"].value() == 600
        explorer.set_style(dataclasses.replace(explorer.style(), dpi=None))
        assert explorer._controls["dpi"].value() == 600

    with subtests.test(msg="an unknown control is stepped over"):
        explorer = _build(qtbot)
        explorer._controls["unknown_probe"] = QWidget(explorer)
        explorer._controls["title"] = explorer._controls.pop("title")
        explorer.set_style(dataclasses.replace(explorer.style(), title="Kept"))
        assert explorer._controls["title"].text() == "Kept"
        explorer._controls["title"].setText("  Typed by hand ")
        explorer._pull_style_from_controls()
        assert explorer.style().title == "Typed by hand"


# --------------------------------------------------------------------------
# colouring by localisation compartment
# --------------------------------------------------------------------------

def test_compartments_are_offered_ticked_and_drawn(qtbot, monkeypatch,
                                                  subtests):
    """Only this screen's compartments; each tick reaches style and plot."""
    frame = _localised_frame()

    with subtests.test(msg="only the compartments this screen has"):
        explorer = _build(qtbot, frame)
        assert explorer.compartments() == _COMPARTMENTS
        assert explorer._controls["localizations"].options() == _COMPARTMENTS

    with subtests.test(msg="unrecognised genes offer no compartments"):
        explorer = _build(qtbot)
        assert explorer.compartments() == []
        assert explorer._controls["localizations"].count() == 0

    with subtests.test(msg="a tick by hand reaches the style, a refill not"):
        explorer = _build(qtbot, frame)
        compartments = explorer._controls["localizations"]
        seen = []
        compartments.changed.connect(lambda: seen.append(1))
        compartments.setValues(())
        assert seen == []
        compartments.item(0).setCheckState(Qt.Checked)
        assert seen == [1]
        assert explorer.style().localizations == ("dense granules",)

    with subtests.test(msg="the panel's ticks reach the style and the plot"):
        explorer = _build(qtbot, frame)
        explorer._controls["localizations"].setValues(tuple(_COMPARTMENTS))
        explorer._on_control_changed()
        assert explorer.style().localizations == tuple(_COMPARTMENTS)
        assert {c.get_label() for c in explorer._panels[0].collections} == set(
            _COMPARTMENTS)

    with subtests.test(msg="two ticks are two colours; one puts grey back"):
        explorer = _build(qtbot, frame)
        explorer._style.localizations = tuple(_COMPARTMENTS)
        explorer.refresh()
        drawn = {c.get_label(): tuple(np.asarray(c.get_facecolor())[0][:3])
                 for c in explorer._panels[0].collections}
        assert set(drawn) == set(_COMPARTMENTS)
        assert len(set(drawn.values())) == 2
        explorer._style.localizations = ("dense granules",)
        explorer.refresh()
        assert {c.get_label() for c in explorer._panels[0].collections} == {
            "dense granules", "elsewhere"}

    with subtests.test(msg="an unreadable reference table costs only ticks"):
        from spacr import localisation

        explorer = _build(qtbot, frame)

        def _unreadable():
            raise OSError("the bundled table is not there")

        monkeypatch.setattr(localisation, "table", _unreadable)
        assert explorer.compartments() == []
        explorer.set_results(explorer.results())
        assert explorer._controls["localizations"].count() == 0
        assert _combo_data(explorer._controls["x_column"]) == [
            "standardized_marginal_effect", "adjusted_p_value"]
