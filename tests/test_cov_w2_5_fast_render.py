"""A run never loses a figure over the renderer it could not use.

Every refusal in this module has to come back as a ``RenderedPanel`` with a
reason on it, because the alternative — raising — throws away a completed fit
for the sake of a picture. These tests take away the pieces one at a time: no
Qt, no pyqtgraph, no QApplication, no columns, no destination, no disk.
"""
from __future__ import annotations

import builtins
from dataclasses import fields
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr.figures import fast_render as fr


@pytest.fixture
def results():
    """A coefficient table with everything the seven panels can want."""
    rng = np.random.default_rng(3)
    n = 120
    frame = pd.DataFrame({
        "feature": [f"gene_fraction:grna[g{i // 3}_{i % 3}]" for i in range(n)],
        "coefficient": rng.normal(0, 0.5, n),
        "p_value": rng.uniform(0, 1, n),
        "std_err": rng.uniform(0.05, 0.3, n),
        "condition": ["nc"] * 20 + ["pc"] * 20 + ["other"] * (n - 40),
    })
    frame.loc[:5, "p_value"] = 1e-6
    frame["q_value"] = frame["p_value"].clip(upper=1.0)
    return frame


@pytest.fixture
def no_columns():
    """A table with a feature column and nothing plottable in it."""
    return pd.DataFrame({"feature": ["gene_fraction:grna[g1_1]",
                                     "gene_fraction:grna[g1_2]"],
                         "note": ["a", "b"]})


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def test_a_record_is_true_only_when_something_was_written():
    """``bool(record)`` answers "is there a file", not "was it attempted"."""
    record = fr.RenderedPanel("qq", path="/tmp/qq.pdf",
                              renderer="matplotlib", drawn=True, reason="")
    for field in fields(fr.RenderedPanel):
        assert f":param {field.name}:" in (fr.RenderedPanel.__doc__ or "")
    assert record.renderer == "matplotlib" and record.reason == ""
    assert bool(record)
    assert not bool(fr.RenderedPanel("qq", path="/tmp/qq.pdf", drawn=False))
    assert not bool(fr.RenderedPanel("qq", path=None, drawn=True))
    assert not bool(fr.RenderedPanel("qq"))


# ---------------------------------------------------------------------------
# asking whether Qt is there without answering yes by accident
# ---------------------------------------------------------------------------

def test_with_qt_never_imported_there_is_no_application(monkeypatch):
    """The question must be answerable without importing a GUI toolkit."""
    monkeypatch.delitem(sys.modules, "PySide6.QtWidgets", raising=False)

    assert fr.qt_application() is None


def test_a_qt_that_cannot_be_asked_answers_none(monkeypatch):
    """A broken Qt module is "no application", not an exception."""
    broken = types.ModuleType("PySide6.QtWidgets")

    class Angry:
        @staticmethod
        def instance():
            raise RuntimeError("this Qt is half-loaded")

    broken.QApplication = Angry
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", broken)

    assert fr.qt_application() is None


def test_a_live_application_is_found(qapp):
    """When Qt really is up, the running application comes back."""
    assert fr.qt_application() is qapp


# ---------------------------------------------------------------------------
# choosing a renderer
# ---------------------------------------------------------------------------

def test_a_misspelt_environment_variable_is_auto(monkeypatch):
    """A typo must not cost the run its figures."""
    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "pyqtgrahp")

    assert fr.requested_renderer() == "auto"


def test_the_environment_can_name_a_renderer(monkeypatch):
    """A recognised value is honoured, case- and space-insensitively."""
    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "  MatPlotLib ")

    assert fr.requested_renderer() == "matplotlib"


def test_a_misspelt_force_argument_falls_back_to_auto(monkeypatch):
    """The same rule applies to the argument as to the variable."""
    monkeypatch.delenv("SPACR_FIGURE_RENDERER", raising=False)

    chosen, why = fr.renderer_for("volcano", "pyqtgrahp")

    assert chosen == "matplotlib"
    assert "no live plot was handed in" in why


def test_a_key_with_no_twin_says_which_key(monkeypatch):
    """The reason names the panel, so the user can find it."""
    chosen, why = fr.renderer_for("not_a_panel", "pyqtgraph")

    assert chosen == "matplotlib"
    assert "'not_a_panel' has no interactive twin" == why


def test_pyqtgraph_that_cannot_be_started_falls_back_with_its_reason(
        monkeypatch):
    """The refusal from ``_pyqtgraph_ready`` is passed through verbatim."""
    monkeypatch.setattr(fr, "_pyqtgraph_ready",
                        lambda create=True: (False, "no Qt on this machine"))

    chosen, why = fr.renderer_for("volcano", "pyqtgraph")

    assert chosen == "matplotlib"
    assert why == "no Qt on this machine"


def test_pyqtgraph_that_is_ready_is_chosen_with_no_excuse(monkeypatch):
    """A renderer that works needs no reason attached."""
    monkeypatch.setattr(fr, "_pyqtgraph_ready", lambda create=True: (True, ""))

    assert fr.renderer_for("volcano", "pyqtgraph") == ("pyqtgraph", "")


# ---------------------------------------------------------------------------
# whether a scene can be built here
# ---------------------------------------------------------------------------

def test_an_import_that_fails_is_a_reason_not_a_traceback(monkeypatch):
    """A broken widget package sends the run back to matplotlib politely."""
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if "fast_plots" in name or "fast_plots" in (fromlist or ()):
            raise ImportError("fast_plots is not installed here")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)
    monkeypatch.delitem(sys.modules, "spacr.qt.widgets.fast_plots",
                        raising=False)

    ok, why = fr._pyqtgraph_ready(create=True)

    monkeypatch.undo()
    assert ok is False
    assert "pyqtgraph plots are unavailable" in why
    assert "not installed here" in why


def test_pyqtgraph_missing_is_named_as_such(monkeypatch):
    """The reason distinguishes "not installed" from "could not import"."""
    from spacr.qt.widgets import fast_plots

    monkeypatch.setattr(fast_plots, "HAVE_PYQTGRAPH", False)

    assert fr._pyqtgraph_ready(create=True) == (False, "pyqtgraph is not "
                                                       "installed")


def test_with_no_display_the_offscreen_platform_is_selected(monkeypatch,
                                                            qapp):
    """A machine with no display still renders, offscreen."""
    monkeypatch.setattr(fr, "qt_application", lambda: None)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    ok, why = fr._pyqtgraph_ready(create=True)

    assert ok is True and why == ""
    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"
    assert os.environ["PYQTGRAPH_QT_LIB"] == "PySide6"


def test_a_qapplication_that_cannot_start_is_a_reason(monkeypatch):
    """No Qt is a fallback, not a crash in the middle of a fit."""
    monkeypatch.setattr(fr, "qt_application", lambda: None)
    real_import = builtins.__import__

    def block(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PySide6.QtWidgets" and "QApplication" in (fromlist or ()):
            raise ImportError("PySide6 will not load")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block)

    ok, why = fr._pyqtgraph_ready(create=True)

    monkeypatch.undo()
    assert ok is False
    assert "no QApplication could be started" in why


# ---------------------------------------------------------------------------
# building a scene from a table
# ---------------------------------------------------------------------------

def test_a_key_with_no_twin_is_a_keyerror_that_lists_the_ones_there_are():
    """A programming mistake here is loud, unlike a missing column."""
    with pytest.raises(KeyError) as caught:
        fr.build_fast_plot("nonsense", pd.DataFrame({"a": [1]}))

    assert "volcano" in str(caught.value)


@pytest.mark.parametrize("frame", [None, pd.DataFrame()])
def test_an_empty_table_builds_no_plot(frame, qapp):
    """Nothing to draw is None, not an empty scene."""
    assert fr.build_fast_plot("volcano", frame) is None


@pytest.mark.parametrize("key", ["volcano", "effect_rank",
                                 "effect_distribution", "p_histogram", "qq",
                                 "controls", "agreement"])
def test_a_table_with_no_usable_columns_supports_no_panel(key, no_columns,
                                                          qapp):
    """Every panel refuses the same way when its column is not there."""
    assert fr.build_fast_plot(key, no_columns) is None


def test_agreement_needs_a_feature_column(qapp):
    """Guide agreement is about features; without them there is no panel."""
    frame = pd.DataFrame({"coefficient": [1.0, 2.0], "p_value": [0.1, 0.2]})

    assert fr.build_fast_plot("agreement", frame) is None


def test_agreement_with_no_guide_terms_at_all_is_no_panel(qapp):
    """A fit with only nuisance terms supports no agreement statement."""
    frame = pd.DataFrame({
        "feature": ["Intercept", "batch"],
        "coefficient": [0.4, -0.3],
        "p_value": [0.01, 0.02],
    })

    from spacr.guide_concordance import guide_support
    assert not len(guide_support(frame))

    assert fr.build_fast_plot("agreement", frame) is None


def test_controls_with_only_one_group_is_no_panel(qapp):
    """A separation plot needs two things to separate."""
    frame = pd.DataFrame({
        "feature": ["gene_fraction:grna[g1_1]"] * 4,
        "coefficient": [0.1, 0.2, 0.3, 0.4],
        "condition": ["nc"] * 4,
    })

    assert fr.build_fast_plot("controls", frame) is None


def test_every_panel_builds_from_a_full_table(results, qapp):
    """The seven keys all produce a widget from one real coefficient table."""
    for key in fr.FAST_PANELS:
        plot = fr.build_fast_plot(key, results)
        assert plot is not None, key
        plot.deleteLater()


def test_keys_are_narrowed_with_the_rows_they_label():
    """A subset of rows carries the subset of names, in order."""
    rows = np.array([True, False, True])

    assert fr._subset(["a", "b", "c"], rows) == ["a", "c"]
    assert fr._subset(None, rows) is None


def test_control_groups_need_both_an_effect_and_a_label_column(results):
    """Either half missing means no groups at all."""
    assert fr._control_groups(results, None) == ({}, {})
    assert fr._control_groups(results.drop(columns=["condition"]),
                              "coefficient") == ({}, {})


def test_control_groups_use_the_paper_figures_own_names(results):
    """``nc`` is "negative" here and in the house-style panel alike."""
    groups, keys = fr._control_groups(results, "coefficient")

    assert set(groups) == {"negative", "positive", "screen"}
    assert len(groups["negative"]) == 20
    assert set(keys) == set(groups)


def test_control_groups_do_not_invent_keys_without_a_feature_column(results):
    """Unlabelled effect rows remain drawable without fabricated row keys."""
    groups, keys = fr._control_groups(results.drop(columns=["feature"]),
                                      "coefficient")

    assert set(groups) == {"negative", "positive", "screen"}
    assert keys == {}


# ---------------------------------------------------------------------------
# writing one
# ---------------------------------------------------------------------------

def test_a_scene_with_no_destination_is_a_reason(results, qapp):
    """The panel says which piece was missing rather than writing nowhere."""
    record = fr._render_with_pyqtgraph("qq", results, None)

    assert record.drawn is False
    assert record.reason == "no destination was given"
    assert record.renderer == "pyqtgraph"


def test_a_table_that_cannot_support_the_panel_is_a_reason(no_columns, qapp,
                                                           tmp_path):
    """A scene that cannot be built is named as such."""
    record = fr._render_with_pyqtgraph("qq", no_columns,
                                       str(tmp_path / "qq.pdf"))

    assert record.drawn is False
    assert record.reason == "this table cannot support the panel"


def test_an_export_that_raises_comes_back_as_its_own_message(results, qapp,
                                                             tmp_path,
                                                             monkeypatch):
    """The exception type and text land on the record, not on the caller."""
    plot = fr.build_fast_plot("qq", results)

    def refuse(destination):
        raise OSError("the disk went away")

    monkeypatch.setattr(plot, "export", refuse)

    record = fr._render_with_pyqtgraph("qq", results,
                                       str(tmp_path / "qq.pdf"), plot=plot)

    assert record.drawn is False
    assert record.reason == "OSError: the disk went away"
    plot.deleteLater()


def test_a_widget_that_will_not_be_deleted_still_yields_the_file(
        results, qapp, tmp_path, monkeypatch):
    """Cleanup failing must not lose a figure that was already written."""
    real_build = fr.build_fast_plot

    def build_then_break(key, frame, **kwargs):
        plot = real_build(key, frame, **kwargs)

        def refuse():
            raise RuntimeError("already destroyed")

        plot.deleteLater = refuse
        return plot

    monkeypatch.setattr(fr, "build_fast_plot", build_then_break)

    record = fr._render_with_pyqtgraph("qq", results,
                                       str(tmp_path / "out" / "qq.png"),
                                       announce=False)

    assert record.drawn is True
    assert os.path.exists(record.path)


def test_a_scene_that_cannot_be_built_still_writes_the_matplotlib_page(
        results, tmp_path, monkeypatch, qapp):
    """Falling back is the point: the run keeps its figure."""
    monkeypatch.setattr(fr, "_pyqtgraph_ready", lambda create=True: (True, ""))
    monkeypatch.setattr(fr, "build_fast_plot", lambda *a, **k: None)

    record = fr.render_panel("qq", results, str(tmp_path / "qq"),
                             renderer="pyqtgraph", announce=False)

    assert record.renderer == "matplotlib"
    assert record.drawn is True
    assert record.reason == "this table cannot support the panel"
    assert os.path.exists(record.path)


def test_a_house_style_panel_that_raises_is_a_reason(results, tmp_path,
                                                     monkeypatch):
    """A failure inside ``build_panel`` is reported, not propagated."""
    from spacr.figures import sheet

    def explode(key, frame):
        raise ValueError("that sheet is broken")

    monkeypatch.setattr(sheet, "build_panel", explode)

    record = fr.render_panel("qq", results, str(tmp_path / "qq"),
                             renderer="matplotlib", announce=False)

    assert record.drawn is False
    assert record.reason == "ValueError: that sheet is broken"


def test_writing_without_announcing_still_lands_on_disk(results, tmp_path):
    """``announce=False`` skips the gallery, not the file."""
    published = []
    from spacr import figure_sink

    figure_sink.set_sink(lambda figure, path=None, title=None:
                         published.append(path))
    try:
        record = fr.render_panel("qq", results, str(tmp_path / "qq"),
                                 renderer="matplotlib", announce=False)
    finally:
        figure_sink.clear_sink()

    assert record.drawn is True
    assert os.path.exists(record.path)
    assert published == []


def test_writing_nowhere_without_announcing_writes_nothing(results):
    """No path and no gallery means the panel is drawn and dropped."""
    record = fr.render_panel("qq", results, None, renderer="matplotlib",
                             announce=False)

    assert record.drawn is True
    assert record.path is None


def test_an_empty_table_keeps_the_renderer_reason(tmp_path):
    """The reason the renderer was chosen survives into the empty answer."""
    record = fr.render_panel("qq", pd.DataFrame(), str(tmp_path / "qq"),
                             renderer="matplotlib", announce=False)

    assert record.drawn is False
    assert record.reason == "matplotlib was asked for"


# ---------------------------------------------------------------------------
# writing the set
# ---------------------------------------------------------------------------

def test_the_summary_names_the_panels_that_were_not_drawn(no_columns,
                                                          tmp_path, capsys):
    """A user who finds a figure missing reads why in the run log."""
    records = fr.write_panels(no_columns, tmp_path, keys=("qq", "volcano"),
                              renderer="matplotlib", verbose=True)

    out = capsys.readouterr().out
    assert all(not record.drawn for record in records)
    assert "0/2 regression panel(s) written" in out
    assert "qq not drawn:" in out
    assert "volcano not drawn:" in out


def test_the_summary_counts_by_renderer(results, tmp_path, capsys):
    """The line says who drew them, so a mismatch is findable."""
    records = fr.write_panels(results, tmp_path, keys=("qq", "p_histogram"),
                              renderer="matplotlib", verbose=True)

    out = capsys.readouterr().out
    assert all(record.drawn for record in records)
    assert "2/2 regression panel(s) written" in out
    assert "2 by matplotlib" in out


def test_a_quiet_run_prints_nothing(results, tmp_path, capsys):
    """``verbose=False`` writes files and says nothing."""
    records = fr.write_panels(results, tmp_path, keys=("qq",),
                              renderer="matplotlib", verbose=False)

    assert capsys.readouterr().out == ""
    assert records[0].drawn is True


def test_handing_in_live_plots_decides_the_renderer_for_the_whole_set(
        results, tmp_path, qapp):
    """One renderer for the set, and a widget in hand settles which."""
    plot = fr.build_fast_plot("qq", results)

    records = fr.write_panels(results, tmp_path, keys=("qq",),
                              plots={"qq": plot}, verbose=False)

    assert [record.renderer for record in records] == ["pyqtgraph"]
    assert records[0].drawn is True
    plot.deleteLater()


def test_an_empty_key_list_still_decides_a_renderer(results, tmp_path,
                                                    monkeypatch):
    """No keys is an empty set of records, not an index error."""
    monkeypatch.delenv("SPACR_FIGURE_RENDERER", raising=False)

    assert fr.write_panels(results, tmp_path, keys=(), verbose=False) == []
