"""Instruction 139 A: the generated figure is a RENDER of the screen's scene.

The seven duplicated plots -- volcano, effect ranking, effect distribution,
control separation, guide agreement, p-value histogram and Q-Q -- are drawn on
screen by pyqtgraph and on disk by matplotlib. These assert the three things
that have to be true of moving the file to the screen's renderer:

  * ONE RUN, ONE RENDERER. Two libraries in one output folder is a worse
    disagreement than the one the move is meant to remove.
  * A HEADLESS RUN STILL WRITES ITS FIGURES. `spacr-run regression` in a
    terminal has no tab to disagree with, so it keeps the page it has always
    written -- and says which renderer drew it.
  * EVERY SAVE HONOURS THE FORMAT PREFERENCE, and is announced.
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from spacr import figure_sink                                    # noqa: E402
from spacr.figures import fast_render                            # noqa: E402
from spacr.figures.panels import SHEET_ORDER                     # noqa: E402
from spacr.plot import figure_path                               # noqa: E402

#: The first four bytes of each format spaCR can be asked for. Checked on the
#: FILE rather than on its name: a PNG written to a `.pdf` name is a file no
#: viewer opens, and it is the failure this project has already shipped once.
MAGIC = {b"%PDF": "pdf", b"\x89PNG": "png"}


@pytest.fixture()
def results():
    rng = np.random.default_rng(0)
    n = 240
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


@pytest.fixture()
def png_preference(monkeypatch):
    monkeypatch.setattr("spacr.plot.figure_output_preferences",
                        lambda: ("png", 150))
    return "png"


def _kinds(folder):
    return {name: MAGIC.get(open(os.path.join(folder, name), "rb").read(4))
            for name in sorted(os.listdir(folder))}


# --------------------------------------------------------------- the claim

def test_the_mapping_names_exactly_the_seven_duplicated_panels():
    """FAST_PANELS IS the claim that the two renderers draw one picture."""
    assert tuple(fast_render.FAST_PANELS) == tuple(SHEET_ORDER)


def test_every_named_class_exists_in_fast_plots():
    fast_plots = pytest.importorskip("spacr.qt.widgets.fast_plots")
    for key, name in fast_render.FAST_PANELS.items():
        assert hasattr(fast_plots, name), f"{key} names a missing {name}"


# ------------------------------------------------ the renderer is decided once

@pytest.fixture()
def qtagg_backend():
    """matplotlib on the backend a real spaCR run uses.

    THE TEST SUITE PINS Agg (``tests/conftest.py`` sets ``MPLBACKEND``), and
    that is exactly why this trap survived a green suite: on Agg nothing
    touches Qt, so the defect below cannot happen in a test that does not ask
    for the production backend. The default backend in this environment is
    ``qtagg``.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    pytest.importorskip("PySide6.QtWidgets")
    previous = matplotlib.get_backend()
    try:
        matplotlib.use("QtAgg", force=True)
    except Exception as error:                                   # noqa: BLE001
        pytest.skip(f"QtAgg is unavailable: {error}")
    yield
    plt.close("all")
    matplotlib.use(previous, force=True)


def test_a_matplotlib_qapplication_does_not_switch_the_renderer(qtagg_backend):
    """THE TRAP THAT COST A WRONG BUILD, and it is measured here.

    matplotlib's QtAgg backend -- the DEFAULT backend in this environment --
    calls ``_create_qApp`` from inside ``plt.figure()`` and constructs a
    ``QApplication(["matplotlib"])``. A rule that read "a QApplication exists"
    as "the GUI is up" therefore flipped renderer after the first panel of a
    headless run.
    """
    import matplotlib.pyplot as plt
    from PySide6.QtWidgets import QApplication

    figure = plt.figure()
    try:
        assert QApplication.instance() is not None, (
            "this test is pointless unless plt.figure() made a QApplication")
        # It did. Nothing has handed us a scene all the same.
        assert fast_render.renderer_for("volcano")[0] == "matplotlib"
    finally:
        plt.close(figure)


def test_one_run_writes_every_panel_with_one_renderer(tmp_path, results,
                                                      qtagg_backend):
    """Seven figures of one run in two libraries is the worse disagreement."""
    records = fast_render.write_panels(results, tmp_path, verbose=False)
    drawn = [r for r in records if r.drawn]
    assert len(drawn) == len(SHEET_ORDER)
    assert len({r.renderer for r in drawn}) == 1, [(r.key, r.renderer)
                                                   for r in drawn]


def test_a_headless_run_writes_its_figures_and_says_who_drew_them(
        tmp_path, results, capsys):
    records = fast_render.write_panels(results, tmp_path)
    assert all(os.path.exists(r.path) for r in records if r.drawn)
    assert {r.renderer for r in records if r.drawn} == {"matplotlib"}
    assert "by matplotlib" in capsys.readouterr().out


def test_the_headless_reason_is_stated_rather_than_implied():
    renderer, reason = fast_render.renderer_for("volcano")
    assert renderer == "matplotlib"
    assert "no live plot" in reason


def test_forcing_matplotlib_says_so():
    assert fast_render.renderer_for("volcano", "matplotlib") == (
        "matplotlib", "matplotlib was asked for")


def test_an_unknown_panel_has_no_twin_and_says_which():
    renderer, reason = fast_render.renderer_for("plate_heatmap")
    assert renderer == "matplotlib"
    assert "no interactive twin" in reason


def test_a_misspelt_renderer_falls_back_rather_than_raising(monkeypatch):
    monkeypatch.setenv("SPACR_FIGURE_RENDERER", "pyqtgrpah")
    assert fast_render.requested_renderer() == "auto"


# ------------------------------------------------------- the format preference

def test_figure_path_follows_the_preference(png_preference):
    assert figure_path("/tmp/run/volcano.pdf").endswith("volcano.png")
    assert figure_path("/tmp/run/volcano").endswith("volcano.png")


def test_an_explicit_format_still_wins(png_preference):
    assert figure_path("/tmp/run/volcano", "pdf").endswith("volcano.pdf")


def test_a_version_number_in_a_stem_is_not_an_extension(png_preference):
    """`plate_2.5_umap` is not a file called `plate_2` with a `.5_umap`."""
    assert figure_path("/tmp/plate_2.5_umap").endswith("plate_2.5_umap.png")


def test_every_file_is_the_format_its_name_claims(tmp_path, results,
                                                  png_preference):
    fast_render.write_panels(results, tmp_path, verbose=False)
    kinds = _kinds(tmp_path)
    assert kinds and all(name.endswith(".png") and kind == "png"
                         for name, kind in kinds.items()), kinds


# ------------------------------------------------------------- the pyqtgraph half

@pytest.fixture()
def pyqtgraph_available():
    pytest.importorskip("pyqtgraph")
    ok, why = fast_render._pyqtgraph_ready(create=True)
    if not ok:
        pytest.skip(why)
    return True


def test_the_scene_renders_with_no_display(tmp_path, results,
                                           pyqtgraph_available):
    """139 A's own worry, answered: offscreen is enough."""
    records = fast_render.write_panels(results, tmp_path,
                                       renderer="pyqtgraph", verbose=False)
    drawn = [r for r in records if r.drawn]
    assert len(drawn) == len(SHEET_ORDER)
    assert {r.renderer for r in drawn} == {"pyqtgraph"}
    assert all(os.path.getsize(r.path) > 1000 for r in drawn)


def test_a_rendered_scene_is_not_a_blank_page(tmp_path, results,
                                              png_preference,
                                              pyqtgraph_available):
    """A widget that never had a layout pass exports one flat colour."""
    Image = pytest.importorskip("PIL.Image")
    record = fast_render.render_panel("volcano", results,
                                      str(tmp_path / "volcano"),
                                      renderer="pyqtgraph")
    assert record.drawn, record.reason
    image = Image.open(record.path).convert("RGBA")
    assert len(set(image.getdata())) > 50


def test_the_rendered_file_is_the_format_the_preference_asked_for(
        tmp_path, results, png_preference, pyqtgraph_available):
    record = fast_render.render_panel("qq", results, str(tmp_path / "qq.pdf"),
                                      renderer="pyqtgraph")
    assert record.path.endswith(".png")
    assert open(record.path, "rb").read(4) == b"\x89PNG"


def test_a_live_widget_is_rendered_rather_than_redrawn(tmp_path, results,
                                                       pyqtgraph_available):
    """THE POINT OF THE MODULE: the file is the widget on screen."""
    from spacr.qt.widgets.fast_plots import VolcanoPlot

    plot = VolcanoPlot()
    plot.set_results(results, effect="coefficient", p_column="p_value")
    record = fast_render.render_panel("volcano", None,
                                      str(tmp_path / "live"), plot=plot)
    assert record.drawn, record.reason
    assert record.renderer == "pyqtgraph"
    assert os.path.getsize(record.path) > 1000


def test_a_live_widget_overrules_the_auto_rule(tmp_path, results,
                                               pyqtgraph_available):
    """Handed a widget, choosing matplotlib because no screen was detected
    would be absurd -- the widget could not exist without one."""
    from spacr.qt.widgets.fast_plots import QQPlot

    plot = QQPlot()
    plot.set_p_values(results["p_value"])
    record = fast_render.render_panel("qq", None, str(tmp_path / "qq"),
                                      plot=plot)
    assert record.renderer == "pyqtgraph"


def test_build_fast_plot_refuses_a_key_with_no_twin(pyqtgraph_available):
    with pytest.raises(KeyError):
        fast_render.build_fast_plot("plate_heatmap", None)


# ------------------------------------------------------ nothing is lost quietly

def test_an_empty_table_is_a_reason_not_an_exception(tmp_path):
    record = fast_render.render_panel("volcano", pd.DataFrame(),
                                      str(tmp_path / "volcano"))
    assert record.drawn is False
    assert record.reason
    assert not os.listdir(tmp_path)


def test_a_panel_this_table_cannot_support_is_named(tmp_path):
    frame = pd.DataFrame({"feature": ["a", "b"], "coefficient": [1.0, 2.0]})
    records = fast_render.write_panels(frame, tmp_path, verbose=False)
    missing = [r for r in records if not r.drawn]
    assert missing, "a table with no p-values should not draw a Q-Q"
    assert all(r.reason for r in missing)


# ------------------------------------------ saved and visible are one event

def test_a_rendered_file_is_announced(tmp_path, results, pyqtgraph_available):
    seen = []
    previous = figure_sink.set_file_sink(lambda path, title=None:
                                         seen.append((path, title)))
    try:
        fast_render.write_panels(results, tmp_path, renderer="pyqtgraph",
                                 verbose=False)
    finally:
        figure_sink.set_file_sink(previous)
    assert len(seen) == len(SHEET_ORDER)
    assert [title for _, title in seen] == list(SHEET_ORDER)


def test_a_sink_that_raises_does_not_lose_the_file(tmp_path):
    def angry(path, title=None):
        raise RuntimeError("the screen has gone away")

    previous = figure_sink.set_file_sink(angry)
    try:
        target = tmp_path / "already_written.png"
        target.write_bytes(b"\x89PNG")
        assert figure_sink.publish_file(str(target)) == str(target)
    finally:
        figure_sink.set_file_sink(previous)


def test_publishing_no_file_announces_nothing():
    seen = []
    previous = figure_sink.set_file_sink(lambda *a, **k: seen.append(a))
    try:
        assert figure_sink.publish_file(None) is None
        assert figure_sink.publish_file("") is None
    finally:
        figure_sink.set_file_sink(previous)
    assert seen == []


def test_clearing_the_sink_clears_both_routes():
    figure_sink.set_sink(lambda *a, **k: None)
    figure_sink.set_file_sink(lambda *a, **k: None)
    figure_sink.clear_sink()
    assert figure_sink.sink() is None
    assert figure_sink.file_sink() is None
