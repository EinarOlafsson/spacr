"""``Z9`` — the regions that used to be bare dark areas, measured.

Six regions across five screens rendered as flat opaque slabs while every
card beside them thinned with the page-opacity slider:

* Align & Stitch's "Choose a folder of tiles and press Plan" canvas
* Plate Viewer's "choose a database, a table, and a measurement, then
  press Render" canvas
* Model Compare's Model A and Model B panels
* Training Runs' plot area, right of "Runs found"
* Classifier Evaluation's area under the tabs
* Run History's bottom tabs and the container below them

Three different causes, so three different fixes, and no stylesheet
string can tell you whether any of them worked:

``a custom-painted canvas``
    Align and Plate Viewer opened their ``paintEvent`` with
    ``fillRect(self.rect(), QColor(palette["surface"]))``. That hex
    carries no alpha, so the fill was opaque by construction whatever the
    preference said. They call :func:`spacr.qt.theme.paint_panel` now.
``a matplotlib canvas``
    Training Runs' ``FigureCanvasQTAgg`` sets ``WA_OpaquePaintEvent`` and
    paints a figure with a solid ``facecolor``. QSS cannot reach either,
    so the panel is drawn in ``paintEvent`` under a transparent figure.
``a QSS rule with no page opacity in it``
    Model Compare's group boxes had ``background: transparent`` — a
    border around a hole — and the shipped ``QTabWidget::pane`` paints
    raw ``surface_alt``. Both are registered blocks now, through
    :func:`spacr.qt.theme.pane_surface`.

Then roughly twenty-five screens landed in a day, most of them written
without any of that, and the file grew a row for each: the six chart
screens that share one ``FigureCanvasQTAgg``, the two layer canvases and
the three orthogonal planes that took ``palette["bg"]`` (the WINDOW
colour, which is not a surface), and the six named containers that had no
rule of their own and so fell through to the blanket
``QWidget {{ background-color: bg }}`` — the same window colour by another
route. A fourth cause joins the three above:

``a QScrollArea whose viewport auto-fills``
    ``autoFillBackground`` is a widget attribute and paints the palette's
    Window brush before any stylesheet is consulted, so no QSS can undo
    it. ``viewport().setAutoFillBackground(False)`` is the only fix.

One thing is deliberately measured the other way round. Opacity reaches
the container and never the picture, so
:func:`test_the_image_itself_stays_opaque` loads a field into a layer
canvas and requires it to read *below* :data:`OPAQUE`.

**Measured, not read.** Sampling a colour cannot tell "opaque black"
from "a dark part of the backdrop", so every number here comes from
rendering the screen over solid black and again over solid white and
solving ``P = a·B + (1-a)·F`` per pixel. ``1.0`` means the backdrop
arrives untouched; ``0.0`` means something fully opaque is in front of
it. At the 30 % page opacity used below a correct single panel passes
about 0.70, and the regions measured 0.000 before the fix.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QRect, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget,
                               QWidget)

from spacr.qt import preferences as prefs

#: A page opacity well below 100 %, so there is something to see through.
OPACITY = 0.30

#: What one translucent panel over a clear page transmits at ``OPACITY``.
EXPECTED = 1.0 - OPACITY

#: A region below this is passing so little of the backdrop that it is the
#: bare dark area this file exists to catch. Every region measured exactly
#: 0.000 before the fix except Classifier Evaluation's, which measured
#: 0.039 — the hairline of its tab bar, and nothing else.
OPAQUE = 0.20


@pytest.fixture(autouse=True)
def _isolated_qsettings(monkeypatch, tmp_path):
    """Never write to the developer's real preferences.

    ``preferences._settings()`` builds ``QSettings(_ORG, _APP)``, which
    resolves to the NATIVE location whatever ``setPath`` says. Replacing
    the accessor is the only isolation that holds; the assertion refuses
    to run if it ever stops working.
    """
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    assert str(tmp_path) in store.fileName(), (
        "QSettings isolation failed; refusing to write to real preferences")
    return store


@pytest.fixture
def app_theme_restored(qt_theme_applied):
    """Undo what this file does to the session-scoped QApplication.

    ``apply_preferences_to_app`` re-palettes and re-stylesheets the whole
    application. Leaving it at 30 % opacity would take out every later
    test that measures a pixel.
    """
    yield
    from spacr.qt.theme import apply_qpalette, stylesheet
    apply_qpalette(qt_theme_applied)
    qt_theme_applied.setStyleSheet(stylesheet())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _show(qtbot, factory):
    """Build a tool screen the way ``MainWindow`` puts one on screen.

    ``MainWindow._theme_screen`` clears a screen's containers before
    showing it. Skipping that leaves every layout container painting the
    opaque blanket window fill, and the measurement then reports "opaque"
    for a reason that has nothing to do with the region under test.
    """
    prefs.set_theme("dark")
    prefs.set_ambient_enabled(True)
    prefs.set_pane_opacity(OPACITY)
    prefs.apply_preferences_to_app(QApplication.instance())

    screen = factory()
    from spacr.qt.theme import clear_container_surfaces
    clear_container_surfaces(screen)

    window = QMainWindow()
    stack = QStackedWidget()
    window.setCentralWidget(stack)
    stack.addWidget(screen)
    qtbot.addWidget(window)
    window.resize(1400, 950)
    window.show()
    QApplication.processEvents()
    # Keep the window alive: qtbot only weak-references it, and a
    # collected window deletes the C++ half of the screen under the test.
    return window, screen


def _transmission(screen):
    """Per-pixel ``alpha`` of everything painted over the backdrop."""
    ambient = getattr(screen, "_ambient", None)
    if ambient is not None:
        # The sweep that makes the containers see-through has already run;
        # swap the animation for a flat colour so the two renders differ
        # by the backdrop and nothing else.
        ambient.hide()

    backdrop = QWidget(screen)
    backdrop.setObjectName("BackdropProbe")
    backdrop.setGeometry(0, 0, screen.width(), screen.height())
    backdrop.lower()
    backdrop.show()

    def render(colour):
        backdrop.setStyleSheet(
            f"QWidget#BackdropProbe {{ background: {colour}; }}")
        backdrop.lower()
        QApplication.processEvents()
        return screen.grab().toImage()

    dark, light = render("#000000"), render("#ffffff")

    def alpha(x, y):
        a, b = QColor(dark.pixel(x, y)), QColor(light.pixel(x, y))
        return ((b.red() - a.red()) + (b.green() - a.green())
                + (b.blue() - a.blue())) / 765.0

    return alpha


def _rect(screen, widget) -> QRect:
    top_left = widget.mapTo(screen, QPoint(0, 0))
    return QRect(top_left.x(), top_left.y(), widget.width(), widget.height())


def _pane_rect(screen, tabs) -> QRect:
    """The pane of a ``QTabWidget``, without the strip of bar above it.

    The bar's background is transparent on purpose — it is the page
    showing between the tabs — so including it in a panel measurement
    reads *more* backdrop than one panel passes and makes a correctly
    filled pane look like a missing one.
    """
    rect = _rect(screen, tabs)
    bar = tabs.tabBar().height()
    return rect.adjusted(0, bar, 0, 0)


def _row_means(alpha, rect, step: int = 3):
    """Mean transmission per pixel row of ``rect``.

    Per row, because a region is not one number: a panel holding input
    fields has opaque rows where the fields are and panel rows between
    them, and averaging the whole rect turns "correctly translucent panel
    with fields on it" into a middling number indistinguishable from
    "uniformly half-opaque slab". The rows are what discriminate.
    """
    out = []
    for y in range(rect.top() + 2, rect.bottom() - 1, step):
        values = [alpha(x, y)
                  for x in range(rect.left() + 2, rect.right() - 1, step)]
        if values:
            out.append(sum(values) / len(values))
    assert out, f"empty measurement region {rect}"
    return out


def _clearest(alpha, rect, share: float = 0.25) -> float:
    """The mean of the clearest ``share`` of rows in ``rect``.

    The panel's own surface, with the fields, tables and text on it
    excluded — those are opaque on purpose and are not what is being
    measured. An opaque region has no clear rows at all, so this stays
    near zero for exactly the fault this file catches.
    """
    rows = sorted(_row_means(alpha, rect), reverse=True)
    keep = max(1, int(round(len(rows) * share)))
    return sum(rows[:keep]) / keep


# ---------------------------------------------------------------------------
# The regions
# ---------------------------------------------------------------------------

def _align(qtbot):
    from spacr.qt.screens.align import AlignScreen
    window, screen = _show(qtbot, lambda: AlignScreen(threaded=False))
    return window, screen, {"plan canvas": screen._layout_view}


def _plate(qtbot):
    from spacr.qt.screens.plate_view import PlateViewScreen
    window, screen = _show(qtbot, PlateViewScreen)
    return window, screen, {"well grid": screen._grid}


def _model_compare(qtbot):
    from spacr.qt.screens.model_compare import ModelCompareScreen
    window, screen = _show(qtbot, ModelCompareScreen)
    # The two mask canvases and the two result tables were reported black a
    # second time, after the Model A / B panels above were fixed. The
    # canvases carried `setStyleSheet("background: " +
    # active_palette()["bg"])` — raw hex, and the WINDOW colour rather than
    # a surface — and the tables had no fill at all, because
    # `clear_container_surfaces` tags every QAbstractScrollArea transparent
    # by type and a QTableWidget is one.
    return window, screen, {"Model A": screen._panel_a,
                            "Model B": screen._panel_b,
                            "mask canvas A": screen._preview_a,
                            "mask canvas B": screen._preview_b,
                            "parameter table": screen._param_table,
                            "per-field table": screen._row_table}


def _train_compare(qtbot):
    from spacr.qt.screens.train_compare import TrainCompareScreen
    window, screen = _show(qtbot, TrainCompareScreen)
    return window, screen, {"curve plot": screen._canvas}


def _classifier(qtbot):
    from spacr.qt.screens.classifier_evaluation import (
        ClassifierEvaluationScreen)
    window, screen = _show(qtbot, ClassifierEvaluationScreen)
    return window, screen, {"under the tabs": screen._tabs}


def _run_history(qtbot):
    from spacr.qt.screens.run_history import RunHistoryScreen
    window, screen = _show(qtbot, RunHistoryScreen)
    return window, screen, {"tabs + container": screen._tabs}


def _region_rect(screen, widget) -> QRect:
    """The part of ``widget`` its own QSS box actually paints.

    A widget's geometry and the box its rule fills are not the same
    rectangle, and the difference is page. :func:`_pane_rect` was the first
    case of this — a tab bar is transparent on purpose — and there are two
    more, both found by a correct panel failing the "one panel thick"
    assertion from *above*:

    * every panel here is rounded, so the corner arcs pass the backdrop.
      On a large region that is a rounding error; on Model Compare's
      360 px mask canvas the arc rows were enough of the clearest quarter
      to report 0.769 for a panel measuring 0.702.
    * a ``QGroupBox`` carries ``margin-top`` in the shared stylesheet, to
      leave room for its title, and a QSS margin shrinks the painted box
      **inside** the geometry. On Model Zoo's 127 px Download box that
      band is most of the clearest rows and it reported 0.807.

    Inset rather than switched to a modal statistic so the twenty rows
    that already passed keep the number they were measured at.
    """
    from PySide6.QtWidgets import QGroupBox, QTabWidget
    from spacr.qt.theme import RADIUS, SPACING

    if isinstance(widget, QTabWidget):
        return _pane_rect(screen, widget)
    rect = _rect(screen, widget)
    if isinstance(widget, QGroupBox):
        rect = rect.adjusted(0, SPACING["md"], 0, 0)
    corner = int(RADIUS["md"])
    return rect.adjusted(corner, corner, -corner, -corner)


# ---------------------------------------------------------------------------
# The modules that landed after Z9
# ---------------------------------------------------------------------------
# Roughly twenty-five screens arrived in a day and most were written without
# the treatment above, so they painted the same slabs for the same reasons.
# Regions are looked up by object name rather than by private attribute: the
# object name is what the QSS keys off, so a rename that breaks the styling
# breaks this file too instead of silently measuring a different widget.

def _named(screen, object_name: str):
    """The one widget called ``object_name`` under ``screen``."""
    from PySide6.QtWidgets import QWidget
    found = screen.findChild(QWidget, object_name)
    assert found is not None, (
        f"no widget named {object_name!r} under {type(screen).__name__} — "
        "renamed? the QSS block keys off the same name")
    return found


def _figure_of(screen):
    """The ``FigureCanvasQTAgg`` inside this screen's ``GraphCanvas``.

    The canvas is the thing that was opaque — ``WA_OpaquePaintEvent`` plus a
    solid figure ``facecolor`` — and it has no object name of its own
    because matplotlib made it.
    """
    return _named(screen, "GraphCanvas")._canvas


def _qc_dashboard(qtbot):
    from spacr.qt.screens.qc_dashboard import QCDashboardScreen
    window, screen = _show(qtbot, QCDashboardScreen)
    return window, screen, {"card column": screen._cards_panel}


def _experiment_design(qtbot):
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen
    window, screen = _show(qtbot, ExperimentDesignScreen)
    return window, screen, {"findings": screen._findings_panel}


def _power(qtbot):
    from spacr.qt.screens.power import PowerScreen
    window, screen = _show(qtbot, lambda: PowerScreen(threaded=False))
    return window, screen, {"cells curve": screen._cells_view,
                            "wells curve": screen._wells_view}


def _graph_builder(qtbot):
    from spacr.qt.screens.graph_builder import GraphBuilderScreen
    window, screen = _show(qtbot, lambda: GraphBuilderScreen(threaded=False))
    return window, screen, {"chart canvas": _figure_of(screen),
                            "shelf": _named(screen, "GraphShelf")}


def _trellis(qtbot):
    from spacr.qt.screens.trellis import TrellisScreen
    window, screen = _show(qtbot, lambda: TrellisScreen(threaded=False))
    return window, screen, {"chart canvas": _figure_of(screen),
                            "shelf": _named(screen, "TrellisShelf")}


def _gate_editor(qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen
    window, screen = _show(qtbot, lambda: GateEditorScreen(threaded=False))
    return window, screen, {"chart canvas": _figure_of(screen)}


def _feature_explorer(qtbot):
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
    window, screen = _show(qtbot,
                           lambda: FeatureExplorerScreen(threaded=False))
    return window, screen, {"distributions": screen.explorer._canvas}


def _tabulate(qtbot):
    from spacr.qt.screens.tabulate import TabulateScreen
    window, screen = _show(qtbot, lambda: TabulateScreen(threaded=False))
    return window, screen, {"chart canvas": _figure_of(screen)}


def _pca(qtbot):
    from spacr.qt.screens.pca import PCAScreen
    window, screen = _show(qtbot, lambda: PCAScreen(threaded=False))
    return window, screen, {"chart canvas": _figure_of(screen),
                            "scree plot": _named(screen, "PCAScreePlot")}


def _image_scatter(qtbot):
    from spacr.qt.screens.image_scatter import (CANVAS_OBJECT,
                                                ImageScatterScreen)
    window, screen = _show(qtbot,
                           lambda: ImageScatterScreen(threaded=False))
    return window, screen, {"point cloud": _named(screen, CANVAS_OBJECT)}


def _curate(qtbot):
    from spacr.qt.screens.curate import CurateScreen
    window, screen = _show(qtbot, CurateScreen)
    return window, screen, {"layer canvas": _named(screen,
                                                   "LayerCanvasFrame")}


def _layer_viewer(qtbot):
    from spacr.qt.layer_viewer import LayerViewer
    window, screen = _show(qtbot, LayerViewer)
    return window, screen, {"layer canvas": _named(screen,
                                                   "LayerCanvasFrame")}


def _control_chart(qtbot):
    from spacr.qt.screens.control_chart import (CONTROLS_OBJECT,
                                                OUTPUT_OBJECT,
                                                ControlChartScreen)
    window, screen = _show(qtbot,
                           lambda: ControlChartScreen(threaded=False))
    # The output column under the chart was the region left over: an
    # ANONYMOUS QWidget, so the sweep tagged it transparent as scaffolding,
    # holding a QPlainTextEdit and a QTableWidget that the same sweep tags
    # by type. Three transparent things stacked read 1.000 — the backdrop
    # untouched, which over a dark window is a black rectangle.
    return window, screen, {"chart canvas": screen.canvas.canvas,
                            "control column": _named(screen,
                                                     CONTROLS_OBJECT),
                            "output column": _named(screen, OUTPUT_OBJECT)}


def _model_zoo(qtbot):
    from spacr.qt.screens.model_zoo import (DETAIL_NAME, GROUP_NAME,
                                            TABLE_NAME, ModelZooScreen)
    window, screen = _show(qtbot, lambda: ModelZooScreen(threaded=False))
    # Five regions, and the page had all three causes at once: the preview
    # painted raw `bg` inline, the two QGroupBoxes took the shipped
    # `background: transparent` (a border drawn round a hole), and the
    # listing and the provenance box were tagged transparent by type.
    groups = screen.findChildren(QWidget, GROUP_NAME)
    assert len(groups) == 2, (
        f"expected two {GROUP_NAME} boxes on Model Zoo, found {len(groups)}")
    return window, screen, {"Download": groups[0],
                            "Test on fields": groups[1],
                            "model listing": _named(screen, TABLE_NAME),
                            "provenance": _named(screen, DETAIL_NAME)}


def _dose_response(qtbot):
    from spacr.qt.screens.dose_response import DoseResponseScreen
    window, screen = _show(qtbot,
                           lambda: DoseResponseScreen(threaded=False))
    return window, screen, {"curve canvas": screen.canvas}


def _outliers(qtbot):
    from spacr.qt.screens.outliers import CONTROLS_OBJECT, OutliersScreen
    window, screen = _show(qtbot, lambda: OutliersScreen(threaded=False))
    return window, screen, {"control column": _named(screen,
                                                     CONTROLS_OBJECT)}


def _ortho_view(qtbot):
    from spacr.qt.ortho_view import OrthoView
    window, screen = _show(qtbot, OrthoView)
    return window, screen, {name: panel
                            for name, panel in screen.panels.items()}


# ---------------------------------------------------------------------------
# The views that had no surface AT ALL
# ---------------------------------------------------------------------------
# `clear_container_surfaces` sweeps every `QAbstractScrollArea` by type, and
# `QAbstractItemView` and `QPlainTextEdit` are both one — so the shipped
# `QTableView/QTreeView { background-color: surface_alt }` rule never landed on
# any view in the application. Where a view sits on a tab pane the pane
# supplies the surface and it looked right by accident; where nothing is
# behind it, the backdrop arrived untouched at 1.000 and the text floated on
# the animation. The screens below say which of their views IS the page, with
# `spacr.qt.theme.mark_surface` — see that function for why the sweep could
# not simply be narrowed instead.
#
# Two regions here are deliberately the OTHER case, and they are the ones that
# make this file able to fail: Control Chart's report and violations table sit
# on the `ControlChartOutput` panel, and Tabulate's well list sits on
# `PivotShelf`. They were correct before this change and must still measure one
# panel thick, not two — `test_each_region_is_one_panel_thick` is what checks
# that they were not swept up by a blanket fix.


def _project_browser(qtbot):
    from spacr.qt.screens.project_browser import ProjectBrowserScreen
    window, screen = _show(qtbot,
                           lambda: ProjectBrowserScreen(threaded=False))
    return window, screen, {"search folders": screen._root_list,
                            "project table": screen._table,
                            "detail": screen._detail}


def _napari_bridge(qtbot):
    from spacr.qt.screens.napari_bridge import NapariBridgeScreen
    window, screen = _show(qtbot, NapariBridgeScreen)
    return window, screen, {"status log": screen.status}


def _hit_list(qtbot):
    from spacr.qt.screens.hit_list import HitListScreen
    window, screen = _show(qtbot, lambda: HitListScreen(threaded=False))
    return window, screen, {"hits": screen._table}


def _lineage(qtbot):
    from spacr.qt.screens.lineage import LineageScreen
    window, screen = _show(qtbot, LineageScreen)
    return window, screen, {"object tree": screen.tree,
                            "orphans": screen.orphan_list}


def _profiler(qtbot):
    from spacr.qt.screens.profiler import ProfilerScreen
    window, screen = _show(qtbot, lambda: ProfilerScreen(threaded=False))
    return window, screen, {"input tree": screen._inputs}


def _pipeline_graph(qtbot):
    from spacr.qt.screens.pipeline_graph import PipelineGraphScreen
    window, screen = _show(qtbot,
                           lambda: PipelineGraphScreen(threaded=False))
    return window, screen, {"details": screen._details}


def _power_views(qtbot):
    from spacr.qt.screens.power import PowerScreen
    window, screen = _show(qtbot, lambda: PowerScreen(threaded=False))
    # The caveat panel is the one region here that is not a scroll area. It
    # is a plain QWidget subclass, and those ignore a QSS background unless
    # `WA_StyledBackground` is set — it measured 1.000 with the rule already
    # in the sheet and matching, which is why `mark_surface` sets it.
    return window, screen, {"caveats": screen._caveats,
                            "sample sizes": screen._table}


def _gate_editor_tree(qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen
    window, screen = _show(qtbot, lambda: GateEditorScreen(threaded=False))
    return window, screen, {"gate hierarchy": _named(screen,
                                                     "GateHierarchy")}


def _feature_explorer_table(qtbot):
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
    window, screen = _show(qtbot,
                           lambda: FeatureExplorerScreen(threaded=False))
    return window, screen, {"ranking table": _named(screen, "ExplorerTable")}


def _tabulate_pivot(qtbot):
    from spacr.qt.screens.tabulate import TabulateScreen
    window, screen = _show(qtbot, lambda: TabulateScreen(threaded=False))
    # The well list is the control case: it sits ON `PivotShelf` and was
    # already correct, so it must not gain a second surface.
    return window, screen, {"pivot grid": _named(screen, "PivotTable"),
                            "well list (on the shelf)":
                                _named(screen, "PivotWellList")}


def _control_chart_output(qtbot):
    from spacr.qt.screens.control_chart import ControlChartScreen
    window, screen = _show(qtbot,
                           lambda: ControlChartScreen(threaded=False))
    # All three are the control case: they sit on `ControlChartOutput` and
    # `ControlChartControls`, which are panels already.
    return window, screen, {"report (on the panel)": screen.report,
                            "violations (on the panel)": screen.violations,
                            "levels (on the panel)": screen._levels}


def _dose_response_views(qtbot):
    from spacr.qt.screens.dose_response import DoseResponseScreen
    window, screen = _show(qtbot,
                           lambda: DoseResponseScreen(threaded=False))
    return window, screen, {"fit table": screen.table,
                            "report": screen.report}


SCREENS = (
    ("Project Browser", _project_browser),
    ("napari Bridge", _napari_bridge),
    ("Hit List", _hit_list),
    ("Lineage", _lineage),
    ("Profiler", _profiler),
    ("Pipeline Graph", _pipeline_graph),
    ("Power views", _power_views),
    ("Gate Editor tree", _gate_editor_tree),
    ("Feature Explorer table", _feature_explorer_table),
    ("Tabulate pivot", _tabulate_pivot),
    ("Control Chart output", _control_chart_output),
    ("Dose Response views", _dose_response_views),
    ("Align & Stitch", _align),
    ("Plate Viewer", _plate),
    ("Model Compare", _model_compare),
    ("Training Runs", _train_compare),
    ("Classifier Evaluation", _classifier),
    ("Run History", _run_history),
    ("QC Panel", _qc_dashboard),
    ("Experiment Design", _experiment_design),
    ("Power", _power),
    ("Graph Builder", _graph_builder),
    ("Trellis", _trellis),
    ("Gate Editor", _gate_editor),
    ("Feature Explorer", _feature_explorer),
    ("Tabulate", _tabulate),
    ("PCA", _pca),
    ("Image Scatter", _image_scatter),
    ("Curate", _curate),
    ("Layer Viewer", _layer_viewer),
    ("Ortho View", _ortho_view),
    ("Control Chart", _control_chart),
    ("Dose Response", _dose_response),
    ("Outliers", _outliers),
    ("Model Zoo", _model_zoo),
)


# ---------------------------------------------------------------------------
# The measurement has to be able to fail
# ---------------------------------------------------------------------------

def test_the_probe_reports_zero_for_an_opaque_canvas(
        qtbot, app_theme_restored, monkeypatch):
    """Guards the guard.

    A backdrop diff that reads 0.7 whatever is painted is measuring
    nothing. Put the old opaque fill back and the same probe must report
    the bare dark area it was reporting before the fix.
    """
    from PySide6.QtGui import QColor as _QColor
    from spacr.qt import theme as theme_mod
    from spacr.qt.screens import align as align_mod

    def opaque(painter, widget, **kwargs):
        painter.fillRect(widget.rect(),
                         _QColor(theme_mod.active_palette()["surface"]))

    monkeypatch.setattr(align_mod, "paint_panel", opaque)
    _window, screen, regions = _align(qtbot)
    measured = _clearest(_transmission(screen),
                         _rect(screen, regions["plan canvas"]))
    assert measured < OPAQUE, (
        f"the plan canvas passes {measured:.3f} of the backdrop even with "
        "the opaque fill restored, so this file's probe is not measuring "
        "opacity")


@pytest.mark.parametrize("name,build", SCREENS, ids=[n for n, _ in SCREENS])
def test_no_named_region_is_a_bare_dark_area(name, build, qtbot,
                                             app_theme_restored):
    """Every Z9 region passes the backdrop like a panel, not like a slab.

    Measured at 30 % page opacity, clearest quarter of the rows::

        Align & Stitch  plan canvas       0.689
        Plate Viewer    well grid         0.691
        Model Compare   Model A           0.703
        Model Compare   Model B           0.703
        Training Runs   curve plot        0.696
        Classifier Ev.  under the tabs    0.697
        Run History     tabs + container  0.701

    Before the fix all seven read 0.000, except Classifier Evaluation's
    0.039 — which was its tab bar hairline and nothing else.

    The modules that landed after Z9, measured the same way::

        QC Panel          card column       0.000 -> 0.702
        Experiment Design findings          0.000 -> 0.702
        Power             cells curve       0.000 -> 0.698
        Power             wells curve       0.000 -> 0.698
        Graph Builder     chart canvas      0.000 -> 0.698
        Graph Builder     shelf             0.699 -> 0.699
        Trellis           chart canvas      0.000 -> 0.698
        Trellis           shelf             0.000 -> 0.700
        Gate Editor       chart canvas      0.000 -> 0.698
        Feature Explorer  distributions     0.000 -> 0.698
        Tabulate          chart canvas      0.216 -> 0.763
        PCA               chart canvas      0.114 -> 0.733
        PCA               scree plot        0.000 -> 0.702
        Image Scatter     point cloud       0.000 -> 0.702
        Curate            layer canvas      0.000 -> 0.702
        Layer Viewer      layer canvas      0.000 -> 0.702
        Ortho View        xy / zx / yz      0.000 -> 0.702
        Control Chart     chart canvas      0.000 -> 0.698
        Control Chart     control column    0.000 -> 0.702
        Dose Response     curve canvas      0.000 -> 0.698
        Outliers          control column    0.000 -> 0.698

    And the three the user reported a second time — Model Compare, Model
    Zoo and Control Charts — after the rows above had gone in::

        Model Compare     mask canvas A     0.231 -> 0.734
        Model Compare     mask canvas B     0.233 -> 0.734
        Model Compare     parameter table   1.000 -> 0.702
        Model Compare     per-field table   1.000 -> 0.702
        Model Zoo         Download          1.000 -> 0.702
        Model Zoo         Test on fields    0.918 -> 0.701
        Model Zoo         model listing     1.000 -> 0.702
        Model Zoo         provenance        1.000 -> 0.702
        Control Charts    output column     1.000 -> 0.699

    Note which way those failed. Four of the nine measured 1.000 — the
    backdrop arriving *untouched*, which is the opposite of a slab and
    reads black for a different reason: a container with no surface over a
    near-black window colour looks exactly like an opaque black box. The
    two mask canvases are the classic version, opaque by construction from
    a raw palette hex.

    Model Compare's Model A / Model B are in that list from the first
    round and were still black in the running app, which is worth
    recording: their fix was correct and unreachable. ``app.py`` imports
    ``spacr.qt.screens.model_compare`` inside the branch that builds the
    screen, minutes after the launch stylesheet was generated, so the
    block it registers at import was never in the sheet that was live.
    See :func:`spacr.qt.theme.ensure_widget_qss_applied`.
    """
    _window, screen, regions = build(qtbot)
    alpha = _transmission(screen)
    measured = {label: _clearest(alpha, _region_rect(screen, widget))
                for label, widget in regions.items()}

    opaque = {k: v for k, v in measured.items() if v < OPAQUE}
    assert not opaque, (
        f"{name}: " + ", ".join(f"{k} passes only {v:.3f} of the backdrop"
                                for k, v in opaque.items()) +
        f" at {OPACITY:.0%} page opacity — a panel passes about "
        f"{EXPECTED:.2f}. Measured: "
        + ", ".join(f"{k}={v:.3f}" for k, v in measured.items()))


def test_the_probe_reports_zero_for_an_opaque_figure_canvas(
        qtbot, app_theme_restored, monkeypatch):
    """The same guard, for the matplotlib mode.

    ``paint_panel`` is not what makes a ``FigureCanvasQTAgg`` translucent —
    ``WA_OpaquePaintEvent`` off and ``figure.patch`` at alpha 0 are — so
    monkeypatching ``paint_panel`` would prove nothing about this one. Put
    the two opaque things back instead and the probe must see the slab.
    """
    from PySide6.QtCore import Qt as _Qt
    from spacr.qt.screens.graph_builder import GraphBuilderScreen  # noqa: F401
    from spacr.qt.widgets import graph_builder as gb

    real = gb._canvas_class()

    class _Opaque(real):
        def __init__(self, figure, **kwargs):
            super().__init__(figure, **kwargs)
            self._spacr_panel = False
            self.setAttribute(_Qt.WA_OpaquePaintEvent, True)
            self.setAttribute(_Qt.WA_TranslucentBackground, False)
            figure.patch.set_alpha(1.0)
            figure.set_facecolor(gb.active_palette()["surface"])

    monkeypatch.setattr(gb, "_canvas_class", lambda: _Opaque)
    _window, screen, regions = _graph_builder(qtbot)
    measured = _clearest(_transmission(screen),
                         _rect(screen, regions["chart canvas"]))
    assert measured < OPAQUE, (
        f"the chart canvas passes {measured:.3f} of the backdrop with the "
        "opaque figure restored, so this file's probe is not measuring what "
        "made the six chart screens slabs")


def test_the_image_itself_stays_opaque(qtbot, app_theme_restored):
    """Opacity reaches the container, never the picture.

    That was the instruction from the start, and it is the one place where
    a *low* number is the pass: an empty layer canvas is a page panel at
    0.70, and the same canvas with a field loaded is the field — drawn as
    an opaque pixmap on top of the panel, which the page-opacity slider is
    not allowed to wash out.
    """
    import numpy as np
    from spacr.layers import LayerStack
    from spacr.qt.layer_viewer import LayerViewer

    stack = LayerStack()
    stack.add_image(np.linspace(0, 1, 64 * 64).reshape(64, 64), name="field")
    _window, screen = _show(qtbot, lambda: LayerViewer(stack))
    canvas = _named(screen, "LayerCanvasFrame")
    measured = _clearest(_transmission(screen), _rect(screen, canvas))
    assert measured < OPAQUE, (
        f"the loaded field passes {measured:.3f} of the backdrop — the "
        "page-opacity preference has reached the image, and it must not")


@pytest.mark.parametrize("name,build", SCREENS, ids=[n for n, _ in SCREENS])
def test_each_region_is_one_panel_thick(name, build, qtbot,
                                        app_theme_restored):
    """And it is ONE panel, not two stacked.

    "Not opaque" is also satisfied by a region that got its panel twice —
    a fill in QSS *and* a fill in ``paintEvent`` — which reads 0.49 at a
    requested 30 % and is a shade no position of the slider can reach.
    """
    _window, screen, regions = build(qtbot)
    alpha = _transmission(screen)
    for label, widget in regions.items():
        measured = _clearest(alpha, _region_rect(screen, widget))
        assert abs(measured - EXPECTED) < 0.06, (
            f"{name}: {label} passes {measured:.3f} of the backdrop where "
            f"one panel at {OPACITY:.0%} passes {EXPECTED:.2f} — "
            f"{'two surfaces stacked' if measured < EXPECTED else 'no surface at all'}")


# ---------------------------------------------------------------------------
# The tab strips take Home's treatment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name,tabs_attr", [
    ("spacr.qt.screens.classifier_evaluation", "_tabs"),
    ("spacr.qt.screens.run_history", "_tabs"),
])
def test_the_tab_strips_are_styled_like_home(module_name, tabs_attr,
                                             qt_theme_applied):
    """Rounded top corners, a dark-grey tab, the accent blue on hover.

    Asserted on the registered block rather than on pixels because a
    ``:hover`` state cannot be rendered without a live pointer — the
    pixel assertions above cover the part that can be.
    """
    import importlib
    from spacr.qt.theme import palette_for, widget_qss_names, _WIDGET_QSS

    module = importlib.import_module(module_name)
    name = module.TABS_NAME
    assert name in widget_qss_names(), (
        f"{module_name} did not register a QSS block for its tab strip")

    palette = dict(palette_for("dark"), theme="dark")
    block = _WIDGET_QSS[name](palette, OPACITY)

    assert f"QTabWidget#{name} > QTabBar::tab" in block
    assert "border-top-left-radius" in block and \
           "border-top-right-radius" in block, "tabs must have rounded tops"
    assert f"QTabWidget#{name} > QTabBar::tab:hover" in block
    hover = block.split("QTabBar::tab:hover")[1].split("}")[0]
    assert palette["accent"] in hover, (
        f"the hover state must be the accent blue; got: {hover.strip()}")
    # Default tab and pane both carry the page opacity, so neither is a
    # slab: `rgba(` is what `pane_surface` returns below 100 %.
    assert block.count("rgba(") >= 2, (
        "the tab and the pane must both carry the page opacity")


# ---------------------------------------------------------------------------
# The opt-out itself, without a screen in the way
# ---------------------------------------------------------------------------

def test_the_sweep_skips_a_view_that_declared_itself_a_surface(qtbot):
    """`mark_surface` is the whole distinction, so assert it directly.

    The pixel tests above prove the outcome on twelve real screens. This
    proves the mechanism: an unmarked view is still tagged transparent
    exactly as it always was — which is what keeps the tables sitting on
    panes from stacking two greys — and a marked one is left alone, along
    with its viewport, which is the half that actually paints.
    """
    from PySide6.QtWidgets import QTableWidget, QVBoxLayout
    from spacr.qt.theme import (SURFACE_PROPERTY, TRANSPARENT_PROPERTY,
                                clear_container_surfaces, is_surface,
                                mark_surface)

    root = QWidget()
    qtbot.addWidget(root)
    layout = QVBoxLayout(root)
    passenger = QTableWidget(1, 1, root)
    page = QTableWidget(1, 1, root)
    layout.addWidget(passenger)
    layout.addWidget(page)

    mark_surface(page)
    assert is_surface(page) and not is_surface(passenger)

    clear_container_surfaces(root)

    for widget, label in ((passenger, "passenger"), (passenger.viewport(),
                                                     "passenger viewport")):
        assert widget.property(TRANSPARENT_PROPERTY) is True, (
            f"the {label} must still be swept — a table on a pane that keeps "
            "a fill of its own stacks two translucent greys")
    for widget, label in ((page, "page"), (page.viewport(),
                                           "page viewport")):
        assert not widget.property(TRANSPARENT_PROPERTY), (
            f"the {label} was declared a surface and the sweep tagged it "
            "anyway, so the backdrop arrives under the text")
        assert widget.property(SURFACE_PROPERTY) is True


def test_the_surface_marker_has_a_rule_in_every_theme(qt_theme_applied):
    """A property nothing styles is a property that does nothing.

    The marker is only half the fix: the sweep leaving a view alone gives
    it whatever rule it already matched, and a bare ``QListWidget`` matches
    nothing but the blanket ``QWidget`` fill — the WINDOW colour, which is
    not a surface. Three of the marked views are exactly that.
    """
    from spacr.qt.theme import SURFACE_PROPERTY, THEMES, stylesheet

    selector = f'*[{SURFACE_PROPERTY}="true"]'
    for name in THEMES:
        qss = stylesheet(name, surface_opacity=OPACITY)
        assert selector in qss, (
            f"the {name} theme has no rule for {selector}")
        block = qss.split(selector, 1)[1].split("}", 1)[0]
        assert "background-color" in block, (
            f"{selector} must set a background in {name}; got {block!r}")
