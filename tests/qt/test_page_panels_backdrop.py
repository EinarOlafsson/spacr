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
    return window, screen, {"Model A": screen._panel_a,
                            "Model B": screen._panel_b}


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
    """The rect to measure for ``widget`` — pane only, for a tab strip."""
    from PySide6.QtWidgets import QTabWidget
    if isinstance(widget, QTabWidget):
        return _pane_rect(screen, widget)
    return _rect(screen, widget)


SCREENS = (
    ("Align & Stitch", _align),
    ("Plate Viewer", _plate),
    ("Model Compare", _model_compare),
    ("Training Runs", _train_compare),
    ("Classifier Evaluation", _classifier),
    ("Run History", _run_history),
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
