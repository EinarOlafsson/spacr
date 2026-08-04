"""Every screen wears the same masthead, and it is Mask's.

The report was that "several of the new modules' naming in the top left is
in too small font", with Mask Generation named as the reference: **the
module name large, the description to its right, a brief instruction
below.**

Mask Generation is an :class:`~spacr.qt.screens.app_screen.AppScreen`, and
``AppScreen`` built that header inline in its own constructor. Roughly
twenty-five screens landed over two days, none of them an ``AppScreen``,
and every one rolled a header of its own. Fifteen used a ``QLabel`` tagged
``ScreenTitle`` — an object name with **no rule anywhere in the
stylesheet**, so it rendered at the body size — and four had no title at
all, opening on a paragraph.

So the fix is not styling: it is that the new screens were not using the
shared header. It is a component now,
:class:`~spacr.qt.screens.app_screen.ModuleHeader`, ``AppScreen`` builds
its own from it, and every screen below is routed through it. Copying the
styling into each of them instead would have set the same trap for the
twenty-sixth.

**Asserted on rendered geometry, never on stylesheet strings.** A rule
saying ``font-size: 30px`` proves nothing about a label that no selector
reaches — which is exactly the bug — so every number here comes off a
laid-out widget: ``QFontMetrics`` for the type size, ``geometry()`` for
the arrangement. A screen has to be shown and its layout activated for
either to mean anything, which is what :func:`_show` is for.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication, QWidget


def _build_control_chart():
    from spacr.qt.screens.control_chart import ControlChartScreen
    return ControlChartScreen(threaded=False)


def _build_data_manager():
    from spacr.qt.screens.data_manager import DataManagerScreen
    return DataManagerScreen()


def _build_dose_response():
    from spacr.qt.screens.dose_response import DoseResponseScreen
    return DoseResponseScreen(threaded=False)


def _build_experiment_design():
    from spacr.qt.screens.experiment_design import ExperimentDesignScreen
    return ExperimentDesignScreen()


def _build_feature_explorer():
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen
    return FeatureExplorerScreen(threaded=False)


def _build_gate_editor():
    from spacr.qt.screens.gate_editor import GateEditorScreen
    return GateEditorScreen(threaded=False)


def _build_graph_builder():
    from spacr.qt.screens.graph_builder import GraphBuilderScreen
    return GraphBuilderScreen(threaded=False)


def _build_hit_list():
    from spacr.qt.screens.hit_list import HitListScreen
    return HitListScreen()


def _build_methods_export():
    from spacr.qt.screens.methods_export import MethodsExportScreen
    return MethodsExportScreen()


def _build_outliers():
    from spacr.qt.screens.outliers import OutliersScreen
    return OutliersScreen(threaded=False)


def _build_pca():
    from spacr.qt.screens.pca import PCAScreen
    return PCAScreen(threaded=False)


def _build_pipeline_graph():
    from spacr.qt.screens.pipeline_graph import PipelineGraphScreen
    return PipelineGraphScreen(threaded=False)


def _build_power():
    from spacr.qt.screens.power import PowerScreen
    return PowerScreen(threaded=False)


def _build_profiler():
    from spacr.qt.screens.profiler import ProfilerScreen
    return ProfilerScreen()


def _build_project_browser():
    from spacr.qt.screens.project_browser import ProjectBrowserScreen
    return ProjectBrowserScreen(threaded=False)


def _build_qc_dashboard():
    from spacr.qt.screens.qc_dashboard import QCDashboardScreen
    return QCDashboardScreen()


def _build_run_compare():
    from spacr.qt.screens.run_compare import RunCompareScreen
    return RunCompareScreen()


def _build_tabulate():
    from spacr.qt.screens.tabulate import TabulateScreen
    return TabulateScreen(threaded=False)


def _build_trellis():
    from spacr.qt.screens.trellis import TrellisScreen
    return TrellisScreen(threaded=False)


#: The nineteen screens that were not using the shared header. Fifteen wore
#: a ``ScreenTitle`` label — an object name with no rule, so body size — and
#: the four marked below opened on a paragraph with no title at all.
SCREENS = (
    ("Control Charts", _build_control_chart),
    ("Data Manager", _build_data_manager),
    ("Dose-Response", _build_dose_response),
    ("Experiment Design", _build_experiment_design),      # had no title
    ("Feature Explorer", _build_feature_explorer),
    ("Gate Editor", _build_gate_editor),
    ("Graph Builder", _build_graph_builder),
    ("Hit List", _build_hit_list),
    ("Methods Export", _build_methods_export),
    ("Outliers", _build_outliers),
    ("PCA", _build_pca),
    ("Pipeline Graph", _build_pipeline_graph),
    ("Power", _build_power),                              # had no title
    ("Profiler", _build_profiler),
    ("Project Browser", _build_project_browser),          # had no title
    ("QC Dashboard", _build_qc_dashboard),                # had no title
    ("Run Compare", _build_run_compare),
    ("Tabulate", _build_tabulate),
    ("Trellis", _build_trellis),
)


def _show(qtbot, factory):
    """Build, show and lay out a screen so its geometry is real.

    An unshown widget reports its children at ``(0, 0, 100, 30)``, so every
    arrangement assertion would be comparing placeholders. Showing it and
    pumping the event loop is what makes ``geometry()`` mean anything.
    """
    screen = factory()
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    QApplication.processEvents()
    screen.layout().activate()
    QApplication.processEvents()
    return screen


def _header(screen):
    from spacr.qt.screens.app_screen import ModuleHeader

    found = screen.findChild(ModuleHeader)
    assert found is not None, (
        f"{type(screen).__name__} does not use ModuleHeader — it is rolling "
        "a header of its own, which is the defect this file exists for")
    return found


def _px(label) -> int:
    """The rendered cap height of ``label``'s type, in pixels.

    ``QFontMetrics`` on the label's **effective** font, so a size that
    arrived through the stylesheet counts and one that never reached the
    widget does not. Reading ``font().pixelSize()`` would not do: a font
    set in points reports ``-1`` there.
    """
    return QFontMetrics(label.font()).height()


@pytest.fixture
def mask_header(qtbot, qt_theme_applied):
    """Mask Generation's header — the reference the report named."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key="mask")
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    QApplication.processEvents()
    return _header(screen)


def test_the_reference_header_is_itself_large(mask_header, qt_theme_applied):
    """Mask's name is much bigger than body text.

    Everything below compares against Mask, so if Mask were not itself
    large the whole file would pass on a page of uniformly small type.
    """
    body = QFontMetrics(QApplication.instance().font()).height()
    assert _px(mask_header.title_label) > body * 1.6, (
        f"the reference module name renders at {_px(mask_header.title_label)}"
        f"px against a {body}px body — it is not large, so nothing this file "
        "compares against it means anything")


@pytest.mark.parametrize("name,factory", SCREENS, ids=[n for n, _ in SCREENS])
def test_the_module_name_is_as_large_as_masks(name, factory, qtbot,
                                              mask_header, qt_theme_applied):
    """Every screen's name renders at the size Mask's does.

    The measurement is the rendered height of the label's own font, not
    the rule that was meant to set it. Before, the fifteen ``ScreenTitle``
    screens rendered at the body height — no selector reached that object
    name — and the four with no title had nothing to measure.
    """
    reference = _px(mask_header.title_label)
    screen = _show(qtbot, factory)
    measured = _px(_header(screen).title_label)
    assert measured == reference, (
        f"{name}: the module name renders at {measured}px where Mask "
        f"Generation's renders at {reference}px")


@pytest.mark.parametrize("name,factory", SCREENS, ids=[n for n, _ in SCREENS])
def test_the_description_is_right_of_the_name(name, factory, qtbot,
                                              qt_theme_applied):
    """"...with description to the right", measured as laid-out geometry.

    Both labels are mapped into the header's own coordinates, so a screen
    that nests its header inside another row is compared on the same terms
    as one that does not.
    """
    screen = _show(qtbot, factory)
    header = _header(screen)
    assert header.description_label is not None, (
        f"{name}: the header has no description")
    title = header.title_label
    blurb = header.description_label
    left = blurb.mapTo(header, blurb.rect().topLeft()).x()
    right = title.mapTo(header, title.rect().topRight()).x()
    assert left >= right, (
        f"{name}: the description starts at x={left} and the module name "
        f"ends at x={right} — it is not to the right of the name")


@pytest.mark.parametrize("name,factory", SCREENS, ids=[n for n, _ in SCREENS])
def test_the_instruction_is_below_the_name(name, factory, qtbot,
                                           qt_theme_applied):
    """"...and brief instruction below", likewise.

    Below the *name*, and short: an instruction that runs to a paragraph
    is a description in the wrong place, so the length is pinned too.
    """
    screen = _show(qtbot, factory)
    header = _header(screen)
    instruction = header.instruction_label
    assert instruction.isVisible() and instruction.text().strip(), (
        f"{name}: the header has no instruction under the module name")
    title_bottom = header.title_label.mapTo(
        header, header.title_label.rect().bottomLeft()).y()
    top = instruction.mapTo(header, instruction.rect().topLeft()).y()
    assert top >= title_bottom, (
        f"{name}: the instruction starts at y={top} and the module name ends "
        f"at y={title_bottom} — it is not below the name")
    assert len(instruction.text()) <= 120, (
        f"{name}: the instruction is {len(instruction.text())} characters — "
        "that is a description, and descriptions go to the right")


def test_no_screen_still_wears_the_ruleless_title():
    """``ScreenTitle`` reaches nothing, so nothing may use it.

    The object name fifteen screens set has no rule in
    :func:`spacr.qt.theme.stylesheet` and never had one, which is why they
    all rendered at body size. A source check rather than a pixel one, on
    purpose: this is the one assertion that catches the *twentieth* screen
    before anybody renders it.
    """
    from pathlib import Path
    import spacr.qt.screens as screens_pkg

    root = Path(screens_pkg.__file__).parent
    offenders = sorted(path.name for path in root.glob("*.py")
                       if '"ScreenTitle"' in path.read_text(encoding="utf-8"))
    assert not offenders, (
        "these screens still name a label 'ScreenTitle', which no stylesheet "
        "rule reaches, so it renders at body size: " + ", ".join(offenders)
        + ". Use spacr.qt.screens.app_screen.ModuleHeader instead.")
