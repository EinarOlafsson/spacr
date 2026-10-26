"""Every settings popup paints a solid, theme-coloured background.

Asked for on 2026-10-24: "the settings window in live model in mask
generation should have a solid non transparent background same for other
settings popup windows."

WHAT IS MEASURED IS THE PIXEL. Each popup is built, glassed by the real
application filter, shown offscreen and grabbed; the grab's pixels inside the
rounded card must be fully opaque and the bare band the rim runs along must be
the theme's surface colour, in the light theme and in the dark one. A plain
popup that is not a settings popup is grabbed the same way and must still be
see-through, so the measurement can tell the two apart.

THE LIST IS CHECKED AGAINST THE SOURCE. Every dialog class whose name or
registration makes it a settings popup, and every dialog marked one by its
``spacrSettingsPopup`` property, must have a builder below.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "spacr"


@pytest.fixture
def glassed(qapp):
    """The application filter that gives every dialog its card."""
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    was = uninstall_glass_everywhere(qapp)
    install_glass_everywhere(qapp)
    yield qapp
    uninstall_glass_everywhere(qapp)
    if was:
        install_glass_everywhere(qapp)


def _themed(app, theme):
    """The theme, with both things that made popups see-through switched on:
    a half-opaque pane and the drifting backdrop behind every popup."""
    from spacr.qt.preferences import (apply_preferences_to_app,
                                      set_pane_opacity, set_popup_backdrop,
                                      set_theme)

    set_theme(theme)
    set_pane_opacity(0.5)
    set_popup_backdrop("drift")
    apply_preferences_to_app(app)


def _a_figure():
    from matplotlib.figure import Figure

    figure = Figure()
    figure.add_subplot(111).plot([1, 2], [3, 4])
    return figure


def _keymap(qtbot):
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import shortcuts

    window = QMainWindow()
    qtbot.addWidget(window)
    shortcuts.install(window)
    return shortcuts._KeymapDialog(window)


def _embeddings(qtbot):
    from spacr.qt.screens.embeddings import EmbeddingsScreen

    screen = EmbeddingsScreen(threaded=False)
    qtbot.addWidget(screen)
    return screen


def _overview_options(qtbot):
    from spacr.qt.widgets.qc_field_browser import _FieldOverview

    overview = _FieldOverview(settings={})
    qtbot.addWidget(overview, before_close_func=lambda widget: widget._cancel())
    return overview._options_window


def _overview_image_settings(qtbot):
    from spacr.qt.widgets.qc_field_browser import (_FieldOverview, _OverviewField,
                                                   _OverviewImageSettings)

    overview = _FieldOverview(settings={})
    qtbot.addWidget(overview, before_close_func=lambda widget: widget._cancel())
    field = _OverviewField(plate="p1", well="A01", row="A", column=1, field="1", files=())
    return _OverviewImageSettings(overview, field, (0,))


def _canvas_settings(qtbot):
    from spacr.qt.widgets.picture_export import _CanvasComposer

    composer = _CanvasComposer()
    qtbot.addWidget(composer)
    return composer.settings_dialog


def _builders():
    """``(source file, class name or objectName) -> builder(qtbot)``."""

    def panel(cls, **kwargs):
        def build(qtbot):
            widget = cls(**kwargs)
            qtbot.addWidget(widget)
            return widget
        return build

    def live(qtbot):
        from spacr.qt.widgets.live_preview import (LivePreviewPanel,
                                                   LiveSettingsDialog)
        return LiveSettingsDialog(panel(LivePreviewPanel)(qtbot))

    def preferences(qtbot):
        from spacr.qt.preferences import PreferencesDialog
        return PreferencesDialog()

    def figure_settings(qtbot):
        from spacr.qt.widgets.figure_settings import FigureSettingsDialog
        return FigureSettingsDialog(_a_figure())

    def queued_figure_settings(qtbot):
        from spacr.qt.widgets.figure_queue import _FigureSettingsDialog
        return _FigureSettingsDialog(_a_figure())

    def gates(qtbot):
        from spacr.qt.widgets.gate_settings import (GateEditorSettings,
                                                    GateSettingsDialog)
        return GateSettingsDialog(GateEditorSettings())

    def clusters(qtbot):
        from spacr.qt.widgets.gate_editor import _ClusterSettingsDialog
        return _ClusterSettingsDialog()

    def crop(qtbot):
        from spacr.qt.widgets.measure_preview import (CropSettingsDialog,
                                                      MeasurePreviewPanel)
        return CropSettingsDialog(panel(MeasurePreviewPanel,
                                        threaded=False)(qtbot))

    def picture(qtbot):
        from spacr.qt.widgets.picture_settings_dialog import \
            PictureSettingsDialog
        return PictureSettingsDialog()

    def plaque(qtbot):
        from spacr.qt.widgets import plaque_preview as ppv
        return ppv.PlaqueSettingsDialog(panel(ppv.PlaquePreviewPanel,
                                              threaded=False)(qtbot))

    def plaque_overlay(qtbot):
        from spacr.qt.widgets.plaque_preview import (OverlayStyle,
                                                     PlaqueOverlayDialog)
        return PlaqueOverlayDialog(OverlayStyle())

    def umap_display(qtbot):
        from spacr.qt.widgets.umap_explorer import UmapDisplaySettings
        return UmapDisplaySettings({})

    def umap_search(qtbot):
        from spacr.qt.screens.hyperparam import (HyperparamPanel,
                                                 UmapSearchSettingsDialog)
        widget = HyperparamPanel("ml_analyze")
        qtbot.addWidget(widget)
        return UmapSearchSettingsDialog(widget)

    def annotate(qtbot):
        from spacr.qt.screens.annotate import AnnotateSettings, _SettingsDialog
        return _SettingsDialog(AnnotateSettings())

    def umap_appearance(qtbot):
        from spacr.qt.widgets.umap_search_viewer import UmapAppearanceDialog
        return UmapAppearanceDialog({})

    def providers(qtbot):
        from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog
        return _ProvidersDialog()

    def execution_profile(qtbot):
        from spacr.qt.screens.distributed_jobs import ExecutionProfileDialog
        return ExecutionProfileDialog()

    def advisor(qtbot):
        from spacr.qt.widgets.settings_advisor_dialog import \
            SettingsAdvisorDialog
        from spacr.settings_advisor import Reading
        return SettingsAdvisorDialog(Reading(), {})

    def aggregation(qtbot):
        import pandas as pd

        from spacr.qt.widgets.aggregation_rules import AggregationRulesDialog
        return AggregationRulesDialog(
            pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))

    def levels(qtbot):
        from spacr.qt.screens.make_masks import _LevelsDialog
        image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        return _LevelsDialog(image, (1.0, 99.0))

    return {
        ("spacr/qt/widgets/live_preview.py", "LiveSettingsDialog"): live,
        ("spacr/qt/preferences.py", "_PreferencesWindow"): preferences,
        ("spacr/qt/widgets/figure_settings.py", "FigureSettingsDialog"):
            figure_settings,
        ("spacr/qt/widgets/figure_queue.py", "_FigureSettingsDialog"):
            queued_figure_settings,
        ("spacr/qt/widgets/gate_settings.py", "GateSettingsDialog"): gates,
        ("spacr/qt/widgets/gate_editor.py", "_ClusterSettingsDialog"): clusters,
        ("spacr/qt/widgets/measure_preview.py", "CropSettingsDialog"): crop,
        ("spacr/qt/widgets/picture_settings_dialog.py",
         "PictureSettingsDialog"): picture,
        ("spacr/qt/widgets/plaque_preview.py", "PlaqueSettingsDialog"): plaque,
        ("spacr/qt/widgets/plaque_preview.py", "PlaqueOverlayDialog"):
            plaque_overlay,
        ("spacr/qt/widgets/umap_explorer.py", "UmapDisplaySettings"):
            umap_display,
        ("spacr/qt/screens/hyperparam.py", "UmapSearchSettingsDialog"):
            umap_search,
        ("spacr/qt/screens/annotate.py", "_SettingsDialog"): annotate,
        ("spacr/qt/widgets/umap_search_viewer.py", "UmapAppearanceDialog"):
            umap_appearance,
        ("spacr/qt/widgets/ai_chat_panel.py", "_ProvidersDialog"): providers,
        ("spacr/qt/shortcuts.py", "_KeymapDialog"): _keymap,
        ("spacr/qt/screens/distributed_jobs.py", "ExecutionProfileDialog"):
            execution_profile,
        ("spacr/qt/widgets/settings_advisor_dialog.py",
         "SettingsAdvisorDialog"): advisor,
        ("spacr/qt/widgets/aggregation_rules.py", "AggregationRulesDialog"):
            aggregation,
        ("spacr/qt/screens/make_masks.py", "_LevelsDialog"): levels,
        ("spacr/qt/widgets/qc_field_browser.py", "OverviewOptionsWindow"):
            _overview_options,
        ("spacr/qt/widgets/qc_field_browser.py", "_OverviewImageSettings"):
            _overview_image_settings,
        ("spacr/qt/widgets/picture_export.py", "_CanvasSettingsDialog"):
            _canvas_settings,
        ("spacr/qt/screens/embeddings.py", "EmbeddingsSubCellChannelsDialog"):
            lambda qtbot: _embeddings(qtbot)._subcell_channels_dialog(),
        ("spacr/qt/screens/embeddings.py", "EmbeddingsCellDinoDialog"):
            lambda qtbot: _embeddings(qtbot)._cell_dino_dialog(),
    }


BUILDERS = _builders()


def _base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _source_settings_popups():
    """Every settings popup the source defines, as ``(file, name)``."""
    from spacr.qt.widgets.glass import (SETTINGS_POPUP, SETTINGS_POPUP_CLASSES,
                                        SETTINGS_POPUP_NAME)

    classes, marked = {}, set()
    for path in sorted(PACKAGE.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[(relative, node.name)] = {
                    _base_name(base) for base in node.bases}
            if (isinstance(node, ast.Call)
                    and _base_name(node.func) == "setProperty"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == SETTINGS_POPUP
                    and isinstance(node.args[1], ast.Constant)
                    and node.args[1].value is True):
                marked.add(relative)
    dialogs = {"QDialog"}
    while True:
        grown = dialogs | {name for (_, name), bases in classes.items()
                           if bases & dialogs}
        if grown == dialogs:
            break
        dialogs = grown
    found = {key for key, bases in classes.items()
             if bases & dialogs and (key[1] in SETTINGS_POPUP_CLASSES
                                     or SETTINGS_POPUP_NAME.search(key[1]))}
    return found, marked, {name for _, name in classes}


def test_every_settings_popup_in_the_source_has_a_builder():
    found, marked, names = _source_settings_popups()
    from spacr.qt.widgets.glass import SETTINGS_POPUP_CLASSES

    assert SETTINGS_POPUP_CLASSES <= names, (
        "registered names with no class", SETTINGS_POPUP_CLASSES - names)
    built_classes = {key for key in BUILDERS if key in found}
    assert found == built_classes, (
        "settings popups without a builder", sorted(found - built_classes))
    by_marks = {key for key in BUILDERS if key not in found}
    assert {path for path, _ in by_marks} == marked
    assert len(BUILDERS) == len(found) + len(by_marks) == 25


def _open(qtbot, app, dialog):
    from PySide6.QtWidgets import QWidget

    if dialog.parent() is None:
        qtbot.addWidget(dialog)
    QWidget.resize(dialog, max(QWidget.size(dialog).width(), 420),
                   max(QWidget.size(dialog).height(), 320))
    dialog.show()
    qtbot.waitExposed(dialog)
    for _ in range(5):
        app.processEvents()
    return dialog


def _pixels(dialog):
    """The grab as an ``(h, w, 4)`` RGBA array, alpha straight."""
    from PySide6.QtGui import QImage

    image = dialog.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    width, height = image.width(), image.height()
    data = np.frombuffer(image.constBits(), dtype=np.uint8,
                         count=image.bytesPerLine() * height)
    rows = data.reshape(height, image.bytesPerLine())[:, :width * 4]
    return rows.reshape(height, width, 4).copy(), image.devicePixelRatio()


def _band(pixels, ratio):
    """Pixels on the bare band the rim runs along, clear of the corners."""
    from spacr.qt.widgets.glass import CARD_RADIUS

    offset = int(round(6 * ratio))
    skip = int(round((CARD_RADIUS + 6) * ratio))
    height, width = pixels.shape[:2]
    return np.concatenate([
        pixels[offset, skip:width - skip], pixels[height - 1 - offset,
                                                  skip:width - skip],
        pixels[skip:height - skip, offset], pixels[skip:height - skip,
                                                   width - 1 - offset]])


def _interior(pixels, ratio):
    from spacr.qt.widgets.glass import CARD_RADIUS

    skip = int(round((CARD_RADIUS + 2) * ratio))
    return pixels[skip:-skip, skip:-skip]


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("key", list(BUILDERS), ids=[n for _, n in BUILDERS])
def test_the_settings_popup_is_solid_in_the_theme_colour(glassed, qtbot,
                                                          monkeypatch, theme,
                                                          key):
    from spacr.qt.theme import palette_for
    from spacr.qt.widgets.glass import (SETTINGS_POPUP, _is_settings_popup,
                                        _settings_popup_colour)
    from spacr.qt.widgets.setup_card import SetupCard

    from spacr.qt.widgets import plaque_preview

    monkeypatch.setattr(plaque_preview, "missing_papers_packages",
                        lambda *a, **k: [])
    _themed(glassed, theme)
    dialog = _open(qtbot, glassed, BUILDERS[key](qtbot))
    try:
        assert _is_settings_popup(dialog), key
        assert dialog.property(SETTINGS_POPUP) is True
        assert dialog.findChildren(SetupCard), "the dialog was never glassed"
        assert getattr(dialog, "_spacr_popup_backdrop", None) is None \
            or dialog._spacr_popup_backdrop.isHidden()

        pixels, ratio = _pixels(dialog)
        inside = _interior(pixels, ratio)
        assert int(inside[..., 3].min()) == 255, (
            key, theme, "see-through pixels", int((inside[..., 3] < 255).sum()))

        expected = _settings_popup_colour(palette_for(theme))
        assert expected.alpha() == 255
        band = _band(pixels, ratio)
        assert int(band[:, 3].min()) == 255
        colour, count = Counter(map(tuple, band[:, :3].tolist())).most_common(1)[0]
        assert colour == (expected.red(), expected.green(), expected.blue()), (
            key, theme, colour, expected.name())
        assert count >= 0.5 * len(band)
    finally:
        dialog.close()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_a_popup_that_is_not_settings_stays_translucent(glassed, qtbot, theme):
    """The control: the same grab tells a see-through card from a solid one.

    See-through here means either alpha below 255 or, with the drifting
    backdrop behind the card, a band that is not the plain surface colour.
    """
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

    from spacr.qt.theme import palette_for
    from spacr.qt.widgets.glass import _is_settings_popup, _settings_popup_colour

    _themed(glassed, theme)
    dialog = QDialog()
    column = QVBoxLayout(dialog)
    column.addWidget(QLabel("about"))
    column.addWidget(QDialogButtonBox(QDialogButtonBox.Close))
    _open(qtbot, glassed, dialog)
    try:
        assert not _is_settings_popup(dialog)
        pixels, ratio = _pixels(dialog)
        band = _band(pixels, ratio)
        expected = _settings_popup_colour(palette_for(theme))
        solid = (expected.red(), expected.green(), expected.blue())
        plain = Counter(map(tuple, band[:, :3].tolist()))[solid]
        see_through = int((band[:, 3] < 255).sum())
        assert see_through > 0 or plain < 0.5 * len(band), (
            theme, "a popup that is not settings reads as solid")
    finally:
        dialog.close()


def test_a_dialog_can_be_marked_or_unmarked():
    from PySide6.QtWidgets import QDialog

    from spacr.qt.widgets.glass import SETTINGS_POPUP, _is_settings_popup

    plain = QDialog()
    assert not _is_settings_popup(plain)
    plain.setProperty(SETTINGS_POPUP, True)
    assert _is_settings_popup(plain)
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog
    named = PictureSettingsDialog()
    assert _is_settings_popup(named)
    named.setProperty(SETTINGS_POPUP, False)
    assert not _is_settings_popup(named)
    assert not _is_settings_popup(None)
    plain.deleteLater()
    named.deleteLater()
