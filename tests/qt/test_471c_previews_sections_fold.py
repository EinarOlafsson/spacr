"""Item 471 slice C: every section inside a live preview folds and drags.

"also any sections inside the figures or live preview containers should also
be colapseable ... whenever anything is collapsed it should auto loch to the
bottom of the container it is in." Mask's image-set table, its images and its
pixel strip; Plaque's pictures and its wells/plaques tables (and the well
picture beside the image); Timelapse's and Motility's settings groups, picture
and summary; image UMAP's sidebar; the annotation UMAP tab's plot, table and
report. Each folds by its heading, a folded one is its heading at the bottom
of its room, a dragged size is remembered, and the lazily-built preview is not
built by any of it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                # noqa: E402
from PySide6.QtWidgets import QApplication                   # noqa: E402

from spacr.qt.widgets.collapsible_splitter import (          # noqa: E402
    EDGE, FoldSection, get_pane_extents)


def _pump(n: int = 10) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _bottom_of(section, container) -> int:
    heading = section.heading
    return heading.mapTo(container, heading.rect().bottomLeft()).y()


def _shown(qtbot, widget, width=1100, height=760):
    qtbot.addWidget(widget)
    widget.resize(width, height)
    widget.show()
    _pump()
    return widget


def _mask(qtbot):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    return _shown(qtbot, LivePreviewPanel(threaded=False))


def _plaque(qtbot, mode="figure"):
    from spacr.qt.widgets.plaque_preview import PlaquePreviewPanel

    panel = PlaquePreviewPanel(threaded=False)
    panel.set_mode(mode)
    return _shown(qtbot, panel)


def _timelapse(qtbot):
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    return _shown(qtbot, TimelapsePreviewPanel(threaded=False))


def _motility(qtbot):
    from spacr.qt.widgets.motility_preview import MotilityPreviewPanel

    return _shown(qtbot, MotilityPreviewPanel(threaded=False))


PREVIEWS = (
    ("mask", _mask, "_table_split",
     ("Image sets", "Images", "Pixel info"), "Pixel info"),
    ("plaque", _plaque, "_section_split",
     ("Pictures", "Wells and plaques"), "Wells and plaques"),
    ("timelapse", _timelapse, "_section_split",
     ("Preview settings", "Movie", "Track quality"), "Track quality"),
    ("motility", _motility, "_section_split",
     ("Preview settings", "Plot", "Summary"), "Summary"),
)


@pytest.mark.parametrize("name,build,split_attr,sections,last", PREVIEWS,
                         ids=[p[0] for p in PREVIEWS])
def test_every_section_of_a_preview_folds_by_its_heading(
        qtbot, name, build, split_attr, sections, last):
    panel = build(qtbot)
    split = getattr(panel, split_attr)
    for section_name in sections:
        section = panel._sections[section_name]
        assert isinstance(section, FoldSection)
        assert split.pane(section_name) is not None, section_name
        section.set_folded(True, by_user=False)
        _pump()
        assert not section.body.isVisible(), section_name
        section.set_folded(False, by_user=False)
        _pump()
        assert section.body.isVisible(), section_name


@pytest.mark.parametrize("name,build,split_attr,sections,last", PREVIEWS,
                         ids=[p[0] for p in PREVIEWS])
def test_folding_everything_stacks_the_headings_at_the_bottom(
        qtbot, name, build, split_attr, sections, last):
    panel = build(qtbot)
    split = getattr(panel, split_attr)
    for section_name in sections:
        panel._sections[section_name].set_folded(True, by_user=False)
    _pump()
    bottom = _bottom_of(panel._sections[last], split)
    assert bottom >= split.height() - 12, (
        f"{name}: the last folded heading ends at {bottom} of "
        f"{split.height()}; it should be locked to the bottom")
    first = panel._sections[sections[0]]
    assert _bottom_of(first, split) > split.height() // 2, (
        f"{name}: the first folded heading stayed at the top")


def test_a_dragged_section_size_is_remembered(qtbot):
    from spacr.qt.widgets.live_preview import LivePreviewPanel

    panel = _mask(qtbot)
    split = panel._table_split
    split.moveSplitter(320, 1)
    _pump()
    stored = get_pane_extents(f"{LivePreviewPanel.SECTION_KEY}::sections")
    assert stored.get("Image sets", 0) > 200

    again = _mask(qtbot)
    assert again._table_split.pane("Image sets").extent == stored["Image sets"]


def test_a_user_fold_of_a_preview_section_survives_a_rebuild(qtbot):
    from spacr.qt.preferences import set_folded_panel

    panel = _timelapse(qtbot)
    panel._sections["Preview settings"].set_folded(True, by_user=True)
    again = _timelapse(qtbot)
    try:
        assert again._sections["Preview settings"].shut
    finally:
        set_folded_panel("timelapse_preview/Preview settings", False)


def test_plaque_mode_hides_the_empty_tables_section(qtbot):
    """In Plaque mode there are no well tables: no empty heading either."""
    panel = _plaque(qtbot, mode="plaque")
    assert panel._sections["Wells and plaques"].isHidden()
    assert not panel._sections["Pictures"].isHidden()
    panel.set_mode("figure")
    _pump()
    assert not panel._sections["Wells and plaques"].isHidden()


def test_plaque_well_picture_collapses_to_the_right_by_its_handle(qtbot):
    panel = _plaque(qtbot)
    pictures = panel._pictures_split
    assert pictures.pane("Well").mode == EDGE
    pictures.toggle_pane("Well", by_user=False)
    _pump()
    assert pictures.sizes()[1] == 0
    assert pictures.handle(1).edge_pane() is pictures.pane("Well")
    pictures.toggle_pane("Well", by_user=False)
    _pump()
    assert pictures.sizes()[1] > 0


def test_the_umap_sidebar_collapses_and_keeps_the_display_width(qtbot):
    from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

    explorer = _shown(qtbot, ImageUmapExplorer(), 1400, 760)
    split = explorer._body_splitter
    split.set_collapsed("Sidebar", True, by_user=False)
    _pump()
    assert split.sizes()[1] == 0
    explorer.apply_display({"sidebar_width": 333})
    _pump()
    assert split.sizes()[1] == 0, "a display change reopened the sidebar"
    assert split.pane("Sidebar").extent == 333
    split.set_collapsed("Sidebar", False, by_user=False)
    _pump()
    assert split.sizes()[1] > 0


def test_the_annotation_tab_folds_its_plot_table_and_report(qtbot):
    from spacr.qt.widgets.annotation_umap_tab import AnnotationUmapTab

    tab = _shown(qtbot, AnnotationUmapTab())
    names = [tab.body.widget(i).heading.text() for i in range(2)]
    assert [n.split(" ", 1)[1] for n in names] == ["Purity plot", "Scores"]
    report = tab.sections.pane("Report").widget
    report.set_folded(True, by_user=False)
    _pump()
    assert _bottom_of(report, tab.sections) >= tab.sections.height() - 12
    assert tab.sections.sizes()[0] > tab.sections.height() - 60


def test_the_hidden_live_preview_stays_unbuilt(qtbot):
    """Items 284/380: the sections live inside the lazily-built preview."""
    from spacr.qt.screens.app_screen import _LIVE_PREVIEW, AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen.resize(1366, 768)
    screen.show()
    _pump()
    assert screen._part_is_owed(_LIVE_PREVIEW)
    screen._figures_card.show()
    _pump()
    assert screen._part_is_owed(_LIVE_PREVIEW), (
        "the fold machinery built the hidden live preview")
    screen._build_owed_part(_LIVE_PREVIEW)
    panel = screen._if_built("_live_preview")
    assert panel is not None and "Image sets" in panel._sections
    assert screen._live_preview_card.isHidden(), (
        "building the preview's sections showed the preview")
