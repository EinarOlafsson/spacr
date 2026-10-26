"""Round-6 Overview export: the docked export section, drag and drop, pick mode, imports, field picking, DPI.

Item 725, maintainer feedback round 6 (verbatim):
1. "for the custom export i want to be able to drag and drop pictures from the overview window onto the canvas."
2. "the export windows need a close button."
3. "the export popup should instead be a section opend to the far right of the overview tab and when it opens
   the settings section should auto colapse."
4. "the user should me able to hold down a button and click on any image on screen (or do the path option but
   that is much more teadeus but keep it as an option) or select any image file and drag and drop it (there
   should be import settings when this or path is used so the user is able to normalize and enhance these
   images as well)"
5. "in guided when puicking fields the user should be ablu to drag to select and right click and select and
   deselect or click hold shift and the normal hotkeys."
6. "the user should be able to go back choose dpi in the last page other wise dpi is set in the main
   application i guess."
7. "for the export the settings should be hidden under a settings button which triggers a popup so the canvs
   can take as much space as possible"
"""
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QPushButton
from pypdf import PdfReader

import spacr.plot  # noqa: F401  -- imported here, on the main thread, before any export worker needs it
from spacr.qt.widgets import qc_field_browser as browser
from spacr.qt.widgets import picture_export


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B02")


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    rows, columns = np.mgrid[0:45, 0:60]
    for index, well in enumerate(WELLS):
        for field in (1, 2, 3):
            for channel in (1, 2):
                pixels = (rows * (channel + 1) + columns * field + 50 * index).astype(np.uint16)
                tifffile.imwrite(folder / f"p1-{well}-f{field}-c{channel}.tif", pixels)
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _retire(view, qtbot):
    view._close_export_dock()
    view._cancel()
    qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=15000)
    for runner in _runners(view):
        runner.shutdown(timeout_ms=3000)
    view.close()


@pytest.fixture
def view(qtbot, source):
    browser._OVERVIEW_PLANE_CACHE.clear()
    browser._OVERVIEW_INFO_CACHE.clear()
    dialog = browser._FieldOverview(settings={
        "source": str(source), "source_scope": "folder", "custom_regex": PATTERN,
        "channels": (1, 2), "geometry": (2, 2), "representatives": 1, "pixel_size_um": .5})
    qtbot.addWidget(dialog, before_close_func=lambda widget: _retire(widget, qtbot))
    dialog.resize(1500, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: len(dialog._fields) == 12 and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    return dialog


def _mouse(widget, kind, point, button=Qt.LeftButton, buttons=Qt.LeftButton, modifiers=Qt.NoModifier):
    event = QMouseEvent(kind, QPointF(point), QPointF(widget.mapToGlobal(point)), button, buttons, modifiers)
    QApplication.sendEvent(widget, event)
    return event


def _drop_on(composer, mime):
    """Deliver a whole drag (enter, move, drop) to the middle of the canvas, as the platform does."""
    viewport = composer.view.viewport()
    point = QPointF(viewport.rect().center())
    actions = Qt.CopyAction
    for event in (QDragEnterEvent(point.toPoint(), actions, mime, Qt.LeftButton, Qt.NoModifier),
                  QDragMoveEvent(point.toPoint(), actions, mime, Qt.LeftButton, Qt.NoModifier),
                  QDropEvent(point, actions, mime, Qt.LeftButton, Qt.NoModifier)):
        QApplication.sendEvent(viewport, event)
    return event.isAccepted()


def _drags_drop_on(drags, composer, monkeypatch):
    """Let every started Overview drag end on the canvas; returns the list of drags started."""
    started = []

    def run(drag):
        started.append(drag.mimeData())
        _drop_on(composer, drag.mimeData())
        return Qt.CopyAction

    monkeypatch.setattr(drags, "_run_drag", run)
    return started


def _sources(composer):
    return [(box["source"].get("well"), box["source"].get("field")) for box in composer.canvas.pages[0]
            if box["kind"] == "image"]


def test_plain_drags_from_plate_well_and_image_views_land_on_the_canvas(view, qtbot, monkeypatch):
    composer = view._open_custom_export()
    drags = view._canvas_drags
    started = _drags_drop_on(drags, composer, monkeypatch)
    view.wells.clearSelection()
    viewport = view.wells.viewport()
    point = view.wells.visualItemRect(view.wells.item(*view._well_positions[(view.plates.currentData(), "B02")])).center()
    _mouse(viewport, QEvent.MouseButtonPress, point)
    _mouse(viewport, QEvent.MouseMove, point + QPoint(30, 0), button=Qt.NoButton)
    _mouse(viewport, QEvent.MouseButtonRelease, point + QPoint(30, 0), buttons=Qt.NoButton)
    assert started == [] and _sources(composer) == []
    view.wells.clearSelection()
    _mouse(viewport, QEvent.MouseButtonPress, point)
    qtbot.wait(browser._OVERVIEW_CANVAS_HOLD_MS + 150)
    assert drags._armed and viewport.cursor().shape() == Qt.DragCopyCursor
    _mouse(viewport, QEvent.MouseMove, point + QPoint(30, 0), button=Qt.NoButton)
    assert len(started) == 1
    assert _sources(composer) == [("B02", "1"), ("B02", "2"), ("B02", "3")]
    assert not drags._armed and viewport.cursor().shape() != Qt.DragCopyCursor
    view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]))
    qtbot.waitUntil(lambda: view.fields.count() == 3, timeout=15000)
    view.fields.clearSelection()
    tile = view.fields.visualItemRect(view.fields.item(1)).center()
    _mouse(view.fields.viewport(), QEvent.MouseButtonPress, tile)
    qtbot.wait(browser._OVERVIEW_CANVAS_HOLD_MS + 150)
    _mouse(view.fields.viewport(), QEvent.MouseMove, tile + QPoint(0, 30), button=Qt.NoButton)
    assert _sources(composer)[-1] == ("A01", "2")
    field = next(field for field in view._fields if (field.well, field.field) == ("A02", "3"))
    view._open_image_views([field])
    drags._sync_handles()
    panel = view._image_views[-1]
    image = panel.view.viewport()
    _mouse(image, QEvent.MouseButtonPress, QPoint(40, 40))
    _mouse(image, QEvent.MouseMove, QPoint(90, 90), button=Qt.NoButton)
    _mouse(image, QEvent.MouseButtonRelease, QPoint(90, 90), buttons=Qt.NoButton)
    assert len(started) == 2, "a plain quick drag on an image still pans it"
    _mouse(image, QEvent.MouseButtonPress, QPoint(40, 40))
    qtbot.wait(browser._OVERVIEW_CANVAS_HOLD_MS + 150)
    _mouse(image, QEvent.MouseMove, QPoint(90, 90), button=Qt.NoButton)
    assert _sources(composer)[-1] == ("A02", "3")
    handle = next(handle for handle, owner in drags._handles.items() if owner is image)
    assert handle.isVisible() and handle.text() == browser.tr("Drag to canvas")
    assert "Ctrl+Shift" in handle.toolTip()
    _mouse(handle, QEvent.MouseButtonPress, QPoint(4, 4))
    _mouse(handle, QEvent.MouseMove, QPoint(40, 40), button=Qt.NoButton)
    assert len(started) == 4 and _sources(composer)[-1] == ("A02", "3")
    view._close_export_dock()
    assert drags._watched == [] and drags._handles == {}


def test_selected_wells_still_drag_at_once_and_alt_drags_the_shown_image(view, qtbot, monkeypatch):
    composer = view._open_custom_export()
    drags = view._canvas_drags
    started = _drags_drop_on(drags, composer, monkeypatch)
    cell = view.wells.item(*view._well_positions[(view.plates.currentData(), "A02")])
    view.wells.clearSelection()
    cell.setSelected(True)
    point = view.wells.visualItemRect(cell).center()
    _mouse(view.wells.viewport(), QEvent.MouseButtonPress, point)
    _mouse(view.wells.viewport(), QEvent.MouseMove, point + QPoint(25, 0), button=Qt.NoButton)
    assert len(started) == 1 and {well for well, _field in _sources(composer)} == {"A02"}
    field = next(field for field in view._fields if (field.well, field.field) == ("B01", "2"))
    view._open_image_views([field])
    image = view._image_views[-1].view.viewport()
    _mouse(image, QEvent.MouseButtonPress, QPoint(20, 20), modifiers=Qt.AltModifier)
    _mouse(image, QEvent.MouseMove, QPoint(60, 20), button=Qt.NoButton, modifiers=Qt.AltModifier)
    assert _sources(composer)[-1] == ("B01", "2")


def test_export_opens_docked_at_the_far_right_with_a_close_button(view, qtbot):
    composer = view._open_custom_export()
    dock = view._export_dock
    assert dock is not None and dock.content is composer and not composer.isWindow()
    host = view._export_splitter
    assert host.widget(host.count() - 1) is dock and host.widget(0) is view._splitter
    qtbot.waitUntil(lambda: dock.isVisible() and dock.width() >= 600, timeout=5000)
    close = dock.findChild(QPushButton, "OverviewExportClose")
    assert close is not None and close.isVisible() and close.text() == browser.tr("Close")
    assert "Ctrl+Shift" in dock.hint.text()
    close.click()
    assert view._export_dock is None and not composer.isVisible()
    guided = view._open_guided_export()
    dock = view._export_dock
    assert dock.content is guided and not guided.isWindow()
    assert dock.title.text() == browser.tr("Guided export")
    custom = guided.open_custom()
    assert view._export_dock.content is custom and host.indexOf(dock) == -1
    custom.close()
    assert view._export_dock is None


def test_the_canvas_fills_the_section_and_the_settings_popup_drives_it(view, qtbot):
    composer = view._open_custom_export()
    dock = view._export_dock
    qtbot.waitUntil(lambda: dock.isVisible() and composer.view.width() > 400, timeout=5000)
    qtbot.wait(50)
    canvas = composer.view.width() * composer.view.height()
    assert canvas > .6 * dock.width() * dock.height()
    assert composer.view.width() > .9 * dock.width()
    visible = {button.text() for button in composer.findChildren(QPushButton) if button.isVisibleTo(composer)}
    assert {browser.tr("Select"), browser.tr("Draw text box"), browser.tr("Add image…"), browser.tr("Undo"),
            browser.tr("Redo"), browser.tr("Export PDF…"), browser.tr("Settings…")} <= visible
    for hidden in (composer.page_size, composer.grid_spin, composer.dpi, composer.scale_bar, composer.import_group,
                   composer.pages):
        assert not hidden.isVisible()
    assert composer.settings_dialog.windowFlags() & Qt.WindowType.Tool
    from spacr.qt.widgets.glass import _is_settings_popup

    assert _is_settings_popup(composer.settings_dialog)
    composer.settings_button.click()
    dialog = composer.settings_dialog
    assert dialog.isVisible() and composer.page_size.isVisible() and composer.dpi.isVisible()
    composer.page_size.setCurrentIndex(composer.page_size.findData("Letter"))
    composer.orientation.setCurrentIndex(composer.orientation.findData("landscape"))
    composer.grid_spin.setValue(30)
    composer.snap.setChecked(False)
    assert composer.canvas.page_size == "Letter" and composer.canvas.orientation == "landscape"
    assert composer.canvas.grid == 30 and not composer.canvas.snap
    assert composer.view.sceneRect().width() > composer.view.sceneRect().height()
    box = composer.add_text(36, 36, 200, 40, text="Title")
    composer.font_size.setValue(18)
    composer.bold.setChecked(True)
    assert composer.canvas.find(box["id"])[1]["font_size"] == 18 and composer.canvas.find(box["id"])[1]["bold"]
    composer.add_page()
    assert composer.page_label.text() == browser.tr("Page {number} of {total}", number=2, total=2)
    composer.previous_page.click()
    assert composer.page_index == 0 and not composer.previous_page.isEnabled()
    dialog.close_button.click()
    assert not dialog.isVisible() and composer.canvas.page_size == "Letter"
    view._close_export_dock()
    assert not dialog.isVisible()


def test_the_settings_section_folds_while_the_export_section_is_open(qtbot, source):
    from spacr.qt.screens.app_screen import AppScreen

    mask = AppScreen("mask")
    qtbot.addWidget(mask)
    mask.resize(1700, 1000)
    mask.show()
    mask._settings_model.set_value_for_key("src", str(source))
    field = mask._settings_model._widgets.get("src")
    next(action for action in field.actions()
         if action.objectName() == "GenerateMasksFieldOverview").trigger()
    tab = mask._overview_tab
    page = tab._overview_page
    wizard = page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == 12
                    and page.browser._normalization_ready, timeout=15000)
    overview = page.browser
    body = tab._body_splitter
    qtbot.wait(50)
    assert not body.is_collapsed("Settings")
    before = body.sizes()
    try:
        composer = overview._open_custom_export()
        assert body.is_collapsed("Settings")
        host = overview._export_splitter
        assert host.widget(host.count() - 1) is overview._export_dock
        overview._export_dock.close_button.click()
        assert not body.is_collapsed("Settings") and body.sizes() == before
        assert not composer.isVisible()
        body.set_collapsed("Settings", True, by_user=True)
        overview._open_guided_export()
        overview._close_export_dock()
        assert body.is_collapsed("Settings")
    finally:
        page._retire_browser()
        mask.close()


def test_pick_mode_adds_any_clicked_image(view, qtbot):
    composer = view._open_custom_export()
    assert "Ctrl+Shift" in composer.status.text()
    point = view.wells.visualItemRect(view.wells.item(*view._well_positions[(view.plates.currentData(), "B01")])).center()
    view.wells.clearSelection()
    pick = Qt.ControlModifier | Qt.ShiftModifier
    _mouse(view.wells.viewport(), QEvent.MouseButtonPress, point, modifiers=pick)
    _mouse(view.wells.viewport(), QEvent.MouseButtonRelease, point, buttons=Qt.NoButton, modifiers=pick)
    assert _sources(composer) == [("B01", "1"), ("B01", "2"), ("B01", "3")]
    assert view.wells.selectedItems() == [], "a pick does not change the plate selection"
    field = next(field for field in view._fields if (field.well, field.field) == ("A01", "2"))
    view._open_image_views([field])
    _mouse(view._image_views[-1].view.viewport(), QEvent.MouseButtonPress, QPoint(30, 30), modifiers=pick)
    assert _sources(composer)[-1] == ("A01", "2")
    assert composer.status.text() == browser.tr("Added {count} images to page {page}.", count=1, page=1)
    view._close_export_dock()
    _mouse(view.wells.viewport(), QEvent.MouseButtonPress, point, modifiers=pick)
    assert len(_sources(composer)) == 4


def _rgb_file(tmp_path):
    rows, columns = np.mgrid[0:40, 0:50]
    pixels = np.stack([(rows * 6).astype(np.uint8), (columns * 5).astype(np.uint8),
                       np.full((40, 50), 30, np.uint8)], axis=-1)
    path = tmp_path / "outside.tif"
    tifffile.imwrite(path, pixels, photometric="rgb")
    return path


def test_dropped_and_chosen_files_get_import_settings_through_the_display_pipeline(view, qtbot, tmp_path,
                                                                                   monkeypatch):
    path = _rgb_file(tmp_path)
    composer = view._open_custom_export()
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    assert _drop_on(composer, mime)
    box = composer.canvas.pages[0][0]
    assert box["source"] == {"type": "file", "path": str(path)}
    assert composer.settings_dialog.isVisible(), "the import settings open with the dropped file"
    assert composer.import_group.isVisible() and composer.import_channel.count() == 4
    plain = picture_export._canvas_box_pixels(box, None, composer._reader())
    assert np.array_equal(plain, picture_export._canvas_file_pixels(str(path)))
    seen = []
    original = browser._overview_pixels

    def spy(field, channels, percentiles, **options):
        seen.append((field.files, channels, options.get("channel_enhancement"), options.get("percentile_channels")))
        return original(field, channels, percentiles, **options)

    monkeypatch.setattr(browser, "_overview_pixels", spy)
    composer.import_channel.setCurrentIndex(composer.import_channel.findData(2))
    composer.import_shown.setChecked(False)
    composer.import_channel.setCurrentIndex(composer.import_channel.findData(0))
    composer.import_steps["clahe"].setChecked(True)
    composer.import_gamma.setValue(.5)
    box = composer.canvas.pages[0][0]
    assert box["import"]["channels"]["2"]["shown"] is False
    assert box["import"]["channels"]["0"]["clahe"] and box["import"]["channels"]["0"]["gamma"] == .5
    pixels = picture_export._canvas_box_pixels(box, None, composer._reader())
    files, channels, chains, normalized = seen[-1]
    assert files == (("", str(path)),) and channels == (0, 1) and normalized == [0, 1]
    assert chains[0].clahe and chains[0].gamma == .5 and chains[1] is None
    assert pixels.shape == (40, 50, 3) and not np.array_equal(pixels, plain)
    assert int(pixels[..., 2].max()) == 0, "the hidden third channel is left out"
    composer.reset_import()
    assert "import" not in composer.canvas.pages[0][0]
    composer.undo()
    assert composer.canvas.pages[0][0]["import"]["channels"]["0"]["clahe"]
    composer.settings_dialog.hide()
    monkeypatch.setattr(picture_export.QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *args, **kwargs: ([str(path)], "")))
    add = next(button for button in composer.findChildren(QPushButton) if button.text() == browser.tr("Add image…"))
    add.click()
    assert len(composer.canvas.pages[0]) == 2 and composer.import_group.isVisible()
    assert composer.status.text() == browser.tr("Use the import settings in Settings… to normalize and enhance "
                                                "the added files, channel by channel.")
    layout_path = tmp_path / "layout.json"
    composer.save_layout(str(layout_path))
    reopened = picture_export._CanvasLayout.load(str(layout_path))
    assert reopened.pages[0][0]["import"] == composer.canvas.pages[0][0]["import"]


def test_files_on_a_canvas_without_the_overview_are_drawn_as_they_are(qtbot, tmp_path):
    path = _rgb_file(tmp_path)
    composer = picture_export._CanvasComposer()
    qtbot.addWidget(composer)
    box = composer.add_images([{"source": {"type": "file", "path": str(path)}, "label": "x"}])[0]
    assert not composer.settings_dialog.isVisible() and composer.import_group.isHidden()
    box["import"] = {"channels": {"0": {"shown": False}}}
    assert np.array_equal(picture_export._canvas_box_pixels(box, None, None),
                          picture_export._canvas_file_pixels(str(path)))


def _guided_at_fields(view):
    guided = view._open_guided_export()
    guided.choose_wells("wells", [("p1", "A01"), ("p1", "B02")])
    assert guided.next()
    return guided


def _ticked(guided):
    return [child.text(0) + "@" + child.parent().text(0)[-3:] for child in guided._field_children()
            if child.checkState(0) == Qt.Checked]


def test_guided_fields_pick_with_drag_shift_ctrl_ctrl_a_and_the_right_click_menu(view, qtbot):
    from PySide6.QtTest import QTest

    guided = _guided_at_fields(view)
    tree = guided.field_tree
    children = guided._field_children()
    assert len(children) == 6 and tree.selectionMode() == tree.SelectionMode.ExtendedSelection
    assert all(child.isSelected() for child in children)
    qtbot.waitUntil(lambda: tree.isVisible(), timeout=5000)
    rect = tree.visualItemRect
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, rect(children[0]).center())
    assert _ticked(guided) == [browser.tr("Field {field}", field=1) + "@A01"]
    assert guided.field_mode.checkedId() == 2
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.ShiftModifier, rect(children[3]).center())
    assert len(_ticked(guided)) == 4
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.ControlModifier, rect(children[1]).center())
    assert len(_ticked(guided)) == 3 and children[1].checkState(0) == Qt.Unchecked
    tree.setFocus()
    QTest.keyClick(tree, Qt.Key_A, Qt.ControlModifier)
    assert len(_ticked(guided)) == 6
    QTest.mouseClick(tree.viewport(), Qt.LeftButton, Qt.NoModifier, rect(children[4]).center())
    start, end = rect(children[0]).center(), rect(children[2]).center()
    _mouse(tree.viewport(), QEvent.MouseButtonPress, start)
    _mouse(tree.viewport(), QEvent.MouseMove, rect(children[1]).center(), button=Qt.NoButton)
    _mouse(tree.viewport(), QEvent.MouseMove, end, button=Qt.NoButton)
    _mouse(tree.viewport(), QEvent.MouseButtonRelease, end, buttons=Qt.NoButton)
    assert [child.checkState(0) == Qt.Checked for child in children] == [True, True, True, False, False, False]
    menu = guided._field_menu(rect(children[1]).center())
    assert [action.text() for action in menu.actions()] == [
        browser.tr("Select"), browser.tr("Deselect"), browser.tr("Select all fields"), browser.tr("Deselect all fields")]
    menu.actions()[1].trigger()
    assert not any(child.checkState(0) == Qt.Checked for child in children[:3])
    menu.close()
    menu = guided._field_menu(rect(children[5]).center())
    menu.actions()[0].trigger()
    menu.close()
    assert [child.checkState(0) == Qt.Checked for child in children] == [False] * 5 + [True]
    well_row = children[3].parent()
    menu = guided._field_menu(tree.visualItemRect(well_row).center())
    menu.actions()[0].trigger()
    menu.close()
    assert [child.checkState(0) == Qt.Checked for child in children] == [False] * 3 + [True] * 3
    children[0].setCheckState(0, Qt.Checked)
    assert children[0].isSelected()
    guided.next()
    assert sorted(field.field for field in guided._fields[("p1", "B02")]) == ["1", "2", "3"]
    assert [field.field for field in guided._fields[("p1", "A01")]] == ["1"]


def test_guided_goes_back_from_every_step_and_chooses_the_dpi_on_the_last(view, qtbot, tmp_path):
    guided = _guided_at_fields(view)
    assert guided.back_button.isVisibleTo(guided)
    guided.back()
    assert guided.steps.currentIndex() == 0 and not guided.back_button.isEnabled()
    guided.next()
    guided.set_fields("all")
    guided.next()
    assert guided.steps.currentIndex() == 2 and guided.back_button.isEnabled()
    guided.place(apply_rest=True)
    guided.next()
    assert guided.steps.currentIndex() == 3
    composer = guided.composer
    assert composer.settings_button.isVisibleTo(guided) and not composer.export_button.isVisibleTo(guided)
    assert guided.export_button.isVisibleTo(guided)
    assert composer.dpi.currentData() == picture_export.picture_dpi()
    composer.settings_button.click()
    assert composer.dpi.isVisible()
    composer.dpi.setCurrentIndex(composer.dpi.findData(0))
    full = tmp_path / "full.pdf"
    assert guided.export_pdf(str(full))
    qtbot.waitUntil(lambda: composer.last_export is not None, timeout=60000)
    assert {image.image.size for page in PdfReader(str(full)).pages for image in page.images} == {(60, 45)}
    composer.last_export = None
    widest = max(box["w"] for page in composer.canvas.pages for box in page if box["kind"] == "image")
    longest = max(1, int(round(widest / 72.0 * 20)))
    composer.dpi.addItem("20", 20)
    composer.dpi.setCurrentIndex(composer.dpi.count() - 1)
    low = tmp_path / "low.pdf"
    assert guided.export_pdf(str(low))
    qtbot.waitUntil(lambda: composer.last_export is not None, timeout=60000)
    sizes = {image.image.size for page in PdfReader(str(low)).pages for image in page.images}
    assert all(width <= longest + 1 and width < 60 for width, _height in sizes)
    guided.back()
    assert guided.steps.currentIndex() == 2 and guided._index == 1
    guided.back()
    assert guided.steps.currentIndex() == 2 and guided._index == 0
    guided.back()
    assert guided.steps.currentIndex() == 1
    guided.back()
    assert guided.steps.currentIndex() == 0


def test_the_dpi_cap_never_enlarges_and_keeps_the_aspect():
    pixels = np.zeros((200, 300, 3), np.uint8)
    box = {"w": 72.0, "h": 72.0}
    assert picture_export._canvas_capped_pixels(pixels, box, None) is pixels
    assert picture_export._canvas_capped_pixels(pixels, box, 600) is pixels
    small = picture_export._canvas_capped_pixels(pixels, box, 150)
    assert small.shape == (100, 150, 3)
    combo = picture_export._canvas_dpi_combo()
    assert combo.currentData() == picture_export.picture_dpi() and combo.findData(0) == combo.count() - 1


def test_the_canvas_toolbar_wraps_to_fit_the_export_section(view, qtbot):
    composer = view._open_custom_export()
    toolbar = composer.findChild(picture_export.QWidget, "CanvasComposerToolbar")
    assert toolbar is not None and toolbar.hasHeightForWidth()
    assert toolbar.heightForWidth(300) > toolbar.heightForWidth(2400)
    assert composer.minimumSizeHint().width() < 600
