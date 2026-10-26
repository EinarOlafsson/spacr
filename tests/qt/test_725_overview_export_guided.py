"""Round-5 Overview export, Guided: wells by condition across plates, fields, placement, captions.

Item 725, maintainer feedback 2026-10-25 round 5, point 8: "the user hits
export, then gets prompted with what wells to choose , then if they want to
pick specific fields, then they get asked where on the pdf to put the image,
and what information to print above below to the left or to the right. then
moove on the the next chosen well."
"""
import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import QPoint, Qt
from pypdf import PdfReader

import spacr.plot  # noqa: F401  -- imported here, on the main thread, before any export worker needs it
from spacr.qt.widgets import qc_field_browser as browser
from spacr.qt.widgets import picture_export


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02", "B01", "B02")
GENOTYPES = {"A01": "WT", "A02": "KO", "B01": "WT", "B02": "KO"}


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    rows, columns = np.mgrid[0:45, 0:60]
    for number, plate in enumerate(("p1", "p2")):
        for index, well in enumerate(WELLS):
            for field in (1, 2, 3):
                for channel in (1, 2):
                    pixels = (rows * (channel + 1) + columns * field + 50 * index + 400 * number).astype(np.uint16)
                    tifffile.imwrite(folder / f"{plate}-{well}-f{field}-c{channel}.tif", pixels)
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs, view._group_jobs)


def _retire(view, qtbot):
    for dialog in (getattr(view, "_guided_export", None), getattr(view, "_canvas_composer", None)):
        if dialog is not None:
            try:
                dialog.close()
            except RuntimeError:
                pass
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
    dialog.resize(1300, 950)
    dialog.show()
    dialog._scan()
    qtbot.waitUntil(lambda: len(dialog._fields) == 24 and dialog._normalization_ready
                    and not dialog._index_jobs.is_busy(), timeout=15000)
    for plate in ("p1", "p2"):
        for well, genotype in GENOTYPES.items():
            dialog._annotations[(plate, well)] = {"genotype": genotype}
    return dialog


def _texts(page):
    return {box["role"]: box["text"] for box in page if box["kind"] == "text"}


def _wells_on(page):
    return [(box["source"]["plate"], box["source"]["well"], box["source"]["field"])
            for box in page if box["kind"] == "image"]


def test_wells_are_chosen_by_condition_across_two_plates(view):
    guided = view._open_guided_export()
    labels = [guided.conditions.item(row).data(Qt.UserRole) for row in range(guided.conditions.count())]
    assert labels == ["genotype: WT", "genotype: KO"]
    assert guided.choose_wells("condition", ["genotype: WT"]) == [
        ("p1", "A01"), ("p1", "B01"), ("p2", "A01"), ("p2", "B01")]
    guided.plates.item(1).setCheckState(Qt.Unchecked)
    guided.choose_wells("condition", ["genotype: KO"])
    assert guided.chosen_wells() == [("p1", "A02"), ("p1", "B02")]
    assert guided.choose_wells("wells", [("p1", "B02"), ("p1", "A01")]) == [("p1", "A01"), ("p1", "B02")]
    assert guided.choose_wells("one", ("p1", "A02")) == [("p1", "A02")]
    assert guided.well_count.text() == browser.tr("{count} wells chosen", count=1)


def test_comparison_groups_are_conditions_and_selected_wells_preselect(view):
    view.group_builder.add_group({"genotype": "KO"})
    view._selected_wells = {("p2", "B02")}
    guided = view._open_guided_export()
    assert guided.mode_group.checkedId() == 1 and guided.chosen_wells() == [("p2", "B02")]
    group = next(guided.conditions.item(row).data(Qt.UserRole) for row in range(guided.conditions.count())
                 if guided.conditions.item(row).data(Qt.UserRole).startswith(browser.tr("Group")))
    assert guided.choose_wells("condition", [group]) == [
        ("p1", "A02"), ("p1", "B02"), ("p2", "A02"), ("p2", "B02")]


def test_guided_flow_end_to_end_with_picked_fields_captions_and_apply_to_remaining(view, qtbot, tmp_path):
    guided = view._open_guided_export()
    guided.choose_wells("condition", ["genotype: WT"])
    assert guided.next() and guided.steps.currentIndex() == 1
    guided.set_fields("pick", picks={("p1", "A01"): ["1", "3"], ("p1", "B01"): ["2"],
                                     ("p2", "A01"): ["1"], ("p2", "B01"): ["2", "3"]})
    assert guided.next() and guided.steps.currentIndex() == 2
    assert guided.picker.selection == (0, 0, 1, 1) and guided._page == 0
    guided.place(cell=(0, 0, 1, 2), sides={"plate": "above", "well": "above", "condition": "below",
                                           "field": "below", "text": "right"}, text="24 h")
    assert guided.next() and guided._index == 1
    assert guided.picker.selection == (1, 0, 1, 2) and guided.picker.occupied == {(0, 0): "A01", (0, 1): "A01"}
    assert guided.free_text.text() == ""
    guided.place(sides={"condition": "left", "field": "none"}, apply_rest=True)
    guided.scale_bars.setChecked(True)
    assert guided.next() and guided.steps.currentIndex() == 3
    layout = guided.composer.canvas
    assert len(layout.pages) == 2
    first, second = layout.pages
    assert _wells_on(first) == [("p1", "A01", "1"), ("p1", "A01", "3"), ("p1", "B01", "2"), ("p2", "A01", "1")]
    assert _wells_on(second) == [("p2", "B01", "2"), ("p2", "B01", "3")]
    captions = [box for box in first if box["kind"] == "text"]
    assert [(box["role"], box["text"]) for box in captions[:3]] == [
        ("above", browser.tr("Plate {plate}", plate="p1") + " · " + browser.tr("Well {well}", well="A01")),
        ("below", "genotype: WT · " + browser.tr("Fields {fields}", fields="1, 3")),
        ("right", "24 h")]
    assert _texts(second) == {"above": browser.tr("Plate {plate}", plate="p2") + " · "
                              + browser.tr("Well {well}", well="B01"), "left": "genotype: WT"}
    width, height = layout.page_points
    cell_height = (height - 72 - 20) / 3
    images = [box for box in first if box["kind"] == "image"]
    assert images[0]["y"] > 36 and images[0]["y"] + images[0]["h"] < 36 + cell_height
    assert images[2]["y"] > 36 + cell_height and images[3]["y"] > 36 + 2 * cell_height
    assert all(box["fit"] and box["scale_bar"] and box["pixel_size_um"] == .5 for box in images)
    path = tmp_path / "guided.pdf"
    guided.composer.export_pdf(str(path))
    qtbot.waitUntil(lambda: guided.composer.last_export is not None, timeout=60000)
    reader = PdfReader(str(path))
    assert len(reader.pages) == 2
    assert [len(page.images) for page in reader.pages] == [4, 2]
    assert {image.image.size for page in reader.pages for image in page.images} == {(60, 45)}
    text = reader.pages[0].extract_text()
    assert "Plate p1" in text and "24 h" in text and "µm" in text


def test_first_fields_one_well_back_and_the_display_settings_page(view):
    guided = view._open_guided_export()
    guided.choose_wells("one", ("p2", "B02"))
    guided.next()
    guided.set_fields("first", 2)
    guided.next()
    guided.back()
    assert guided.steps.currentIndex() == 1
    guided.next()
    guided.place(cell=(1, 1, 2, 1), page=1, sides={"plate": "none", "well": "none", "condition": "none",
                                                   "channels": "below"})
    guided.display_note.setChecked(True)
    guided.next()
    layout = guided.composer.canvas
    assert [len(page) for page in layout.pages][0] == 0
    assert _wells_on(layout.pages[1]) == [("p2", "B02", "1"), ("p2", "B02", "2")]
    assert _texts(layout.pages[1]) == {"below": browser.tr("Channels: {channels}",
                                                           channels=", ".join(view._canvas_channel_names()))}
    note = _texts(layout.pages[2])["note"]
    assert note.startswith(browser.tr("Display settings")) and "0.5" in note
    guided.back()
    assert guided.steps.currentIndex() == 2 and guided._placements == []


def test_nothing_chosen_stops_at_its_step(view):
    guided = view._open_guided_export()
    guided.choose_wells("wells", [])
    assert not guided.next() and guided.status.text() == browser.tr("Choose at least one well.")
    guided.choose_wells("one", ("p1", "A01"))
    guided.next()
    guided.set_fields("pick", picks={})
    assert not guided.next() and guided.status.text() == browser.tr("Choose at least one field.")


def test_measurements_print_the_mean_of_the_wells_chosen_fields(view):
    guided = view._open_guided_export()
    guided.choose_wells("one", ("p1", "A01"))
    guided.next()
    guided.set_fields("first", 2)
    guided.next()
    guided._measurements_loaded(pd.DataFrame({
        "plateID": ["p1", "p1", "p1", "p2"], "wellID": ["A01", "A01", "A01", "A01"],
        "fieldID": ["1", "2", "3", "1"], "mean_intensity": [10., 20., 90., 500.]}))
    assert guided.sides["measurements"].isEnabled()
    guided.place(sides={"measurements": "right", "plate": "none", "well": "none", "condition": "none"},
                 measurements=["mean_intensity"])
    guided.next()
    assert _texts(guided.composer.canvas.pages[0]) == {"right": "mean_intensity: 15"}


def test_guided_hands_its_layout_to_custom_mid_way_and_from_the_preview(view):
    guided = view._open_guided_export()
    guided.choose_wells("condition", ["genotype: KO"])
    guided.next()
    guided.next()
    guided.place(cell=(2, 0, 1, 1))
    custom = guided.open_custom()
    assert isinstance(custom, picture_export._CanvasComposer) and view._canvas_composer is custom
    assert _wells_on(custom.canvas.pages[0]) == [("p1", "A02", str(field)) for field in (1, 2, 3)]
    assert not guided.isVisible()
    custom.close()
    guided = view._open_guided_export()
    guided.choose_wells("one", ("p1", "B01"))
    guided.next()
    guided.next()
    guided.next()
    guided.composer.add_text(300, 700, 200, 36, text="fine-tuned")
    custom = guided.open_custom()
    assert custom.canvas.to_dict()["pages"][0][-1]["text"] == "fine-tuned"
    assert custom.canvas is not guided.composer.canvas


def test_canvas_images_render_through_the_display_pipeline(view, monkeypatch):
    import threading

    seen = []
    original = browser._overview_pixels
    caller = threading.get_ident()

    def spy(field, channels, limits, **options):
        # the snapshot runs synchronously on this thread; the browser's background
        # thumbnail runners call the same function from their own threads
        if threading.get_ident() == caller:
            seen.append((field.plate, field.well, channels, options.get("edge"), "image_overrides" in options))
        return original(field, channels, limits, **options)

    field = next(field for field in view._fields if (field.plate, field.well, field.field) == ("p2", "A02", "3"))
    key = browser._overview_image_key(field)
    view._image_overrides = {key: {"gamma": 2.0}}
    monkeypatch.setattr(browser, "_overview_pixels", spy)
    snapshot = view._canvas_snapshot()
    box = {"source": browser._overview_canvas_image(field)["source"], "label": "x"}
    full = snapshot(box, None)
    assert full.shape == (45, 60, 3) and full.dtype == np.uint8
    assert seen == [("p2", "A02", (1, 2), None, True)]
    with pytest.raises(ValueError):
        snapshot({"source": {"type": "overview_field", "key": ["", "p9", "Z99", "1", "", ""]}}, None)


def test_the_export_button_offers_guided_custom_and_contact_sheet(view, qtbot):
    from PySide6.QtWidgets import QPushButton

    button = next(button for button in view.findChildren(QPushButton) if button.text() == browser.tr("Export…"))
    button.click()
    menu = view._export_menu_shown
    assert [action.text() for action in menu.actions()] == [
        browser.tr("Guided export…"), browser.tr("Custom export…"), browser.tr("Contact sheet PDFs")]
    menu.actions()[1].trigger()
    composer = view._canvas_composer
    assert composer.isVisible() and composer.windowTitle() == browser.tr("Custom export")
    menu.close()
    menu = view._export_menu()
    menu.actions()[0].trigger()
    assert view._guided_export.isVisible()
    menu.close()
    assert [button.text() for button in view._module_action_buttons][0] == browser.tr("Export…")


def test_selected_fields_and_wells_go_on_the_canvas_by_button_and_by_drag(view, qtbot):
    composer = view._open_custom_export()
    assert view._canvas_drags._watched
    view._selected_wells = {("p1", "B02")}
    add = next(button for button in composer.findChildren(browser.QPushButton)
               if button.text() == browser.tr("Add selected"))
    add.click()
    assert [box["source"]["field"] for box in composer.canvas.pages[0]] == ["1", "2", "3"]
    drags = view._canvas_drags
    view._well(view.wells.item(*view._well_positions[(view.plates.currentData(), "A01")]))
    qtbot.waitUntil(lambda: view.fields.count() >= 1, timeout=15000)
    drags.watch()
    item = view.fields.item(0)
    item.setSelected(True)
    images = drags._images_at(view.fields.viewport(), view.fields.visualItemRect(item).center(), False)
    assert [image["source"]["well"] for image in images] == ["A01"]
    assert drags._images_at(view.view.viewport(), QPoint(1, 1), False) == []
    view.wells.clearSelection()
    cell = view.wells.item(*view._well_positions[(view.plates.currentData(), "B01")])
    cell.setSelected(True)
    rect = view.wells.visualItemRect(cell)
    images = drags._images_at(view.wells.viewport(), rect.center(), False)
    assert {(image["source"]["plate"], image["source"]["well"]) for image in images} == {("p1", "B01")}
    dropped = picture_export._canvas_dropped_images(picture_export._canvas_mime(images))
    assert composer.add_images(dropped)[0]["source"]["well"] == "B01"
    composer.close()
    assert drags._watched == []
