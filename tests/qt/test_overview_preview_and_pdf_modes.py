"""Each preview-resolution, PDF-quality and PDF-DPI value, driven through the Overview.

The settings panel offers these values; each is set on the Overview tab's
setting row, reaches the browser's control, and is checked for its effect: a
preview resolution of '256' or '512' loads a stacked image with exactly that
longest edge, a PDF quality of '3200' bounds each exported field's edge, and a
PDF DPI of '150' bounds the pixel density of the bounded-quality images
embedded on the page; full source resolution keeps every pixel.
"""
import json
import re

import numpy as np
import pytest
import tifffile

from spacr.qt.widgets import qc_field_browser as browser


PATTERN = r"(?P<plateID>p\d+)-(?P<wellID>[A-Z]\d+)-f(?P<fieldID>\d+)-c(?P<chanID>\d+)\.tif"
WELLS = ("A01", "A02")
WIDTH, HEIGHT = 4096, 3072


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    row = (np.arange(WIDTH, dtype=np.uint32) % 4000).astype(np.uint16)
    for index, well in enumerate(WELLS):
        tifffile.imwrite(folder / f"p1-{well}-f1-c1.tif", np.tile(row + 50 * index, (HEIGHT, 1)))
    return folder


def _runners(view):
    return (view._index_jobs, view._image_jobs, view._thumbnail_jobs, view._field_thumbnail_jobs,
            view._montage_jobs, view._export_jobs, view._metadata_jobs, view._measurement_jobs,
            view._native_jobs, view._image_stack_jobs)


def _close(mask, qtbot):
    tab = getattr(mask, "_overview_tab", None)
    page = getattr(tab, "_overview_page", None)
    if page is not None and page.browser is not None:
        view = page.browser
        view._cancel()
        qtbot.waitUntil(lambda: all(runner.active_jobs() == 0 for runner in _runners(view)), timeout=30000)
        for runner in _runners(view):
            runner.shutdown(timeout_ms=3000)
        page.retire()
    from spacr.qt.bridge import prune_parked_threads
    qtbot.waitUntil(lambda: prune_parked_threads() == 0, timeout=15000)
    if tab is not None:
        tab.close()
    mask.close()


@pytest.fixture
def mask(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen, before_close_func=lambda widget: _close(widget, qtbot))
    screen.resize(1400, 1000)
    screen.show()
    return screen


def _open(mask, qtbot, source):
    mask._settings_model.set_value_for_key("src", str(source))
    field = mask._settings_model._widgets.get("src")
    next(action for action in field.actions()
         if action.objectName() == "GenerateMasksFieldOverview").trigger()
    tab = mask._overview_tab
    wizard = tab._overview_page.wizard
    wizard.preset.setCurrentIndex(wizard.preset.count() - 1)
    wizard.regex.setText(PATTERN)
    wizard._scan()
    qtbot.waitUntil(lambda: wizard.next.isEnabled(), timeout=15000)
    wizard._next()
    page = tab._overview_page
    qtbot.waitUntil(lambda: page.browser is not None and len(page.browser._fields) == len(WELLS)
                    and page.browser._normalization_ready, timeout=30000)
    return tab, page.browser


def _export_one_field(tab, view, qtbot, source):
    tab._settings_model.set_value_for_key("overview_export_scope", "selection")
    field = view._fields[0]
    view._selected_fields = {field.identity: field}
    view._export()
    qtbot.waitUntil(lambda: not view._export_jobs.is_busy()
                    and view.status.text().startswith("Wrote"), timeout=60000)
    folder = max((source / "results").glob("overview_*"))
    with open(folder / "overview_selection.json", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return sorted(folder.glob("overview_*.pdf")), metadata


def _embedded_images(pdf):
    """Each image drawn on each page: (pixel width, pixel height, printed width in inches)."""
    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    reader = PdfReader(str(pdf))
    found = []
    for page in reader.pages:
        resources = page["/Resources"]
        objects = resources.get("/XObject", {})
        matrix = None
        for operands, operator in ContentStream(page.get_contents(), reader).operations:
            if operator == b"cm":
                matrix = [float(value) for value in operands]
            elif operator == b"Do":
                image = objects[operands[0]].get_object()
                if image.get("/Subtype") == "/Image" and matrix is not None:
                    found.append((int(image["/Width"]), int(image["/Height"]),
                                  abs(matrix[0]) / 72.))
    return found


def test_full_source_quality_keeps_every_pixel_whatever_the_dpi(mask, qtbot, source):
    tab, view = _open(mask, qtbot, source)
    tab._settings_model.set_value_for_key("overview_pdf_quality", "full")
    tab._settings_model.set_value_for_key("overview_pdf_dpi", 150)
    pdfs, metadata = _export_one_field(tab, view, qtbot, source)
    assert metadata["dpi"] == 150 and metadata["quality_edge"] is None
    assert [image[:2] for pdf in pdfs for image in _embedded_images(pdf)] == [(WIDTH, HEIGHT)]


@pytest.mark.parametrize("value", ["256", "512"])
def test_a_preview_resolution_loads_images_with_that_longest_edge(mask, qtbot, source, value):
    tab, view = _open(mask, qtbot, source)
    tab._settings_model.set_value_for_key("overview_preview_resolution", int(value))
    assert view.preview_resolution.currentData() == int(value)
    view._open_image_views([view._fields[0]])
    panel = view._image_views[-1]
    qtbot.waitUntil(lambda: panel.view._item is not None, timeout=30000)
    pixmap = panel.view._item.pixmap()
    assert max(pixmap.width(), pixmap.height()) == int(value)
    assert pixmap.width() * HEIGHT == pixmap.height() * WIDTH


def test_pdf_quality_3200_bounds_each_exported_field_edge(mask, qtbot, source):
    tab, view = _open(mask, qtbot, source)
    tab._settings_model.set_value_for_key("overview_pdf_quality", "3200")
    tab._settings_model.set_value_for_key("overview_pdf_dpi", 600)
    assert view.pdf_quality.currentData() == 3200 and view.pdf_dpi.currentData() == 600
    pdfs, metadata = _export_one_field(tab, view, qtbot, source)
    assert metadata["quality_edge"] == 3200 and metadata["written_field_count"] == 1
    images = [image for pdf in pdfs for image in _embedded_images(pdf)]
    assert len(images) == 1
    width, height, _inches = images[0]
    assert max(width, height) <= 3200
    assert (width, height) == (WIDTH // 2, HEIGHT // 2)


def test_pdf_dpi_150_bounds_the_embedded_pixel_density(mask, qtbot, source):
    tab, view = _open(mask, qtbot, source)
    tab._settings_model.set_value_for_key("overview_pdf_quality", "3200")
    tab._settings_model.set_value_for_key("overview_pdf_dpi", 150)
    assert view.pdf_quality.currentData() == 3200 and view.pdf_dpi.currentData() == 150
    pdfs, metadata = _export_one_field(tab, view, qtbot, source)
    assert metadata["dpi"] == 150 and metadata["quality_edge"] == 3200
    images = [image for pdf in pdfs for image in _embedded_images(pdf)]
    assert len(images) == 1
    width, height, inches = images[0]
    density = width / inches
    assert 75 < density <= 150 + 1e-6, density
    assert width < WIDTH // 2 and width * HEIGHT == pytest.approx(height * WIDTH, rel=0.01)
    assert re.fullmatch(r"overview_\d{4}\.pdf", pdfs[0].name)
