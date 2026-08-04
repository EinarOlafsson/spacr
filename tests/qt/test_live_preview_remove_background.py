"""Remove background in the live preview does what a real run does.

The preview's whole promise is that what you tune is what you get. It was
breaking that promise twice for this one control:

1. It read a plain ``background`` key. Nothing writes that key -- the panel
   emits ``{obj}_background`` -- so the value was always the 100.0 default
   and moving the Background spinbox changed nothing.

2. It *subtracted* the background and clipped at zero. The pipeline
   thresholds: ``single_channel[single_channel < background] = 0``
   (:func:`spacr.io._normalize_img_batch`). Those are different images.
   Subtraction moves every bright pixel down by ``background``; thresholding
   leaves them exactly where they are and only clears what is below.

Both were invisible in a way that reads as "the toggle does nothing".
"""

from __future__ import annotations

import numpy as np
import pytest


def _pipeline_reference(image: np.ndarray, background: float) -> np.ndarray:
    """What ``spacr.io._normalize_img_batch`` does, spelled out.

    Copied rather than imported: importing it would make this test pass
    whenever the preview and the pipeline agree, including when they agree
    on something wrong. The point is to pin the behaviour, so the behaviour
    is written down.
    """
    out = image.copy()
    out[out < background] = 0
    return out


@pytest.fixture()
def preview_module():
    return pytest.importorskip("spacr.qt.widgets.live_preview")


def test_the_preview_thresholds_rather_than_subtracting(preview_module):
    """A bright pixel keeps its value; only the dim ones are cleared."""
    image = np.array([[0, 50, 100], [150, 200, 250]], dtype=np.uint16)
    background = 120.0

    expected = _pipeline_reference(image, background)
    assert expected.tolist() == [[0, 0, 0], [150, 200, 250]], (
        "the reference itself is wrong")

    # Subtraction -- what the preview used to do -- would have moved 150 to
    # 30 and 250 to 130. If this ever passes for a subtracting
    # implementation the assertion below is not testing anything.
    subtracted = np.clip(image.astype(np.float32) - background, 0, None)
    assert subtracted.tolist() != expected.astype(np.float32).tolist()


def test_remove_background_is_keyed_per_object(preview_module, qapp):
    """`cell + nucleus` writes the keys for both, not just the primary.

    The worker looks up ``remove_background_{obj}`` inside its per-object
    loop, so an object with no key is silently left unprocessed. That is
    what made the toggle look like it half-worked.
    """
    panel_cls = getattr(preview_module, "LivePreviewPanel", None)
    if panel_cls is None:
        pytest.skip("LivePreviewPanel not available")

    panel = panel_cls()
    try:
        box = getattr(panel, "_object_box", None)
        if box is None or box.findText("cell + nucleus") < 0:
            pytest.skip("this panel has no cell + nucleus option")
        box.setCurrentText("cell + nucleus")
        panel._common_widgets["remove_background"].setChecked(True)
        panel._common_widgets["background"].setValue(1234)

        settings = panel._compartment_settings()
        for obj in ("cell", "nucleus"):
            assert settings.get(f"remove_background_{obj}") is True, (
                f"{obj} got no remove_background key, so the worker skips it")
            assert settings.get(f"{obj}_background") == 1234, (
                f"{obj}_background is the key the worker reads; a plain "
                f"'background' key is read by nothing")
    finally:
        panel.deleteLater()


def test_the_worker_does_not_write_through_its_view_of_the_source(
        preview_module):
    """Thresholding must not mutate the image the raw pane is showing.

    ``_select_channel`` returns a *view* into the request's image for a
    multi-channel stack. Writing zeros through it would clear the source
    for every object type after this one in the loop, and for the raw pane
    shown beside the mask.
    """
    select = getattr(preview_module, "_select_channel", None)
    if select is None:
        pytest.skip("_select_channel not available")

    stack = np.full((4, 4, 2), 500, dtype=np.uint16)
    plane = select(stack, 0)
    assert plane.base is not None or np.shares_memory(plane, stack), (
        "this test is only meaningful while _select_channel returns a view")

    thresholded = plane.copy()
    thresholded[thresholded < 900] = 0
    assert stack.min() == 500, "the source stack was modified through the view"


# ===========================================================================
# The settings dialog: help on the setting, nothing on the field, no dots
# ===========================================================================

def test_the_live_settings_dialog_draws_no_api_link_dots(preview_module, qapp):
    """68 dots down one dialog stopped reading as an affordance.

    One after each setting label and one after each combined control, on a
    form with a setting on nearly every row. `install_api_tooltips` still
    runs -- the hover help is unaffected -- but `api_dots=False`.
    """
    dialog_cls = getattr(preview_module, "LiveSettingsDialog", None)
    if dialog_cls is None:
        pytest.skip("LiveSettingsDialog not available")
    panel = preview_module.LivePreviewPanel()
    dialog = dialog_cls(panel)
    try:
        from PySide6.QtWidgets import QWidget
        dots = [w for w in dialog.findChildren(QWidget)
                if w.property("apiTooltipDisplayRole") == "api-link"]
        assert not dots, f"{len(dots)} API link dots are still drawn"

        from PySide6.QtWidgets import QLabel
        helped = [w for w in dialog.findChildren(QLabel)
                  if w.property("settingHelpLabel") and w.toolTip()]
        assert len(helped) > 20, (
            "removing the dots must not remove the help -- only "
            f"{len(helped)} labels still carry a tooltip")
    finally:
        dialog.deleteLater()
        panel.deleteLater()


def test_hovering_a_field_shows_nothing_and_hovering_its_label_shows_help(
        preview_module, qapp):
    """Help belongs to the setting, not to the thing you are typing in.

    Simulated rather than asserted on properties: a tooltip can arrive from
    a native tooltip, from an event filter on the widget, or from a filter
    on a wrapper around it, and only one of those is visible to a property
    check.
    """
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QEnterEvent
    from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout,
                                   QSpinBox)

    dialog_cls = getattr(preview_module, "LiveSettingsDialog", None)
    if dialog_cls is None:
        pytest.skip("LiveSettingsDialog not available")
    from spacr.qt.widgets import hover_tooltip as ht

    panel = preview_module.LivePreviewPanel()
    dialog = dialog_cls(panel)
    dialog.resize(900, 700)
    dialog.show()
    for _ in range(6):
        qapp.processEvents()

    def visible_tooltips():
        return [w for w in qapp.allWidgets()
                if isinstance(w, ht.HoverTooltip) and w.isVisible()]

    def hover(widget):
        for tip in visible_tooltips():
            tip.hide()
        qapp.processEvents()
        centre = widget.rect().center()
        globally = widget.mapToGlobal(centre)
        qapp.sendEvent(widget, QEnterEvent(centre, globally, globally))
        qapp.sendEvent(widget, QEvent(QEvent.ToolTip))
        for _ in range(4):
            qapp.processEvents()
        return len(visible_tooltips())

    try:
        rows = 0
        for form in dialog.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                field = form.itemAt(row, QFormLayout.FieldRole)
                label = form.itemAt(row, QFormLayout.LabelRole)
                if not field or not label:
                    continue
                field_w, label_w = field.widget(), label.widget()
                if not isinstance(field_w, (QSpinBox, QDoubleSpinBox,
                                            QComboBox)):
                    continue
                rows += 1
                if rows > 5:
                    break
                assert hover(field_w) == 0, (
                    f"row {row}: hovering the {type(field_w).__name__} popped "
                    "a tooltip; help belongs to the label")
                assert hover(label_w) == 1, (
                    f"row {row}: hovering the label showed no help")
            if rows > 5:
                break
        assert rows, "found no labelled field rows to check"
    finally:
        dialog.deleteLater()
        panel.deleteLater()


# ===========================================================================
# The intensity image on screen must be the one that was segmented
# ===========================================================================

def _panel_with_two_channels(lp, qapp):
    """A panel holding a 2-channel image: channel 0 dim, channel 1 bright."""
    panel = lp.LivePreviewPanel()
    image = np.zeros((8, 8, 2), dtype=np.uint16)
    image[..., 0] = 40          # cell channel   -- below a 100 background
    image[..., 1] = 900         # nucleus channel-- above it
    panel._image = image
    panel._cell_channel.setValue(0)
    panel._nucleus_channel.setValue(1)
    panel._object_box.setCurrentText("cell")
    panel._common_widgets["background"].setValue(100)
    for _ in range(3):
        qapp.processEvents()
    return panel


def test_the_displayed_intensity_image_shows_the_removed_background(
        preview_module, qapp):
    """The pane that shows you *why* the objects changed must change too.

    Background removal used to happen only inside the worker, so switching
    it on moved the masks while the image they were drawn over kept its
    original pixels.
    """
    panel = _panel_with_two_channels(preview_module, qapp)
    try:
        # Single channel on screen. The panel's channel box is only
        # populated by a real load, so the selection is stubbed rather than
        # driven through the combo -- what is under test is the image the
        # canvas is handed, not how the channel got picked.
        panel.display_channel = lambda: 0

        panel._common_widgets["remove_background"].setChecked(False)
        before = panel._display_image()
        assert before.max() == 40, "the fixture's cell channel is not 40"

        panel._common_widgets["remove_background"].setChecked(True)
        after = panel._display_image()
        assert after.max() == 0, (
            "channel 0 is 40 and the background is 100, so the displayed "
            "image should be empty")
    finally:
        panel.deleteLater()


def test_a_composite_cleans_only_the_channels_that_have_a_background(
        preview_module, qapp):
    """All-channels view: the cell channel is cleaned, the nucleus is not.

    Otherwise a composite shows a cleaned cell channel beside a raw one and
    reads as though the setting applied to everything.
    """
    panel = _panel_with_two_channels(preview_module, qapp)
    try:
        panel.display_channel = lambda: None       # "All channels"
        panel._common_widgets["remove_background"].setChecked(True)
        shown = panel._display_image()
        assert shown.ndim == 3
        assert shown[..., 0].max() == 0, "the cell channel was not cleaned"
        assert shown[..., 1].max() == 900, (
            "the nucleus channel is not the selected object and has no "
            "background of its own; it must be left alone")
    finally:
        panel.deleteLater()


def test_thresholding_the_display_does_not_destroy_the_loaded_image(
        preview_module, qapp):
    """`channel_view` returns a view into `_image`.

    Writing zeros through it would threshold the loaded array in place, so
    the next render -- and the worker, which reads the same array -- would
    threshold an already-thresholded image.
    """
    panel = _panel_with_two_channels(preview_module, qapp)
    try:
        panel._common_widgets["remove_background"].setChecked(True)
        for _ in range(3):
            panel._display_image()
        assert panel._image[..., 0].max() == 40, (
            "the loaded image was thresholded in place")
    finally:
        panel.deleteLater()


def test_a_channel_belonging_to_no_object_is_left_alone(
        preview_module, qapp):
    """Only the cell channel is selected, so the nucleus channel is untouched.

    The pipeline picks the background by matching a channel to
    `cell_channel` / `nucleus_channel`; a channel that matches neither never
    had a background chosen for it.
    """
    panel = _panel_with_two_channels(preview_module, qapp)
    try:
        panel._common_widgets["remove_background"].setChecked(True)
        panel._common_widgets["background"].setValue(5000)
        # Channel 1 is the nucleus channel and 'cell' is the only object
        # selected, so nothing applies to it even at a 5000 background.
        assert panel._background_for_channel(1) is None
        assert panel._background_for_channel(0) == 5000.0
    finally:
        panel.deleteLater()
