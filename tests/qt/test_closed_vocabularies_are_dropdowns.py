"""Settings with a small closed set of values render as dropdowns.

Each key below was a free-text box although the engine accepts only a fixed
list of values. The dropdown offers exactly that list, opens on the same
default, and a saved settings file carrying any of the old strings loads
the same value back.
"""

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QComboBox

from spacr.qt.screens.settings_model import SettingsWidgets
from spacr.settings_spec import convert_settings_dict_for_gui


CONVERTED = {
    ("spatial_masks", "spatial_platform"): (["auto", "xenium", "visium_hd"], "auto"),
    ("spatial_map", "spatial_platform"): (["auto", "xenium", "visium_hd"], "auto"),
    ("spatial_masks", "spatial_window_unit"): (["um", "px"], "um"),
    ("spatial_masks", "spatial_segmenter"): (["cellpose", "threshold"], "cellpose"),
    ("spatial_masks", "spatial_stain"): (["auto", "dapi", "hematoxylin", "rgb"], "auto"),
    ("spatial_map", "spatial_bin_um"): ([2, 8, 16], 2),
    ("spatial_map", "spatial_objects"): (["cell,nucleus", "cell", "nucleus"], "cell,nucleus"),
    ("mask", "z_segmentation_mode"): (["project", "stitch", "volumetric"], "project"),
    ("mask", "z_projection"): (["max", "mean", "sum", "best_focus", None], "max"),
    ("mask", "t_track_backend"): (["iou", "centroid", "trackpy", "btrack", "trackastra", "ultrack"], "iou"),
    ("mask", "pipeline_style"): (["v1", "v2"], "v1"),
    ("illumination", "illumination_estimator"): (["polynomial", "smooth"], "polynomial"),
    ("illumination", "illumination_on_missing"): (["error", "skip"], "error"),
    ("invasion", "outside_threshold_method"): (["otsu", "triangle", "li", "yen", "mean"], "otsu"),
    ("invasion", "extracellular_class"): (["attached", "exclude", "classify"], "attached"),
    ("classify_merged", "stream_method"): (["column", "array"], "column"),
    ("classify_merged", "crop_shape"): (["bounding_box", "object"], "bounding_box"),
}


def _build(app, key, value=None):
    """The panel for ``app`` and the control it builds for ``key``."""
    model = SettingsWidgets(app)
    assert key in model._defaults, f"{app} does not hold {key}"
    if value is not None:
        model._defaults[key] = value
    kind, options, default = convert_settings_dict_for_gui(
        {key: model._defaults[key]})[key]
    widget = model._widget_for(kind, options, default, key)
    model._widgets[key] = widget
    return model, widget


@pytest.mark.parametrize("app, key", sorted(CONVERTED))
def test_the_key_is_a_dropdown_of_exactly_its_choices(qapp, app, key):
    choices, default = CONVERTED[(app, key)]
    model, widget = _build(app, key)
    assert isinstance(widget, QComboBox)
    assert [widget.itemData(i) for i in range(widget.count())] == choices
    assert widget.currentData() == default
    assert model.collect()[key] == default
    widget.deleteLater()


@pytest.mark.parametrize("app, key", sorted(CONVERTED))
def test_every_old_string_loads_as_the_same_value(qapp, app, key):
    choices, _default = CONVERTED[(app, key)]
    for value in choices:
        if value is None:
            continue
        model, widget = _build(app, key, value)
        assert widget.count() == len(choices)
        assert model.collect()[key] == value
        widget.deleteLater()


def test_a_value_outside_the_list_is_kept_rather_than_replaced(qapp):
    """Standard Visium is accepted by the engine though not offered."""
    model, widget = _build("spatial_map", "spatial_platform", "visium")
    assert widget.currentData() == "visium"
    assert model.collect()["spatial_platform"] == "visium"
    widget.deleteLater()
