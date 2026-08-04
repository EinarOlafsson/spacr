"""Qt integration for the External Masks data module."""
from __future__ import annotations

import numpy as np
import tifffile

from spacr.qt import maturity
from spacr.qt.app import APPS, SECTION_DATA, STAGE_BETA, app_stage
from spacr.qt.bridge import resolve_pipeline_entry
from spacr.qt.dnd_handlers import ExternalMasksDropHandler, get_handler
from spacr.qt.screens.settings_model import SettingsWidgets
from spacr.qt.widgets.external_mask_inputs import ExternalMaskInputWidget


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.asarray(array), photometric="minisblack")


def test_module_is_a_beta_data_app_with_a_real_pipeline_entry():
    """Beta, not alpha, and the reason is recorded next to the decision.

    `spacr.qt.maturity` assessed all twenty-six alpha modules against the
    evidence in the repository. This one keeps a caveat rather than going
    straight to stable -- a real `spacr-run external_masks` pipeline with a
    defaults entry and a tutorial lesson, but 49 assertions across two small
    files, the thinnest test evidence of any CLI-backed module in the batch.

    The promotions are applied by `register_self_registering_modules`, which
    runs at every launch. A bare test process may not have called it, so
    `maturity.apply` is called directly rather than depending on whether
    some earlier test happened to -- and `apply` alone, because it touches
    only APP_STAGE and cannot re-register anything.
    """
    maturity.apply()

    record = next(app for app in APPS if app[0] == "external_masks")
    assert record[3] == SECTION_DATA
    assert app_stage("external_masks") == STAGE_BETA
    assert maturity.reason_for("external_masks")
    assert resolve_pipeline_entry("external_masks").__name__
    assert isinstance(get_handler("external_masks"), ExternalMasksDropHandler)


def test_settings_reuse_measure_and_add_editable_input_mapping(qtbot):
    model = SettingsWidgets("external_masks")
    sections = dict(model.build_sections())
    qtbot.addWidget(model._widgets["inputs"])

    assert "Input mapping" in sections
    assert "Measurements" in sections
    assert "Filter settings" in sections
    assert isinstance(
        model._widgets["inputs"], ExternalMaskInputWidget)
    settings = model.collect()
    assert settings["save_measurements"] is True
    assert settings["save_png"] is True
    assert settings["inputs"] == []


def test_input_widget_detects_and_allows_role_corrections(tmp_path, qtbot):
    folder = tmp_path / "masks"
    label = np.zeros((16, 16), dtype=np.uint16)
    label[2:9, 2:9] = 1
    _write(folder / "field_cell_mask.tif", label)
    widget = ExternalMaskInputWidget()
    qtbot.addWidget(widget)

    assert widget.add_paths([folder]) == 1
    assert widget.group_count() == 1
    assert widget.groups()[0].role == "mask"
    assert widget.groups()[0].object_type == "cell"
    assert widget.set_group_object_type(0, "pathogen")
    assert widget.get_value()[0]["object_type"] == "pathogen"
    assert widget.set_group_role(0, "ignore")
    assert widget.get_value() == []
