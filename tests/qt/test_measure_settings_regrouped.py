"""Measure's settings are grouped the way the maintainer asked (item 595).

Cloud hangs under Input & Experiment, which also holds the preview and
diagnostics settings; the pixel corrections hang under Image Preprocessing,
the assays under Features, and the run's plumbing under Postprocessing.
"""

import pytest

pytest.importorskip("PySide6")


def _tree(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    panel = SettingsWidgets("measure")
    sections = panel.build_sections()
    for widget in panel._widgets.values():
        qtbot.addWidget(widget)
    return {section.title: section for section in sections}


def _children(section):
    return [child.title for child in section.children]


def _own_keys(section):
    return {widget.property("settingKey") for _label, widget in
            section.own_rows}


def test_the_top_level_reads_in_run_order(qtbot):
    assert list(_tree(qtbot)) == [
        "Input & Experiment", "Mask & Channel Mapping", "Image Preprocessing",
        "Features", "Object Filtering", "Crop Output",
        "3D Calibration (Beta)", "Postprocessing"]


def test_input_holds_preview_and_diagnostics_and_nests_cloud(qtbot):
    section = _tree(qtbot)["Input & Experiment"]
    assert {"src", "experiment", "plot", "test_mode", "test_nr"} <= (
        _own_keys(section))
    assert _children(section) == ["Cloud α"]


def test_image_preprocessing_nests_the_corrections(qtbot):
    assert _children(_tree(qtbot)["Image Preprocessing"]) == [
        "Bleach Correction α", "Spectral Unmixing α",
        "Image Deconvolution (PSF)", "Illumination Correction",
        "Intensity Calibration α", "Plate Barcode Linkage α"]


def test_features_keeps_its_rows_and_nests_the_assays(qtbot):
    section = _tree(qtbot)["Features"]
    assert "save_measurements" in _own_keys(section)
    assert _children(section) == [
        "Confluency α", "Cell Cycle α", "Wound Closure α",
        "Viability α", "CellProfiler α",
        "GPU Measurement α", "Time To Event α"]


def test_postprocessing_nests_runtime_and_profiling(qtbot):
    assert _children(_tree(qtbot)["Postprocessing"]) == [
        "Runtime & Reliability", "Profiling α",
        "Measurement Backend α"]


def test_other_modules_keep_their_own_groups(qtbot):
    """The nesting is Measure's; Mask keeps its own groups."""
    from spacr.qt.screens.settings_model import SettingsWidgets

    panel = SettingsWidgets("mask")
    titles = [section.title for section in panel.build_sections()]
    for widget in panel._widgets.values():
        qtbot.addWidget(widget)
    assert "Input & Metadata" in titles and "Features" not in titles
    assert "Postprocessing" not in titles
