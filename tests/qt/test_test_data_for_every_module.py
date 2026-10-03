"""Alpha "Load test data…" buttons on the screens that had none.

Each of these screens reads a measurements database (or, for Data Manager, a
plate folder) and now offers the shared example plate through
:mod:`spacr.qt.widgets.measurements_example`. The buttons are registered in
``spacr.settings.ALPHA_FEATURES``: hidden while Preferences -> "Show alpha
features" is off, shown when it is on, and a load made while hidden still
reaches the screen.
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QPushButton

from spacr.qt.widgets import measurements_example as mx

from .test_modules_that_read_a_measurements_db_offer_test_data import (
    _build as _build_from_app, _write_plate)

#: Screen key -> the alpha button's object name.
BUTTONS = {
    "trellis": "TrellisTestDataButton",
    "feature_explorer": "FeatureExplorerTestDataButton",
    "outliers": "OutliersTestDataButton",
    "control_chart": "ControlChartTestDataButton",
    "data_manager": "DataManagerTestDataButton",
    "embeddings": "EmbeddingsTestDataButton",
    "power": "PowerTestDataButton",
    "pipeline_graph": "PipelineGraphTestDataButton",
    "project_browser": "ProjectBrowserTestDataButton",
}

#: Screen key -> the alpha button that opens Import's test fields.
IMPORT_BUTTONS = {
    "convert": "ConvertTestDataButton",
    "layer_viewer": "LayerViewerTestDataButton",
    "external_masks": "ExternalMasksTestDataButton",
}


#: Screens whose catalog entry is an alpha stage, built from their own factory.
FACTORY_BUILT = ("trellis", "feature_explorer", "outliers", "control_chart",
                 "embeddings")


def _build(qtbot, key):
    """Build ``key``'s screen the way the app does, or from its factory."""
    if key not in FACTORY_BUILT:
        return _build_from_app(qtbot, key)
    import importlib

    module = importlib.import_module(f"spacr.qt.screens.{key}")
    screen = getattr(module, f"make_{key}_screen")(key)
    qtbot.addWidget(screen)
    return screen


def _reached(key, screen, folder, database) -> bool:
    """Whether the example reached the screen's own source field."""
    if key in ("trellis", "feature_explorer", "outliers", "control_chart"):
        return screen._path == str(database)
    if key == "data_manager":
        return screen._root == str(folder)
    if key == "embeddings":
        return screen._path.text() == str(database)
    if key == "pipeline_graph":
        return screen._project_edit.text() == str(folder)
    if key == "project_browser":
        return str(folder.parent) in screen._roots
    if key == "power":
        return (screen._pilot_path.text() == str(database)
                and screen._pilot_table.text() == "cell")
    raise AssertionError(key)


def test_every_button_is_registered_as_alpha():
    """The registry names exactly the buttons this item added."""
    from spacr.settings import ALPHA_FEATURES

    assert set(ALPHA_FEATURES[633]["widgets"]) == (
        set(BUTTONS.values()) | set(IMPORT_BUTTONS.values()))


@pytest.mark.parametrize("key", sorted(BUTTONS))
def test_hidden_without_alpha_shown_with_it_and_loads_while_hidden(
        qtbot, tmp_path, monkeypatch, key):
    """Off hides it, on shows it, and a load while hidden still lands."""
    from spacr.qt import preferences

    folder = tmp_path / "example_data" / "plate1"
    monkeypatch.setattr(mx, "example_measurements_folder", lambda: folder)
    database = _write_plate(folder)

    screen = _build(qtbot, key)
    found = screen.findChildren(QPushButton, BUTTONS[key])
    assert len(found) == 1
    button = found[0]
    assert "test data" in button.text().lower()

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    preferences._apply_alpha_widgets(screen)
    assert button.isHidden()

    def refuse(*_args):
        """Fail if the cached plate is downloaded again."""
        raise AssertionError("downloaded a cached plate")

    handed = mx.load_test_data(screen, ask=refuse)
    assert handed["db"] == str(database)
    qtbot.waitUntil(lambda: _reached(key, screen, folder, database),
                    timeout=20_000)

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    preferences._apply_alpha_widgets(screen)
    assert not button.isHidden()


def _write_import_variant(plate, key):
    """Write one Import variant: one field, three channels and two masks."""
    import csv

    import numpy as np
    import tifffile

    root = plate / "import_example"
    variant = root / "variants" / key
    files = [variant / "plate1" / "E01" / f"fov09_ch{c}.tif" for c in (1, 2, 3)]
    files += [variant / "masks" / role / "E01" / "fov09_ch1.tif"
              for role in ("cell", "nucleus")]
    labels = np.zeros((32, 32), dtype=np.uint16)
    labels[4:12, 4:12] = 1
    labels[18:28, 16:30] = 2
    for index, path in enumerate(files):
        path.parent.mkdir(parents=True, exist_ok=True)
        image = (labels if "masks" in path.parts else
                 (np.arange(1024, dtype=np.uint16).reshape(32, 32) + index))
        tifffile.imwrite(path, image)
    with (root / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "path"])
        for path in files:
            writer.writerow([key, path.relative_to(root).as_posix()])
    return variant / "plate1"


def _owner(button):
    """The widget the shared Import button helper stored its state on."""
    holder = button
    while holder is not None and not hasattr(holder, "_import_test_data"):
        holder = holder.parent()
    return holder


def _import_reached(key, screen, images) -> bool:
    """Whether Import's test field reached the screen."""
    if key == "convert":
        return screen.source_path() == str(images)
    if key == "layer_viewer":
        return len(screen.stack) == 5
    if key == "external_masks":
        model = screen._settings_model
        widget = model._widgets["inputs"]
        return (len(widget._groups) >= 2
                and model._read_widget(model._widgets["dst"])
                == str(images) + "_spacr")
    raise AssertionError(key)


@pytest.mark.parametrize("key", sorted(IMPORT_BUTTONS))
def test_import_fields_button_is_alpha_and_loads_while_hidden(
        qtbot, tmp_path, monkeypatch, key):
    """Off hides it, on shows it, and a load while hidden still lands."""
    from spacr.qt import import_demo, preferences

    variant = "nikon_nd2" if key == "convert" else "auto"
    images = _write_import_variant(tmp_path, variant)

    screen = _build_from_app(qtbot, key)
    found = screen.findChildren(QPushButton, IMPORT_BUTTONS[key])
    assert len(found) == 1
    button = found[0]
    assert "test data" in button.text().lower()

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    preferences._apply_alpha_widgets(screen)
    assert button.isHidden()

    def refuse(*_args):
        """Fail if the cached variant is downloaded again."""
        raise AssertionError("downloaded a cached variant")

    assert import_demo._load_import_variant(
        _owner(button), ask=refuse, plate=tmp_path) is True
    qtbot.waitUntil(lambda: _import_reached(key, screen, images),
                    timeout=20_000)

    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    preferences._apply_alpha_widgets(screen)
    assert not button.isHidden()
