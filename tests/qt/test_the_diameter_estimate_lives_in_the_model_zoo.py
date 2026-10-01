"""Item 533: the diameter estimate moves into the Model zoo popup.

    "in mask generation the, diamiter calculation should be a button in the
     model zoo button popup window not on the main screen. pressing the
     button should bring another pupup with all the information in the
     current container."

Asserted on the real Mask generation screen, built the way the window
builds it: the panel is gone from it, the Model zoo opened from it has a
"Measure diameters…" button, and that button's popup shows the estimates
with their evidence and notes and writes them into this screen's settings.
The estimator is faked, so nothing here reads an image.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt                                   # noqa: E402
from PySide6.QtWidgets import QLabel, QPushButton                # noqa: E402

from spacr.qt import prerun                                      # noqa: E402
from spacr.qt.widgets import model_zoo_picker as mzp             # noqa: E402


def _estimate(object_type, diameter, usable=True):
    from spacr.diameter import DiameterEstimate

    if usable:
        return DiameterEstimate(
            object_type=object_type, diameter=diameter, low=diameter - 4,
            high=diameter + 4, n_objects=137, n_fields=5,
            method="threshold_otsu", confidence="high",
            note=f"{object_type} objects were separated cleanly")
    return DiameterEstimate(
        object_type=object_type, diameter=float("nan"), low=float("nan"),
        high=float("nan"), n_objects=0, n_fields=0, method="none",
        confidence="low", note="no usable signal in this channel")


@pytest.fixture
def registered():
    from spacr.qt import chaining

    chaining.register()
    prerun.register()
    try:
        yield
    finally:
        prerun.unregister()


@pytest.fixture
def mask_screen(qtbot, registered, tmp_path):
    from spacr.qt.app import APP_FACTORIES, _call_screen_factory

    factory = APP_FACTORIES.get("mask")
    assert factory is not None
    screen = _call_screen_factory(factory, "mask", None)
    qtbot.addWidget(screen)
    model = screen._settings_model
    assert model.set_value_for_key("src", str(tmp_path))
    assert model.set_value_for_key("cell_channel", 0)
    assert model.set_value_for_key("nucleus_channel", 1)
    return screen


@pytest.fixture
def quiet_zoo(monkeypatch, tmp_path):
    """The picker without its network warm-ups."""
    monkeypatch.setattr(mzp.ModelZooPicker, "_warm_the_community_catalogue",
                        lambda self: None)
    monkeypatch.setattr(mzp.ModelZooPicker, "_warm_bioimageio",
                        lambda self: None)
    monkeypatch.setattr(mzp, "remembered_model_dir", lambda: str(tmp_path))


@pytest.fixture
def zoo(qtbot, quiet_zoo, mask_screen):
    picker = mzp.ModelZooPicker(kinds=("cellpose",), parent=mask_screen)
    yield picker
    picker.reject()


@pytest.fixture
def fake_estimator(monkeypatch):
    """Replace the estimator the popup calls, and record what it was asked."""
    import spacr.diameter as diameter

    calls = []

    def fake(src, channels, **kwargs):
        calls.append((src, dict(channels), kwargs))
        return {"cell": _estimate("cell", 31.6),
                "nucleus": _estimate("nucleus", 12.2),
                "pathogen": _estimate("pathogen", 0, usable=False)}

    monkeypatch.setattr(diameter, "estimate_diameters", fake)
    return calls


def _texts(widget):
    return "\n".join(label.text() for label in widget.findChildren(QLabel)
                     if label.text())


def test_the_mask_screen_no_longer_carries_the_panel(mask_screen):
    assert mask_screen.findChildren(prerun.DiameterPanel) == []
    assert "Diameter — measure it" not in _texts(mask_screen)


def test_the_zoo_opened_from_mask_generation_has_the_button(zoo):
    button = zoo.diameter_button
    assert isinstance(button, QPushButton)
    assert button.text() == "Measure diameters…"
    assert button.window() is zoo
    assert zoo._diameter_screen is zoo.parentWidget()


def test_the_zoo_finds_the_screen_from_a_control_inside_it(qtbot, quiet_zoo,
                                                          mask_screen):
    """The per-object model cell opens the zoo with itself as parent."""
    inner = mask_screen._settings_model._widgets["cell_diameter"]
    picker = mzp.ModelZooPicker(kinds=("cellpose",), parent=inner)
    try:
        assert picker._diameter_screen is mask_screen
        assert picker.diameter_button is not None
    finally:
        picker.reject()


def test_a_zoo_opened_anywhere_else_has_no_such_button(qtbot, quiet_zoo):
    picker = mzp.ModelZooPicker(kinds=("cellpose",))
    try:
        assert picker.diameter_button is None
        assert all(b.text() != "Measure diameters…"
                   for b in picker.findChildren(QPushButton))
    finally:
        picker.reject()


def test_the_button_opens_the_popup_that_measures_and_applies(
        qtbot, zoo, mask_screen, fake_estimator):
    qtbot.mouseClick(zoo.diameter_button, Qt.LeftButton)
    dialog = zoo._diameter_dialog
    assert isinstance(dialog, prerun.DiameterDialog)
    assert dialog.isVisible()
    assert dialog.parentWidget() is zoo
    panel = dialog.panel
    assert panel._screen is mask_screen
    assert "30/diameter" in _texts(dialog)

    with qtbot.waitSignal(panel.estimated, timeout=30000) as caught:
        qtbot.mouseClick(panel._btn_measure, Qt.LeftButton)
    assert sorted(caught.args[0]) == ["cell", "nucleus"]
    assert fake_estimator[0][1] == {"cell": 0, "nucleus": 1}

    text = _texts(dialog)
    assert "cell: 31.6 px" in text
    assert "nucleus: 12.2 px" in text
    assert "pathogen: no estimate" in text
    assert "measured on 137 object(s) across 5 field(s)" in text
    assert "cell objects were separated cleanly" in text
    assert "no usable signal in this channel" in text

    model = mask_screen._settings_model
    use_cell = [b for b in dialog.findChildren(QPushButton)
                if b.text() == "Use 32"]
    assert len(use_cell) == 1
    qtbot.mouseClick(use_cell[0], Qt.LeftButton)
    assert model.collect().get("cell_diameter") == 32
    assert model.collect().get("nucleus_diameter") != 12

    qtbot.mouseClick(panel._btn_use_all, Qt.LeftButton)
    assert model.collect().get("nucleus_diameter") == 12
    assert "Set cell_diameter, nucleus_diameter." in _texts(dialog)


def test_the_popup_is_reused_and_closes_with_the_zoo(qtbot, zoo):
    zoo._measure_diameters()
    first = zoo._diameter_dialog
    first.close_button.click()
    assert not first.isVisible()
    zoo._measure_diameters()
    assert zoo._diameter_dialog is first
    assert first.isVisible()
    zoo.reject()
    assert not first.isVisible()


def test_the_popup_wears_spacrs_dress(qtbot, mask_screen):
    from spacr.qt.widgets.glass import wants_glass

    dialog = prerun.diameter_dialog(mask_screen)
    qtbot.addWidget(dialog)
    assert wants_glass(dialog) is True
    assert dialog.close_button.objectName() == "DangerButton"
    assert dialog.close_button.autoDefault() is False
