"""Mouse influence is opt-in and survives only an accepted Preferences save."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QComboBox, QDialogButtonBox, QSlider

from spacr.qt import preferences as prefs
from spacr.qt.widgets.ambient import AmbientWidget


@pytest.fixture
def radius_store(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "radius.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    return store


@pytest.mark.parametrize("stored, expected", [
    (None, 0.15), ("broken", 0.0), ("nan", 0.0), ("inf", 0.0),
    (-0.4, 0.0), (0.35, 0.35), (1.4, 1.0),
])
def test_gravity_radius_read_does_not_rewrite_saved_settings(radius_store, stored, expected):
    if stored is not None:
        radius_store.setValue("prefs/ambient_gravity_radius", stored)
    assert prefs._ambient_gravity_radius() == expected
    assert radius_store.value("prefs/ambient_gravity_radius") == stored


@pytest.mark.parametrize("value, expected", [("bad", 0.0), (float("nan"), 0.0),
                                             (float("inf"), 0.0), (-1, 0.0),
                                             (0.7, 0.7), (3, 1.0)])
def test_gravity_radius_saved_value_is_finite_and_bounded(radius_store, value, expected):
    prefs._set_ambient_gravity_radius(value)
    assert prefs._ambient_gravity_radius() == expected
    assert float(radius_store.value("prefs/ambient_gravity_radius")) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_widget_radius_cannot_enable_hover_gravity(qtbot, value):
    widget = AmbientWidget(theme="data_art_impulse_lens", gravity_radius=value)
    qtbot.addWidget(widget)
    widget.resize(200, 120)
    widget.show()
    assert widget.gravity_radius() == 0
    assert widget._interaction_app is not None
    qtbot.mouseClick(widget, Qt.LeftButton)
    assert not widget._pending_art_impulses
    assert not widget.engine._gravity_impulses
    widget.set_gravity_radius(0.5)
    assert widget._interaction_app is not None
    widget.set_gravity_radius(value)
    assert widget.gravity_radius() == 0
    assert widget._interaction_app is not None
    assert not widget._pending_art_impulses
    assert not widget.engine._gravity_impulses


def test_gravity_radius_defaults_to_fifteen_percent_cancel_preserves_and_save_persists(
    radius_store, qtbot, qt_theme_applied,
):
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    radius = dialog.findChild(QSlider, "AmbientGravityRadius")
    assert radius is not None
    assert (radius.minimum(), radius.maximum(), radius.value()) == (0, 100, 15)
    radius.setValue(65)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel).click()
    assert prefs._ambient_gravity_radius() == 0.15

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    radius = dialog.findChild(QSlider, "AmbientGravityRadius")
    assert radius.value() == 15
    radius.setValue(65)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert prefs._ambient_gravity_radius() == 0.65

    reopened = prefs.PreferencesDialog()
    qtbot.addWidget(reopened)
    assert reopened.findChild(QSlider, "AmbientGravityRadius").value() == 65
    theme = reopened.findChild(QComboBox, "AmbientTheme")
    theme.setCurrentIndex(next(index for index in range(theme.count())
                              if theme.itemData(index) == "none"))
    assert not reopened.findChild(QSlider, "AmbientGravityRadius").isEnabled()
