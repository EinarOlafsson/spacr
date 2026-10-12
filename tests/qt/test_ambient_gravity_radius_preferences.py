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
    (None, 0.10), ("broken", 0.0), ("nan", 0.0), ("inf", 0.0),
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
    widget = AmbientWidget(theme="data_art_impulse_lens", gravity_radius=value,
                           ripples_enabled=False)
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


def test_gravity_radius_defaults_to_ten_percent_cancel_preserves_and_save_persists(
    radius_store, qtbot, qt_theme_applied,
):
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    radius = dialog.findChild(QSlider, "AmbientGravityRadius")
    assert radius is not None
    assert (radius.minimum(), radius.maximum(), radius.value()) == (0, 100, 10)
    radius.setValue(65)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel).click()
    assert prefs._ambient_gravity_radius() == 0.10

    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    radius = dialog.findChild(QSlider, "AmbientGravityRadius")
    assert radius.value() == 10
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


def test_the_retired_fifteen_percent_default_becomes_ten_once(radius_store):
    radius_store.setValue("prefs/ambient_gravity_radius", 0.15)
    assert prefs._ambient_gravity_radius() == pytest.approx(0.10)
    assert not radius_store.contains("prefs/ambient_gravity_radius")
    assert prefs._ambient_gravity_radius() == pytest.approx(0.10)


def test_a_fifteen_percent_chosen_after_the_change_is_kept(radius_store):
    prefs._set_ambient_gravity_radius(0.15)
    assert prefs._ambient_gravity_radius() == pytest.approx(0.15)
    radius_store.setValue("prefs/ambient_gravity_radius", 0.35)
    assert prefs._ambient_gravity_radius() == pytest.approx(0.35)
    radius_store.setValue("prefs/ambient_gravity_radius", 0.15)
    assert prefs._ambient_gravity_radius() == pytest.approx(0.15)


def test_another_saved_radius_is_never_migrated(radius_store):
    radius_store.setValue("prefs/ambient_gravity_radius", 0.35)
    assert prefs._ambient_gravity_radius() == pytest.approx(0.35)
    radius_store.setValue("prefs/ambient_gravity_radius", 0.15)
    assert prefs._ambient_gravity_radius() == pytest.approx(0.15)
