"""Opening geometry and mode defaults on a newly created native window."""

from PySide6.QtCore import QSettings

from spacr.qt import preferences as prefs


def test_field_default_is_ten_percent():
    from spacr.qt.widgets.ambient import _default_density_for

    assert _default_density_for("data_art_impulse_lens") == 0.1


def test_leaving_minimal_mode_without_a_stash_restores_field(monkeypatch, tmp_path):
    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    prefs.set_spacr_mode("extra_performance")
    store.remove(prefs._KEY_MODE_VISUAL_STASH)
    prefs.set_spacr_mode("balanced")
    assert prefs.get_ambient_animation() == "data_art_impulse_lens"
    assert prefs.get_ambient_density() == 0.1


def test_setup_card_is_laid_out_before_any_next_click(qtbot, monkeypatch):
    from spacr.qt.widgets.setup_slides import SetupSlides

    monkeypatch.setattr(SetupSlides, "_install_backdrop", lambda self: None)
    window = SetupSlides()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(100)
    assert window.card.width() >= window.width() - 4
    assert window.card.height() >= window.height() - 4
    first = window._pages.geometry()
    window._settle_opening_layout()
    assert window._pages.geometry() == first


def test_mac_dissolve_captures_the_window_without_the_cover(qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QGuiApplication, QPalette
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets import loading_screen as ls

    host = QWidget()
    host.resize(400, 300)
    palette = host.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#c02917"))
    host.setPalette(palette)
    host.setAutoFillBackground(True)
    qtbot.addWidget(host)
    host.show()
    screen = ls.LoadingScreen(parent=host)
    assert screen.show_mosaic()
    screen.show()
    screen.grab()
    monkeypatch.setattr(QGuiApplication, "platformName", lambda: "cocoa")
    monkeypatch.setattr(ls, "_MOSAIC_MIN_HOLD_S", 0)
    assert screen.blow_away(force=True)
    screen._begin()
    assert screen.testAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
    image = screen._underlay.toImage()
    assert image.pixelColor(image.width() // 2, image.height() // 2) == QColor("#c02917")
    # At the end of the fade the window, rather than an opaque black cover,
    # must remain visible, including after the host has been resized.
    screen._began_at = screen._clock() - 3.5
    screen.resize(500, 350)
    image = screen.grab().toImage()
    assert image.pixelColor(0, 0) == QColor("#c02917")
    screen._finish()
    assert screen._underlay is None
