"""Live popup backdrops keep their own choice, overlay and readable card."""

import pytest
import threading

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDialog, QLabel, QSlider, QVBoxLayout, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient, glass
from spacr.qt.widgets.setup_card import SetupCard
from tests.qt.test_preferences_apply import private_preferences, _dialog, _apply, _answer


def _popup(qtbot, owner):
    popup = QDialog(owner)
    qtbot.addWidget(popup)
    QVBoxLayout(popup).addWidget(QLabel("Readable settings", popup))
    popup.resize(520, 360)
    assert glass.glass(popup)
    popup.show()
    qtbot.waitExposed(popup)
    return popup


def test_spaceout_preferences_keeps_its_popup_drift_without_another_fractal(
        private_preferences, qtbot, qapp, monkeypatch):
    from spacr.qt import theme
    from spacr.qt.app import MainWindow

    built = []

    class Fractal(QWidget):
        backend_name = "cpu"

        def __init__(self):
            super().__init__()
            self._spaceout_built_from = ("test",)
            built.append(self)

        def shutdown(self):
            self.hide()

    monkeypatch.setattr(ambient, "_build_the_spaceout_fractal",
                        lambda *_args, **_kwargs: Fractal())
    was_spaceout = theme.spaceout_enabled()
    window = None
    dialog = None
    theme.enable_spaceout()
    try:
        prefs.set_ambient_animation(ambient.SPACEOUT_THEME)
        prefs.set_popup_backdrop("drift")
        prefs.set_fractal_settings(backend="cpu")
        window = MainWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.waitExposed(window)
        original = tuple(ambient._live_spaceout_fractals())
        assert len(original) == 1
        before = len(built)
        assert before >= 1

        dialog = prefs.PreferencesDialog(window)
        qtbot.addWidget(dialog)
        assert glass.glass(dialog)
        dialog.show()
        qtbot.waitExposed(dialog)
        backdrop = dialog._spacr_popup_backdrop
        assert isinstance(backdrop, ambient.AmbientWidget)
        assert backdrop.theme() == "drift"
        assert backdrop.palette_name() == "spacr"
        assert tuple(ambient._live_spaceout_fractals()) == original
        assert len(built) == before

        prefs.apply_ambient_preferences(qapp)
        assert dialog._spacr_popup_backdrop is backdrop
        assert backdrop.theme() == "drift"
        assert backdrop.palette_name() == "spacr"
        assert tuple(ambient._live_spaceout_fractals()) == original
        assert len(built) == before
        dialog.reject()
    finally:
        if dialog is not None:
            backdrop = getattr(dialog, "_spacr_popup_backdrop", None)
            if isinstance(backdrop, ambient.AmbientWidget):
                backdrop.stop()
            dialog.reject()
        if window is not None:
            ambient._retire_fractals_on(window)
            window.close()
        qapp.processEvents()
        theme.enable_spaceout() if was_spaceout else theme.disable_spaceout()



@pytest.mark.parametrize("initial", ["off", "blobs", "drift"])
def test_apply_keeps_none_independent_of_main_animation(private_preferences, qtbot, initial):
    prefs.set_ambient_animation("data_art_impulse_lens")
    prefs.set_popup_backdrop(initial)
    dialog, owner, _ = _dialog(qtbot)
    popup = _popup(qtbot, owner)
    combo = dialog.findChild(QComboBox, "PopupBackdrop")
    combo.setCurrentIndex(combo.findData("off"))
    main = dialog.findChild(QComboBox, "AmbientTheme")
    assert main is not None
    main.setCurrentIndex(main.findData("blobs"))
    question = _apply(dialog, qtbot)
    backdrop = getattr(popup, "_spacr_popup_backdrop", None)
    if backdrop is not None:
        assert backdrop.isHidden() and not backdrop.is_running()
    assert not getattr(dialog, "_spacr_popup_backdrop", None) or dialog._spacr_popup_backdrop.isHidden()
    _answer(question, "Revert", qtbot)
    assert prefs.get_popup_backdrop() == initial
    if initial != "off":
        assert popup._spacr_popup_backdrop.theme() == initial
        assert not popup._spacr_popup_backdrop.isHidden()


def test_apply_creates_and_reuses_popup_theme_separately(private_preferences, qtbot):
    prefs.set_ambient_animation("data_art_impulse_lens")
    prefs.set_popup_backdrop("off")
    dialog, owner, _ = _dialog(qtbot)
    popup = _popup(qtbot, owner)
    assert not popup.findChildren(ambient.AmbientWidget)
    combo = dialog.findChild(QComboBox, "PopupBackdrop")
    combo.setCurrentIndex(combo.findData("blobs"))
    question = _apply(dialog, qtbot)
    backdrop = popup._spacr_popup_backdrop
    assert backdrop.theme() == "blobs"
    assert prefs.get_ambient_theme() == "data_art_impulse_lens"
    _answer(question, "Keep", qtbot)
    question = _apply(dialog, qtbot)
    assert popup._spacr_popup_backdrop is backdrop
    assert len(popup.findChildren(ambient.AmbientWidget)) == 1
    _answer(question, "Keep", qtbot)


def test_card_stays_above_animation_after_resize_and_reopen(private_preferences, qtbot):
    prefs.set_ambient_animation("blobs")
    prefs.set_popup_backdrop("blobs")
    dialog, owner, _ = _dialog(qtbot)
    popup = _popup(qtbot, owner)
    card = popup.findChild(SetupCard)
    backdrop = popup._spacr_popup_backdrop
    card._paint_accent = lambda painter, color, rect: painter.fillRect(
        0, 0, card.width(), 12, Qt.blue)
    for size in [(700, 460), (560, 390)]:
        popup.resize(*size)
        popup.hide()
        popup.show()
        qtbot.waitExposed(popup)
        image = popup.grab().toImage()
        pixel = image.pixelColor(card.x() + card.width() // 2, card.y() + 6)
        assert pixel.blue() > 240 and pixel.red() < 15
        assert backdrop.geometry() == popup.rect()


def test_popup_darkness_apply_revert_keep_and_reset(private_preferences, qtbot):
    prefs.set_ambient_animation("blobs")
    prefs.set_popup_backdrop("blobs")
    prefs._set_popup_backdrop_darkness(0.4)
    dialog, owner, _ = _dialog(qtbot)
    popup = _popup(qtbot, owner)
    slider = dialog.findChild(QSlider, "PopupBackdropDarkness")
    assert slider.isEnabled() and slider.value() == 40
    slider.setValue(95)
    question = _apply(dialog, qtbot)
    assert prefs._popup_backdrop_darkness() == 0.95
    _answer(question, "Revert", qtbot)
    assert prefs._popup_backdrop_darkness() == 0.4
    assert dialog.isVisible() and slider.value() == 95
    question = _apply(dialog, qtbot)
    _answer(question, "Keep", qtbot)
    new, _, _ = _dialog(qtbot)
    assert new.findChild(QSlider, "PopupBackdropDarkness").value() == 95
    combo = new.findChild(QComboBox, "PopupBackdrop")
    combo.setCurrentIndex(combo.findData("off"))
    assert not new.findChild(QSlider, "PopupBackdropDarkness").isEnabled()


@pytest.mark.parametrize("value,expected", [(0, 0), (1, 1), (0.4, 0.4),
                                          (-1, 0), (2, 1), (float("nan"), 0.85),
                                          (float("inf"), 0.85), ("bad", 0.85)])
def test_popup_darkness_validates_values(private_preferences, value, expected):
    prefs._set_popup_backdrop_darkness(value)
    assert prefs._popup_backdrop_darkness() == expected


@pytest.mark.parametrize("theme", ["blobs", "aurora"])
def test_classic_animation_moves_after_switching_from_field(private_preferences, qtbot, theme):
    widget = ambient.AmbientWidget(theme="data_art_impulse_lens", palette="spacr",
                                  background="#101010", seed=17, density=0.1)
    qtbot.addWidget(widget)
    widget.resize(640, 480)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_theme(theme)
    before = bytes(widget.grab().toImage().constBits())
    qtbot.waitUntil(lambda: widget.time() > 0.4, timeout=5000)
    after = bytes(widget.grab().toImage().constBits())
    assert widget.is_running()
    assert widget.time() > 0.4
    assert after != before
    widget.set_animating(False)


def test_blobs_shading_consumes_time_when_gui_ticks_cannot_get_the_lock(private_preferences, qtbot):
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=17)
    qtbot.addWidget(widget)
    widget.resize(640, 480)
    widget.show()
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0)
    widget._timer.stop()
    acquired = threading.Event()
    release = threading.Event()

    def hold_engine():
        with widget._engine_lock:
            acquired.set()
            release.wait(5)

    thread = threading.Thread(target=hold_engine)
    thread.start()
    assert acquired.wait(1)
    try:
        before = widget.time()
        widget._clock.start()
        qtbot.wait(30)
        widget._on_tick()
        assert widget.time() == before
        assert widget._pending_dt > 0
    finally:
        release.set()
        thread.join(1)
    qtbot.waitUntil(lambda: widget.time() > before, timeout=3000)
    assert not widget._timer.isActive()
    widget.set_animating(False)


def test_darkness_changes_the_rendered_card_without_changing_page_opacity(private_preferences, qtbot):
    prefs.set_ambient_animation("blobs")
    prefs.set_popup_backdrop("blobs")
    prefs.set_theme_choice("dark")
    prefs.apply_preferences_to_app()
    dialog, owner, _ = _dialog(qtbot)
    popup = _popup(qtbot, owner)
    backdrop = popup._spacr_popup_backdrop
    backdrop.set_animating(False)
    backdrop.set_background_color("#808080")
    card = popup.findChild(SetupCard)
    card._paint_accent = lambda *_args: None
    page_opacity = prefs.get_pane_opacity()
    pixels = []
    for darkness in (0, 0.5, 1):
        prefs._set_popup_backdrop_darkness(darkness)
        card.update()
        pixels.append(popup.grab().toImage().pixelColor(40, 90).lightness())
    assert pixels[0] > pixels[1] > pixels[2]
    assert prefs.get_pane_opacity() == page_opacity


@pytest.mark.parametrize('theme', ['off', 'blobs'])
def test_page_opacity_changes_the_popup_card_with_and_without_animation(
        private_preferences, qtbot, theme):
    prefs.set_ambient_animation('blobs')
    prefs.set_popup_backdrop(theme)
    prefs.set_theme_choice('dark')
    prefs.apply_preferences_to_app()
    dialog, owner, _ = _dialog(qtbot)
    popup = _popup(qtbot, owner)
    backdrop = getattr(popup, '_spacr_popup_backdrop', None)
    if backdrop is not None:
        backdrop.set_animating(False)
        backdrop.set_background_color('#808080')
    card = popup.findChild(SetupCard)
    card._paint_accent = lambda *_args: None
    pixels = []
    for opacity in (.2, .6, 1):
        prefs.set_pane_opacity(opacity)
        prefs.apply_preferences_to_app()
        card.update()
        pixel = popup.grab().toImage().pixelColor(40, 90)
        pixels.append((pixel.lightness(), pixel.alpha()))
    if theme == 'off':
        assert pixels[0][1] < pixels[1][1] < pixels[2][1], pixels
    else:
        assert pixels[0][0] > pixels[1][0] > pixels[2][0], pixels
