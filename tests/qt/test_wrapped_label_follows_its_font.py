"""The empty state's wrapped label re-measures itself when its font changes."""
import pytest

from PySide6.QtCore import QEvent
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.empty_state import _WrappedLabel


SENTENCE = ("Drop a folder of images anywhere on this window or type a path "
            "into the Source field below.")


def test_a_larger_font_grows_the_fixed_height(qtbot):
    label = _WrappedLabel(SENTENCE, wrap_width=200)
    qtbot.addWidget(label)
    before = label.height()
    before_width = label.width()
    from spacr.qt import gui_scale
    factor = gui_scale._factor_for(label) if gui_scale.installed() else 1.0
    assert before_width == gui_scale.scale_int(200, factor)
    font = QFont(label.font())
    font.setPointSizeF(font.pointSizeF() * 2)
    label.setFont(font)
    assert label.height() > before
    assert label.height() == label._wrapped_height()
    assert label.width() == before_width


def test_a_style_change_with_the_same_font_keeps_the_height(qtbot):
    label = _WrappedLabel(SENTENCE, wrap_width=200)
    qtbot.addWidget(label)
    before = label.height()
    QApplication.sendEvent(label, QEvent(QEvent.StyleChange))
    assert label.height() == before == label._wrapped_height()
    QApplication.sendEvent(label, QEvent(QEvent.EnabledChange))
    assert label.height() == before


@pytest.mark.parametrize("scale", (0.75, 1.25))
def test_wrapped_text_fits_actual_width_through_zoom_font_style_and_text(
        qtbot, qt_theme_applied, monkeypatch, scale):
    from spacr.qt import gui_scale, preferences

    settings = preferences._settings()
    keys = ("prefs/gui_scale", "prefs/font_scale")
    previous = {key: settings.value(key) for key in keys if settings.contains(key)}
    previous_live = gui_scale.current_scale()
    gui_scale.install_scaling_layer()
    with monkeypatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", scale)
        try:
            preferences.set_gui_scale(scale)
            preferences.set_font_scale(1.0)
            label = _WrappedLabel(SENTENCE, wrap_width=200)
            qtbot.addWidget(label)
            label.show()
            QApplication.processEvents()
            assert label.width() == gui_scale.scale_int(200, scale)
            assert label.height() == label._wrapped_height()
            assert label.height() >= label.heightForWidth(label.width())

            zoom = 1.25 if scale == 0.75 else 0.75
            gui_scale.set_gui_scale_live(zoom)
            QApplication.processEvents()
            assert label.width() == gui_scale.scale_int(200, zoom)
            assert label.height() == label._wrapped_height()
            assert label.height() >= label.heightForWidth(label.width())

            before = label.height()
            width = label.width()
            font = QFont(label.font())
            font.setPixelSize(label.fontInfo().pixelSize() * 2)
            label.setFont(font)
            QApplication.processEvents()
            assert label.width() == width
            assert label.height() > before
            assert label.height() == label._wrapped_height()
            assert label.height() >= label.heightForWidth(label.width())
            before = label.height()
            QApplication.sendEvent(label, QEvent(QEvent.StyleChange))
            assert label.height() == before
            label.setText(SENTENCE * 3)
            assert label.height() > before
            assert label.height() == label._wrapped_height()
            assert label.height() >= label.heightForWidth(label.width())
            label.close()
        finally:
            gui_scale.set_gui_scale_live(previous_live if previous_live is not None else 1.0)
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


def test_zoom_skips_a_deleted_wrapped_label_with_a_retained_python_wrapper(
        qtbot, qt_theme_applied, monkeypatch, caplog):
    import logging

    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QWidget
    from shiboken6 import isValid
    from spacr.qt import gui_scale, preferences

    settings = preferences._settings()
    keys = ("prefs/gui_scale", "prefs/font_scale")
    previous = {key: settings.value(key) for key in keys if settings.contains(key)}
    previous_live = gui_scale.current_scale()
    gui_scale.install_scaling_layer()
    with monkeypatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 0.75)
        try:
            preferences.set_gui_scale(0.75)
            preferences.set_font_scale(1.0)
            parent = QWidget()
            qtbot.addWidget(parent)
            label = _WrappedLabel(SENTENCE, wrap_width=120, parent=parent)
            parent.show()
            QApplication.processEvents()
            label.deleteLater()
            QCoreApplication.sendPostedEvents(label, QEvent.DeferredDelete)
            assert isValid(parent)
            assert not isValid(label)
            assert any(
                getattr(ref(), "__self__", None) is label
                for ref in gui_scale._LISTENERS
            )
            before_logs = len(caplog.records)
            with caplog.at_level(logging.DEBUG, logger=gui_scale.LOG.name):
                with qtbot.captureExceptions() as exceptions:
                    assert gui_scale.set_gui_scale_live(1.25) == 1.25
                    QApplication.processEvents()
            assert not exceptions
            assert not any(
                record.name == gui_scale.LOG.name
                and record.getMessage() == "a GUI scale listener failed"
                for record in caplog.records[before_logs:]
            )
            assert not isValid(label)
            parent.close()
        finally:
            gui_scale.set_gui_scale_live(previous_live if previous_live is not None else 1.0)
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


def test_nested_wrapped_text_reconciles_native_bounds_after_zoom_replay(
        qtbot, qt_theme_applied, monkeypatch):
    from PySide6.QtWidgets import QWidget
    from spacr.qt import gui_scale, preferences

    settings = preferences._settings()
    keys = ("prefs/gui_scale", "prefs/font_scale")
    previous = {key: settings.value(key) for key in keys if settings.contains(key)}
    previous_live = gui_scale.current_scale()
    gui_scale.install_scaling_layer()
    with monkeypatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 0.75)
        try:
            preferences.set_gui_scale(0.75)
            preferences.set_font_scale(1.0)
            parent = QWidget()
            qtbot.addWidget(parent)
            label = _WrappedLabel(SENTENCE, wrap_width=120, parent=parent)
            parent.show()
            QApplication.processEvents()
            assert label.width() == gui_scale.scale_int(120, 0.75)
            assert label.height() == label._wrapped_height()
            gui_scale.set_gui_scale_live(1.25)
            QApplication.processEvents()
            measured = label._wrapped_height()
            assert label.parentWidget() is parent
            assert label.width() == gui_scale.scale_int(120, 1.25)
            assert label.height() == measured
            assert gui_scale._actual_minimum_height(label) == measured
            maximum = gui_scale._ORIGINAL[(QWidget, "maximumHeight")](label)
            assert maximum == measured
            assert label.height() >= label.heightForWidth(label.width())
            parent.close()
        finally:
            gui_scale.set_gui_scale_live(previous_live if previous_live is not None else 1.0)
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)
