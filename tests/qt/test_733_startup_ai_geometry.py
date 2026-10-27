"""Startup AI guidance stays reachable on a laptop-sized native screen."""
from PySide6.QtCore import QPoint, QRect
from PySide6.QtGui import QScreen
from PySide6.QtWidgets import QLabel, QScrollArea

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


def test_how_it_runs_guidance_is_reachable_on_720p(qtbot, private_ai_profile, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides
    preferences.set_gui_scale(1.0)
    preferences.set_font_scale(1.0)
    preferences.apply_preferences_to_app()
    monkeypatch.setattr(QScreen, 'availableGeometry', lambda self: QRect(0, 0, 1280, 720))
    dialog = SetupSlides()
    qtbot.addWidget(dialog)
    dialog._show_slide(next(i for i, row in enumerate(SLIDES) if row[0] == 'How it runs'))
    dialog.show()
    qtbot.wait(100)
    guidance = dialog.findChildren(QLabel, 'StartupAiGuidance')
    assert len(guidance) == 4
    for label in guidance:
        assert label.height() >= label.heightForWidth(label.width())
    assert dialog.width() <= 1280 and dialog.height() <= 720
    for control in [*guidance, dialog._next, dialog._back]:
        ancestors = []
        parent = control.parentWidget()
        while parent and parent is not dialog:
            if isinstance(parent, QScrollArea):
                ancestors.append(parent)
            parent = parent.parentWidget()
        for scroll in ancestors:
            scroll.ensureWidgetVisible(control)
        qtbot.wait(10)
        top_left = control.mapTo(dialog, QPoint(0, 0))
        assert dialog.rect().contains(QRect(top_left, control.size()))
        assert control.isVisible()
