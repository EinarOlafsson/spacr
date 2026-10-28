"""Generate Masks category visibility stays stable as deferred settings arrive."""
import pytest
from PySide6.QtCore import QEvent, QObject

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


class _BodyVisibility(QObject):
    def __init__(self, body):
        super().__init__(body)
        self.events = []
        body.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Show, QEvent.Hide):
            self.events.append(event.type())
        return False


@pytest.fixture
def mask_window(qtbot, private_ai_profile, monkeypatch):
    from spacr.qt import preferences, settings_search
    from spacr.qt.app import MainWindow

    preferences.set_easy_mode(False)
    preferences.set_reduce_motion(True)
    preferences.set_ambient_enabled(False)
    monkeypatch.setattr(settings_search, '_settings', preferences._settings)
    preferences._settings().setValue('settings/disclosure/mask', settings_search.ESSENTIALS)
    window = MainWindow(initial_app='mask')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    yield window
    window.close()


def test_generate_masks_hidden_category_does_not_flash_during_idle_build(mask_window, qtbot):
    screen = mask_window._screens['mask']
    section = next(section for section in screen.rendered_settings_sections()
                   if section.property('settingsCategorySource') == 'Per-object settings')
    assert section.isHidden() and section.is_expanded()
    observer = _BodyVisibility(section._body)
    mask_window.show()
    qtbot.wait(1200)
    assert observer.events == []
    assert section.isHidden() and section.is_expanded()


def test_filter_release_preserves_user_fold_and_channel_values(mask_window, qtbot):
    from spacr.qt import settings_search

    screen = mask_window._screens['mask']
    bar = screen._settings_search
    section = next(section for section in screen.rendered_settings_sections()
                   if section.property('settingsCategorySource') == 'Per-object settings')
    mask_window.show()
    bar.set_level(settings_search.ALL)
    assert not section.isHidden()
    section.set_expanded(False)
    values = {'cell_channel': 0, 'nucleus_channel': 1,
              'pathogen_channel': None, 'cell_diameter': 33}
    screen.apply_settings_dict(values)
    for _ in range(3):
        screen.refresh_maturity_visibility()
        screen._settings_model.refresh_object_visibility()
    assert not section.is_expanded()
    bar.set_query('cell_diameter')
    assert not section.isHidden() and section.is_expanded()
    bar.set_query('not-a-real-setting')
    assert section.isHidden()
    observer = _BodyVisibility(section._body)
    for _ in range(3):
        screen.refresh_maturity_visibility()
        screen._settings_model.refresh_object_visibility()
    assert observer.events == []
    bar.set_query('')
    assert not section.isHidden() and not section.is_expanded()
    collected = screen._settings_model.collect()
    assert {key: collected[key] for key in values} == values
