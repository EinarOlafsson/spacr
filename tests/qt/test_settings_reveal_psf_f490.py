"""Help-style PSF navigation must expose the whole nested Measure row."""
import pytest

from spacr.qt.settings_search import (
    ALL,
    ESSENTIALS,
    disclosure_for,
    forget_disclosure,
    install,
    remember_disclosure,
)
from spacr.qt.widgets.section import Section, _logical_parent


@pytest.mark.parametrize('level', [ESSENTIALS, ALL])
def test_reveal_opens_psf_ancestors_without_changing_values_or_preferences(
        qtbot, qt_theme_applied, monkeypatch, level):
    from spacr.qt import preferences
    from spacr.qt.screens.app_screen import AppScreen

    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: False)
    remember_disclosure('measure', level)
    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    screen.resize(1400, 1000)
    screen.show()
    screen.reveal_settings()
    bar = install(screen) or screen._settings_search
    key = 'psf_measurement_source'
    try:
        # Build this lazily-created group before deliberately shutting every
        # heading. The regression then exercises real nested section geometry.
        assert bar.reveal(key)
        field = screen._settings_model._widgets[key]
        ancestors = []
        parent = _logical_parent(field)
        while parent is not None:
            if isinstance(parent, Section):
                ancestors.append(parent)
            parent = _logical_parent(parent)
        assert len(ancestors) >= 2
        before = screen._settings_model.collect()
        for section in bar._sections:
            section.set_expanded(False)
        assert not field.isVisible()

        assert bar.reveal(key)
        qtbot.waitUntil(field.isVisible)
        assert all(section.is_expanded() for section in ancestors)
        assert all(not section.is_expanded() for section in bar._sections
                   if section not in ancestors)
        assert screen._settings_model.collect() == before
        assert disclosure_for('measure') == level
        assert bar.revealed_key() == key
        assert not bar.reveal('not_a_real_psf_setting')
        assert field.isVisible()
    finally:
        screen.close()
        forget_disclosure('measure')
