"""The public organism guide survives the actual startup maturity pass."""
from PySide6.QtCore import QSettings

from spacr.qt import app, register_self_registering_modules
from spacr.qt.widgets.home import AppTile


def test_normal_registration_keeps_reference_guide_on_real_home(qtbot, tmp_path, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, '_settings',
                        lambda: QSettings(str(tmp_path / 'private.ini'), QSettings.IniFormat))
    preferences.set_easy_mode(False)
    preferences.set_show_alpha(False)
    preferences._set_show_alpha_features(False)
    preferences._set_show_alpha_species(False)
    monkeypatch.setattr(app, 'APP_STAGE', dict(app.APP_STAGE))
    register_self_registering_modules()
    home = app.make_home_page()
    qtbot.addWidget(home)
    tiles = {tile.property('moduleAppKey') for tile in home.findChildren(AppTile)}
    assert app.app_stage('toxoplasma') == app.STAGE_STABLE
    assert 'toxoplasma' in tiles
    assert not tiles.intersection({'plasmodium', 'candida', 'trypanosoma',
                                   'leishmania', 'giardia', 'virus', 'mammalian'})
