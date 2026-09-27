"""The Preferences Plugins tab, a plugin and recipe catalogue, is alpha.

Its widgets are registered in ``spacr.settings.ALPHA_FEATURES``: with Show
alpha features off the tab is not built, with it on the tab lists the
catalogue and installs and uninstalls from it. A catalogue remembered while
the tab is hidden is the one it opens with, and a plugin installed from it
keeps loading whatever the gate says.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings                              # noqa: E402
from PySide6.QtWidgets import QTabWidget, QWidget                 # noqa: E402

from spacr import plugins                                         # noqa: E402
from spacr.settings import ALPHA_FEATURES                         # noqa: E402
from tests.test_plugin_catalogue import _catalogue                # noqa: E402

WIDGETS = ALPHA_FEATURES[582]["widgets"]


@pytest.fixture
def prefs(tmp_path, monkeypatch, qt_theme_applied):
    from spacr.qt import preferences

    path = tmp_path / "plugins.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setenv("SPACR_PLUGIN_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SPACR_PLUGIN_CATALOGUE", raising=False)
    monkeypatch.delenv("SPACR_PLUGIN_MODULES", raising=False)
    monkeypatch.delenv("SPACR_DISABLE_PLUGINS", raising=False)
    plugins.reload_plugins()
    yield preferences
    plugins._forget_modules_under(
        str(tmp_path / "home" / "site" / "catalogue_probe"))
    monkeypatch.delenv("SPACR_PLUGIN_HOME")
    plugins.reload_plugins()


def _tab_titles(dlg):
    tabs = dlg.findChild(QTabWidget, "PreferencesTabs")
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_hidden_until_alpha_features_are_shown(qtbot, prefs):
    for shown in (False, True):
        prefs._set_show_alpha_features(shown)
        dlg = prefs.PreferencesDialog()
        qtbot.addWidget(dlg)
        assert ("Plugins" in _tab_titles(dlg)) is shown
        for name in WIDGETS:
            assert (dlg.findChild(QWidget, name) is not None) is shown, name


def test_a_catalogue_saved_while_hidden_opens_installs_and_uninstalls(
        qtbot, prefs, tmp_path):
    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    prefs._set_show_alpha_features(False)
    prefs._set_plugin_catalogue(str(source))

    prefs._set_show_alpha_features(True)
    dlg = prefs.PreferencesDialog()
    qtbot.addWidget(dlg)
    table = dlg.findChild(QWidget, "PluginCatalogueTable")
    assert dlg.findChild(QWidget, "PluginCatalogueSource").text() == str(source)
    assert table.rowCount() == 2
    assert table.item(0, 1).text() == "Catalogue probe"
    assert table.item(0, 5).text() == "Test Lab"
    assert table.item(0, 6).text() == "MIT"
    install = dlg.findChild(QWidget, "PluginCatalogueInstall")
    uninstall = dlg.findChild(QWidget, "PluginCatalogueUninstall")
    assert not install.isEnabled()

    table.selectRow(0)
    assert install.isEnabled() and not uninstall.isEnabled()
    install.click()
    assert "Installed Catalogue probe 1.0.0" in dlg.findChild(
        QWidget, "PluginCatalogueStatus").text()
    assert table.item(0, 3).text() == "1.0.0"
    assert table.item(0, 4).text() == "installed"

    prefs._set_show_alpha_features(False)
    assert plugins.reload_plugins()
    assert plugins.get_app("catalogue_probe_app") is not None

    table.selectRow(0)
    uninstall.click()
    assert plugins.get_app("catalogue_probe_app") is None
    assert table.item(0, 4).text() == "available"


def test_an_unreadable_catalogue_says_so(qtbot, prefs, tmp_path):
    prefs._set_show_alpha_features(True)
    dlg = prefs.PreferencesDialog()
    qtbot.addWidget(dlg)
    dlg.findChild(QWidget, "PluginCatalogueSource").setText(
        str(tmp_path / "missing.json"))
    dlg.findChild(QWidget, "PluginCatalogueLoad").click()
    assert "Could not read the catalogue" in dlg.findChild(
        QWidget, "PluginCatalogueStatus").text()
    assert dlg.findChild(QWidget, "PluginCatalogueTable").rowCount() == 0


def test_sorting_keeps_installation_and_reselection_on_the_chosen_entry(
        qtbot, prefs, tmp_path):
    """Sorting the recipe above the plugin must not install the plugin."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFormLayout

    source = tmp_path / "cat"
    source.mkdir()
    _catalogue(source)
    prefs._set_plugin_catalogue(str(source))
    host = QWidget()
    qtbot.addWidget(host)
    page = prefs._PluginCataloguePage(QFormLayout(host), host)
    qtbot.wait(1)
    page.table.sortItems(1, Qt.DescendingOrder)
    assert page.table.item(0, 1).text() == "Toxoplasma infection assay"
    page.table.selectRow(0)
    assert page.selected()["key"] == "toxo_infection"
    assert page.install_selected()
    qtbot.wait(1)
    assert set(plugins._catalogue_installed()) == {"toxo_infection"}
    assert page.selected()["key"] == "toxo_infection"
    assert page.selected()["installed"] == "0.2"
    for row in range(page.table.rowCount()):
        keys = {page.table.item(row, column).data(Qt.UserRole)
                for column in range(page.table.columnCount())}
        assert len(keys) == 1
    assert page.table.item(0, 1).text() == "Toxoplasma infection assay"
    assert page.table.item(0, 3).text() == "0.2"
    assert page.uninstall_selected()
    assert plugins._catalogue_installed() == {}
