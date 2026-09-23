"""Host–Pathogen Analysis is accessible with typed settings and CLI dispatch."""
import pytest
pytest.importorskip('PySide6')
pytestmark = pytest.mark.qt


def test_toxoplasma_route_cli_and_settings_share_one_entry(qapp):
    from spacr.qt.app import APPS
    from spacr.qt.organisms import ORGANISMS
    from spacr.qt.screens.settings_model import resolve_default_settings, categories_for_app
    from spacr import settings
    from spacr.cli import MODULES
    assert 'host_pathogen' in {row[0] for row in ORGANISMS['toxoplasma']['modules']}
    assert 'host_pathogen' in {row[0] for row in APPS}
    defaults = resolve_default_settings('host_pathogen')
    groups = categories_for_app('host_pathogen', settings.categories)
    assert set(defaults) <= {key for values in groups.values() for key in values}
    for key in defaults:
        assert key in settings.expected_types and key in settings.tooltips
    module = MODULES['host_pathogen']
    assert module.entry == 'spacr.host_pathogen:analyze_host_pathogen'


def test_host_pathogen_settings_widgets_are_buildable(qapp, qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets
    widgets = SettingsWidgets('host_pathogen')
    widgets.build_sections()
    assert 'hp_marker_channels' in widgets._widgets
    assert 'hp_parasite_parent' in widgets._widgets
