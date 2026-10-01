"""Live alpha exposure without changing an already selected segmentation model."""
import pytest

from spacr.qt.screens import make_masks as mm
from spacr.qt import preferences


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch):
    state = {'shown': False}
    monkeypatch.setattr(preferences, '_get_show_alpha_features', lambda: state['shown'])
    monkeypatch.setattr(mm, '_MAGNIFIER_BACKENDS', dict(mm._MAGNIFIER_BACKENDS))
    monkeypatch.setattr(mm, '_MAGNIFIER_SEGMENTERS', dict(mm._MAGNIFIER_SEGMENTERS))
    monkeypatch.setattr(mm, '_backend_ready', lambda _mode: True)
    monkeypatch.setattr(mm, '_state_ready', lambda _backend: True)
    widget = mm.MakeMasksScreen()
    qtbot.addWidget(widget)
    return widget, state


def test_existing_screen_off_on_off_on_refreshes_choices(screen):
    widget, state = screen
    mode = 'stardist:2D_versatile_fluo'
    assert widget._mag_mode.findData(mode) == -1
    state['shown'] = True
    widget._refresh_alpha_visibility()
    for combo in (widget._mag_mode, widget._cp_model, widget._uncertainty_ensemble):
        index = combo.findData(mode)
        assert index >= 0, (combo is widget._mag_mode, combo is widget._cp_model,
                            [combo.itemData(i) for i in range(combo.count())])
        assert not combo.view().isRowHidden(index)
        assert combo.model().item(index).isEnabled()
    counts = [combo.count() for combo in (widget._mag_mode, widget._cp_model)]
    state['shown'] = False
    widget._refresh_alpha_visibility()
    for combo in (widget._mag_mode, widget._cp_model):
        index = combo.findData(mode)
        assert combo.view().isRowHidden(index)
        assert not combo.model().item(index).isEnabled()
    state['shown'] = True
    widget._refresh_alpha_visibility()
    assert [combo.count() for combo in (widget._mag_mode, widget._cp_model)] == counts
    for combo in (widget._mag_mode, widget._cp_model):
        index = combo.findData(mode)
        assert not combo.view().isRowHidden(index)
        assert combo.model().item(index).isEnabled()


def test_hidden_selected_model_stays_selected_and_executable(screen):
    widget, state = screen
    mode = 'stardist:2D_versatile_fluo'
    state['shown'] = True
    widget._refresh_alpha_visibility()
    widget._mag_mode.setCurrentIndex(widget._mag_mode.findData(mode))
    widget._cp_model.setCurrentIndex(widget._cp_model.findData(mode))
    before = (widget._mag_mode.currentData(), widget._cp_model.currentData(), widget._magnifier.mode)
    state['shown'] = False
    widget._refresh_alpha_visibility()
    assert (widget._mag_mode.currentData(), widget._cp_model.currentData(), widget._magnifier.mode) == before
    assert mode in mm._MAGNIFIER_SEGMENTERS
    assert widget._mag_mode.view().isRowHidden(widget._mag_mode.currentIndex())
