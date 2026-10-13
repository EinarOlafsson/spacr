"""Dictionary chrome and collective organelle descriptions stay readable."""
import pytest
pytest.importorskip('PySide6')
from PySide6.QtWidgets import QPushButton
from spacr.qt.widgets.feature_dictionary import FeatureDictionaryDialog
from spacr.feature_dict import parse_column

pytestmark = pytest.mark.qt


def test_one_organelle_filter_and_compact_scope_preserve_concrete_lookup(qtbot):
    dialog = FeatureDictionaryDialog()
    qtbot.addWidget(dialog)
    panel = dialog.panel
    choices = [panel._object.itemText(i) for i in range(panel._object.count())]
    assert sum('organelle' in label.lower() for label in choices) == 1
    panel.set_query('area')
    assert 'organelle(?:[b-z]|[a-z]{2})?' in panel.detail_text()
    assert 'organelleb,' not in panel.detail_text()
    panel.show_column('organellez_area')
    assert panel.current_doc() is not None
    assert parse_column('organellez_area').object_type == 'organellez'
    button = dialog.findChild(QPushButton, 'DangerButton')
    assert button is not None and button.property('buttonActionRole') == 'negative'
    assert dialog.property('spacrNoGlass')
    assert dialog.windowOpacity() == 1


@pytest.mark.parametrize('theme', ['dark', 'light', 'blobs'])
@pytest.mark.parametrize('opacity', [.4, .8])
def test_dictionary_styles_keep_the_prepared_theme_surface(theme, opacity, caplog, monkeypatch):
    from spacr.qt import theme as styles
    from spacr.qt.widgets.feature_dictionary import OBJECT_NAME, _panel_qss
    monkeypatch.setitem(styles._WIDGET_QSS, OBJECT_NAME, _panel_qss)
    palette = styles._widget_qss_palette(theme, 1., opacity)
    block = _panel_qss(palette, opacity)
    assert f"background: {palette['surface_alt']}" in block
    sheet = styles.stylesheet(theme, surface_opacity=opacity)
    assert f'QWidget#{OBJECT_NAME} QTextBrowser#FeatureDictionaryDetail' in sheet
    assert 'FeatureDictionary failed to render' not in caplog.text
