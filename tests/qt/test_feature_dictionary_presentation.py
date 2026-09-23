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
