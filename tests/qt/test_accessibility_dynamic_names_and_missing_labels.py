"""Changing visible captions and missing labels preserve accessible help."""

import pytest

pytest.importorskip("PySide6")
from PySide6.QtGui import QAccessible
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMenu,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from spacr.qt import i18n
from spacr.qt.screens import settings_model
from spacr.qt.screens.settings_model import retarget_field_tooltips


pytestmark = pytest.mark.qt


@pytest.mark.parametrize("caption_source", ("button", "window", "group"))
def test_native_caption_changes_preserve_usable_accessible_names(
        qtbot, monkeypatch, caption_source):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "en")
    if caption_source == "window":
        host = field = QLineEdit()
        set_caption = field.setWindowTitle
        set_caption("Current selection")
        native = QAccessible.queryAccessibleInterface(field)
        assert native is not None
        assert native.text(QAccessible.Name) == ""
        set_caption("")
    else:
        host = QGroupBox() if caption_source == "group" else QWidget()
        field = QLineEdit(host) if caption_source == "group" else QToolButton(host)
        QVBoxLayout(host).addWidget(field)
        set_caption = host.setTitle if caption_source == "group" else field.setText
    qtbot.addWidget(host)
    field.setToolTip("Details")
    i18n.retranslate_widget_tree(host)
    assert field.accessibleName() == "Details"
    assert field.property(i18n._A11Y_DERIVED_NAME) == "Details"

    set_caption("Current selection")
    i18n.retranslate_widget_tree(host)

    interface = QAccessible.queryAccessibleInterface(field)
    assert interface is not None
    if caption_source == "window":
        assert field.windowTitle() == "Current selection"
        assert field.accessibleName() == "Details"
        assert interface.text(QAccessible.Name) == "Details"
    else:
        assert field.accessibleName() == ""
        assert interface.text(QAccessible.Name) == "Current selection"
    assert field.toolTip() == "Details"
    i18n.retranslate_widget_tree(host)
    if caption_source == "window":
        assert field.accessibleName() == "Details"
        assert interface.text(QAccessible.Name) == "Details"
    else:
        assert field.accessibleName() == ""
        assert interface.text(QAccessible.Name) == "Current selection"


def test_clearing_a_menu_title_retires_its_derived_accessible_name(
        qtbot, monkeypatch):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "en")
    host = QWidget()
    qtbot.addWidget(host)
    menu = QMenu("Details", host)
    i18n.retranslate_widget_tree(host)
    assert menu.accessibleName() == "Details"

    menu.setTitle("")
    i18n.retranslate_widget_tree(host)

    assert menu.title() == ""
    assert menu.accessibleName() == ""
    interface = QAccessible.queryAccessibleInterface(menu)
    assert interface is not None
    assert interface.text(QAccessible.Name) == ""


def test_scroll_area_uses_the_caption_inside_an_untitled_group(
        qtbot, monkeypatch):
    monkeypatch.setenv(i18n.ENV_LANGUAGE, "en")
    area = QScrollArea()
    qtbot.addWidget(area)
    group = QGroupBox()
    caption = QLabel("Acquisition filters")
    QVBoxLayout(group).addWidget(caption)
    area.setWidget(group)

    i18n.retranslate_widget_tree(area)

    assert group.title() == ""
    assert area.accessibleName() == caption.text() == "Acquisition filters"


@pytest.mark.parametrize("blank_label", (False, True))
def test_a_field_without_a_usable_caption_keeps_its_help(qtbot, blank_label):
    host = QWidget()
    qtbot.addWidget(host)
    form = QFormLayout(host)
    field = QLineEdit()
    help_text = "Keep this help with the unnamed setting."
    field.setToolTip(help_text)
    if blank_label:
        label = QLabel("")
        form.addRow(label, field)
    else:
        form.addRow(field)

    assert retarget_field_tooltips(host) == 0

    assert field.toolTip() == help_text
    assert field.parentWidget() is host
    if blank_label:
        assert label.toolTip() == ""
        assert label.text() == ""
    else:
        assert form.labelForField(field) is None


def test_a_navigation_label_does_not_take_the_adjacent_fields_help(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    label = QLabel('<a href="https://example.invalid/help">Open documentation</a>')
    label.setCursor(Qt.PointingHandCursor)
    field = QLineEdit()
    field.setToolTip("Number of observations to retain.")
    QFormLayout(host).addRow(label, field)
    caption = label.text()

    assert retarget_field_tooltips(host) == 0

    assert field.toolTip() == "Number of observations to retain."
    assert label.toolTip() == ""
    assert label.text() == caption
    assert label.cursor().shape() == Qt.PointingHandCursor


def test_an_undeclared_plugin_count_remains_an_integer_editor(
        qtbot, monkeypatch):
    from spacr import plugins
    from tests.qt.test_cov_settings_model import _FakeApp

    app = _FakeApp(defaults="local_plugin:defaults")
    monkeypatch.setattr(plugins, "get_app", lambda _key: app)
    monkeypatch.setattr(plugins, "load_object", lambda _reference: (
        lambda _settings=None: {"plugin_iterations": 7}))
    host = QWidget()
    qtbot.addWidget(host)
    model = settings_model.SettingsWidgets("integer_plugin", parent=host)
    assert model.build_sections()
    field = model._widgets["plugin_iterations"]
    assert isinstance(field, QSpinBox)
    assert field.value() == 7

    field.setValue(19)

    saved = model.collect()
    assert saved["plugin_iterations"] == 19
    assert type(saved["plugin_iterations"]) is int
