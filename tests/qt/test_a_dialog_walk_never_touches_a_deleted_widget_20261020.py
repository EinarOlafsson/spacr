"""A dialog's language pass never reaches a widget that has been deleted.

Lesson 71 crashed (SIGSEGV on the GUI thread) when Investigate hit opened a
file picker: the dialog filter retranslated the picker from inside its Show
event, and the accessible-naming walk that ends every pass
(``_name_for_assistive_tech`` -> ``_a11y_source`` -> ``_a11y_label_beside``)
used widget wrappers collected before the translation phase, when part of
the tree had been torn down in between.

Now the pass for a shown dialog runs on the next event-loop turn, is
dropped with the dialog if the dialog goes first, and the naming walk reads
the tree afresh and skips anything shiboken reports as gone.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
shiboken6 = pytest.importorskip("shiboken6")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QHBoxLayout, QLabel, QLineEdit, QWidget,
)

from spacr.qt import i18n  # noqa: E402


class _RebuildsOnRetranslate(QWidget):
    """A composite field that replaces its editor when the language changes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.addWidget(QLabel("Fractions", self))
        self.editor = QLineEdit(self)
        row.addWidget(self.editor)
        self.gone = []

    def retranslate_dynamic_content(self, _code):
        old = self.editor
        self.gone.append(id(old))
        self.layout().removeWidget(old)
        shiboken6.delete(old)
        self.editor = QLineEdit(self)
        self.layout().addWidget(self.editor)


@pytest.fixture
def dialog_filter(qapp):
    previous = getattr(qapp, "_spacr_dialog_i18n_filter", None)
    qapp._spacr_dialog_i18n_filter = None
    i18n.install_dialog_translation(qapp)
    yield qapp._spacr_dialog_i18n_filter
    from spacr.qt.gil_priority import _stop_watching_application_events

    installed = qapp._spacr_dialog_i18n_filter
    if installed is not None and installed is not previous:
        _stop_watching_application_events(qapp, installed)
        installed.deleteLater()
    qapp._spacr_dialog_i18n_filter = previous


def test_the_naming_walk_reads_the_tree_after_the_rebuild(qapp, monkeypatch):
    host = QDialog()
    field = _RebuildsOnRetranslate(host)
    named = []
    real = i18n._name_for_assistive_tech

    def recording(widgets, stamp=None, only_new=False):
        widgets = list(widgets)
        named.append(widgets)
        return real(widgets, stamp, only_new)

    monkeypatch.setattr(i18n, "_name_for_assistive_tech", recording)
    try:
        i18n.retranslate_widget_tree(host)
        assert field.gone, "the field did not rebuild during the pass"
        assert named and named[-1]
        assert all(shiboken6.isValid(w) for w in named[-1])
        assert not {id(w) for w in named[-1]} & set(field.gone)
        assert field.editor in named[-1]
        assert field.editor.accessibleName() == "Fractions"
    finally:
        shiboken6.delete(host)


def test_a_dialog_is_walked_after_its_show_not_inside_it(
        qapp, dialog_filter, monkeypatch):
    walked = []
    monkeypatch.setattr(i18n, "retranslate_widget_tree",
                        lambda widget, *a, **k: walked.append(widget))
    dialog = QDialog()
    try:
        dialog.show()
        assert walked == [], "the walk ran inside the Show event"
        qapp.processEvents()
        assert walked == [dialog]
    finally:
        shiboken6.delete(dialog)


def test_a_dialog_closed_before_its_walk_is_never_walked(
        qapp, dialog_filter, monkeypatch):
    walked = []
    monkeypatch.setattr(i18n, "retranslate_widget_tree",
                        lambda widget, *a, **k: walked.append(widget))
    deleted = QDialog()
    deleted.show()
    shiboken6.delete(deleted)
    closed = QDialog()
    closed.show()
    closed.close()
    try:
        QApplication.processEvents()
        assert walked == []
    finally:
        shiboken6.delete(closed)


def test_a_picker_that_rebuilds_while_shown_survives_its_walk(
        qapp, dialog_filter):
    for _ in range(3):
        picker = QDialog()
        field = _RebuildsOnRetranslate(picker)
        picker.show()
        qapp.processEvents()
        assert field.gone
        assert shiboken6.isValid(field.editor)
        picker.close()
        shiboken6.delete(picker)
        qapp.processEvents()


def test_accessibility_walk_skips_a_child_deleted_after_the_parent_snapshot(qapp):
    class RebuildingParent(QWidget):
        def children(self):
            children = super().children()
            shiboken6.delete(self.obsolete)
            return children

    from PySide6.QtWidgets import QFileDialog

    parent = RebuildingParent()
    parent.obsolete = QLabel('Gone', parent)
    live = QLineEdit(parent)
    picker = QFileDialog(parent)
    try:
        walked = i18n._a11y_live_tree(parent)
        assert live in walked and picker in walked
        assert all(shiboken6.isValid(widget) for widget in walked)
        assert all(widget.parentWidget() is not picker for widget in walked)
    finally:
        shiboken6.delete(parent)


def test_accessibility_name_probe_failure_preserves_an_explicit_control_name(qapp, monkeypatch):
    from PySide6.QtGui import QAccessible
    from PySide6.QtWidgets import QComboBox

    combo = QComboBox()
    combo.setAccessibleName('Physical channel')

    def unavailable(widget):
        raise RuntimeError('native accessibility interface unavailable')

    monkeypatch.setattr(QAccessible, 'queryAccessibleInterface', unavailable)
    try:
        assert i18n._a11y_effective_name(combo) == 'Physical channel'
        field = QLineEdit()
        field.setAccessibleName('Object identity')
        try:
            assert i18n._a11y_effective_name(field) == 'Object identity'
        finally:
            shiboken6.delete(field)
    finally:
        shiboken6.delete(combo)


def test_accessibility_ignores_a_layout_retired_before_its_label_lookup(qapp):
    class RebuildingParent(QWidget):
        def layout(self):
            return self.saved_layout

    parent = RebuildingParent()
    layout = QHBoxLayout(parent)
    field = QLineEdit(parent)
    layout.addWidget(field)
    parent.saved_layout = layout
    shiboken6.delete(layout)
    try:
        assert i18n._a11y_label_beside(field) is None
        field.setAccessibleName('Stable field name')
        assert field.accessibleName() == 'Stable field name'
    finally:
        shiboken6.delete(parent)


def test_accessibility_ignores_a_label_removed_during_the_form_lookup(qapp):
    from PySide6.QtWidgets import QFormLayout

    class CachedLabelForm(QFormLayout):
        def labelForField(self, field):
            return self.retired_label

    parent = QWidget()
    layout = CachedLabelForm(parent)
    field = QLineEdit(parent)
    layout.addRow('Live label', field)
    retired = QLabel('Retired cached label')
    shiboken6.delete(retired)
    layout.retired_label = retired
    try:
        assert i18n._a11y_label_beside(field) is None
        assert shiboken6.isValid(field)
        field.setText('Preserved value')
        assert field.text() == 'Preserved value'
        assert layout.itemAt(0, QFormLayout.LabelRole).widget().text() == 'Live label'
    finally:
        shiboken6.delete(parent)


def test_dialog_retirement_during_its_queued_translation_is_not_stamped(qapp, monkeypatch):
    dialog = QDialog()
    dialog.setProperty(i18n.TRANSLATE_ONCE_PER_LANGUAGE, True)
    dialog.show()
    translated = []

    def retire(root):
        translated.append(root)
        shiboken6.delete(root)

    monkeypatch.setattr(i18n, 'retranslate_widget_tree', retire)
    i18n._translate_shown_dialog(dialog)
    assert translated == [dialog]
    assert not shiboken6.isValid(dialog)



def test_message_translation_skips_a_button_retired_after_the_button_snapshot(qapp):
    from PySide6.QtWidgets import QMessageBox, QPushButton

    class CachedButtonBox(QMessageBox):
        def buttons(self):
            return super().buttons() + [self.retired_button]

    box = CachedButtonBox()
    box.setText('Cancel')
    box.setStandardButtons(QMessageBox.Yes)
    live = box.button(QMessageBox.Yes)
    retired = QPushButton('Retired cached action')
    shiboken6.delete(retired)
    box.retired_button = retired
    try:
        i18n.retranslate_widget_tree(box, 'sv')
        assert box.text() == 'Avbryt'
        assert not shiboken6.isValid(retired)
        assert shiboken6.isValid(live) and live in box.buttons()
    finally:
        shiboken6.delete(box)
