"""i18n assistive-naming fallbacks and portable-mode edges."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QObject  # noqa: E402
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QLineEdit,  # noqa: E402
                               QPushButton, QVBoxLayout, QWidget)

from spacr import logging_util as lu  # noqa: E402
from spacr.qt import i18n  # noqa: E402


def test_a_composite_with_an_empty_side_is_left_alone():
    assert i18n._composite_translation(
        f"Organelle 1{i18n._COMPOSITE_SEPARATOR} ", "sv") is None


def test_effective_names_fall_back_to_the_widget(qtbot, monkeypatch):
    from PySide6.QtGui import QAccessible

    edit = QLineEdit()
    qtbot.addWidget(edit)
    edit.setAccessibleName("plate")
    monkeypatch.setattr(QAccessible, "queryAccessibleInterface",
                        lambda widget: (_ for _ in ()).throw(RuntimeError()))
    assert i18n._a11y_effective_name(edit) == "plate"
    monkeypatch.setattr(QAccessible, "queryAccessibleInterface", lambda widget: None)
    assert i18n._a11y_effective_name(edit) == "plate"


def test_a_deleted_ancestor_ends_the_setting_label_search():
    class _Gone:
        def property(self, name):
            raise RuntimeError("deleted")

    assert i18n._a11y_setting_label(_Gone(), {}) is None


def test_naming_without_widgets_types_does_nothing(monkeypatch):
    monkeypatch.setattr(i18n, "_a11y_interactive_types",
                        lambda: (_ for _ in ()).throw(ImportError()))
    assert i18n._name_for_assistive_tech([]) == 0


def test_deleted_widgets_are_skipped_while_naming(qtbot):
    class _Gone(QLabel):
        def property(self, name):
            raise RuntimeError("deleted")

    label = _Gone()
    qtbot.addWidget(label)
    assert i18n._name_for_assistive_tech([label]) == 0


def test_retranslating_skips_widgets_that_break(qtbot):
    class _Broken(QWidget):
        def retranslate_dynamic_content(self, code):
            raise RuntimeError("half built")

        def setProperty(self, name, value):
            raise RuntimeError("deleted")

    widget = _Broken()
    qtbot.addWidget(widget)
    i18n.retranslate_widget_tree(widget)
    assert widget is not None


def test_app_folders_are_listed_once(monkeypatch):
    monkeypatch.setenv("SPACR_LAUNCHER_DIR", str(Path(sys.prefix)))
    folders = lu._app_folders()
    assert len(folders) == len(set(folders))


def test_an_unreadable_folder_is_not_portable_and_data_may_be_unwritable(
        monkeypatch, tmp_path):
    real = Path.is_file

    def refuse(self):
        if self.name == lu._PORTABLE_MARKER:
            raise OSError("permission denied")
        return real(self)

    lu._portable_root_for.cache_clear()
    monkeypatch.setattr(Path, "is_file", refuse)
    assert lu._portable_root_for("", "x-unreadable") is None
    lu._portable_root_for.cache_clear()
    monkeypatch.undo()
    monkeypatch.setattr(lu, "_portable_root", lambda: tmp_path)
    names = ("SPACR_HOME", "SPACR_LOG_DIR", "SPACR_BACKENDS_DIR",
             "SPACR_PLUGIN_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME",
             "TORCH_HOME", "HF_HOME", "MPLCONFIGDIR",
             "CELLPOSE_LOCAL_MODELS_PATH")
    for name in names:
        monkeypatch.delenv(name, raising=False)

    def refuse_mkdir(self, *a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(Path, "mkdir", refuse_mkdir)
    try:
        data = lu._apply_portable_mode()
        assert data == tmp_path / lu._PORTABLE_DATA
    finally:
        for name in names:
            os.environ.pop(name, None)
    assert os


def test_a_new_only_pass_skips_widgets_it_cannot_read(qtbot):
    class _Unreadable(QWidget):
        def property(self, name):
            raise RuntimeError("deleted")

    root = QWidget()
    qtbot.addWidget(root)
    _Unreadable(root)
    edit = QLineEdit(root)
    i18n.retranslate_widget_tree(root, only_new=True)
    assert edit is not None


def test_the_layout_search_steps_over_children_that_are_not_layouts(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    outer = QVBoxLayout(host)
    bystander = QObject(outer)            # a child of the layout that is no layout
    inner = QHBoxLayout()
    outer.addLayout(inner)
    button = QPushButton("Go", host)
    inner.addWidget(button)

    assert bystander.parent() is outer
    assert i18n._a11y_layout_holding(outer, button) is inner
    assert i18n._a11y_layout_holding(outer, QPushButton("elsewhere")) is None


def test_a_layout_with_no_owner_holds_no_widgets():
    assert i18n._a11y_widgets_laid_out_by(QHBoxLayout()) == []


def test_a_layout_lists_the_widgets_it_holds_in_item_order(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    row = QHBoxLayout(host)
    first, second = QPushButton("a", host), QPushButton("b", host)
    row.addWidget(first)
    row.addWidget(second)

    assert i18n._a11y_widgets_laid_out_by(row) == [(0, first), (1, second)]


def test_a_child_whose_children_cannot_be_listed_is_still_collected(
        qtbot, monkeypatch):
    class _Unlistable(QWidget):
        def children(self):
            raise RuntimeError("deleted while walking")

    root = QWidget()
    qtbot.addWidget(root)
    bad = _Unlistable(root)
    good = QPushButton("ok", root)
    monkeypatch.setattr(i18n, "_a11y_holds_a_system_dialog", lambda widget: True)

    found = i18n._a11y_spacr_tree(root, QWidget)

    assert root in found and bad in found and good in found


def test_a_control_whose_caption_cannot_be_read_is_skipped_and_the_rest_named(
        qtbot, monkeypatch):
    root = QWidget()
    qtbot.addWidget(root)
    broken, fine = QLineEdit(root), QLineEdit(root)
    fine.setPlaceholderText("Plate")
    real = i18n._a11y_source

    def _source(widget, labels_by_key):
        if widget is broken:
            raise TypeError("caption is not text")
        return real(widget, labels_by_key)

    monkeypatch.setattr(i18n, "_a11y_source", _source)

    changed = i18n._name_for_assistive_tech([broken, fine])

    assert changed >= 1
    assert broken.accessibleName() == ""
    assert fine.accessibleName() == "Plate"
