"""i18n assistive-naming fallbacks and portable-mode edges."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QLabel, QLineEdit, QWidget  # noqa: E402

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
