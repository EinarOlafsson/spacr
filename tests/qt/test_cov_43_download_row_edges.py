"""The Download row's alignment when the screen is not built the usual way.

A screen missing one of the four buttons, or holding them in two different
rows, is left alone; a refit is skipped while one is running or before any
column was aligned; and neither a failed alignment on show nor a failed refit
reaches the user.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (QPushButton, QVBoxLayout,        # noqa: E402
                               QWidget)

from spacr.qt.widgets import file_list as fl                     # noqa: E402

pytestmark = pytest.mark.qt


def _host(qtbot, *, names=None, split=False):
    host = QWidget()
    qtbot.addWidget(host)
    column = QVBoxLayout(host)
    rows = [QWidget(host), QWidget(host)]
    for index, (name, _column) in enumerate(fl.PairedFileTableWidget.DOWNLOAD_BUTTONS):
        if names is not None and name not in names:
            continue
        row = rows[1] if split and index == 1 else rows[0]
        setattr(host, name, QPushButton(name, row))
    table = fl.PairedFileTableWidget(parent=host)
    column.addWidget(table)
    return host, table


def test_a_screen_missing_a_download_button_is_left_alone(qtbot):
    first = fl.PairedFileTableWidget.DOWNLOAD_BUTTONS[0][0]
    _host_widget, table = _host(qtbot, names={first})
    assert table.align_download_buttons() is False


def test_buttons_in_two_different_rows_are_left_alone(qtbot):
    _host_widget, table = _host(qtbot, split=True)
    assert table.align_download_buttons() is False


def test_a_refit_is_skipped_while_running_or_before_any_alignment(qtbot,
                                                                  monkeypatch):
    table = fl.PairedFileTableWidget()
    qtbot.addWidget(table)
    widened = []
    monkeypatch.setattr(table, "_widen_columns_for", widened.append)
    table._refit_download_columns()
    table._download_columns = [(QPushButton("x"), 1)]
    table._refitting = True
    table._refit_download_columns()
    assert widened == []
    table._refitting = False
    table._refit_download_columns()
    assert len(widened) == 1


def test_failures_while_aligning_or_refitting_are_only_logged(qtbot,
                                                              monkeypatch,
                                                              caplog):
    table = fl.PairedFileTableWidget()
    qtbot.addWidget(table)

    def boom(*args, **kwargs):
        raise RuntimeError("the header went away")

    monkeypatch.setattr(table, "align_download_buttons", boom)
    monkeypatch.setattr(table, "_widen_columns_for", boom)
    table._download_columns = [(QPushButton("x"), 1)]
    with caplog.at_level(logging.DEBUG, logger=fl.LOG.name):
        table.show()
        table._refit_download_columns()
    assert "could not align the Download row" in caplog.text
    assert "could not re-fit the Download columns" in caplog.text
    assert table._refitting is False
