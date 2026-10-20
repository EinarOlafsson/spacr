"""Sortable table cells decide "missing" and "text role" without pandas.

A results table fills one cell per value, 22,770 of them for lesson 13's
coefficients, and every cell asked ``pandas.isna`` twice and compared its
data role with Qt's enum members, which together cost a few microseconds a
cell. Numbers are now answered directly and the role is compared by
identity; the answers, and when the sort key is re-read, are unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from spacr.qt.widgets import sortable_table as S

pytestmark = pytest.mark.qt

NUMBERS = [0.5, float("nan"), float("inf"), np.float64(2.5),
           np.float64("nan"), 3, True, np.int64(4), np.uint8(1),
           np.bool_(False), np.float32(1.5)]


def test_numbers_do_not_ask_pandas(monkeypatch):
    asked = []
    real = pd.isna
    monkeypatch.setattr(pd, "isna",
                        lambda value: (asked.append(value), real(value))[1])

    answers = [S.is_missing(value) for value in NUMBERS]

    assert answers == [bool(real(value)) for value in NUMBERS]
    assert not asked, asked


@pytest.mark.parametrize("value", [None, "", "  ", "x", pd.NA, pd.NaT,
                                   np.datetime64("NaT"),
                                   np.array(np.nan), np.array(3)])
def test_everything_else_answers_as_pandas_does(value):
    expected = True if value is None else (
        not value.strip() if isinstance(value, str) else bool(pd.isna(value)))

    assert S.is_missing(value) is expected


@pytest.mark.parametrize("role", [Qt.DisplayRole, Qt.EditRole, 0, 2])
def test_a_text_role_still_rereads_the_sort_key(qapp, role):
    item = S.table_item("5")
    item.setData(role, "12")

    assert item._resolve_key() == (False, 12.0)


@pytest.mark.parametrize("role", [Qt.UserRole, Qt.ToolTipRole, 3, 256])
def test_another_role_leaves_the_sort_key(qapp, role):
    item = S.table_item("5")
    serial = item._sort_serial
    item.setData(role, "12")

    assert item._resolve_key() == (False, 5.0)
    assert item._sort_serial == serial
    assert not math.isnan(item._sort_number)
