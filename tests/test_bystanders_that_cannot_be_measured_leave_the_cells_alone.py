"""When the bystander split cannot be made -- the classifier fails, or it
answers without a neighbourhood -- the cell table is written without the
bystander columns and the run says why, rather than failing the field."""
from __future__ import annotations

import pandas as pd

from spacr import bystanders
from tests.test_a_bystander_column_reaches_the_measurement_table import (
    COLUMNS, _cells, _field)


def test_a_failed_classification_is_reported_and_leaves_no_columns(
        monkeypatch, capsys):
    def refuse(*_a, **_k):
        raise ValueError("labels are not contiguous")

    monkeypatch.setattr(bystanders, "classify", refuse)
    cells = _cells(*_field())
    assert not set(COLUMNS) & set(cells.columns)
    assert len(cells) == 4
    assert ("bystanders were not measured: ValueError: labels are not "
            "contiguous") in capsys.readouterr().out


def test_an_answer_without_a_neighbourhood_adds_nothing(monkeypatch):
    monkeypatch.setattr(bystanders, "classify", lambda *a, **k: pd.DataFrame(
        {"label": [1, 2, 3, 4]}))
    cells = _cells(*_field())
    assert not set(COLUMNS) & set(cells.columns)
    assert sorted(cells["label"]) == [1, 2, 3, 4]
