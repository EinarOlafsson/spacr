"""A prediction file may spell its crop-path column the way spaCR does.

``_read_cells`` joins predictions to objects through ``png_list`` when the
prediction file carries no ``prcfo``. Both frames name their path column, and
an export produced by spaCR itself names it ``png_path`` -- the same name
``png_list`` uses.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from spacr import hit_investigation as hi


@pytest.fixture
def cells_from(monkeypatch):
    """Let a test supply the object frame ``_read_and_join_tables`` returns."""
    def _install(frame):
        from spacr import io
        monkeypatch.setattr(io, "_read_and_join_tables", lambda *a, **k: frame)
    return _install


def _png_list_db(tmp_path, rows):
    database = tmp_path / "measurements.db"
    with sqlite3.connect(database) as connection:
        pd.DataFrame(rows).to_sql("png_list", connection, index=False)
    return database


def test_a_prediction_file_may_spell_its_crop_column_png_path(tmp_path,
                                                              cells_from):
    """Merging the two frames whole suffixed ``png_path`` to ``_x``/``_y``,
    and the join that follows asked for a column that no longer existed --
    so the whole crop-name route died on a ``KeyError`` for the one column
    name a spaCR user is most likely to have."""
    cells_from(pd.DataFrame({"prcfo": ["a", "b"], "cell_area": [1.0, 2.0]}))
    database = _png_list_db(tmp_path, [
        {"prcfo": "a", "png_path": "/old/crops/a.png"},
        {"prcfo": "b", "png_path": "/old/crops/b.png"},
    ])
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"png_path": ["/moved/a.png", "/moved/b.png"],
                  "score": [0.25, 0.75]}).to_csv(predictions, index=False)

    out = hi._read_cells(str(database), str(predictions), "score", "png_path")

    assert list(out["png_path"]) == ["/old/crops/a.png", "/old/crops/b.png"]
    assert list(out["score"]) == [0.25, 0.75]
    assert list(out["cell_area"]) == [1.0, 2.0]


def test_the_join_still_carries_a_differently_named_path_column(tmp_path,
                                                                cells_from):
    """The default spelling is ``path``, and it must keep working: the
    result's ``png_path`` is png_list's recorded location, not the path the
    classifier happened to read the crop from."""
    cells_from(pd.DataFrame({"prcfo": ["a"], "cell_area": [1.0]}))
    database = _png_list_db(tmp_path, [
        {"prcfo": "a", "png_path": "/old/crops/a.png"}])
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame({"path": ["/elsewhere/a.png"], "score": [0.5]}).to_csv(
        predictions, index=False)

    out = hi._read_cells(str(database), str(predictions), "score", "path")

    assert list(out["png_path"]) == ["/old/crops/a.png"]
    assert list(out["score"]) == [0.5]
