"""Tests for the threshold-filter fetch in annotate_engine."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from spacr.qt import annotate_engine as engine


@pytest.fixture
def db_with_measurements(tmp_path: Path) -> tuple[str, list[str]]:
    """A `measurements/measurements.db` where png_list + cell tables
    share a `prcfo` join key. Some rows have cell_area > 500, some < 500;
    the threshold filter should split them cleanly.
    """
    src = tmp_path / "expt"
    (src / "measurements").mkdir(parents=True)
    (src / "images").mkdir(parents=True)
    png_paths = []
    rng = np.random.default_rng(0)
    for i in range(6):
        p = src / "images" / f"cell_{i:02d}.png"
        arr = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(p)
        png_paths.append(str(p))
    prcfos = [f"plate1_A01_{i}_{i}" for i in range(6)]
    db = src / "measurements" / "measurements.db"
    conn = sqlite3.connect(db)
    try:
        # png_list — prcfo + png_path
        pd.DataFrame({"prcfo": prcfos, "png_path": png_paths}) \
            .to_sql("png_list", conn, index=False)
        # cell — prcfo + cell_area
        cell = pd.DataFrame({
            "prcfo": prcfos,
            "cell_area": [100, 200, 800, 1500, 300, 900],
        })
        cell.to_sql("cell", conn, index=False)
    finally:
        conn.close()
    return str(db), png_paths


def test_apply_threshold_higher_and_lower():
    df = pd.DataFrame({"x": [1, 5, 10]})
    assert list(engine._apply_threshold(df, "x", 5, "higher")["x"]) == [10]
    assert list(engine._apply_threshold(df, "x", 5, "lower")["x"]) == [1]
    # No-op branches
    assert engine._apply_threshold(df, "missing", 5, "higher").equals(df)
    assert engine._apply_threshold(df, "x", None, "higher").equals(df)


def test_fetch_filtered_paths_higher_threshold(db_with_measurements, monkeypatch):
    """fetch_filtered_paths delegates the DB join to spacr.io helpers.
    We monkey-patch those helpers to return a compact DataFrame so this
    test doesn't need the full spacr measurement schema."""
    db, all_paths = db_with_measurements
    engine.ensure_annotation_column(db, "annotate")

    def _fake_read_and_join_tables(_db):
        return pd.DataFrame({
            "prcfo": [f"plate1_A01_{i}_{i}" for i in range(6)],
            "png_path": all_paths,
            "cell_area": [100, 200, 800, 1500, 300, 900],
        })

    def _fake_read_db(_db, tables=None):
        return [pd.DataFrame({
            "prcfo": [f"plate1_A01_{i}_{i}" for i in range(6)],
            "png_path": all_paths,
        })]

    import spacr.io as spacr_io
    monkeypatch.setattr(spacr_io, "_read_and_join_tables", _fake_read_and_join_tables)
    monkeypatch.setattr(spacr_io, "_read_db", _fake_read_db)

    rows = engine.fetch_filtered_paths(
        db_path=db,
        annotation_column="annotate",
        measurements=["cell_area"],
        thresholds=[500.0],
        directions=["higher"],
    )
    kept_paths = {r[0] for r in rows}
    # cell_area values [100, 200, 800, 1500, 300, 900] — indices 2,3,5 > 500
    expected = {all_paths[2], all_paths[3], all_paths[5]}
    assert kept_paths == expected


def test_fetch_filtered_paths_empty_when_no_filter(db_with_measurements):
    db, _ = db_with_measurements
    rows = engine.fetch_filtered_paths(
        db_path=db,
        annotation_column="annotate",
        measurements=[],
        thresholds=[],
        directions=[],
    )
    assert rows == []


def test_fetch_filtered_paths_missing_db_returns_empty(tmp_path: Path):
    rows = engine.fetch_filtered_paths(
        db_path=str(tmp_path / "does-not-exist.db"),
        annotation_column="annotate",
        measurements=["cell_area"],
        thresholds=[500.0],
        directions=["higher"],
    )
    assert rows == []


def test_default_channels_are_rgb_and_normalized():
    """Object crops must be visible out of the box: show + normalise R,G,B by
    default so a dim/unnormalised crop doesn't render as a grey square."""
    import numpy as np
    from PIL import Image
    from spacr.qt.annotate_engine import (
        AnnotateSettings, normalize_pil, filter_channels_pil)
    s = AnnotateSettings()
    assert s.channels == ["r", "g", "b"]
    assert s.normalize_channels == ["r", "g", "b"]
    dim = Image.fromarray(
        np.random.RandomState(0).randint(0, 30, (16, 16, 3)).astype("uint8"))
    out = filter_channels_pil(
        normalize_pil(dim, s.percentiles, s.normalize_channels), s.channels)
    assert np.array(out).max() > 200   # stretched to a visible range
