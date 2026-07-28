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


# ---------------------------------------------------------------------------
# Display channel order
# ---------------------------------------------------------------------------

def _slot_values(img):
    """Return the three channel values of a uniform test image."""
    import numpy as np
    arr = np.asarray(img)
    return [int(arr[0, 0, i]) for i in range(3)]


def _stain_image(first=10, second=20, third=30):
    """A uniform crop whose three channels are individually identifiable."""
    import numpy as np
    from PIL import Image
    arr = np.zeros((4, 4, 3), "uint8")
    arr[..., 0], arr[..., 1], arr[..., 2] = first, second, third
    return Image.fromarray(arr)


def test_display_order_default_is_identity():
    """An untouched project must render exactly as it did before the setting
    existed -- the identity permutation returns the same image object."""
    from spacr.qt.annotate_engine import (
        AnnotateSettings, DEFAULT_DISPLAY_ORDER, reorder_channels_pil)
    s = AnnotateSettings()
    assert s.display_order == ["r", "g", "b"]
    assert tuple(s.display_order) == DEFAULT_DISPLAY_ORDER
    img = _stain_image()
    assert reorder_channels_pil(img, s.display_order) is img
    assert _slot_values(reorder_channels_pil(img, s.display_order)) == [10, 20, 30]


def test_display_order_bgr_swaps_red_and_blue():
    """b,g,r draws the source's blue channel as red. This is what a project
    whose png_dims predates the 341f446 crop-format correction needs to get
    its first stain back into the red slot."""
    from spacr.qt.annotate_engine import reorder_channels_pil
    out = reorder_channels_pil(_stain_image(), ["b", "g", "r"])
    assert _slot_values(out) == [30, 20, 10]


def test_display_order_applies_before_channel_filter():
    """The permutation has to run first, or "Show channels" would name slots
    the user isn't looking at. With b,g,r selecting r,g must light up the
    source's third and second channels."""
    from spacr.qt.annotate_engine import filter_channels_pil, reorder_channels_pil
    out = filter_channels_pil(
        reorder_channels_pil(_stain_image(), ["b", "g", "r"]), ["r", "g"])
    assert _slot_values(out) == [30, 20, 0]


def test_display_order_supports_arbitrary_permutation():
    """Not just the reversal -- any of the six orders is expressible."""
    from spacr.qt.annotate_engine import reorder_channels_pil
    assert _slot_values(reorder_channels_pil(_stain_image(), ["g", "b", "r"])) \
        == [20, 30, 10]


def test_display_order_ignores_malformed_input():
    """A half-typed or nonsense order must leave the image alone rather than
    blank the screen mid-edit."""
    from spacr.qt.annotate_engine import reorder_channels_pil
    img = _stain_image()
    for bad in (None, [], ["r", "g"], ["x", "y", "z"], ["r", "g", "b", "r"], ["", " "]):
        assert _slot_values(reorder_channels_pil(img, bad)) == [10, 20, 30]


def test_display_order_is_case_and_space_insensitive():
    """Typed settings arrive with whatever spacing and case the user used."""
    from spacr.qt.annotate_engine import reorder_channels_pil
    out = reorder_channels_pil(_stain_image(), [" B ", "G", "r"])
    assert _slot_values(out) == [30, 20, 10]


def test_display_order_leaves_non_rgb_untouched():
    """Only an RGB image has three slots to permute."""
    from PIL import Image
    from spacr.qt.annotate_engine import reorder_channels_pil
    grey = Image.new("L", (4, 4), 12)
    assert reorder_channels_pil(grey, ["b", "g", "r"]) is grey
