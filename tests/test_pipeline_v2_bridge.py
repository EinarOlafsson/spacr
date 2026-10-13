"""Tests for spacr._v1_v2_bridge — settings translation + disk reporter."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# v2_channels_from_settings
# ---------------------------------------------------------------------------

def test_settings_translation_picks_all_four_channels():
    from spacr._v1_v2_bridge import v2_channels_from_settings
    s = {"nucleus_channel": 0, "cell_channel": 1,
         "pathogen_channel": 2, "organelle_channel": 3}
    chans, names = v2_channels_from_settings(s)
    assert chans == [0, 1, 2, 3]
    assert names == ["nucleus", "cell", "pathogen", "organelle"]


def test_settings_translation_skips_none_channels():
    from spacr._v1_v2_bridge import v2_channels_from_settings
    s = {"cell_channel": 1, "nucleus_channel": None,
         "pathogen_channel": 2}
    chans, names = v2_channels_from_settings(s)
    # Order is fixed (nucleus, cell, pathogen, organelle) — nucleus
    # is skipped, so we get cell then pathogen.
    assert chans == [1, 2]
    assert names == ["cell", "pathogen"]


def test_settings_translation_falls_back_to_channels_list():
    from spacr._v1_v2_bridge import v2_channels_from_settings
    s = {"channels": [0, 2, 3]}
    chans, names = v2_channels_from_settings(s)
    assert chans == [0, 2, 3]
    assert names == ["ch0", "ch1", "ch2"]


def test_settings_translation_defaults_when_empty():
    from spacr._v1_v2_bridge import v2_channels_from_settings
    chans, names = v2_channels_from_settings({})
    assert chans == [0, 1, 2, 3]
    assert names == ["ch0", "ch1", "ch2", "ch3"]


def test_settings_translation_ignores_non_int_values():
    from spacr._v1_v2_bridge import v2_channels_from_settings
    s = {"nucleus_channel": "bogus", "cell_channel": 1}
    chans, names = v2_channels_from_settings(s)
    assert chans == [1]
    assert names == ["cell"]


# ---------------------------------------------------------------------------
# report_disk_savings
# ---------------------------------------------------------------------------

def test_report_disk_savings_sums_stack_bytes(tmp_path, caplog):
    from spacr._v1_v2_bridge import report_disk_savings
    from spacr.pipeline_v2 import StackFile

    # Write two fake stack files with known sizes
    (tmp_path / "merged").mkdir()
    p1 = tmp_path / "merged" / "stack_0000.npy"
    p2 = tmp_path / "merged" / "stack_0001.npy"
    np.save(p1, np.zeros((100, 100, 4), dtype=np.uint16))
    np.save(p2, np.zeros((100, 100, 4), dtype=np.uint16))

    stacks = [
        StackFile(field_id="0000", path=p1, shape=(100, 100, 4),
                    channels=["a"]),
        StackFile(field_id="0001", path=p2, shape=(100, 100, 4),
                    channels=["a"]),
    ]

    with caplog.at_level(logging.INFO, logger="spacr.pipeline_v2.bridge"):
        result = report_disk_savings(tmp_path, stacks)

    assert result["v2_bytes"] > 0
    assert result["v1_estimated_bytes"] == result["v2_bytes"] * 4
    assert 70 <= result["saved_pct"] <= 80    # 3/4 = 75%
    # Log message mentions "Saved:"
    assert any("Saved" in r.message for r in caplog.records)


def test_report_disk_savings_includes_sidecars(tmp_path):
    from spacr._v1_v2_bridge import report_disk_savings
    from spacr.pipeline_v2 import StackFile

    (tmp_path / "merged").mkdir()
    p = tmp_path / "merged" / "stack_0000.npy"
    np.save(p, np.zeros((10, 10, 2), dtype=np.uint16))
    stack = StackFile(field_id="0000", path=p, shape=(10, 10, 2),
                        channels=["a", "b"])

    # No sidecars — baseline
    baseline = report_disk_savings(tmp_path, [stack])["v2_bytes"]

    # Add sidecars — should bump the v2 count
    (tmp_path / "filename_map.csv").write_text("x" * 1000)
    (tmp_path / "filename_map.json").write_text("y" * 500)
    (tmp_path / "merged" / "channel_order.json").write_text("z" * 300)

    bumped = report_disk_savings(tmp_path, [stack])["v2_bytes"]
    assert bumped - baseline >= 1800


def test_report_disk_savings_handles_empty_stack_list():
    """No crash + finite saved_pct even when nothing was written."""
    from spacr._v1_v2_bridge import report_disk_savings
    result = report_disk_savings(Path("/tmp/nowhere"), [])
    assert result["v2_bytes"] == 0
    assert result["v1_estimated_bytes"] == 0


# ---------------------------------------------------------------------------
# _human
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected_unit", [
    (500, "B"),
    (5_000, "KB"),
    (5_000_000, "MB"),
    (5_000_000_000, "GB"),
    (5_000_000_000_000, "TB"),
])
def test_human_readable_bytes(n, expected_unit):
    from spacr._v1_v2_bridge import _human
    assert expected_unit in _human(n)
