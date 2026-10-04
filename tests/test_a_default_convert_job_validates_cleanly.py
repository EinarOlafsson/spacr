"""A Batch Runner convert job built from convert's own defaults validates with no warnings.

Convert keeps its defaults in :func:`spacr.convert.default_settings`, not in
:mod:`spacr.settings`, so ``map_name`` and ``preview_rows`` were once reported
as "not a setting spaCR knows" by the queue's Validate step.
"""
from __future__ import annotations

from spacr import batch as bt
from spacr import cli
from spacr.validate import validate_settings


def _convert_defaults(tmp_path):
    settings = cli.module_defaults(cli.MODULES["convert"])
    src = tmp_path / "plate"
    src.mkdir()
    (src / "plate_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"")
    settings["src"] = str(src)
    return settings


def test_a_default_convert_job_validates_with_no_warnings(tmp_path):
    settings = _convert_defaults(tmp_path)
    assert {"map_name", "preview_rows"} <= set(settings)
    job = bt.Job(module="convert", settings=settings, label="convert")
    problems = bt.validate_job(job)
    assert problems == [], [p.message for p in problems]


def test_convert_default_keys_are_known_but_a_stranger_is_still_flagged(tmp_path):
    settings = _convert_defaults(tmp_path)
    settings["not_a_real_convert_key"] = 3
    messages = [p.message for p in validate_settings(settings, "convert")]
    assert not any("map_name" in m or "preview_rows" in m for m in messages)
    assert any("not_a_real_convert_key" in m for m in messages)
