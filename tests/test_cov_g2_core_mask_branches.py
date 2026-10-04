"""Mask pipeline branches: cloud sources, robustness, events and relationships."""
from __future__ import annotations

import sqlite3

import pytest

from tests.test_core_mask_orchestration import (_mask_settings, run_dir,  # noqa: F401
                                                stubs)


def test_a_cloud_source_is_handed_to_the_cloud_runner(monkeypatch, tmp_path):
    import spacr.ome_zarr as oz
    from spacr.core import preprocess_generate_masks

    monkeypatch.setattr(oz, "_needs_cloud_run", lambda settings: True)
    monkeypatch.setattr(oz, "_run_with_cloud_sources",
                        lambda fn, settings, module: ("cloud", module))
    assert preprocess_generate_masks({"src": "s3://bucket/plate"}) == ("cloud", "mask")


def test_v2_says_mask_parallel_does_not_apply(monkeypatch, tmp_path, capsys):
    import spacr.pipeline_v2 as v2
    from spacr.core import preprocess_generate_masks

    def stop(*a, **k):
        raise RuntimeError("stop after the notice")

    monkeypatch.setattr(v2, "run_v2", stop)
    with pytest.raises(RuntimeError, match="stop after the notice"):
        preprocess_generate_masks(_mask_settings(
            tmp_path, pipeline_style="v2", mask_parallel=True))
    assert "mask_parallel applies to the v1" in capsys.readouterr().out


def test_robustness_reports_cover_every_object_channel(run_dir, stubs,  # noqa: F811
                                                       monkeypatch):
    import spacr.object as sobj
    from spacr.core import preprocess_generate_masks

    roles = []
    monkeypatch.setattr(sobj, "_run_robustness_report",
                        lambda src, settings, role: roles.append(role))
    preprocess_generate_masks(_mask_settings(run_dir, robustness_report=True))
    assert roles == ["cell", "nucleus"]


def test_relationship_failures_are_warnings(run_dir, stubs, monkeypatch,  # noqa: F811
                                            capsys):
    import spacr.filters as filters
    from spacr.core import preprocess_generate_masks

    def broken(*a, **k):
        raise ValueError("no tables")

    meas = run_dir / "measurements"
    meas.mkdir()
    sqlite3.connect(str(meas / "measurements.db")).close()
    monkeypatch.setattr(filters, "_write_object_relationships", broken)
    monkeypatch.setattr(filters, "object_tables", lambda db: ["cell"])
    monkeypatch.setattr(filters, "write_relationships", broken)
    preprocess_generate_masks(_mask_settings(run_dir))
    out = capsys.readouterr().out
    assert "could not write the object relationships table" in out
    assert "could not write the relationships table" in out


def test_relationships_are_written_when_measure_left_object_tables(
        run_dir, stubs, monkeypatch):  # noqa: F811
    import spacr.filters as filters
    from spacr.core import preprocess_generate_masks

    written = []
    meas = run_dir / "measurements"
    meas.mkdir()
    sqlite3.connect(str(meas / "measurements.db")).close()
    monkeypatch.setattr(filters, "object_tables", lambda db: ["cell"])
    monkeypatch.setattr(filters, "write_relationships", written.append)
    preprocess_generate_masks(_mask_settings(run_dir))
    assert written


def test_timelapse_events_are_detected_after_the_merge(run_dir, stubs,  # noqa: F811
                                                       monkeypatch):
    from spacr import core, timelapse

    detected = []
    monkeypatch.setattr(timelapse, "_run_event_detection_step",
                        lambda src, settings: detected.append(src))
    core.preprocess_generate_masks(_mask_settings(
        run_dir, timelapse=True, timelapse_events=True))
    assert detected == [str(run_dir)]
