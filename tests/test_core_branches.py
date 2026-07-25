"""Fast branch coverage for spacr.core.preprocess_generate_masks.

These exercise the guard clauses, dispatch and setting-normalisation paths
that run before (or instead of) the expensive Cellpose segmentation, so
they stay CPU-only and CI-friendly. The full segmentation path is covered
by the slow/gpu e2e suite (tests/test_pipeline_e2e.py).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)


def _base_settings(src, **over):
    s = {
        "src": str(src),
        "metadata_type": "cellvoyager",
        "channels": [0, 1, 2],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": None,
        "organelle_channel": None,
        "preprocess": False, "masks": False, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False, "all_to_mip": False,
        "batch_size": 10, "save": True, "pathogen_model": None,
        "custom_regex": None, "randomize": True,
    }
    s.update(over)
    return s


def test_requires_at_least_one_channel(tmp_path, capsys):
    """All object channels None → prints an error and returns without work."""
    from spacr.core import preprocess_generate_masks
    src = tmp_path / "plate1"; src.mkdir()
    out = preprocess_generate_masks(_base_settings(
        src, cell_channel=None, nucleus_channel=None,
        pathogen_channel=None, organelle_channel=None))
    assert out is None
    printed = capsys.readouterr().out
    assert "at least one of" in printed.lower()


def test_save_bool_is_expanded_to_list(tmp_path):
    """settings['save']=True is normalised to a 3-element list."""
    from spacr.core import preprocess_generate_masks
    src = tmp_path / "plate1"; src.mkdir()
    s = _base_settings(src, save=True)
    preprocess_generate_masks(s)
    assert isinstance(s["save"], list) and len(s["save"]) == 3


def test_timelapse_disables_randomize(tmp_path):
    from spacr.core import preprocess_generate_masks
    src = tmp_path / "plate1"; src.mkdir()
    s = _base_settings(src, timelapse=True, randomize=True)
    preprocess_generate_masks(s)
    assert s["randomize"] is False


def test_settings_written_to_settings_folder(tmp_path):
    """save_settings writes gen_mask_settings.csv under settings/."""
    from spacr.core import preprocess_generate_masks
    src = tmp_path / "plate1"; src.mkdir()
    preprocess_generate_masks(_base_settings(src))
    assert os.path.isdir(src / "settings")


def test_verbose_pretty_prints_settings(tmp_path, capsys):
    from spacr.core import preprocess_generate_masks
    src = tmp_path / "plate1"; src.mkdir()
    preprocess_generate_masks(_base_settings(src, verbose=True))
    assert "Mask Generation Settings" in capsys.readouterr().out


def test_test_mode_announced(tmp_path, capsys):
    from spacr.core import preprocess_generate_masks
    src = tmp_path / "plate1"; src.mkdir()
    preprocess_generate_masks(_base_settings(src, test_mode=True))
    assert "Test mode" in capsys.readouterr().out


def test_preprocess_without_masks_warns(tmp_path, capsys, monkeypatch):
    """preprocess=True + masks=False emits the mismatch warning."""
    import spacr.core as core
    import spacr.io as sio
    monkeypatch.setattr(sio, "preprocess_img_data",
                        lambda s: (s, s["src"]), raising=True)
    src = tmp_path / "plate1"; src.mkdir()
    core.preprocess_generate_masks(
        _base_settings(src, preprocess=True, masks=False))
    assert "preprocess = True" in capsys.readouterr().out


def test_src_accepts_a_list_of_folders(tmp_path):
    from spacr.core import preprocess_generate_masks
    a = tmp_path / "p1"; a.mkdir()
    b = tmp_path / "p2"; b.mkdir()
    s = _base_settings(a)
    s["src"] = [str(a), str(b)]
    preprocess_generate_masks(s)
    assert (a / "settings").is_dir() and (b / "settings").is_dir()


def test_consolidate_moves_images(tmp_path, monkeypatch):
    """consolidate=True redirects src into a consolidated/ subfolder."""
    import spacr.core as core
    src = tmp_path / "plate1"; src.mkdir()
    (src / "consolidated").mkdir()
    seen = {}
    import spacr.utils as su
    monkeypatch.setattr(su, "generate_image_path_map", lambda s: {})
    monkeypatch.setattr(su, "copy_images_to_consolidated",
                        lambda m, s: seen.setdefault("called", True))
    core.preprocess_generate_masks(_base_settings(src, consolidate=True))
    assert seen.get("called") is True


def test_v2_pipeline_dispatch(tmp_path, monkeypatch):
    """pipeline_style='v2' routes to pipeline_v2.run_v2 and returns early."""
    import spacr.pipeline_v2 as pv2
    import spacr._v1_v2_bridge as bridge
    calls = {}
    def _fake_run_v2(*a, **k):
        calls["run"] = True
        return {"stacks": []}
    monkeypatch.setattr(pv2, "run_v2", _fake_run_v2)
    monkeypatch.setattr(bridge, "report_disk_savings", lambda *a, **k: None)
    from spacr.core import preprocess_generate_masks
    src = tmp_path / "plate1"; src.mkdir()
    preprocess_generate_masks(_base_settings(src, pipeline_style="v2"))
    assert calls.get("run") is True


def test_metadata_auto_calls_converter(tmp_path, monkeypatch):
    """metadata_type='auto' without a custom regex calls convert_to_yokogawa."""
    import spacr.core as core
    calls = {}
    import spacr.io as sio
    monkeypatch.setattr(sio, "convert_to_yokogawa",
                        lambda folder: calls.setdefault("plain", folder))
    src = tmp_path / "plate1"; src.mkdir()
    core.preprocess_generate_masks(_base_settings(src, metadata_type="auto"))
    assert "plain" in calls


def test_metadata_auto_with_custom_regex(tmp_path, monkeypatch):
    """A custom regex routes through convert_separate_files_to_yokogawa."""
    import spacr.core as core
    calls = {}
    import spacr.io as sio
    monkeypatch.setattr(sio, "convert_separate_files_to_yokogawa",
                        lambda folder, regex: calls.setdefault("regex", regex))
    src = tmp_path / "plate1"; src.mkdir()
    core.preprocess_generate_masks(_base_settings(
        src, metadata_type="auto", custom_regex=r"(?P<plateID>.*)"))
    assert "regex" in calls


def test_metadata_auto_converter_failure_returns(tmp_path, monkeypatch, capsys):
    """Both converters failing prints an error and aborts the run."""
    import spacr.core as core

    def _boom(*a, **k):
        raise RuntimeError("no converter")
    import spacr.io as sio
    monkeypatch.setattr(sio, "convert_to_yokogawa", _boom)
    monkeypatch.setattr(sio, "convert_separate_files_to_yokogawa", _boom)
    src = tmp_path / "plate1"; src.mkdir()
    out = core.preprocess_generate_masks(
        _base_settings(src, metadata_type="auto"))
    assert out is None
    assert "Error" in capsys.readouterr().out
