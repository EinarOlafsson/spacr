"""Pipelined and staged phenotype fields, and the GPU self-check (item 872, B1, B2, B7, B15).

A synthetic four-field well is segmented by the threshold stand-in for
Cellpose and measured on the CPU, once the way the phases always ran
(segment the whole well, then measure it) and once field by field: with
``ops_ph_pipeline`` (a field is measured while the next is segmented) and
with ``ops_ph_stage_dir`` (each field's projection is staged locally and
the raw files are read once). The masks and tables must be identical, the
raw files opened once per field when staged, and the staging folder never
hold more than ``ops_ph_stage_ahead`` fields.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
tifffile = pytest.importorskip("tifffile")

from spacr import ops_engine                                         # noqa: E402
from spacr.ops_run_infra import FieldStager, compare_tables          # noqa: E402
from tests.test_ops_phenotype_run_v2 import (                        # noqa: E402,F401
    PLATE, SETTINGS, _placed, _write_tiles, models)

SITES = (0, 1, 2, 3)
PHASES = ("ph_segment", "ph_measure")


def _prepared(root):
    """Tiles for four fields and their placement, under ``root``."""
    _write_tiles(root / "tiles", sites=SITES)
    (root / "run").mkdir(parents=True)
    _placed(str(root / "run" / "measurements.db"), sites=SITES)
    return root


def _run(root, **extra):
    """``run_ops`` over the four-field well."""
    settings = {**SETTINGS, "dst_root": str(root / "run"),
                "phenotype_source": str(root / "tiles"), **extra}
    return ops_engine.run_ops(settings, phases=PHASES)


def _outputs(root):
    """Every field's masks and the well's cell table."""
    masks = {}
    for site in SITES:
        with np.load(root / "run" / "A1" / "phenotype_masks"
                     / f"site-{site}.npz") as saved:
            masks[site] = (saved["nuclei"].copy(), saved["cells"].copy())
    table = pd.read_parquet(ops_engine._ph_cells_path(str(root / "run"), "A1"))
    return masks, table


@pytest.fixture
def opens(monkeypatch):
    """Count every TIFF file the field reader opens, by path."""
    counts = {}
    lock = threading.Lock()
    original = tifffile.TiffFile

    class Counting(original):
        """``tifffile.TiffFile`` that counts the paths it is opened on."""

        def __init__(self, path, *args, **kwargs):
            """Count and open."""
            if isinstance(path, (str, os.PathLike)):
                with lock:
                    counts[str(path)] = counts.get(str(path), 0) + 1
            super().__init__(path, *args, **kwargs)

    monkeypatch.setattr(tifffile, "TiffFile", Counting)
    return counts


@pytest.fixture
def sequential(tmp_path, models):
    """The well run the way the phases always ran."""
    root = _prepared(tmp_path / "sequential")
    out = _run(root)
    return root, out


def _same(first, second):
    """Masks and tables of two runs are identical."""
    masks_a, table_a = _outputs(first)
    masks_b, table_b = _outputs(second)
    for site in SITES:
        assert np.array_equal(masks_a[site][0], masks_b[site][0])
        assert np.array_equal(masks_a[site][1], masks_b[site][1])
    pd.testing.assert_frame_equal(table_a, table_b, check_exact=True)
    assert len(table_a) == 16


def test_the_pipelined_run_gives_the_sequential_masks_and_tables(
        tmp_path, sequential):
    root = _prepared(tmp_path / "pipelined")
    out = _run(root, ops_ph_pipeline=True, ops_ph_measure_workers=1)
    _same(sequential[0], root)
    fields = out["wells"]["A1"]["ph_fields"]
    assert fields["ph_stream"]["pipelined"] is True
    assert fields["ph_stream"]["staged"] is False
    assert fields["ph_stream"]["wall_seconds"] >= 0
    assert fields["ph_segment"]["segmented"] == 4
    assert fields["ph_measure"]["measured"] == 4
    assert "ph_stream" not in sequential[1]["wells"]["A1"]["ph_fields"]


def test_staging_reads_each_raw_file_once_and_bounds_the_local_copies(
        tmp_path, sequential, opens):
    root = _prepared(tmp_path / "staged")
    stage = tmp_path / "stage"
    out = _run(root, ops_ph_stage_dir=str(stage), ops_ph_stage_ahead=2,
               ops_ph_stage_threads=2)
    _same(sequential[0], root)
    raw = [p for p in opens if p.startswith(str(root / "tiles"))]
    assert len(raw) == 4 * 3
    assert all(opens[p] == 1 for p in raw)
    stream = out["wells"]["A1"]["ph_fields"]["ph_stream"]
    assert stream["staged"] is True and stream["pipelined"] is False
    assert 1 <= stream["stage_peak_files"] <= 2
    assert stream["stage_peak_bytes"] <= 2 * 64 * 64 * 3 * 4 + 2 * 256
    assert os.listdir(stage / PLATE / "A1") == []


def test_the_unstaged_run_reads_each_raw_file_twice(tmp_path, models, opens):
    root = _prepared(tmp_path / "twice")
    _run(root)
    raw = [p for p in opens if p.startswith(str(root / "tiles"))]
    assert len(raw) == 12 and all(opens[p] == 2 for p in raw)


def test_staging_and_pipelining_together_and_a_measure_only_resume(
        tmp_path, sequential, opens):
    root = _prepared(tmp_path / "both")
    stage = tmp_path / "stage"
    _run(root, ops_ph_stage_dir=str(stage), ops_ph_stage_ahead=1,
         ops_ph_pipeline=True)
    _same(sequential[0], root)
    for site in (1, 3):
        os.remove(root / "run" / "A1" / "phenotype_cells" / f"site-{site}.parquet")
    opens.clear()
    settings = {**SETTINGS, "dst_root": str(root / "run"),
                "phenotype_source": str(root / "tiles"),
                "ops_ph_stage_dir": str(stage)}
    out = ops_engine.run_ops(settings, phases=("ph_measure",))
    report = out["wells"]["A1"]["ph_fields"]
    assert report["ph_measure"]["measured"] == 2
    assert report["ph_measure"]["resumed"] == 2
    raw = [p for p in opens if p.startswith(str(root / "tiles"))]
    assert sorted({os.path.basename(p).rsplit("-", 1)[1] for p in raw}) == [
        "1.tif", "3.tif"]
    _same(sequential[0], root)


def test_the_stager_frees_unreadable_fields_and_cleans_up(tmp_path):
    def read(site):
        """Site 1 is unreadable; the others are small stacks."""
        if site == 1:
            return None, [("x", "missing")]
        return np.full((4, 4, 2), site, np.float32), []

    stager = FieldStager(str(tmp_path / "s"), [0, 1, 2, 3], read, ahead=1,
                         threads=1)
    seen = []
    for site in (0, 1, 2, 3):
        stack, lost = stager.load(site)
        if stack is None:
            assert lost == [("x", "missing")]
            continue
        seen.append(float(stack[0, 0, 0]))
        stager.release(site)
        stager.release(site)
    stager.close()
    assert seen == [0.0, 2.0, 3.0]
    assert stager.peak_files == 1
    assert os.listdir(tmp_path / "s") == []


def test_closing_the_stager_early_deletes_what_it_staged(tmp_path):
    stager = FieldStager(str(tmp_path / "s"), [0, 1, 2],
                         lambda s: (np.zeros((2, 2, 1), np.float32), []),
                         ahead=2, threads=2)
    assert stager.path(0)[0].endswith("site-0.npy")
    stager.close()
    assert os.listdir(tmp_path / "s") == []


def test_a_read_error_in_the_stager_frees_its_slot(tmp_path):
    def read(site):
        """Every read fails."""
        raise OSError("gone")

    stager = FieldStager(str(tmp_path / "s"), [0, 1], read, ahead=1)
    with pytest.raises(OSError):
        stager.path(0)
    with pytest.raises(OSError):
        stager.path(1)
    stager.close()


def test_the_gpu_self_check_reports_the_columns_beyond_the_tolerance(
        tmp_path, models, monkeypatch):
    from spacr import ops_sample

    original = ops_sample.measure_phenotype_field

    def drifting(*args, gpu=False, **kwargs):
        """The CPU table, with one column moved by 0.1 % on the 'card'."""
        table = original(*args, gpu=False, **kwargs)
        if gpu:
            table["cell_gfp_mean_intensity"] *= 1.001
        return table

    monkeypatch.setattr(ops_sample, "measure_phenotype_field", drifting)
    monkeypatch.setitem(ops_engine._PH_WORKER_FIELDS, "count", 0)
    monkeypatch.delenv("CUPY_CACHE_DIR", raising=False)
    root = _prepared(tmp_path / "check")
    out = _run(root, ops_ph_measure_gpu=True)
    fields = out["wells"]["A1"]["ph_fields"]
    check = fields["ph_measure_gpu_check"]
    assert check["site"] == 0 and check["rows"] == 4
    assert [c["column"] for c in check["outside"]] == ["cell_gfp_mean_intensity"]
    assert check["outside"][0]["max_relative"] == pytest.approx(1e-3, rel=1e-6)
    assert check["columns"] > 50
    assert fields["ph_measure"]["first_field_seconds_per_worker"] and len(
        fields["ph_measure"]["first_field_seconds_per_worker"]) == 1
    assert fields["ph_measure"]["later_field_seconds_median"] is not None
    assert os.environ["CUPY_CACHE_DIR"].endswith(os.path.join(".spacr",
                                                              "cupy_cache"))

    root = _prepared(tmp_path / "off")
    out = _run(root, ops_ph_measure_gpu=True, ops_ph_measure_gpu_check=False,
               ops_ph_pipeline=True)
    assert "ph_measure_gpu_check" not in out["wells"]["A1"]["ph_fields"]


def test_the_gpu_self_check_runs_on_the_first_streamed_field(
        tmp_path, models, monkeypatch):
    calls = []
    monkeypatch.setattr(ops_engine, "_ph_gpu_check",
                        lambda task: calls.append(task) or {"site": task["site"]})
    root = _prepared(tmp_path / "streamed")
    out = _run(root, ops_ph_measure_gpu=True,
               ops_ph_stage_dir=str(tmp_path / "stage"))
    assert len(calls) == 1 and calls[0]["staged"].endswith("site-0.npy")
    assert out["wells"]["A1"]["ph_fields"]["ph_measure_gpu_check"] == {"site": 0}


def test_compare_tables_matches_rows_by_key_and_names_extra_columns():
    test = pd.DataFrame({"site": [0, 0], "label": [2, 1], "a": [1.0, np.nan],
                         "b": [0.0, 5.0], "only": [1, 1], "text": ["x", "y"]})
    reference = pd.DataFrame({"site": [0, 0], "label": [1, 2], "a": [np.nan, 1.0],
                              "b": [5.0, 1.0], "gone": [1, 2],
                              "text": ["y", "x"]})
    report = compare_tables(test, reference)
    assert report["rows"] == 2 and report["columns"] == 2
    assert report["outside"] == [{"column": "b", "max_relative": 1.0,
                                  "rows_outside": 1}]
    assert report["only_in_test"] == ["only"]
    assert report["only_in_reference"] == ["gone"]
    zero = compare_tables(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"a": [0.0]}),
                          keys=())
    assert zero["outside"][0]["max_relative"] == float("inf")


def test_measure_workers_follow_the_setting_and_the_vram_cap(monkeypatch):
    assert ops_engine._ph_measure_workers({"n_workers": 3}, 10, False) == 3
    assert ops_engine._ph_measure_workers({"n_workers": 3,
                                           "ops_ph_measure_workers": 5}, 4,
                                          False) == 4
    from spacr import gpu_admission

    monkeypatch.setattr(gpu_admission, "query_gpu", lambda: gpu_admission.GpuSample(
        utilisation=[0.0], memory_used_mib=[30 * 1024.0]))
    capped = {"ops_ph_measure_workers": 8, "ops_gpu_admission": True,
              "ops_gpu_vram_cap_gb": 22}
    assert ops_engine._ph_measure_workers(capped, 20, True) == 4
    assert ops_engine._ph_measure_workers(capped, 20, False) == 8
