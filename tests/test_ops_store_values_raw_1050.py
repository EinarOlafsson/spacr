"""Item 1050: ``ops_store_values_raw`` stores the raw read under any threshold mode.

The read mode decides the calls and the threshold mode decides the
thresholds; storing each spot's unfiltered read (``values_raw``) in the field
cache depends on neither. Generation runs manual thresholds with the log
read and still needs the raw read stored, which ``ops_read_mode`` 'auto'
refuses under manual thresholds.
"""
import glob
import hashlib
import os
import re

import numpy as np
import pandas as pd
import pytest

from spacr import ops_engine, ops_settings


def _hash_tables(out):
    digests = {}
    for path in sorted(glob.glob(os.path.join(out, "**", "*.parquet"), recursive=True),
                       key=lambda p: (os.path.dirname(p), int(re.findall(r"(\d+)", os.path.basename(p))[-1])
                                      if re.findall(r"(\d+)", os.path.basename(p)) else -1)):
        if "decode_cache" in path or "__keys" in path:
            continue
        frame = pd.read_parquet(path)
        key = re.sub(r"part-\d+\.parquet$", "part.parquet", os.path.relpath(path, out))
        digests[key] = hashlib.sha256(
            pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
    return digests


def _caches(out):
    return {os.path.basename(p): dict(np.load(p))
            for p in sorted(glob.glob(os.path.join(out, "*", "decode_cache", "*.npz")))}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The item-873 synthetic well decoded with manual thresholds three ways."""
    import tests.test_the_ops_engine_runs_a_well_end_to_end as base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ops_engine, "_RASTER_OVERLAP", base.OVERLAP)
    monkeypatch.setitem(ops_settings.OPS_DEFAULTS, "ops_raster_overlap", base.OVERLAP)
    monkeypatch.setattr(ops_engine, "_WINDOW", 256)
    monkeypatch.setattr(ops_engine, "_WINDOW_OVERLAP", 48)
    monkeypatch.setattr(ops_engine, "_cellpose_model",
                        lambda settings, gpu: base._ThresholdModel())
    root = tmp_path_factory.mktemp("ops_store_values_raw_1050")
    raw = root / "raw"
    raw.mkdir()
    _nuclei, codes, _centre, _tops = base._plate(raw)
    library = sorted(set(codes)) + ["TTTT", "GGGG"]
    common = {"genotype_source": str(raw), "plate": "synthetic", "ops_gpu": False,
              "n_workers": 1, "ops_decode_preset": "rule_a_s2",
              "ops_decode_threshold_mode": "manual", "ops_store_field_cache": True}
    outs = {}
    for name, extra in (("log", {}), ("stored", {"ops_store_values_raw": True}),
                        ("raw", {"ops_read_mode": "raw"})):
        out = str(root / f"out_{name}")
        phases = None if name == "log" else ("decode",)
        if phases:
            import shutil
            shutil.copytree(str(root / "out_log"), out)
        ops_engine.run_ops({**common, **extra, "dst_root": out},
                           library=library, **({"phases": phases} if phases else {}))
        outs[name] = out
    yield outs
    monkeypatch.undo()


def test_the_default_is_off():
    assert ops_settings.OPS_DEFAULTS["ops_store_values_raw"] is False


def test_manual_log_with_the_raw_read_stored_calls_exactly_as_manual_log(runs):
    assert _hash_tables(runs["stored"]) == _hash_tables(runs["log"])


def test_the_raw_read_is_stored_and_equals_the_raw_modes(runs):
    log, stored, raw = (_caches(runs[k]) for k in ("log", "stored", "raw"))
    assert log and set(log) == set(stored) == set(raw)
    for site in log:
        assert "values_raw" not in log[site]
        assert set(stored[site]) == set(log[site]) | {"values_raw"}
        for key in log[site]:
            np.testing.assert_array_equal(stored[site][key], log[site][key])
        np.testing.assert_array_equal(stored[site]["values_raw"], raw[site]["values_raw"])
