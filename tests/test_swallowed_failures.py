"""Failures spaCR swallows with ``except Exception: pass`` that the user needs.

Each test here forces one guarded statement in product code to fail and then
asserts what the user *should* be left with. They are all
``xfail(strict=True)``: the product code has not been changed, so each one
fails today, and each one starts passing the moment its named site stops
swallowing. The site and the one-line fix are named in the ``reason``.

The bar for inclusion is the same in every case — the swallow either loses
data, reports success while having done less than it said, hides a
misconfiguration the user could fix, or substitutes a fallback for a failed
computation and reports the resulting number as if it were the real one.
Cleanup, cosmetics, optional imports and probes whose caller re-checks the
same thing are deliberately absent: they swallow nothing worth telling
anybody about.

CPU-only, offline, deterministic.
"""
from __future__ import annotations

import csv
import importlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# spacr/cli.py:736 — module_defaults() returns {} when the defaults module
# cannot be imported, and resolve_settings() is the *run* path, not just
# --describe. The pipeline then runs on a settings dict with no defaults in
# it at all, and --set naming one of that module's own keys is rejected as
# "a setting that does not exist" — pointing the user at their command line
# instead of at the missing dependency.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "spacr/cli.py:736 swallows the ImportError from the defaults module and "
    "returns {}, so spacr-run and the batch queue resolve settings with no "
    "defaults and blame the user's --set. Fix: let the ImportError out of "
    "module_defaults (or re-raise it as a SettingsError naming the module)."))
def test_module_defaults_reports_a_defaults_module_that_will_not_import(
        monkeypatch):
    """A missing optional dependency must not read as an empty defaults dict."""
    from spacr import cli

    module = cli.resolve_module("convert")
    assert module is not None
    assert module.defaults_entry == "spacr.convert:default_settings"

    real_import = importlib.import_module

    def refuse(name, *args, **kwargs):
        if name == "spacr.convert":
            raise ImportError("No module named 'nd2reader'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", refuse)

    with pytest.raises(Exception) as caught:
        cli.resolve_settings(module, None, ["z_handling=max"])

    # The user has to be told what is actually broken. Today the settings
    # resolve silently to {'z_handling': 'max'} — no defaults, no complaint —
    # or fail with "names a setting that does not exist".
    assert "nd2reader" in str(caught.value)


# ---------------------------------------------------------------------------
# spacr/notebook_export.py:63 — export_run() calls _read_settings() at line
# 175 with the comment "Validate that the recorded settings exist and parse".
# _read_settings swallows the parse error and returns {}, so the validation
# is a no-op and the exported notebook's first code cell —
# json.loads((RUN_DIR / 'settings.json').read_text()) — blows up in the
# user's face instead.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "spacr/notebook_export.py:63 swallows the JSON error, so the "
    "'validate that the recorded settings parse' call at line 175 validates "
    "nothing and the notebook is exported with a first cell that cannot run. "
    "Fix: let json.JSONDecodeError out of _read_settings (or re-raise it as "
    "the documented FileNotFoundError/ValueError from export_run)."))
def test_export_run_refuses_a_run_whose_settings_do_not_parse(tmp_path):
    """A notebook that cannot load its own settings is not an export."""
    from spacr import notebook_export

    run_dir = tmp_path / "2026-01-01_000000_deadbeef__mask"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "app_key": "mask", "status": "ok", "start_utc": "2026-01-01T00:00:00",
    }))
    # What a kill -9 mid-write leaves behind.
    (run_dir / "settings.json").write_text('{"src": "/data/plate1",')

    with pytest.raises(Exception):
        notebook_export.export_run(run_dir)


# ---------------------------------------------------------------------------
# spacr/batch.py:1410 — an artifact whose run-status stamp cannot be read is
# dropped from the job summary. The other artifact's clean stamp then decides
# the verdict, and the queue reports the job 'complete' having never read the
# stamp that says a plate failed.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "spacr/batch.py:1410 swallows RunStatusUnreadable, so a locked or "
    "truncated artifact silently leaves the queue's per-job verdict resting "
    "on the artifacts it could read — 'complete' for a job that may have "
    "failed. Fix: record the unreadable artifact on the summary and refuse "
    "the 'complete' verdict (a warning-level log at minimum)."))
def test_batch_job_summary_does_not_report_complete_over_an_unreadable_stamp(
        tmp_path, monkeypatch):
    """'complete' must mean every stamp was read, not every stamp that opened."""
    from spacr import batch

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    good = src / "measurements" / "measurements.db"
    bad = src / "measurements" / "other.db"
    good.write_bytes(b"")
    bad.write_bytes(b"")

    def read_stamp(artifact):
        if str(artifact) == str(good):
            return [{"name": "measure_crop", "n_attempted": 4,
                     "n_succeeded": 4, "n_failed": 0}]
        raise batch.SpacrError(f"{artifact} could not be read")

    monkeypatch.setattr(batch, "read_run_status", read_stamp)

    summary = batch._collect_run_status({"src": str(src)}, {})

    assert summary is not None
    assert summary["status"] != "complete", (
        "one of the two artifacts was never read, so 'complete' is a claim "
        "the queue cannot make")


# ---------------------------------------------------------------------------
# spacr/run_journal.py:1081 and :870 — a run folder whose manifest.json does
# not parse is dropped from the Home dashboard's totals and from the run
# history, with nothing anywhere saying a folder was skipped. The user sees a
# run count that is quietly short and a run that has vanished.
# ---------------------------------------------------------------------------

def _run_folder(root: Path, name: str, manifest: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(manifest)
    return d


@pytest.mark.xfail(strict=True, reason=(
    "spacr/run_journal.py:1081 swallows the manifest parse error, so "
    "journal_totals() under-counts runs with no indication that a folder was "
    "unreadable. Fix: LOG.warning the folder that could not be read."))
def test_journal_totals_says_when_a_run_folder_could_not_be_read(
        tmp_path, monkeypatch, caplog):
    """A total that silently skips runs is a wrong number."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    _run_folder(root, "a__mask", json.dumps({"app_key": "mask"}))
    _run_folder(root, "b__mask", json.dumps({"app_key": "mask"}))
    broken = _run_folder(root, "c__measure", '{"app_key": "measu')

    monkeypatch.setattr(run_journal, "runs_root", lambda: root)

    with caplog.at_level(logging.WARNING, logger="spacr.run_journal"):
        totals = run_journal.journal_totals()

    assert totals["total_runs"] == 2  # the count itself is honest
    assert any(broken.name in record.getMessage() for record in caplog.records), (
        "journal_totals reported 2 of 3 run folders and said nothing about "
        "the third")


@pytest.mark.xfail(strict=True, reason=(
    "spacr/run_journal.py:870 swallows the manifest parse error, so a run "
    "disappears from Run History with no trace. Fix: LOG.warning the folder "
    "that could not be read."))
def test_recent_runs_says_when_a_run_folder_could_not_be_read(
        tmp_path, monkeypatch, caplog):
    """A run the user can see on disk must not vanish from the history in silence."""
    from spacr import run_journal

    root = tmp_path / "runs"
    root.mkdir()
    _run_folder(root, "a__mask", json.dumps(
        {"app_key": "mask", "start_utc": "2026-01-01T00:00:00"}))
    broken = _run_folder(root, "b__measure", "not json at all")

    monkeypatch.setattr(run_journal, "runs_root", lambda: root)

    with caplog.at_level(logging.WARNING, logger="spacr.run_journal"):
        entries = run_journal.recent_runs(limit=10)

    assert len(entries) == 1
    assert any(broken.name in record.getMessage() for record in caplog.records), (
        "recent_runs dropped a run folder without telling anybody")


# ---------------------------------------------------------------------------
# spacr/model_compare.py:1121 — a field that cannot be read is skipped, so
# the model comparison is computed and drawn over fewer fields than the user
# asked for, and nothing on the figure or in the log says so.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "spacr/model_compare.py:1121 swallows the read error, so load_fields "
    "returns fewer fields than requested and the comparison is drawn over a "
    "subset the user never chose. Fix: log the unreadable field at warning "
    "(the total-failure case already raises ValueError)."))
def test_load_fields_says_which_field_it_could_not_read(tmp_path, caplog):
    """Comparing 2 of the 3 requested fields is not the comparison asked for."""
    from spacr import model_compare

    folder = tmp_path / "fields"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for name in ("field_000.npy", "field_002.npy"):
        np.save(folder / name, rng.integers(0, 255, (16, 16)).astype(np.uint16))
    # Truncated .npy — exactly what an interrupted write leaves.
    (folder / "field_001.npy").write_bytes(b"\x93NUMPY\x01\x00")

    with caplog.at_level(logging.WARNING):
        names, images = model_compare.load_fields(str(folder), n_fields=3)

    assert len(images) == 2
    assert any("field_001" in record.getMessage() for record in caplog.records), (
        "load_fields quietly compared 2 fields where 3 were asked for")


# ---------------------------------------------------------------------------
# spacr/spacrops.py:1442 — a mosaic-manifest row whose transform will not
# parse is dropped from the composite. The tile is missing from the stitched
# output, the file is written anyway, and the run returns the path as if the
# mosaic were whole.
# ---------------------------------------------------------------------------

_MANIFEST_COLUMNS = ["path", "H", "W", "M00", "M01", "M02",
                     "M10", "M11", "M12", "canvas_x", "canvas_y"]


def _manifest_row(path, x, y, m00="1"):
    return {"path": str(path), "H": "32", "W": "32",
            "M00": m00, "M01": "0", "M02": str(x),
            "M10": "0", "M11": "1", "M12": str(y),
            "canvas_x": str(x), "canvas_y": str(y)}


@pytest.mark.xfail(strict=True, reason=(
    "spacr/spacrops.py:1442 swallows the malformed transform, so the tile is "
    "silently missing from the mosaic and the file is written as if complete. "
    "Fix: raise RuntimeError naming the manifest row (the builder already "
    "raises for a manifest with no usable rows at all)."))
def test_mosaic_builder_refuses_a_manifest_row_it_cannot_parse(tmp_path):
    """A tile dropped from a stitched mosaic is data the user never sees again."""
    pytest.importorskip("cv2")
    tifffile = pytest.importorskip("tifffile")
    from spacr.spacrops import spacrStitcher

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    rng = np.random.default_rng(1)
    for index in (0, 1):
        tifffile.imwrite(
            tiles / f"tile_{index}.tif",
            rng.integers(0, 4000, (32, 32)).astype(np.uint16))

    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(_manifest_row(tiles / "tile_0.tif", 0, 0))
        # A transform column that did not survive whatever wrote the manifest.
        writer.writerow(_manifest_row(tiles / "tile_1.tif", 32, 0, m00=""))

    stitcher = spacrStitcher(outdir=str(tmp_path / "out"), save_qc=False,
                             feature_cache_mode="ram")

    with pytest.raises(RuntimeError, match="tile_1"):
        stitcher.build_multichannel_mosaic_from_manifest(
            str(manifest), str(tmp_path / "mosaic.tif"),
            tmp_dir=str(tmp_path / "tmp"))


# ---------------------------------------------------------------------------
# spacr/ml.py:726 — a bootstrap resample that will not fit is skipped. The
# guard below it only catches the case where *every* resample failed; 199 of
# 200 failing produces a standard deviation taken over one draw, which is
# zero, which makes every p-value exactly 1.0 — "no significant gRNAs" for a
# screen, reported with no hint that the inference never happened.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "spacr/ml.py:726 swallows every per-resample fit failure and only the "
    "all-failed case raises, so a hinge regression can report p = 1 for every "
    "gRNA off a single successful draw. Fix: count the skipped resamples and "
    "log/raise when they are most of n_boot."))
def test_bootstrap_p_values_do_not_hide_a_collapsed_resample_count(
        monkeypatch, caplog):
    """P-values from 1 of 200 resamples must not read like a clean null result."""
    sklearn_svm = pytest.importorskip("sklearn.svm")
    from spacr.ml import _bootstrap_wald_p_values

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] + rng.normal(scale=0.1, size=60) > 0).astype(float)

    model = sklearn_svm.LinearSVC(dual="auto", max_iter=5000)
    model.fit(X, y)

    real_fit = sklearn_svm.LinearSVC.fit
    calls = {"n": 0}

    def flaky_fit(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:            # every resample after the first fails
            raise RuntimeError("solver did not converge")
        return real_fit(self, *args, **kwargs)

    monkeypatch.setattr(sklearn_svm.LinearSVC, "fit", flaky_fit)

    with caplog.at_level(logging.WARNING):
        p_values = _bootstrap_wald_p_values(model, X, y, n_boot=200,
                                            random_state=0)

    assert np.allclose(p_values, 1.0), (
        "precondition: one usable draw gives zero spread and p = 1 throughout")
    assert caplog.records, (
        "199 of 200 resamples were dropped and the p-values came back as a "
        "clean 'no hits' with nothing logged")
