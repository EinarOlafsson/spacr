"""Defensive branch coverage for the item-60 core/utils sweep."""
from __future__ import annotations

import builtins
import importlib
import io
import os
import sqlite3
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg", force=True)

from spacr import core as C
from spacr import utils as U


def test_square_footprint_uses_installed_skimage_api():
    footprint = U._square_footprint(3)
    assert footprint.shape == (3, 3)


def test_square_footprint_uses_new_skimage_api_when_available(monkeypatch):
    from skimage import morphology
    monkeypatch.setattr(morphology, "footprint_rectangle",
                        lambda shape: np.ones(shape, dtype=np.uint8), raising=False)
    reloaded = importlib.reload(U)
    assert reloaded._square_footprint(2).shape == (2, 2)
    monkeypatch.delattr(morphology, "footprint_rectangle", raising=False)
    importlib.reload(U)


def test_utils_import_survives_ipython_display_failure(monkeypatch):
    """The notebook helper is optional even when IPython is mid-import."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "IPython.display":
            raise RuntimeError("partially initialized IPython")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    reloaded = importlib.reload(U)
    assert reloaded.display("ignored") is None
    monkeypatch.setattr(builtins, "__import__", real_import)
    importlib.reload(U)


def test_release_version_rejects_non_version_text():
    assert U._release_version("development") == ()


def test_lazy_module_explicitly_blocked_root(monkeypatch):
    proxy = U._LazyModule("item60_absent.child")
    monkeypatch.setitem(sys.modules, "item60_absent", None)
    with pytest.raises(ModuleNotFoundError, match="None in sys.modules"):
        proxy._load()


def test_lazy_module_metadata_probe_failure_falls_through(monkeypatch):
    proxy = U._LazyModule(
        "item60_fake", minimum_distribution=("fake-dist", "2.0", "reason"))
    module = SimpleNamespace(answer=42)
    monkeypatch.setattr(U, "_distribution_version",
                        lambda _name: (_ for _ in ()).throw(RuntimeError("metadata")))
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    assert proxy.answer == 42


def test_lazy_module_rejects_unsupported_installed_version(monkeypatch):
    proxy = U._LazyModule(
        "item60_old", minimum_distribution=("old-dist", "2.0", "reason"))
    monkeypatch.setattr(U, "_distribution_version", lambda _name: "1.9")
    with pytest.raises(U.OptionalDependencyCompatibilityError, match="2.0 or newer"):
        proxy._load()


@pytest.mark.parametrize("blocked", [False, True])
def test_lazy_module_failed_import_cleans_partial_modules(monkeypatch, blocked):
    root = "item60_partial"
    proxy = U._LazyModule(root + ".child", block_roots=("tensorflow",) if blocked else ())

    def fail(_name):
        sys.modules[root + ".leaked"] = SimpleNamespace()
        raise RuntimeError("broken import")

    monkeypatch.setattr(importlib, "import_module", fail)
    with pytest.raises(RuntimeError, match="broken import"):
        proxy._load()
    assert root + ".leaked" not in sys.modules
    assert proxy.__dict__["_module"] is None


def test_lazy_module_assignment_loads_then_sets(monkeypatch):
    module = SimpleNamespace()
    proxy = U._LazyModule("item60_assignment")
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    proxy.value = 9
    assert module.value == 9


def test_lazy_module_tolerates_blocker_removed_by_import(monkeypatch):
    class VanishingBlockerList(list):
        def remove(self, value):
            if isinstance(value, U._BlockTensorFlowFinder):
                super().remove(value)
                raise ValueError("already removed")
            return super().remove(value)

    module = SimpleNamespace(answer=1)
    monkeypatch.setattr(sys, "meta_path", VanishingBlockerList(sys.meta_path))
    monkeypatch.setattr(importlib, "import_module", lambda _name: module)
    proxy = U._LazyModule("item60_blocked", block_roots=("tensorflow",))
    assert proxy.answer == 1


@pytest.mark.parametrize("jobs,batch", [("bad", None), (1, "bad")])
def test_print_progress_bad_parallelism_values_fall_back(capsys, jobs, batch):
    U.print_progress(1, 4, jobs, time_ls=[2.0], batch_size=batch)
    assert "Time_left" in capsys.readouterr().out


def test_update_database_duplicate_metadata_closes_and_raises(tmp_path):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        pd.DataFrame({"prcfo": ["p_o1"]}).to_sql("png_list", conn, index=False)
    duplicate = pd.DataFrame({"prcfo": ["p_o1", "p_o1"], "condition": ["a", "b"]})
    with pytest.raises(pd.errors.MergeError):
        U._update_database_with_merged_info(
            str(db), duplicate, columns=["prcfo", "condition"])
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM png_list").fetchone()[0] == 1


def test_generate_names_rejects_unknown_crop_role(tmp_path):
    with pytest.raises(ValueError, match="no naming rule"):
        U._generate_names(
            "plate_A01_f1", np.array([1]), np.array([]), np.array([]), str(tmp_path),
            "unknown-role")


def test_stamp_identity_incomplete_stamp_is_legacy():
    assert U._stamp_identity({"measurement_ndim": 3}) == U._LEGACY_STAMP


def test_release_imported_rows_retries_locked_database(monkeypatch, tmp_path, capsys):
    db = tmp_path / "measurements.db"
    db.touch()
    calls = iter([sqlite3.OperationalError("database is locked"), 3])

    def attempt(*_args, **_kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(U, "_release_imported_rows_once", attempt)
    monkeypatch.setattr(U.time, "sleep", lambda _delay: None)
    assert U._release_imported_rows_for_field(
        str(db), "cell", pd.DataFrame({"prcf": ["p"]})) == 3
    assert "retrying" in capsys.readouterr().out


def test_release_imported_rows_does_not_retry_other_errors(monkeypatch, tmp_path):
    db = tmp_path / "measurements.db"
    db.touch()
    monkeypatch.setattr(
        U, "_release_imported_rows_once",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.OperationalError("readonly")))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        U._release_imported_rows_for_field(str(db), "cell", pd.DataFrame())


def test_release_imported_rows_refuses_table_without_field_key(tmp_path, monkeypatch):
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE cell (object_label INTEGER, imported_from TEXT)")
        conn.execute("INSERT INTO cell VALUES (1, 'foreign')")
    from spacr import resume
    monkeypatch.setattr(resume, "importer_rows_clause",
                        lambda _conn, _table: "s.imported_from IS NOT NULL")
    with pytest.raises(U.ImportedCopyNotReleased, match="cannot be established"):
        U._release_imported_rows_once(
            str(db), "cell", pd.DataFrame({"prcf": ["p"]}))


def test_widen_table_returns_empty_for_new_table():
    conn = sqlite3.connect(":memory:")
    try:
        assert U._widen_table_for(conn, "missing", pd.DataFrame({"a": [1]})) == []
    finally:
        conn.close()


def test_widen_table_propagates_non_duplicate_error():
    class Broken:
        def execute(self, sql):
            if sql.startswith("PRAGMA"):
                return [(0, "old")]
            raise sqlite3.OperationalError("disk I/O error")

    with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        U._widen_table_for(Broken(), "cell", pd.DataFrame({"new": [1]}))


@pytest.mark.parametrize("required", [True, False])
def test_append_database_exhausts_locked_retries(monkeypatch, tmp_path, required, capsys):
    from spacr import database_concurrency
    monkeypatch.setattr(U, "DB_WRITE_ATTEMPTS", 2)
    monkeypatch.setattr(U.time, "sleep", lambda _delay: None)
    monkeypatch.setattr(
        database_concurrency, "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    frame = pd.DataFrame({"a": [1]})
    if required:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            U._append_to_measurements_db(str(tmp_path / "x.db"), "cell", frame)
    else:
        U._append_to_measurements_db(
            str(tmp_path / "x.db"), "png_list", frame, required=False)
        assert "giving up" in capsys.readouterr().out


def test_pick_best_model_direct_missing_and_empty(tmp_path):
    direct = tmp_path / "direct.pth"
    direct.write_bytes(b"checkpoint")
    assert U.pick_best_model(str(direct)) == str(direct.resolve())
    with pytest.raises(FileNotFoundError, match="does not exist"):
        U.pick_best_model(str(tmp_path / "absent"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No .pth"):
        U.pick_best_model(str(empty))


def test_pick_best_model_uses_checkpoint_metadata(tmp_path):
    milestone = tmp_path / "odd_epoch_7.pth"
    best = tmp_path / "plain.pth"
    torch.save({"artifact_role": "milestone", "metrics": {"accuracy": .99},
                "training_state": {"epoch": 7}}, milestone)
    torch.save({"artifact_role": "best", "metrics": {"accuracy": .5},
                "training_state": {"epoch": 2}}, best)
    assert U.pick_best_model(str(tmp_path)) == str(best)


def test_pick_best_model_uses_epoch_from_broken_legacy_filename(tmp_path):
    older = tmp_path / "model_epoch_2.pth"
    newer = tmp_path / "model_epoch_7.pth"
    older.write_bytes(b"broken")
    newer.write_bytes(b"broken")
    assert U.pick_best_model(str(tmp_path)) == str(newer)


def test_augment_single_image_rejects_unreadable_input(tmp_path):
    with pytest.raises(ValueError, match="Could not read"):
        U.augment_single_image((str(tmp_path / "missing.png"), str(tmp_path)))


def test_training_advice_scalar_accepts_duplicate_index_series(tmp_path, monkeypatch):
    """Exercise the guard by making iloc return a one-element Series."""
    # The helper is nested; a tiny real progress file reaches it repeatedly.
    pd.DataFrame({"epoch": [1, 2], "loss": [2., 1.], "accuracy": [.2, .4]}).to_csv(
        tmp_path / "train.csv", index=False)
    pd.DataFrame({"epoch": [1, 2], "loss": [2.5, 1.5], "accuracy": [.1, .3]}).to_csv(
        tmp_path / "val.csv", index=False)
    from pandas.core.indexing import _iLocIndexer
    real_getitem = _iLocIndexer.__getitem__

    def series_for_accuracy(self, key):
        value = real_getitem(self, key)
        if (isinstance(self.obj, pd.Series) and self.obj.name == "accuracy"
                and key == -1 and not isinstance(value, pd.Series)):
            return pd.Series([value])
        return value

    monkeypatch.setattr(_iLocIndexer, "__getitem__", series_for_accuracy)
    assert isinstance(U.suggest_training_changes(str(tmp_path)), dict)


def test_installed_cellpose_models_import_failure_is_empty(monkeypatch):
    class Failing:
        def __getattr__(self, _name):
            raise ImportError("cellpose unavailable")
    monkeypatch.setattr(U, "cp_models", Failing())
    assert U._installed_cellpose_models() == ()


def test_resolve_cellpose_reports_nondefault_stock(monkeypatch, capsys):
    monkeypatch.setattr(U, "_installed_cellpose_models",
                        lambda: (U.CPSAM_MODEL, "cpsam_v2"))
    U.reset_cellpose_model_reports()
    assert U._resolve_cellpose_pretrained("cpsam_v2", "cell") == "cpsam_v2"
    assert "cpsam_v2" in capsys.readouterr().out


def test_plot_theme_ignores_invalid_role_color():
    colors = U._plot_theme_colors(False, {"background": "not-a-color",
                                          "foreground": "#123456"})
    assert colors["background"] == "white"
    assert colors["foreground"] == "#123456"


def test_plot_clusters_ignores_invalid_fixed_color():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    U.plot_clusters(
        ax, np.array([[0., 0.], [1., 1.]]), np.array([0, 0]), ["red"],
        [(0.5, 0.5)], False, True, False, point_color="not-a-color")
    assert ax.collections
    plt.close(fig)


def test_measure_test_mode_rejects_empty_source(tmp_path):
    with pytest.raises(ValueError, match="nothing can be sampled"):
        U.measure_test_mode({"src": str(tmp_path), "test_mode": True,
                             "test_nr": 2})


def test_measure_test_mode_reports_small_source(tmp_path, capsys):
    source = tmp_path / "merged"
    source.mkdir()
    (source / "one.npy").write_bytes(b"x")
    out = U.measure_test_mode({"src": str(source), "test_mode": True,
                               "test_nr": 5})
    assert "fewer than test_nr" in capsys.readouterr().out
    assert os.path.isfile(os.path.join(out["src"], "one.npy"))


def test_feature_filter_matches_morphology_and_channel_list():
    columns = ["cell_area", "cell_channel_0_mean_intensity",
               "cell_channel_2_mean_intensity"]
    assert U._feature_filter_matches(columns, "morphology") == ["cell_area"]
    assert U._feature_filter_matches(columns, [0, 2]) == columns[1:]


def test_preprocess_data_reports_filter_that_becomes_empty(monkeypatch):
    frame = pd.DataFrame({"cell_area": [1., 2.]})
    monkeypatch.setattr(U, "filter_dataframe_features",
                        lambda df, **_kwargs: (df.iloc[:, 0:0], []))
    with pytest.raises(ValueError, match="initially matched"):
        U.preprocess_data(frame, "morphology", False, False, None)


def test_preprocess_data_prints_batch_warning(monkeypatch, capsys):
    from spacr import batch_correction
    report = SimpleNamespace(method="center", batches=("a",),
                             centroid_spread_before=2.0,
                             centroid_spread_after=0.0,
                             warnings=("small batch",))
    monkeypatch.setattr(batch_correction, "correct_from_metadata",
                        lambda data, metadata, **kwargs: (data, report))
    out = U.preprocess_data(pd.DataFrame({"cell_area": [1., 2.]}), None,
                            False, False, None, batch_correction="center")
    assert out.shape == (2, 1)
    assert "Warning: batch correction: small batch" in capsys.readouterr().out


def test_adjust_cell_masks_default_worker_count(monkeypatch, tmp_path):
    for name in ("parasite", "cell", "nucleus"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(U, "cpu_count", lambda: 3)
    # n_jobs=None derives max(1, cpu_count() - 2) = 1, and one worker runs
    # INLINE rather than starting a child. Spying on the pool is what makes
    # that observable -- passing None straight through would raise inside
    # Pool, and simply calling the function proved neither.
    started = []
    monkeypatch.setattr(U, "Pool",
                        lambda *a, **k: started.append(a) or (_ for _ in ()))
    U.adjust_cell_masks(str(tmp_path / "parasite"), str(tmp_path / "cell"),
                        str(tmp_path / "nucleus"), n_jobs=None)
    assert started == [], "one derived worker must run inline"


def test_merge_regression_metadata_collapses_duplicate_gene(tmp_path, capsys):
    results = tmp_path / "results.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame({"feature": ["C(gene)[T.123_x]"]}).to_csv(results, index=False)
    pd.DataFrame({"Gene ID": ["TGGT_123", "TGGT_123"],
                  "note": ["first", "second"]}).to_csv(metadata, index=False)
    merged = U.merge_regression_res_with_metadata(str(results), str(metadata))
    assert len(merged) == 1 and merged.loc[0, "note"] == "first"
    assert "Keeping the first row" in capsys.readouterr().out


def test_remove_outliers_rejects_negative_threshold():
    with pytest.raises(ValueError, match="threshold must be"):
        U.remove_outliers_by_group(pd.DataFrame({"g": ["a"], "v": [1.]}),
                                   "g", "v", threshold=-1)
