"""Table-driven tests of how ``spacr.core`` orchestrates a Mask run and an image UMAP.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture:

* ``preprocess_generate_masks`` refuses a missing or mistyped ``src``, and on a
  valid run decides what is segmented, plotted, adjusted and consolidated:
  the v2 dispatch's Cellpose planes, the per-object "already generated" skip,
  cell adjustment with an organelle folder that may not exist, a
  ``test_mode`` overlay budget counted in fields, an unlistable ``merged/``
  folder, and per-plate consolidation;
* ``generate_image_umap`` warns when two source databases share a plate id and
  only then, never stops a run over its own failed check, and filters
  excluded conditions and DBSCAN noise while keeping the frame aligned.

The Cellpose-SAM forward pass, the array merge, overlay rendering and the
reducer are the only things stubbed; everything else runs on real folders and
real SQLite files under ``tmp_path``.
"""
from __future__ import annotations

import os
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)

import spacr.core as core  # noqa: E402


class _Spy:
    """Records every call and returns ``ret``; stands in for an externality."""

    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.ret

    @property
    def n(self):
        return len(self.calls)


class _Stop(Exception):
    """Ends the run once the branch under test has been passed."""


def _stop(*args, **kwargs):
    raise _Stop()


def _case_dir(tmp_path, name):
    """A fresh per-case directory under ``tmp_path``."""
    folder = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    folder.mkdir()
    return folder


# ---------------------------------------------------------------------------
# preprocess_generate_masks
# ---------------------------------------------------------------------------

def _mask_settings(src, **over):
    settings = {
        "src": str(src), "metadata_type": "cellvoyager",
        "channels": [0, 1, 2, 3],
        "cell_channel": 1, "nucleus_channel": 0, "pathogen_channel": 2,
        "organelle_channel": None,
        "preprocess": False, "masks": True, "plot": False, "verbose": False,
        "test_mode": False, "timelapse": False, "n_jobs": 1,
        "adjust_cells": False, "consolidate": False,
        "batch_size": 10, "save": True, "custom_regex": None,
        "randomize": True, "examples_to_plot": 1,
    }
    settings.update(over)
    return settings


def _stub_mask_externals(mp):
    """Stub the GPU segmentation and the disk-heavy merge/plot collaborators."""
    import spacr.io as sio
    import spacr.object as sobj
    import spacr.plot as splot
    import spacr.utils as su

    spies = {"cellpose": _Spy(), "organelle": _Spy(), "concat": _Spy(),
             "overlay": _Spy(), "arrays": _Spy(), "adjust": _Spy(),
             "pivot": _Spy(), "cleanup": _Spy(ret=[])}
    mp.setattr(sobj, "generate_cellpose_masks_sam", spies["cellpose"])
    mp.setattr(sobj, "generate_organelle_masks_sam", spies["organelle"])
    mp.setattr(sio, "_load_and_concatenate_arrays", spies["concat"])
    mp.setattr(splot, "plot_image_mask_overlay", spies["overlay"])
    mp.setattr(splot, "plot_arrays", spies["arrays"])
    mp.setattr(su, "adjust_cell_masks", spies["adjust"])
    mp.setattr(su, "_pivot_counts_table", spies["pivot"])
    mp.setattr(su, "cleanup_pipeline_folders", spies["cleanup"])
    return spies


def _run_folder(root, *, fields=3, merged=0, nested=False):
    """A v1 run folder with ``fields`` stack arrays and ``merged`` merged ones."""
    src = (root / "a" / "deeply" / "nested" / "project" / "plate1" if nested
           else root / "plate1")
    (src / "stack").mkdir(parents=True)
    for i in range(fields):
        np.save(src / "stack" / f"f{i}.npy", np.zeros((4, 8, 8), np.uint16))
    if merged:
        (src / "merged").mkdir()
        for i in range(merged):
            np.save(src / "merged" / f"m{i}.npy", np.zeros((5, 8, 8), np.uint16))
    return src


def _mask_folder(src, name, n=3):
    folder = src / "masks" / name
    folder.mkdir(parents=True)
    for i in range(n):
        np.save(folder / f"f{i}.npy", np.ones((8, 8), np.uint16))
    return folder


def test_preprocess_generate_masks_refuses_a_missing_or_mistyped_src(subtests):
    """``src`` is required and must be a string or a list of strings."""
    cases = [
        ("no src key", {}, "src is a required parameter"),
        ("an integer src", {"src": 12345},
         "src must be a string or a list of strings"),
        ("a None src", {"src": None},
         "src must be a string or a list of strings"),
    ]
    for case_id, settings, message in cases:
        with subtests.test(msg=case_id):
            with pytest.raises(ValueError) as excinfo:
                core.preprocess_generate_masks(settings)
            assert str(excinfo.value) == message


def _case_v2_planes(folder, mp, spies, capsys):
    """The v2 dispatch points Cellpose at the planes the v2 stack really has."""
    import spacr._v1_v2_bridge as bridge
    import spacr.pipeline_v2 as pv2

    run = _Spy(ret={"stacks": []})
    mp.setattr(pv2, "run_v2", run)
    mp.setattr(bridge, "report_disk_savings", lambda *a, **k: None)
    seen = []
    for nucleus in (None, 0):
        run.calls.clear()
        core.preprocess_generate_masks(_mask_settings(
            folder, pipeline_style="v2", cell_channel=1,
            nucleus_channel=nucleus, pathogen_channel=None))
        assert run.n == 1
        kw = run.calls[0][1]
        seen.append((kw["channel_names"], kw["channels_for_cellpose"]))
    assert seen == [(["cell"], (0,)), (["nucleus", "cell"], (1, 0))]
    assert spies["cellpose"].n == 0


def _case_skip_complete_objects(folder, mp, spies, capsys):
    """Only the object type whose mask folder is missing is segmented again."""
    src = _run_folder(folder)
    names = ("nucleus_mask_stack", "pathogen_mask_stack", "organelle_mask_stack")
    for name in names:
        _mask_folder(src, name)
    core.preprocess_generate_masks(_mask_settings(src, organelle_channel=3))
    assert [c[0][2] for c in spies["cellpose"].calls] == ["cell"]
    assert spies["organelle"].n == 0
    printed = capsys.readouterr().out
    for name in names:
        assert f"All masks have been generated for {name}" in printed
    assert "All masks have been generated for cell_mask_stack" not in printed


def _case_adjust_without_organelle_folder(folder, mp, spies, capsys):
    """Adjustment gets an organelle path only once that folder exists."""
    src = _run_folder(folder)
    masks = os.path.join(str(src), "masks")
    settings = dict(adjust_cells=True, organelle_channel=3, n_jobs=2)
    core.preprocess_generate_masks(_mask_settings(src, **settings))
    (src / "masks" / "organelle_mask_stack").mkdir(parents=True, exist_ok=True)
    core.preprocess_generate_masks(_mask_settings(src, **settings))
    stacks = [os.path.join(masks, f"{n}_mask_stack")
              for n in ("pathogen", "cell", "nucleus", "organelle")]
    assert [call[0][:4] for call in spies["adjust"].calls] == [
        (stacks[0], stacks[1], stacks[2], None), tuple(stacks)]
    assert [call[1]["n_jobs"] for call in spies["adjust"].calls] == [2, 2]


def _case_test_mode_budget(folder, mp, spies, capsys):
    """``test_mode`` draws one overlay per merged field, however deep the path."""
    src = _run_folder(folder, fields=1, merged=2, nested=True)
    settings = _mask_settings(src, channels=[0, 1, 2], pathogen_channel=None,
                              plot=True, test_mode=True, examples_to_plot=1)
    core.preprocess_generate_masks(settings)
    assert settings["examples_to_plot"] == 2
    assert spies["overlay"].n == 2


def _case_unlistable_merged(folder, mp, spies, capsys):
    """An unlistable ``merged/`` is reported, no overlay is drawn, the run completes."""
    src = _run_folder(folder, fields=1, merged=3)
    real_listdir = os.listdir

    def _refuse_merged(path, *args, **kwargs):
        if str(path).endswith("merged"):
            raise PermissionError("merged is unreadable")
        return real_listdir(path, *args, **kwargs)

    mp.setattr(core.os, "listdir", _refuse_merged)
    core.preprocess_generate_masks(_mask_settings(
        src, channels=[0, 1, 2], pathogen_channel=None, plot=True,
        examples_to_plot=3))
    printed = capsys.readouterr().out
    assert "Failed to plot image mask overly." in printed
    assert "merged is unreadable" in printed
    assert "Successfully completed run" in printed
    assert spies["overlay"].n == 0


def _case_consolidate_per_plate(folder, mp, spies, capsys):
    """A list of plates is consolidated plate by plate, in order."""
    import spacr.utils as su

    roots = [folder / "plate1", folder / "plate2"]
    for root in roots:
        root.mkdir()
    calls = []

    def image_map(source):
        calls.append(("map", source))
        return {f"{source}/raw.tif": f"{source}/renamed.tif"}

    def consolidate(mapping, source):
        calls.append(("copy", source, mapping))
        os.makedirs(os.path.join(source, "consolidated"))

    mp.setattr(su, "generate_image_path_map", image_map)
    mp.setattr(su, "copy_images_to_consolidated", consolidate)
    settings = _mask_settings(roots[0], masks=False, consolidate=True)
    settings["src"] = [str(root) for root in roots]
    core.preprocess_generate_masks(settings)
    expected = []
    for root in map(str, roots):
        expected += [("map", root),
                     ("copy", root, {f"{root}/raw.tif": f"{root}/renamed.tif"})]
    assert calls == expected
    for root in roots:
        assert (root / "consolidated" / "settings").is_dir()


MASK_RUN_CASES = [
    ("v2 dispatch segments on the planes that exist", _case_v2_planes, {}),
    ("complete object types are not segmented again",
     _case_skip_complete_objects, {"cellpose": 1, "concat": 1, "cleanup": 1}),
    ("adjustment follows the organelle folder, not the setting",
     _case_adjust_without_organelle_folder,
     {"cellpose": 6, "organelle": 2, "concat": 2, "adjust": 2, "cleanup": 2}),
    ("test_mode overlay budget counts fields", _case_test_mode_budget,
     {"cellpose": 2, "concat": 1, "overlay": 2, "cleanup": 1}),
    ("an unlistable merged folder is reported", _case_unlistable_merged,
     {"cellpose": 2, "concat": 1, "cleanup": 1}),
    ("consolidation keeps each plate apart", _case_consolidate_per_plate,
     {"cleanup": 2}),
]


def test_preprocess_generate_masks_segments_plots_and_adjusts_only_what_is_due(
        subtests, tmp_path, monkeypatch, capsys):
    """Each orchestration scenario calls its collaborators exactly as the case states."""
    for case_id, case, calls in MASK_RUN_CASES:
        with subtests.test(msg=case_id):
            capsys.readouterr()
            with monkeypatch.context() as mp:
                spies = _stub_mask_externals(mp)
                case(_case_dir(tmp_path, case_id), mp, spies, capsys)
            assert {name: spy.n for name, spy in spies.items() if spy.calls} == calls


# ---------------------------------------------------------------------------
# generate_image_umap
# ---------------------------------------------------------------------------

def _measurement_db(path, plate):
    frame = pd.DataFrame({"plateID": [plate] * 3,
                          "rowID": [f"r{i + 1}" for i in range(3)],
                          "columnID": ["c1"] * 3, "area": range(3)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as db:
        frame.to_sql("cell", db, index=False)
    return str(path)


def test_generate_image_umap_warns_only_when_two_sources_share_a_plate(
        subtests, tmp_path, monkeypatch, capsys):
    """The plate-collision advisory fires on a shared id, names it, and never ends a run."""
    import spacr.io as sio
    import spacr.multi_database as multi_database
    import spacr.utils as su

    def _no_cell_table(*args, **kwargs):
        raise KeyError("cell")

    cases = [
        ("a shared plate id warns", ("plate1", "plate1"), None, True),
        ("different plates stay quiet", ("plate1", "plate2"), None, False),
        ("a failed check stays quiet and continues", ("plate1", "plate1"),
         _no_cell_table, False),
    ]
    for case_id, plates, describe, warns in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            runs = [folder / "runA", folder / "runB"]
            paths = [_measurement_db(run / "measurements" / "measurements.db",
                                     plate) for run, plate in zip(runs, plates)]
            capsys.readouterr()
            with monkeypatch.context() as mp:
                mp.setattr(su, "get_db_paths", lambda srcs, _p=paths: _p)
                mp.setattr(sio, "open_crop_source", _stop)
                if describe is not None:
                    mp.setattr(multi_database, "describe_merge", describe)
                with pytest.raises(_Stop):
                    core.generate_image_umap({"src": [str(r) for r in runs],
                                              "plot_images": False})
            printed = capsys.readouterr().out
            assert "Generating Image UMAP" in printed
            assert ("the same plate id appears in more than one" in printed) \
                is warns
            if warns:
                assert "'plate1'" in printed
                assert "runA" in printed and "runB" in printed
                assert core.UMAP_SOURCE_COLUMN in printed


def _feature_frame(n=8):
    """A measurement frame with two conditions and no crop columns at all."""
    return pd.DataFrame({
        "columnID": ["c1" if i % 2 == 0 else "c2" for i in range(n)],
        "cell_area": np.linspace(10.0, 90.0, n),
        "cell_perimeter": np.linspace(4.0, 40.0, n),
        "prcfo": [f"plate1_r1_c{(i % 2) + 1}_f1_o{i}" for i in range(n)],
    })


def test_generate_image_umap_filters_rows_and_keeps_the_frame_aligned(
        subtests, tmp_path, monkeypatch, capsys):
    """Excluded conditions and noise rows leave the frame together; ``verbose`` only reports."""
    import spacr.io as sio
    import spacr.utils as su

    area = _feature_frame(8)["cell_area"].to_numpy()
    noise = np.array([0, 0, -1, 0, -1, 1, 1, 1])
    cases = [
        # case id, extra settings, reducer labels, rows reduced,
        # expected cell_area, clusters, conds, printed / not printed
        ("string exclusion, quiet", dict(exclude_conditions="pos"), None, 4,
         area[::2], [0] * 4, ["neg"], [], ["Excluded 4 rows"]),
        ("list exclusion, verbose",
         dict(exclude_conditions=["pos"], verbose=True), None, 4, area[::2],
         [0] * 4, ["neg"], ["Excluded 4 rows after excluding: ['pos']"], []),
        ("noise removal without crops",
         dict(remove_cluster_noise=True, plot_images=True), noise, 8,
         area[noise != -1], [0, 0, 0, 1, 1, 1], ["neg", "pos"],
         ["plotting points only"], []),
    ]
    for (case_id, over, labels, n_reduced, kept_area, clusters, conds,
         said, unsaid) in cases:
        with subtests.test(msg=case_id):
            seen = {}

            def _cluster(numeric_data, *a, _labels=labels, **k):
                seen["rows"] = len(numeric_data)
                lab = (np.zeros(len(numeric_data), dtype=int)
                       if _labels is None else np.asarray(_labels))
                return np.zeros((len(numeric_data), 2)), lab, None

            settings = {"src": str(tmp_path), "tables": ["cell"], "pos": "c2",
                        "neg": "c1", "mix": "c3", "plot_images": False,
                        "verbose": False, "exclude_conditions": None,
                        "remove_cluster_noise": False, "n_jobs": 1,
                        "save_figure": False, "filter_by": None,
                        "row_limit": None, **over}
            capsys.readouterr()
            with monkeypatch.context() as mp:
                mp.setattr(su, "get_db_paths", lambda _src: [np.nan])
                mp.setattr(core, "_validate_umap_source_db",
                           lambda *a, **k: None)
                mp.setattr(sio, "_read_and_join_tables",
                           lambda *a, **k: _feature_frame(8))
                mp.setattr(su, "correct_paths", lambda df, _src: (df, []))
                mp.setattr(sio, "open_crop_source", lambda *a, **k: None)
                mp.setattr(su, "reduction_and_clustering", _cluster)
                out = core.generate_image_umap(settings)
            printed = capsys.readouterr().out
            assert seen["rows"] == n_reduced
            assert out["cell_area"].to_numpy() == pytest.approx(kept_area)
            assert list(out["cluster"]) == clusters
            assert sorted(out["cond"].unique()) == conds
            for text in said:
                assert text in printed
            for text in unsaid:
                assert text not in printed
