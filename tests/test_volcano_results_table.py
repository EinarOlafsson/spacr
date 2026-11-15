"""Volcano Explorer: which results CSV a path names, and how it is trimmed.

:func:`spacr.qt.screens.volcano.find_results_table` is handed real files and
folders on disk (a CSV, a non-CSV, a missing path, folders with several
candidate names, a run folder whose table sits in a leaf behind empty
siblings, a ``~`` path); :func:`spacr.qt.screens.volcano.load_results` reads
stacked permutation and multi-response tables and must come back with one
support family, one response and an index from zero.  Neither needs Qt.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from spacr.qt.screens import volcano


def _results_frame(n=24, seed=5):
    """A regression results table with ``n`` guides over ``n // 3`` genes."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "guide": [f"g{i}" for i in range(n)],
        "gene": [f"TGGT1_{200000 + i // 3}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(0, 0.08, n),
        "adjusted_p_value": np.clip(rng.beta(0.6, 6, n), 1e-8, 1),
        "wells_with_guide": rng.integers(1, 12, n),
        "significant": False,
        "alpha": 0.05,
    })


def _write_tree(root, files):
    """Create ``files`` (relative path -> 'csv' | 'pdf' | 'dir' | text) under ``root``."""
    for relative, kind in files.items():
        path = root / relative
        if kind == "dir":
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "csv":
            _results_frame().to_csv(path, index=False)
        elif kind == "pdf":
            path.write_bytes(b"%PDF-1.4\n")
        else:
            path.write_text(kind, encoding="utf-8")


def test_find_results_table_resolves_each_kind_of_path(tmp_path, monkeypatch,
                                                       subtests):
    """A CSV is itself, a folder yields its best-named table, anything else None."""
    cases = [
        # (case id, files under the case folder, path asked for, expected)
        ("csv handed in directly", {"run/results.csv": "csv"},
         "run/results.csv", "run/results.csv"),
        ("upper-case extension is still a csv", {"RUN.CSV": "csv"},
         "RUN.CSV", "RUN.CSV"),
        ("a pdf is not a table", {"volcano.pdf": "pdf"},
         "volcano.pdf", None),
        ("a missing path is not a table", {}, "never_ran", None),
        ("guide-level permutation table wins",
         {"both/results.csv": "csv", "both/results_grna.csv": "csv",
          "both/guide_permutation_results_long.csv": "csv"},
         "both", "both/guide_permutation_results_long.csv"),
        ("results_grna beats results",
         {"pair/results.csv": "csv", "pair/results_grna.csv": "csv"},
         "pair", "pair/results_grna.csv"),
        ("a folder of the wrong csvs holds no table",
         {"settings_only/settings.csv": "key,value\n"},
         "settings_only", None),
        ("the run folder finds the table in its leaf",
         {"run_2024/results.csv": "csv"}, ".", "run_2024/results.csv"),
        ("empty subfolders sorting first do not stop the search",
         {"run/aaa_figures": "dir", "run/bbb_logs": "dir",
          "run/zzz_guide_permutation/results_grna.csv": "csv"},
         "run", "run/zzz_guide_permutation/results_grna.csv"),
        ("a run folder with no table anywhere",
         {"run/figures/volcano.pdf": "pdf", "run/logs": "dir"}, "run", None),
    ]
    for index, (case_id, files, asked, expected) in enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / f"case{index}"
            root.mkdir()
            _write_tree(root, files)
            want = None if expected is None else os.path.abspath(root / expected)
            assert volcano.find_results_table(root / asked) == want
            # A str path and a PathLike name the same table.
            assert volcano.find_results_table(str(root / asked)) == want

    with subtests.test(msg="a ~ path reaches the same table"):
        home = tmp_path / "home"
        _write_tree(home, {"run_2024/results.csv": "csv"})
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        assert volcano.find_results_table("~/run_2024") == os.path.abspath(
            home / "run_2024" / "results.csv")


def test_load_results_keeps_one_family_and_one_response(tmp_path, subtests):
    """The lowest support family and the first response survive, indexed from 0."""
    def stacked(column, values, n, seeds=None):
        parts = []
        for position, value in enumerate(values):
            part = _results_frame(n=n, seed=(seeds or [5] * len(values))[position])
            part[column] = value
            parts.append(part)
        return pd.concat(parts)

    cases = [
        # (case id, frame written, expected column -> set, expected guides)
        ("four stacked support families keep the smallest",
         stacked("minimum_wells_threshold", [2, 3, 4, 5], 12),
         ("minimum_wells_threshold", {2}), [f"g{i}" for i in range(12)]),
        ("the smallest family wins even when it comes last",
         stacked("minimum_wells_threshold", [9, 2], 6, seeds=[5, 11]),
         ("minimum_wells_threshold", {2}), [f"g{i}" for i in range(6)]),
        ("two responses: only the first plots",
         stacked("outcome", ["infected", "parasite_count"], 8, seeds=[5, 9]),
         ("outcome", {"infected"}), [f"g{i}" for i in range(8)]),
        ("one response everywhere is not split",
         stacked("outcome", ["infected"], 8),
         ("outcome", {"infected"}), [f"g{i}" for i in range(8)]),
        ("a plain table is read as is",
         _results_frame(n=5), ("alpha", {0.05}), [f"g{i}" for i in range(5)]),
    ]
    for index, (case_id, frame, (column, values), guides) in enumerate(cases):
        with subtests.test(msg=case_id):
            path = tmp_path / f"results_{index}.csv"
            frame.to_csv(path, index=False)
            loaded = volcano.load_results(path)
            assert set(loaded[column]) == values
            assert list(loaded["guide"]) == guides
            assert list(loaded.index) == list(range(len(guides)))
            assert list(loaded.columns) == list(frame.columns)

    with subtests.test(msg="the kept family's rows are the file's own rows"):
        early = _results_frame(n=6)
        early["minimum_wells_threshold"] = 9
        late = _results_frame(n=6, seed=11)
        late["minimum_wells_threshold"] = 2
        path = tmp_path / "gapped.csv"
        pd.concat([early, late]).to_csv(path, index=False)
        expected = pd.read_csv(path).iloc[6:].reset_index(drop=True)
        pd.testing.assert_frame_equal(volcano.load_results(path), expected)
