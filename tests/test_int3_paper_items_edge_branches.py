"""Edge branches of the third paper-item integration and the Model Zoo fixes.

The E-test power settings fall back to their defaults when left blank; the
retrieval power curve writes an empty test table when no paired draw can be
tested (identical scores, or a reference table without a reference column);
and the Model Zoo scorecard and column-width helpers degrade quietly when the
preference store, the curated scorecard or the model card is unavailable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr import gene_measurement_compare as G
from spacr.ops_engine import run_ops
from tests.test_ops_etest_power_917 import _cells_folder


def test_blank_power_settings_take_the_defaults():
    assert G._etest_list(None, [10, 25]) == [10, 25]
    assert G._etest_list("  ", (3,)) == [3]


def _retrieval(tmp_path, monkeypatch, per_gene_columns):
    """Run the retrieval power curve with a fixed per-gene table."""
    from spacr import ops_reference

    def evaluate(names, pcs, fitness, reference_dir, settings, arm=""):
        genes = [n for n in names if not str(n).startswith("nontargeting")]
        table = pd.DataFrame({"gene": genes, "ap": np.linspace(0.1, 0.9, len(genes)),
                              "reference": "CORUM"})
        return (pd.DataFrame({"metric": ["mAP"], "value": [0.5], "arm": [arm]}),
                table[per_gene_columns])

    monkeypatch.setattr(ops_reference, "evaluate_gene_profiles", evaluate)
    folder = _cells_folder(tmp_path)
    run_ops({"dst_root": str(tmp_path / "run"), "ops_etest_cells": str(folder),
             "ops_etest_reference_cells": 500, "ops_etest_null": 19, "ops_etest_holdout": 10,
             "ops_power_cells": [10, 50], "ops_power_seeds": [0], "ops_power_retrieval": True,
             "ops_reference_dir": str(tmp_path), "ops_profile_min_cells": 5}, phases=["etest"])
    return pd.read_csv(tmp_path / "run" / "results" / "ops_etest" / "power_retrieval_tests.csv")


def test_identical_retrieval_scores_are_not_tested(tmp_path, monkeypatch):
    tests = _retrieval(tmp_path, monkeypatch, ["gene", "ap", "reference"])
    assert tests.empty and "p_wilcoxon" in tests.columns


def test_a_per_gene_table_without_references_writes_no_tests(tmp_path, monkeypatch):
    tests = _retrieval(tmp_path, monkeypatch, ["gene", "ap"])
    assert tests.empty and list(tests.columns)[:2] == ["reference", "n"]


pytest.importorskip("PySide6")


def test_the_zoo_helpers_degrade_quietly(monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.screens import model_zoo as mz
    from spacr.qt.widgets import model_zoo_picker as mzp

    def broken(*_a, **_k):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(preferences, "scaled_px", broken)
    assert mz._px(42) == 42
    assert mz._score_text(None) == "" and mz._score_text("n/a") == "n/a"
    assert mz._score_text(0.5) == "0.500"
    assert mz._score_difference(None, 0.2) == ""
    monkeypatch.setattr(mzp, "_store", broken)
    assert mz._remembered_widths() == []
    mz._remember_widths([100, 200])


def test_a_scorecard_without_curated_rows_or_card_is_empty(monkeypatch):
    from spacr import model_cards
    from spacr import model_zoo as zoo
    from spacr.qt.screens import model_zoo as mz

    class Entry:
        """A curated entry whose metrics hold no score pairs."""
        _curated_card = True
        metrics = {"f1": 0.7}

    monkeypatch.setattr(zoo, "_curated_scorecard_rows", lambda: (("F1", "f1"),))
    monkeypatch.setattr(model_cards, "card_for_entry",
                        lambda entry: (_ for _ in ()).throw(RuntimeError("no card")))
    assert mz._scorecard_rows(Entry()) == []

    def broken():
        raise RuntimeError("no curated rows")

    monkeypatch.setattr(zoo, "_curated_scorecard_rows", broken)
    card = {"scores": {"rows": [{"label": "F1", "this": 0.8, "stock": 0.5}]}}
    assert mz._scorecard_rows(Entry(), card) == [("F1", "0.800", "0.500", "+0.300")]


def test_a_second_show_does_not_queue_another_read(qtbot, monkeypatch):
    from spacr import model_zoo as zoo
    from spacr.qt.screens.model_zoo import ModelZooScreen

    calls = []
    monkeypatch.setattr(zoo, "catalogue", lambda *a, **k: calls.append(1) or [])
    screen = ModelZooScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.show()
    qtbot.wait(50)
    screen.hide()
    screen.show()
    qtbot.wait(50)
    assert screen._first_load_queued and len(calls) <= 1
