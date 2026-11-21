"""The OPS cell-cycle option (item 990, alpha): stage, probability and flag per cell.

Synthetic nuclei carry planted 2N and 4N DNA content, condensed 4N metaphases, 2N
ana_telo_cyto daughter pairs, telophase daughters and G2 prophase cells. G1/S/G2 come from
item 995's field-centred fit (``spacr.measure._cell_table_phases_measure_fc``), called for real.
"""
from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import ops_cell_cycle as cc
from spacr import ops_engine

PLATE = "P1"


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """Few permutations."""
    monkeypatch.setattr(cc, "PERMUTATIONS", 20)


SHARES = {"G1": 0.5, "S": 0.2, "G2": 0.16, "g2_prophase": 0.03, "prometaphase": 0.02,
          "metaphase": 0.03, "ana_telo_cyto": 0.03, "telophase_daughters": 0.03}


def _cells(seed=0, wells=("A1", "A2"), sites=6, per_site=220, shuffle_stage=False):
    """Synthetic cells with their true stage; daughters sit next to each other."""
    rng = np.random.default_rng(seed)
    rows = []
    names, shares = list(SHARES), np.array(list(SHARES.values()))
    for well in wells:
        for site in range(sites):
            label = 0
            for _ in range(per_site):
                stage = names[rng.choice(len(names), p=shares / shares.sum())]
                content = {"G1": 2.0, "S": rng.uniform(2.2, 3.8), "G2": 4.0, "g2_prophase": 4.0,
                           "prometaphase": 4.0, "metaphase": 4.0, "ana_telo_cyto": 2.0,
                           "telophase_daughters": 2.0}[stage]
                content *= np.exp(rng.normal(0, 0.05))
                condensed = stage in ("prometaphase", "metaphase", "ana_telo_cyto",
                                      "telophase_daughters")
                area = rng.normal({"prometaphase": 140, "metaphase": 80, "ana_telo_cyto": 90,
                                   "telophase_daughters": 120, "g2_prophase": 230}.get(stage, 200), 8)
                y, x = rng.uniform(0, 2000, 2)
                spots = rng.normal({"g2_prophase": 8, "prometaphase": 5, "metaphase": 2,
                                    "ana_telo_cyto": 2}.get(stage, 1), 0.5)
                for twin in range(2 if stage in ("ana_telo_cyto", "telophase_daughters") else 1):
                    label += 1
                    rows.append({"well": well, "site": site, "label": label,
                                 "stage": "interphase" if stage in ("G1", "S", "G2") else stage,
                                 "true_phase": stage, "nucleus_area": area,
                                 "nucleus_eccentricity": rng.normal(
                                     {"prometaphase": 0.65, "metaphase": 0.9,
                                      "telophase_daughters": 0.4}.get(
                                         stage, 0.8 if condensed else 0.5), 0.04),
                                 "nucleus_centroid_y": y + 12 * twin, "nucleus_centroid_x": x,
                                 "nucleus_dapi_integrated_intensity": 1000 * content,
                                 "nucleus_dapi_mean_intensity": 1000 * content / area,
                                 "nucleus_dapi_std_intensity": spots * 10 + rng.normal(0, 1)})
    frame = pd.DataFrame(rows)
    if shuffle_stage:
        frame["stage"] = rng.permutation(frame["stage"].to_numpy())
    return frame


def _base(tmp_path, seed=0):
    """A base checkpoint trained on planted labels."""
    cells = _cells(seed=seed, wells=("B1",), sites=8)
    bundle = cc.train_cell_cycle_model(cells, cells["stage"], input_mode="DNA", dna="dapi",
                                       seed=0, repeats=1, permutations=5)
    return cc.save_cell_cycle_checkpoint(bundle, str(tmp_path / "models"))


def _run(tmp_path, base, *, labels=None, folder="run", seed=3):
    """The ph_cell_cycle phase on a two-well plate."""
    destination = tmp_path / folder
    destination.mkdir(exist_ok=True)
    cells = _cells(seed=seed)
    for well, rows in cells.groupby("well"):
        rows.drop(columns=["well", "stage", "true_phase"]).to_parquet(
            ops_engine._ph_cells_path(str(destination), well), index=False)
    settings = {"ops_cell_cycle": "DNA", "ops_cell_cycle_base_model": base,
                "nucleus_channel": "dapi", "ops_cell_cycle_labels": labels or ""}
    db = str(destination / "measurements.db")
    out = cc.run_cell_cycle_phase(db, PLATE, ["A1", "A2"], settings, str(destination))
    from spacr.ops_store import read_table

    return out, cells, read_table(db, "ops_cell_cycle"), read_table(db, "ops_cell_cycle_checks")


def test_the_collapse_rule_follows_the_parent_map():
    assert cc.collapse_stage("metaphase", ["metaphase"]) == "mitotic"
    assert cc.collapse_stage("metaphase", ["metaphase", "mitotic"]) == "G2/M"
    assert cc.collapse_stage("g2_prophase", ["g2_prophase"]) == "G2"
    assert cc.collapse_stage("S", ["G1", "S", "G2"]) == "interphase"
    assert cc.collapse_stage("G2", ["G2", "interphase"]) == "interphase"
    assert cc.collapse_stage("artefact", ["artefact"]) == "artefact"
    assert cc.screen_collapses({"P1": ["metaphase"], "P2": ["ana_telo_cyto"]}) == [
        "ana_telo_cyto", "metaphase"]


def test_failed_checks_name_the_stages_to_collapse():
    def checks(**passed):
        return pd.DataFrame([{"check": k, "passed": v, "value": 0.95 if k == "S6" else 1.0}
                             for k, v in passed.items()])

    assert cc.failed_stages(checks(S1=True, S2=True, S4=True, S6=True)) == []
    assert cc.failed_stages(checks(S1=False, S2=False, S4=True)) == [
        "ana_telo_cyto", "metaphase", "prometaphase", "telophase_daughters"]
    assert "mitotic" in cc.failed_stages(checks(S1=False, S2=True, S4=False))
    assert cc.failed_stages(checks(S6=False)) == ["S/G2"]
    low = pd.DataFrame([{"check": "S6", "passed": False, "value": 0.5}])
    assert cc.failed_stages(low) == ["G1", "G2", "S"]
    assert cc.failed_stages(checks(S1=None, S2=None, S3=None)) == []
    assert cc.failed_stages(checks(S3=False)) == [
        "ana_telo_cyto", "g2_prophase", "metaphase", "prometaphase", "telophase_daughters"]


def test_planted_stages_pass_their_self_checks_and_are_called(tmp_path):
    out, cells, table, checks = _run(tmp_path, _base(tmp_path))
    assert out["collapsed"] == [] and set(table["cell_cycle_flag"]) == {"auto"}
    passed = dict(zip(checks["check"], checks["passed"]))
    assert passed["S1"] == 1 and passed["S2"] == 1 and passed["S6"] == 1
    called = cells[["well", "site", "label", "true_phase"]].merge(
        table[["well", "site", "label", "cell_cycle_stage"]], on=["well", "site", "label"])
    for stage in ("metaphase", "ana_telo_cyto", "telophase_daughters", "G1", "G2"):
        rows = called[called["true_phase"] == stage]
        assert (rows["cell_cycle_stage"] == stage).mean() > 0.8, stage
    assert {"p_metaphase", "dna_log2_fc", "dna_phase_posterior_G1", "model_id"} <= set(table)
    assert (tmp_path / "run" / f"ops_cell_cycle_report_{PLATE}.html").is_file()


def test_shuffled_labels_fail_the_shuffled_check_and_collapse_to_parents(tmp_path):
    cells = _cells(seed=3, shuffle_stage=True)
    labels = cells.sample(300, random_state=0)[["well", "site", "label", "stage"]]
    labels.insert(0, "plate", PLATE)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    out, _cells_, table, checks = _run(tmp_path, _base(tmp_path),
                                       labels=str(tmp_path / "labels.csv"))
    assert dict(zip(checks["check"], checks["passed"]))["S3"] == 0
    assert {"metaphase", "prometaphase", "ana_telo_cyto", "telophase_daughters"} <= set(
        out["collapsed"])
    assert not table["cell_cycle_stage"].isin(["metaphase", "ana_telo_cyto",
                                               "telophase_daughters"]).any()
    assert (table.loc[table["cell_cycle_stage"] == "mitotic", "cell_cycle_flag"]
            == "unresolved").all()


def test_true_labels_verify_their_stages(tmp_path):
    cells = _cells(seed=3)
    labels = cells.groupby("stage", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 60), random_state=0))[["well", "site", "label", "stage"]]
    labels.insert(0, "plate", PLATE)
    labels.to_csv(tmp_path / "labels.csv", index=False)
    _out, _c, table, checks = _run(tmp_path, _base(tmp_path), labels=str(tmp_path / "labels.csv"))
    assert dict(zip(checks["check"], checks["passed"]))["S3"] == 1
    assert (table.loc[table["cell_cycle_stage"] == "metaphase", "cell_cycle_flag"]
            == "verified").all()


def test_a_plate_collapse_reaches_every_plate_of_the_screen(tmp_path):
    base = _base(tmp_path)
    _run(tmp_path, base)
    db = str(tmp_path / "run" / "measurements.db")
    from spacr.ops_store import read_table, write_table

    checks = read_table(db, "ops_cell_cycle_checks")
    write_table(db, "ops_cell_cycle_checks",
                pd.concat([checks, checks.assign(plate="P0", collapsed="metaphase")]),
                if_exists="replace")
    out, _c, table, _k = _run(tmp_path, base)
    assert out["collapsed"] == ["metaphase"] and out["plate_collapsed"] == []
    assert not (table["cell_cycle_stage"] == "metaphase").any()


def test_a_rerun_of_the_same_settings_is_identical(tmp_path):
    base = _base(tmp_path)
    assert _base(tmp_path) == base
    first = _run(tmp_path, base, folder="a")
    second = _run(tmp_path, base, folder="b")
    pd.testing.assert_frame_equal(first[2], second[2])
    pd.testing.assert_frame_equal(first[3], second[3])


def test_a_checkpoint_with_a_wrong_hash_or_format_is_refused(tmp_path):
    base = _base(tmp_path)
    assert cc.load_cell_cycle_checkpoint(base)["card"]["format"] == cc.CHECKPOINT_FORMAT
    with open(os.path.join(base, "model.joblib"), "ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        cc.load_cell_cycle_checkpoint(base)
    other = _base(tmp_path / "second")
    card = json.load(open(os.path.join(other, "card.json")))
    card["format"] = "something-else"
    json.dump(card, open(os.path.join(other, "card.json"), "w"))
    with pytest.raises(ValueError):
        cc.load_cell_cycle_checkpoint(other)


def test_cnn_falls_back_to_the_tabular_learner_and_says_so(tmp_path, monkeypatch):
    base = _base(tmp_path)
    destination = tmp_path / "run"
    destination.mkdir()
    for well, rows in _cells(seed=3).groupby("well"):
        rows.drop(columns=["well", "stage", "true_phase"]).to_parquet(
            ops_engine._ph_cells_path(str(destination), well), index=False)
    out = cc.run_cell_cycle_phase(str(destination / "measurements.db"), PLATE, ["A1", "A2"],
                                  {"ops_cell_cycle": "DNA", "ops_cell_cycle_base_model": base,
                                   "ops_cell_cycle_model_family": "cnn",
                                   "nucleus_channel": "dapi"}, str(destination))
    assert any("tabular" in note and "cnn" in note for note in out["notes"])


def test_the_knockout_check_needs_the_arrest_set_enriched(tmp_path):
    cells = _cells(seed=5, wells=("A1",), sites=4, per_site=400)
    stage = np.full(len(cells), "interphase", object)
    rng = np.random.default_rng(0)
    genes = np.array([f"G{i}" for i in rng.integers(0, 40, len(cells))], object)
    genes[:300] = "KIF11"
    genes[300:600] = "nontargeting"
    stages = pd.DataFrame({"model_stage": stage}, index=cells.index)
    dna = pd.DataFrame({"dna_content_centred": cells["nucleus_dapi_integrated_intensity"]})
    from spacr.measure import _cell_table_phases_measure_fc

    fit = _cell_table_phases_measure_fc(
        cells.assign(_dna_integrated=cells["nucleus_dapi_integrated_intensity"], mitotic=False))[1]
    results = {}
    for fold in (1, 3):
        stage = np.full(len(cells), "interphase", object)
        stage[rng.random(len(cells)) < 0.05] = "metaphase"
        stage[:300] = np.where(rng.random(300) < 0.05 * fold, "metaphase", "interphase")
        stages["model_stage"] = stage
        checks = cc.self_checks(cells.assign(gene=genes), stages, dna, fit, card={},
                                permutations=100, seed=1)
        results[fold] = checks.set_index("check").loc["S5", "passed"]
    assert results == {1: False, 3: True}


def test_none_leaves_the_run_unchanged(tmp_path):
    """ops_cell_cycle None writes the same files as a run without the key."""
    from tests.test_ops_phenotype_path import test_merge_and_aggregate_run_without_a_sequencing_folder as run

    outputs = {}
    for tag, extra in (("without", {}), ("none", {"ops_cell_cycle": None})):
        folder = tmp_path / tag
        folder.mkdir()
        original = ops_engine.run_ops

        def patched(settings, **kwargs):
            """run_ops with the extra settings."""
            return original({**settings, **extra}, **kwargs)

        ops_engine.run_ops = patched
        try:
            run(folder)
        finally:
            ops_engine.run_ops = original
        outputs[tag] = {p.name: (gzip.decompress(p.read_bytes()) if p.suffix == ".gz"
                                 else p.read_bytes())
                        for p in sorted(folder.iterdir())
                        if p.suffix in (".csv", ".parquet") and "report" not in p.name}
        assert not list(folder.glob("ops_ph_cell_cycle_*.json"))
    assert outputs["without"] == outputs["none"] and outputs["none"]


def test_the_settings_are_alpha_with_short_tooltips():
    from spacr.ops_settings import OPS_CATEGORIES, OPS_DEFAULTS, OPS_TOOLTIPS
    from spacr.settings import ALPHA_FEATURES

    keys = ALPHA_FEATURES[990]["settings"]
    assert set(keys) == {k for k in OPS_DEFAULTS if k.startswith("ops_cell_cycle")}
    assert set(keys) == set(OPS_CATEGORIES["Cell-cycle stages"])
    assert OPS_DEFAULTS["ops_cell_cycle"] is None and OPS_DEFAULTS["ops_cell_cycle_tabular"] == "hgb"
    for key in keys:
        assert len(OPS_TOOLTIPS[key]) <= 600 and "Alpha" in OPS_TOOLTIPS[key]
    assert "ops_mitotic_source 'cell_cycle'" in OPS_TOOLTIPS["ops_cell_cycle"]


def _staged(n=400, seed=0):
    """An ops_cell_cycle-like table with probabilities and model stages."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(len(cc.CLASSES)) * 0.3, n)
    table = pd.DataFrame(p, columns=[f"p_{c}" for c in cc.CLASSES])
    table["model_stage"] = [cc.CLASSES[i] for i in p.argmax(axis=1)]
    table["plate"] = np.where(np.arange(n) < n - 3, "P1", "P2")
    table["well"] = "A1"
    table["site"] = np.arange(n) // 50
    table["label"] = np.arange(n) % 50 + 1
    return table


def test_the_label_queue_covers_every_stage_and_plate_and_is_seeded():
    table = _staged()
    queue = cc.build_label_queue(table, 40, seed=1)
    assert len(queue) == 40 and queue.equals(cc.build_label_queue(table, 40, seed=1))
    for stage in cc.CLASSES:
        assert (queue["model_stage"] == stage).sum() >= min(4, (table["model_stage"] == stage).sum())
    assert set(queue["plate"]) == {"P1", "P2"}


def _queue_folder(tmp_path, monkeypatch):
    """A written label queue over a stand-in crop store."""
    from spacr import ops_run_infra

    table = _staged(60)
    queue = cc.build_label_queue(table, 12, seed=0)
    index = queue.assign(shard=str(tmp_path / "data" / "cell" / "P1" / "A1" / "shard-0000.tar"),
                         offset_data=0, size=0)[["shard", "site", "label", "offset_data", "size"]]
    monkeypatch.setattr(ops_run_infra, "crop_index", lambda root, plate, wells=None: index)
    rng = np.random.default_rng(0)
    monkeypatch.setattr(ops_engine, "_ph_read_crop",
                        lambda shard, offset, size: (rng.integers(0, 4000, (2, 16, 16)), None))
    folder = cc.write_label_queue(queue, str(tmp_path), str(tmp_path / "queue"),
                                  labels=str(tmp_path / "labels.csv"))
    return queue, folder


def test_the_queue_round_trips_through_annotate_into_the_labels_file(tmp_path, monkeypatch):
    import sqlite3

    queue, folder = _queue_folder(tmp_path, monkeypatch)
    db = os.path.join(folder, "measurements", "measurements.db")
    with sqlite3.connect(db) as connection:
        paths = [r[0] for r in connection.execute("SELECT png_path FROM png_list")]
        assert len(paths) == 12 and all(os.path.isfile(p) for p in paths)
        connection.executemany("UPDATE png_list SET annotate = ? WHERE png_path = ?",
                               [(4, paths[0]), (1, paths[1])])
    assert cc.collect_label_queue(folder, user="einar") == 2
    labels = cc.read_cell_cycle_labels(str(tmp_path / "labels.csv"))
    assert sorted(labels["stage"]) == ["interphase", "metaphase"]
    assert cc.collect_label_queue(folder, user="einar") == 2
    raw = pd.read_csv(tmp_path / "labels.csv")
    assert len(raw) == 4 and set(raw["user"]) == {"einar"}


def test_the_home_job_is_registered_and_retired(tmp_path, monkeypatch, qapp):
    from spacr.qt.bridge import registry

    _queue, folder = _queue_folder(tmp_path, monkeypatch)
    handle = cc.track_label_job(folder, 12)
    assert handle in registry().active() and handle.app_key == "Cell cycle: label 12 cells"
    reran = []
    assert cc.finish_label_job(handle, rerun=reran.append) == 0
    assert handle not in registry().active() and reran == []


def test_a_run_with_a_label_queue_writes_the_pending_queue(tmp_path):
    destination = tmp_path / "run"
    destination.mkdir()
    for well, rows in _cells(seed=3).groupby("well"):
        rows.drop(columns=["well", "stage", "true_phase"]).to_parquet(
            ops_engine._ph_cells_path(str(destination), well), index=False)
    out = cc.run_cell_cycle_phase(str(destination / "measurements.db"), PLATE, ["A1", "A2"],
                                  {"ops_cell_cycle": "DNA", "nucleus_channel": "dapi",
                                   "ops_cell_cycle_base_model": _base(tmp_path),
                                   "ops_cell_cycle_label_queue": 50}, str(destination))
    pending = pd.read_csv(os.path.join(out["label_queue"]["folder"], "queue_pending.csv"))
    assert len(pending) == 50 and out["label_queue"]["labels"].endswith("labels.csv")


def test_the_report_shows_the_dna_fit_and_the_checks_against_their_controls(tmp_path):
    out, _cells_, _table, _checks = _run(tmp_path, _base(tmp_path))
    page = open(out["report"], encoding="utf-8").read()
    assert page.count("<img src='data:image/png;base64,") == 2
    assert "against its control" in page and "S2" in page


class _Screen:
    """A stand-in Annotate screen that records what it opened."""

    def __init__(self):
        self.opened = []

    def _open_source(self, folder):
        """Record the folder."""
        self.opened.append(folder)


class _Window:
    """A stand-in main window: open_module and its built screens."""

    def __init__(self):
        self.modules, self._screens = [], {"annotate": _Screen()}

    def open_module(self, key):
        """Record the module and return its key."""
        self.modules.append(key)
        return key


def test_a_run_with_a_queue_puts_a_job_on_home_that_opens_annotate(tmp_path, monkeypatch, qapp):
    from types import SimpleNamespace

    from spacr.qt.bridge import registry
    from spacr.qt.screens.app_screen import AppScreen

    _queue, folder = _queue_folder(tmp_path, monkeypatch)
    window = _Window()
    screen = SimpleNamespace(window=lambda: window)
    payload = {"ph_cell_cycle": {"label_queue": {
        "folder": folder, "cells": 12, "pending": os.path.join(folder, "queue.csv"),
        "crop_root": str(tmp_path), "labels": str(tmp_path / "labels.csv")}}}
    before = set(registry().active())
    AppScreen._offer_cell_cycle_label_job(screen, payload)
    (handle,) = set(registry().active()) - before
    assert handle.app_key == "Cell cycle: label 12 cells"
    handle.open_job()
    assert window.modules == ["annotate"] and window._screens["annotate"].opened == [folder]
    AppScreen._offer_cell_cycle_label_job(screen, {"ph_cell_cycle": {"label_queue": None}})
    assert set(registry().active()) - before == {handle}
    handle.retire()


def test_home_open_runs_a_jobs_own_opener(qapp):
    from types import SimpleNamespace

    from spacr.qt.widgets.home import RunningBanner

    calls, emitted = [], []
    banner = SimpleNamespace(_handle=SimpleNamespace(app_key="x", open_job=lambda: calls.append(1)),
                             open_requested=SimpleNamespace(emit=emitted.append))
    RunningBanner._on_open(banner)
    assert calls == [1] and emitted == []
    banner._handle = SimpleNamespace(app_key="ops")
    RunningBanner._on_open(banner)
    assert emitted == ["ops"]


def test_the_finished_job_journals_the_labels_file_hash(tmp_path, monkeypatch, qapp):
    import hashlib
    import sqlite3

    monkeypatch.setenv("SPACR_HOME", str(tmp_path / "home"))
    _queue, folder = _queue_folder(tmp_path, monkeypatch)
    with sqlite3.connect(os.path.join(folder, "measurements", "measurements.db")) as connection:
        first = connection.execute("SELECT png_path FROM png_list").fetchone()[0]
        connection.execute("UPDATE png_list SET annotate = 4 WHERE png_path = ?", (first,))
    handle = cc.track_label_job(folder, 12)
    reran = []
    assert cc.finish_label_job(handle, rerun=reran.append, settings={"plate": PLATE}) == 1
    labels = str(tmp_path / "labels.csv")
    assert reran == [labels]
    manifest = json.load(open(os.path.join(handle.journal, "manifest.json")))
    digest = hashlib.sha256(open(labels, "rb").read()).hexdigest()
    assert digest in json.dumps(manifest)


def test_a_checkpoint_on_the_old_classes_is_refused_with_a_reason(tmp_path):
    base = _base(tmp_path)
    card = json.load(open(os.path.join(base, "card.json")))
    card["classes"] = [c for c in cc.CLASSES if c != "telophase_daughters"]
    json.dump(card, open(os.path.join(base, "card.json"), "w"))
    import hashlib

    sums = {name: hashlib.sha256(open(os.path.join(base, name), "rb").read()).hexdigest()
            for name in ("card.json", "model.joblib")}
    open(os.path.join(base, "SHA256SUMS"), "w").writelines(
        f"{v}  {k}\n" for k, v in sorted(sums.items()))
    with pytest.raises(ValueError, match="telophase_daughters became a stage"):
        cc.load_cell_cycle_checkpoint(base)


def test_telophase_daughters_count_as_mitotic_and_collapse_to_mitotic():
    assert "telophase_daughters" in cc.MITOTIC_STAGES
    assert cc.collapse_stage("telophase_daughters", ["telophase_daughters"]) == "mitotic"


def test_the_store_gets_a_cell_cycle_part(tmp_path, monkeypatch):
    from spacr import store
    from spacr.ops_store import write_table

    db = str(tmp_path / "measurements.db")
    write_table(db, "ops_cell_cycle", pd.DataFrame({
        "plate": PLATE, "well": "A1", "site": [2, 1], "label": [1, 1],
        "cell_cycle_stage": ["G1", "metaphase"]}))
    parts = []
    monkeypatch.setattr(store, "write_part", lambda project, name, rows, **kw: parts.append(
        (name, list(rows["site"]), kw["key_columns"])))
    monkeypatch.setattr(store, "register", lambda project, run=None: 1)
    ops_engine._ops_store_write(db, PLATE, ["A1"], {}, str(tmp_path))
    assert parts == [("cell_cycle", [2, 1], ("site", "label"))]


def _two_plate_labels(tmp_path):
    """Labels for two plates in one labels file."""
    rows = []
    for plate, seed in (("P1", 3), ("P2", 4)):
        cells = _cells(seed=seed)
        picked = cells.groupby("stage", group_keys=False).apply(
            lambda g: g.sample(min(len(g), 20), random_state=0))
        rows.append(picked[["well", "site", "label", "stage"]].assign(plate=plate))
    pd.concat(rows).to_csv(tmp_path / "labels.csv", index=False)
    return str(tmp_path / "labels.csv")


def _plate_run(tmp_path, base, labels, plate, seed, finetune):
    """The phase on one plate of a two-plate screen."""
    destination = tmp_path / plate
    destination.mkdir(exist_ok=True)
    for well, rows in _cells(seed=seed).groupby("well"):
        rows.drop(columns=["well", "stage", "true_phase"]).to_parquet(
            ops_engine._ph_cells_path(str(destination), well), index=False)
    return cc.run_cell_cycle_phase(str(destination / "measurements.db"), plate, ["A1", "A2"],
                                   {"ops_cell_cycle": "DNA", "nucleus_channel": "dapi",
                                    "ops_cell_cycle_base_model": base,
                                    "ops_cell_cycle_labels": labels,
                                    "ops_cell_cycle_finetune": finetune}, str(destination))


def test_screen_fine_tune_pools_the_plates_and_plate_fine_tune_does_not(tmp_path):
    base = _base(tmp_path)
    labels = _two_plate_labels(tmp_path)
    _plate_run(tmp_path, base, labels, "P1", 3, "screen")
    second = _plate_run(tmp_path, base, labels, "P2", 4, "screen")
    assert any("of other plates" in n and not n.endswith(" 0 of other plates")
               for n in second["notes"])
    pool = pd.read_parquet(labels + ".features.parquet")
    assert set(pool["plate"].astype(str)) == {"P1", "P2"}
    os.remove(labels + ".features.parquet")
    alone = _plate_run(tmp_path, base, labels, "P2", 4, "plate")
    assert not os.path.isfile(labels + ".features.parquet")
    assert not any("other plates" in n for n in alone["notes"])


def test_two_plates_give_a_plate_held_out_score():
    cells = pd.concat([_cells(seed=1, wells=("A1",), sites=3).assign(plate="P1"),
                       _cells(seed=2, wells=("A1",), sites=3).assign(plate="P2")],
                      ignore_index=True)
    bundle = cc.train_cell_cycle_model(cells, cells["stage"], seed=0, repeats=1, permutations=2)
    held = bundle["card"]["cv"]["plate_held_out"]
    assert held is not None and held["f1"]["metaphase"] > 0.7
    one = cc.train_cell_cycle_model(cells[cells["plate"] == "P1"], cells.loc[
        cells["plate"] == "P1", "stage"], seed=0, repeats=1, permutations=2)
    assert one["card"]["cv"]["plate_held_out"] is None
